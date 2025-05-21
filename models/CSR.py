
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from models.PointWiseFeedForward import PointWiseFeedForward
#from utils import pairwise_cos_distance, set_random_seed
from utils import *

def _l2_norm(x: Tensor, eps: float = 1e-8) -> Tensor:
    """L2‑norm along last dim, keeps shape."""
    return x.norm(p=2, dim=-1, keepdim=True).clamp(min = eps)

class CSR_Attention(nn.Module):
    """Linear Attention head + (history, interests) support.

    Parameters
    ----------
    use_csn : bool, default True
        If True, Cauchy–Schwarz Normalization path is used (NAM). Otherwise
        original dot‑product division path 사용.
    """
    def __init__(self, num_user, in_dim, out_dim, drop_rate, device):
        super(CSR_Attention, self).__init__()
        
        self.device = device
        self.use_csn  = True   # toggled from parent

        self.W_q = nn.Linear(in_dim, out_dim)
        self.W_k = nn.Linear(in_dim, out_dim)
        self.W_v = nn.Linear(in_dim, out_dim)
        self.elu = nn.ELU()
        
        self.z_ln = nn.LayerNorm(out_dim)
        self.s_ln = nn.LayerNorm([out_dim, out_dim])
        self.nam_ln = nn.LayerNorm([out_dim, out_dim]) # NAM: Normalized Attention Memory
        self.na_ln = nn.LayerNorm(out_dim) # NA: Normalized Attention
        self.dropout = nn.Dropout(drop_rate)

        self.register_buffer('prev_z', torch.zeros(num_user + 1, out_dim))
        self.register_buffer('prev_s', torch.zeros(num_user + 1, out_dim, out_dim))
    
    @torch.no_grad()
    def update_history_new_users(self, new_uid: int, nbr_uids: Tensor, w: Tensor):
        """Initialize new user's history as weighted sum of neighbors (w softmax)."""
        self.prev_z[new_uid] = (self.prev_z[nbr_uids] * w.view(-1, 1)).sum(0)
        self.prev_s[new_uid] = (self.prev_s[nbr_uids] * w.view(-1, 1, 1)).sum(0)
        
    def toggle_csn(self, flag: bool):
        self.use_csn = flag

    def get_zs(self, seqs_emb : Tensor, eps = 1e-8):
        """Return cumulative z, s for sequence embedding x (B,L,D)."""
        z = self.elu(self.W_k(seqs_emb))
        s = torch.einsum('abp, abq -> abpq', z, self.W_v(seqs_emb))
        z = torch.cumsum(z, dim = 1) # B x L x D
        s = torch.cumsum(s, dim = 1) # B x L x D x D
        return self.z_ln(z + eps), self.s_ln(s + eps)
    
    def _na_csn(self, norm_q: Tensor, z: Tensor, s: Tensor) -> Tensor:
        nam = s / _l2_norm(z).unsqueeze(-1)
        return torch.einsum('bld, bldp -> blp', norm_q, self.nam_ln(nam + 1e-8)) 

    def _na_dot(self, q: Tensor, z: Tensor, s: Tensor) -> Tensor:
        numer = torch.einsum('bld, bldp -> blp', q, s)
        denom = (q * z).sum(-1, keepdim=True)
        return numer / (denom + 1e-8)
    
    def forward(self, 
                user_id : Tensor, 
                seqs_emb : Tensor,
                use_history = False,
                update_history = False,
                **Prompts
        ) -> tuple[Tensor, Tensor]: # (NA, std)
        
        q = self.elu(self.W_q(seqs_emb)) # q: query (B x L x D)
        z, s = self.get_zs(seqs_emb) # z: Normalizer Memory (B x L x D) and s: Attention Memory (B x L x D x D)
            
        # current Prompt
        if Prompts['P'] is not None:
            P_z, P_s = self.get_zs(Prompts['P'])
            z = z + P_z[:, -1:, :]
            s = s + P_s[:, -1:, :, :]        
        
        # choose NA path
        if self.use_csn:
            norm_q = q / _l2_norm(q).detach()
            na = self._na_csn(norm_q, z, s)
        else:
            na = self._na_dot(q, z, s)   
                 
        # history & historical interest
        if use_history:
            prev_z, prev_s  = self.prev_z[user_id], self.prev_s[user_id]
            if Prompts['S'] is not None:
                S_z, S_s = self.get_zs(Prompts['S'])
                prev_z = prev_z + S_z[:, -1, :]
                prev_s = prev_s + S_s[:, -1, :, :]
            if self.use_csn:
                prev_nam = prev_s / (_l2_norm(prev_z).unsqueeze(-1) + 1e-8)
                prev_na = torch.einsum('blp, bpq -> blq', norm_q, self.nam_ln(prev_nam + 1e-8))
            else:
                numer = torch.einsum('blp, bpq -> blq', q, prev_s)
                denom = (q * prev_z.unsqueeze(1)).sum(dim=-1, keepdim=True)
                prev_na = numer / (denom + 1e-8)
            na = na + self.dropout(prev_na) # cur + prev
        
        na = self.na_ln(na)
        std = torch.std(q / torch.clamp(z, min=1e-6))
        
        if update_history:
            self.prev_z[user_id] += z[:, -1, :]
            self.prev_s[user_id] += s[:, -1, :, :]
        
        return na, std
        
class CSR_Layer(nn.Module):
    def __init__(self, num_user, hidden_dims, head_dims, num_heads, drop_rate, device):
        super(CSR_Layer, self).__init__()
        
        self.init_layer_norm = nn.LayerNorm(hidden_dims)
        self.heads = nn.ModuleList([CSR_Attention(num_user, hidden_dims, head_dims, drop_rate, device) for _ in range(num_heads)])
        self.W_O = nn.Linear(hidden_dims, hidden_dims)
        self.dropout1 = nn.Dropout(drop_rate)
        self.layer_norm1 = nn.LayerNorm(hidden_dims)
        
        self.feed_forward = PointWiseFeedForward(hidden_dims, drop_rate)
        self.dropout2 = nn.Dropout(drop_rate)
        self.layer_norm2 = nn.LayerNorm(hidden_dims)
        self.device = device
    
    @torch.no_grad()
    def update_history_new_users(self, new_uid: int, nbr_uids: Tensor, w: Tensor):
        for h in self.heads: h.update_history_new_users(new_uid, nbr_uids, w)
        
    def toggle_csn(self, flag: bool):
        for h in self.heads: h.toggle_csn(flag)
    
    def forward(self, user_id, seqs_emb, use_history = False, update_history = False, **Prompts):
        
        seqs_emb = self.init_layer_norm(seqs_emb)
        attention_outs = []
        layer_std = torch.tensor(0., device = self.device)
                
        for head in self.heads:
            att_out, head_std = head(user_id, seqs_emb, use_history, update_history, **Prompts)
            attention_outs.append(att_out)
            layer_std += head_std
            
        attention_out  = torch.cat(attention_outs, dim = -1)
        attention_out  = self.W_O(attention_out)
        
        seqs_emb = self.layer_norm1(seqs_emb + self.dropout1(attention_out))
        ff_out   = self.feed_forward(seqs_emb)
        seqs_emb = self.layer_norm2(seqs_emb + self.dropout2(ff_out))
        
        return seqs_emb, layer_std
    
class CSR(nn.Module):
    def __init__(self, num_user, num_item, hidden_dims, num_layer, num_head, drop_rate, window_size, std_lambda, 
                 num_neighbor, temperature, use_current, use_historical, num_C, C_length, num_H, H_length, matching_loss_lambda, device = "cpu"):
        super(CSR, self).__init__()
                        
        head_dims = hidden_dims // num_head
        assert head_dims * num_head == hidden_dims, "hidden_dims must be divisible by num_head"
        
        self.hidden_dims = hidden_dims
        self.device = device
        self.block_id = 0
        self.window_size = window_size
        self.std_lambda = std_lambda
        self.use_history = False
        
        # embeddings & layers ---------------------------------------------------------
        self.item_emb = nn.Embedding(num_item + 1, hidden_dims, padding_idx = 0)
        self.positional_emb = nn.Embedding(window_size, hidden_dims)
        self.emb_dropout = nn.Dropout(drop_rate)        
        self.layers = nn.ModuleList([CSR_Layer(num_user, hidden_dims, head_dims, num_head, drop_rate, device) for _ in range(num_layer)])
        self.apply(self._init_weights) # initalziation should be called in advance before executing various interests setup.
        
        # interests ---------------------------------------------------------------------
        self.use_historical = use_historical          
        self.S_interests, self.S_keys, self.S_ln = self._init_interests_and_keys(num_H + 1, H_length, hidden_dims)
        self.register_buffer('S_table', torch.zeros(num_user + 1))
        
        self.use_current = use_current
        self.P_interests, self.P_keys, self.P_ln = self._init_interests_and_keys(num_C + 1, C_length, hidden_dims)
        self.register_buffer('P_table', torch.zeros(num_user + 1))

        if not use_historical:
            del self.S_interests, self.S_keys
        
        if not use_current:
            del self.P_interests, self.P_keys
        
        self.cos = nn.CosineSimilarity(dim = 1)
        self.matching_loss_lambda = matching_loss_lambda
            
        # update_history_new_users ----------------------------------------------------
        self.num_neighbor = num_neighbor
        self.temperature = temperature

    def toggle_csn(self, flag: bool):
        print(f"[use_csn] {flag}")
        """Enable / disable CSN path in all heads."""
        for lyr in self.layers: lyr.toggle_csn(flag)
    
    def _interests(self, uid: Tensor, use_history: bool):
        if self.block_id == 0: return {"P": None, "S": None}
        S_ids, P_ids = self.S_table[uid].long(), self.P_table[uid].long()
        return {
            "S": self.S_interests[S_ids] if self.use_historical and use_history else None,
            "S_keys": self.S_keys[S_ids] if self.use_historical and use_history else None,
            "P": self.P_interests[P_ids] if self.use_current else None,
            "P_keys": self.P_keys[P_ids] if self.use_current else None
        }

    def _init_weights(self, module):
        """ Initialize the weights """        
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=(1. / module.in_features))
            module.bias.data.zero_()
            
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=(1. / module.embedding_dim))
        
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
        
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_uniform_(module.weight)
            module.bias.data.zero_()
    
    def _init_interests_and_keys(self, num_Crompts, length, hidden_dims):
        p = nn.Parameter(torch.randn(num_Crompts, length, hidden_dims))
        k = nn.Parameter(torch.randn(num_Crompts, hidden_dims))
        ln = nn.LayerNorm(hidden_dims)
        nn.init.normal_(p, mean = 0.0, std = (1. / self.hidden_dims))
        nn.init.normal_(k, mean = 0.0, std = (1. / self.hidden_dims))
        return p, k, ln
    
    def _run_layer(self, user_id, seqs_emb, timeline_mask, use_history, update_history, **Prompts):
        total_std = torch.tensor(0., device = self.device)
        for layer in self.layers:
            seqs_emb, layer_std = layer(user_id, seqs_emb, use_history, update_history, **Prompts)
            if timeline_mask is not None:
                seqs_emb *= timeline_mask
            total_std += layer_std
        return seqs_emb, total_std

    def loss(self, criterion, **batch):
                
        # forward
        pos_logits, neg_logits, std_loss, matching_loss = self.forward(**batch)
        
        # loss
        pos_labels, neg_labels  = torch.ones_like(pos_logits, device = self.device), torch.zeros_like(neg_logits, device = self.device)
        pos_indices = batch['pos_seq'] != 0
        neg_indices = pos_indices.unsqueeze(1).expand_as(neg_logits)
        
        pos_loss = criterion(pos_logits[pos_indices], pos_labels[pos_indices])
        neg_loss = criterion(neg_logits[neg_indices], neg_labels[neg_indices])
        total_loss = pos_loss + neg_loss
        
        # std loss for Cauchy–Schwarz Inequality
        std_loss = self.std_lambda * torch.log(std_loss + 1e-8)
        total_loss += std_loss
        
        # matching_loss for current interests
        if self.block_id >= 1 and self.use_current:
            matching_loss = self.matching_loss_lambda * matching_loss
            total_loss += matching_loss
        
        return {'total_loss' : total_loss, 'pos_loss' : pos_loss, 'neg_loss' : neg_loss, 'std_loss' : std_loss, 'matching_loss' : matching_loss}
    
    def forward(self, **batch):
        
        seqs_feats, std_loss, matching_loss = self.log2feats(batch['user_id'], batch['train_seq'], use_history = self.use_history, update_history = False)
        pos_embs = self.item_emb(batch['pos_seq'])
        neg_embs = self.item_emb(batch['neg_seq'])
        
        pos_logits = (seqs_feats * pos_embs).sum(dim=-1)  # B x L
        neg_logits = (seqs_feats.unsqueeze(1) * neg_embs).sum(dim=-1)  # B x num_neg x L
        
        return pos_logits, neg_logits, std_loss, matching_loss
        
    def log2feats(self, user_id, seqs, use_history, update_history = False):
        
        # hidden states & interests
        seqs_emb = self.item_emb(seqs)
        seqs_emb *= self.item_emb.embedding_dim ** 0.5
        seqs_emb += self.positional_emb(torch.arange(seqs.size(1)).to(self.device))
        timeline_mask = (seqs != 0).unsqueeze(-1)
        seqs_emb = self.emb_dropout(seqs_emb)
        seqs_emb *= timeline_mask
        Prompts = self._interests(user_id, use_history)
        
        # encoding
        feats_main, total_std = self._run_layer(user_id, seqs_emb, timeline_mask, use_history, update_history, **Prompts)
        
        # matching loss
        matching_loss = torch.tensor(0., device=self.device)
        if self.training and self.block_id > 0:
            S_diatances, C_diatances = torch.tensor(0., device=self.device), torch.tensor(0., device=self.device)
            
            if Prompts['S'] is not None:
                Q_H = feats_main[:, -1, :]
                S_diatances = (1 - self.cos(self.S_ln(Q_H), Prompts['S_keys'])).mean()
                
            if Prompts['P'] is not None:
                if self.use_history:
                    feats_other, _ = self._run_layer(user_id, seqs_emb, timeline_mask, not use_history, update_history, **Prompts)
                    Q_C = feats_other[:, -1, :]
                else:
                    Q_C = feats_main[:, -1, :]
                C_diatances = (1 - self.cos(self.P_ln(Q_C), Prompts['P_keys'])).mean()
            
            matching_loss += S_diatances + C_diatances

        return feats_main, total_std, matching_loss
        
    def predict(self, eval_mode, return_score = True, **batch):
        seqs_feats, _, _ = self.log2feats(batch['user_id'], batch['seq'], use_history = self.use_history, update_history = False)
        final_feat = seqs_feats[:, -1, :]
        logits = torch.mm(final_feat, self.item_emb.weight.T) if eval_mode == "Full" else torch.bmm(self.item_emb(batch['target_item']), final_feat.unsqueeze(-1)).squeeze()
        return torch.sigmoid(logits) if return_score else logits
    
    def _interpolate(self, seq_len):
        positional_emb = self.positional_emb.weight.permute(1, 0).unsqueeze(0)
        positional_emb = F.interpolate(positional_emb, size = seq_len, mode = 'linear', align_corners = True)
        return positional_emb.permute(0, 2, 1)
    
    @torch.no_grad()
    def get_user_emb(self, dataloder, use_history = False, update_history = False):
        
        total_user_emb, id2idx, idx2id = list(), dict(), dict()
        Prompts = {'S' : None, 'S_keys' : None, 'P' : None, 'P_keys' : None}
        
        self.eval()
        for idx, (user_id, items) in enumerate(dataloder.dataset.User.items()):
            
            id2idx[user_id], idx2id[idx] = idx, user_id                
            user_id = torch.tensor(user_id).to(self.device).unsqueeze(0)
            seqs_emb = self.item_emb(torch.tensor(items).to(self.device)).unsqueeze(0) # 1 X I x D
            seqs_emb *= self.item_emb.embedding_dim ** 0.5
            
            seq_len = len(items)
            pos_emb = self.positional_emb(torch.arange(seq_len).to(self.device)) if seq_len <= self.window_size else self._interpolate(seq_len)
            seqs_emb += pos_emb
            
            feats_main, _ = self._run_layer(user_id, seqs_emb, None, use_history, update_history, **Prompts)
            total_user_emb.append(feats_main.squeeze()[-1])
                
        return torch.stack(total_user_emb, dim = 0), id2idx, idx2id
    
    @torch.no_grad()
    def update_history(self, prev_dataloader):
        print(f"\t[Update History] self.use_history: {self.use_history}", end = " ")
        _ = self.get_user_emb(prev_dataloader, use_history = False, update_history = True)
        self.use_history = True
        print(f"-> {self.use_history}")
    
    @torch.no_grad()
    def update_history_new_users(self, cur_dataloader, new_user_ids):
        print(f"\t[update_history_new_users] num_neighbor = {self.num_neighbor}, temperature = {self.temperature}\n")
        cur_user_emb, id2idx, idx2id = self.get_user_emb(cur_dataloader, use_history = False)
        scores = torch.mm(cur_user_emb, cur_user_emb.T)
        new_user_idxs = torch.tensor([id2idx[id] for id in new_user_ids])
        scores[:, new_user_idxs] = -torch.inf
        scores.fill_diagonal_(-torch.inf)
        values, neighbor_users = torch.topk(scores, k = self.num_neighbor, dim = -1)
        weights = torch.softmax(values / self.temperature, dim = -1)
    
        for new_user_id in new_user_ids:
            new_user_idx = id2idx[new_user_id]
            neighbor_user_ids = torch.tensor([idx2id[neighbor_user_idx.item()] for neighbor_user_idx in neighbor_users[new_user_idx]]).to(self.device)
            neighbor_weights = weights[new_user_idx]
            for layer in self.layers:
                layer.update_history_new_users(new_user_id, neighbor_user_ids, neighbor_weights)
    
    @torch.no_grad()
    def get_current_interests(self, cur_dataloader):
        print("\t[Get_current_interests]\n")
        cur_user_emb, id2idx, _ = self.get_user_emb(cur_dataloader, use_history = False)
        cos_distance = pairwise_cos_distance(self.P_ln(cur_user_emb), self.P_keys)
        cos_distance[:, 0] = float('inf') # padding
        _, interest_ids = torch.topk(cos_distance, k = 1, largest = False)
                        
        for i, user_id in enumerate(id2idx.keys()):
            self.P_table[user_id] = interest_ids[i].long()
    
    @torch.no_grad()
    def get_historical_interests(self, cur_dataloader):
        print("\t[Get_historical_interests]\n")
        cur_user_emb, id2idx, _ = self.get_user_emb(cur_dataloader, use_history = True)
        cos_distance = pairwise_cos_distance(self.S_ln(cur_user_emb), self.S_keys)
        cos_distance[:, 0] = float('inf') # padding
        _, interest_ids = torch.topk(cos_distance, k = 1, largest = False)
                
        for i, user_id in enumerate(id2idx.keys()):
            self.S_table[user_id] = interest_ids[i].long()
            

# python -u main_eval.py --d yelp --m CSR --use_csn --us --up --ad_update --update --lr 0.001 --reg 1e-4 --dr 0.1 --CSR_nei 25 --CSR_T 0.5 --num_C 50 --P_l 10 --num_H 30 --S_l 50 --mll 1e-3 --eval_RA --rs 0
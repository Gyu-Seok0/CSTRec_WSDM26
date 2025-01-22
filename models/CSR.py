from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.PointWiseFeedForward import PointWiseFeedForward
from utils import pairwise_cos_distance, set_random_seed

class CSR_Attention(nn.Module):
    def __init__(self, num_user, in_dims, out_dims, drop_rate, device, block_id
                 ):
        super(CSR_Attention, self).__init__()
        
        self.block_id = block_id
        self.device = device

        self.W_q = nn.Linear(in_dims, out_dims)
        self.W_k = nn.Linear(in_dims, out_dims)
        self.W_v = nn.Linear(in_dims, out_dims)
        self.elu = nn.ELU()
        
        self.z_layernorm = nn.LayerNorm(out_dims)
        self.s_layernorm = nn.LayerNorm([out_dims, out_dims])
        self.NAM_layernorm = nn.LayerNorm([out_dims, out_dims])
        self.NA_layernorm = nn.LayerNorm(out_dims)
        
        # Adaptively utilizing historical knowledge
        self.fc1 = nn.Linear(out_dims, out_dims)
        self.fc2 = nn.Linear(out_dims, out_dims)
        self.dropout = nn.Dropout(drop_rate)
        
        # self.prev_z = torch.zeros(num_user + 1, out_dims).to(device).detach()
        # self.prev_s = torch.zeros(num_user + 1, out_dims, out_dims).to(device).detach()
        
        self.register_buffer('prev_z', torch.zeros(num_user + 1, out_dims))
        self.register_buffer('prev_s', torch.zeros(num_user + 1, out_dims, out_dims))

        self.dev = device

    def get_L2_norm_sqrt(self, x):
        return torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True))

    def get_zs(self, seqs_emb):
        
        # Linear Attention
        z_ = self.elu(self.W_k(seqs_emb))
        s_ = torch.einsum('abp, abq -> abpq', z_, self.W_v(seqs_emb))
        
        z_ = torch.cumsum(z_, dim = 1) # B x L x D
        s_ = torch.cumsum(s_, dim = 1) # B x L x D x D
        
        z_ = self.z_layernorm(z_ + 1e-8)
        s_ = self.s_layernorm(s_ + 1e-8)
    
        return z_, s_
    
    def forward(self, user_id, seqs_emb, use_history = False, update_history = False, **Prompts):
        
        # if self.block_id == 2:
        #     print("self.block_id = 2")
        #     import pdb; pdb.set_trace()
        
        # query
        q_ = self.elu(self.W_q(seqs_emb)) # B x L x D
        normalized_q = q_ / (self.get_L2_norm_sqrt(q_) + 1e-8).detach()
        
        # Normalizer Memory and Attention Memory
        z_, s_ = self.get_zs(seqs_emb)
                
        # Plasticity Prompt
        if Prompts['P'] is not None:
            P_z, P_s = self.get_zs(Prompts['P'])
            z_ += P_z[:, -1:, :]
            s_ += P_s[:, -1:, :, :]
        
        # Normalized Attention Memory (NAM)
        NAM = s_ / (self.get_L2_norm_sqrt(z_).unsqueeze(-1) + 1e-8).detach()
        NAM = self.NAM_layernorm(NAM + 1e-8)
        NA = torch.einsum('blp, blpq -> blq', normalized_q, NAM)
        
        if use_history:
            prev_z, prev_s  = self.prev_z[user_id], self.prev_s[user_id]
            user_history_mask = (prev_z.sum(dim=-1) != 0)
            
            # # Stability Prompt
            if Prompts['S'] is not None:
                S_z, S_s = self.get_zs(Prompts['S'])
                z_updated = prev_z + S_z[:, -1, :]
                s_updated = prev_s + S_s[:, -1, :, :]
            else:
                z_updated = prev_z
                s_updated = prev_s
            
            prev_NAM = s_updated / (self.get_L2_norm_sqrt(z_updated).unsqueeze(-1) + 1e-8).detach()
            #prev_NAM = self.NAM_layernorm(prev_NAM + 1e-8)
            #prev_NAM = self.fc1(prev_NAM)
            prev_NA = torch.einsum('blp, bpq -> blq', normalized_q, prev_NAM)
            #prev_NA = self.fc2(prev_NA)
            
            if user_history_mask.any(): # --update but --no-ad_update
                prev_NA *= (user_history_mask).view(-1, 1, 1).float()
            
            NA = NA + self.dropout(prev_NA)
                
        NA = self.NA_layernorm(NA)
        z_clamped = torch.clamp(z_, min=1e-6)  # z_ 값이 최소 1e-6 이하로 떨어지지 않도록 제한
        std = torch.std(q_ / z_clamped)
        
        # update history for exisiting users
        if update_history:
            self.prev_z[user_id] += z_[:, -1, :]
            self.prev_s[user_id] += s_[:, -1, :, :]
        
        return NA, std
    
    # def forward(self, user_id, seqs_emb, use_history = False, update_history = False, **Prompts):
        
    #     # query
    #     q_ = self.elu(self.W_q(seqs_emb)) # B x L x D
    #     normalized_q = q_ / (self.get_L2_norm_sqrt(q_) + 1e-8).detach()
        
    #     # Normalized Attention Memory (NAM)
    #     z_, s_ = self.get_zs(seqs_emb)
        
    #     # update history for exisiting users
    #     if update_history:
    #         if use_history:
    #             self.prev_z[user_id] = z_[:, -1, :]
    #             self.prev_s[user_id] = s_[:, -1, :, :]
    #         else:
    #             self.prev_z[user_id] += z_[:, -1, :]
    #             self.prev_s[user_id] += s_[:, -1, :, :]
                
    #     # Plasticity Prompt
    #     if Prompts['P'] is not None:
    #         P_z, P_s = self.get_zs(Prompts['P'])
    #         z_ += P_z[:, -1:, :]
    #         s_ += P_s[:, -1:, :, :]
            
    #     # 추가
    #     if use_history:
    #         prev_z, prev_s = self.prev_z[user_id], self.prev_s[user_id]
    #         z_ += prev_z.unsqueeze(1)
    #         s_ += prev_s.unsqueeze(1)
        
    #     NAM = s_ / (self.get_L2_norm_sqrt(z_).unsqueeze(-1) + 1e-8).detach()
    #     NAM = self.NAM_layernorm(NAM + 1e-8)
        
    #     # Normalized Attention (NA)
    #     NA = torch.einsum('blp, blpq -> blq', normalized_q, NAM)
        
    #     if use_history:
    #         prev_z, prev_s  = self.prev_z[user_id], self.prev_s[user_id]
    #         user_history_mask = (prev_z.sum(dim=-1) != 0)
            
    #         # # Stability Prompt
    #         if Prompts['S'] is not None:
    #             S_z, S_s = self.get_zs(Prompts['S'])
                
    #             z_updated = prev_z + S_z[:, -1, :]
    #             s_updated = prev_s + S_s[:, -1, :, :]
    #         else:
    #             z_updated = prev_z
    #             s_updated = prev_s
            
    #         prev_NAM = s_updated / (self.get_L2_norm_sqrt(z_updated).unsqueeze(-1) + 1e-8).detach()
    #         prev_NAM = self.NAM_layernorm(prev_NAM + 1e-8)
    #         prev_NAM = self.fc1(prev_NAM)
    #         prev_NA = torch.einsum('blp, bpq -> blq', normalized_q, prev_NAM)
    #         prev_NA = self.fc2(prev_NA)
            
    #         if user_history_mask.any(): # --update but --no-ad_update
    #             prev_NA *= (user_history_mask).view(-1, 1, 1).float()
            
    #         NA = NA + self.dropout(prev_NA)
                
    #     NA = self.NA_layernorm(NA)
    #     std = torch.std(q_ / (z_ + 1e-8))
        
    #     if torch.isnan(NA).any():
    #         import pdb; pdb.set_trace()
        
    #     return NA, std
     
    def update_hisotry_new_users(self, new_user_id, neighbor_user_ids, neighbor_weights):
        self.prev_z[new_user_id] = (self.prev_z[neighbor_user_ids] * neighbor_weights.view(-1, 1)).sum(0)
        self.prev_s[new_user_id] = (self.prev_s[neighbor_user_ids] * neighbor_weights.view(-1, 1, 1)).sum(0)
        
class CSR_Layer(nn.Module):
    def __init__(self, num_user, hidden_dims, head_dims, num_heads, drop_rate, device, block_id):
        super(CSR_Layer, self).__init__()
        
        self.block_id = block_id
        self.init_layer_norm = nn.LayerNorm(hidden_dims)
        self.attention_heads = nn.ModuleList([CSR_Attention(num_user, hidden_dims, head_dims, drop_rate, device, block_id) for _ in range(num_heads)])
        self.W_O = nn.Linear(hidden_dims, hidden_dims)
        self.dropout1 = nn.Dropout(drop_rate)
        self.layer_norm1 = nn.LayerNorm(hidden_dims)
        
        self.feed_forward = PointWiseFeedForward(hidden_dims, drop_rate)
        self.dropout2 = nn.Dropout(drop_rate)
        self.layer_norm2 = nn.LayerNorm(hidden_dims)
        self.dev = device
    
    def forward(self, user_id, seqs_emb, use_history = False, update_history = False, **Prompts):
        
        seqs_emb = self.init_layer_norm(seqs_emb)
        attention_outs = []
        layer_std = 0.
        
        for head in self.attention_heads:
            att_out, head_std = head(user_id, seqs_emb, use_history, update_history, **Prompts)
            attention_outs.append(att_out)
            layer_std += head_std
            
        attention_out  = torch.concat(attention_outs, dim = -1)
        attention_out  = self.W_O(attention_out)
        
        seqs_emb = self.layer_norm1(seqs_emb + self.dropout1(attention_out))
        ff_out   = self.feed_forward(seqs_emb)
        seqs_emb = self.layer_norm2(seqs_emb + self.dropout2(ff_out))
        
        return seqs_emb, layer_std
    
    def update_hisotry_new_users(self, new_user_id, neighbor_user_ids, neighbor_weights):
        for head in self.attention_heads:
            head.update_hisotry_new_users(new_user_id, neighbor_user_ids, neighbor_weights)
            
class CSR(nn.Module):
    def __init__(self, num_user, num_item, hidden_dims, num_layer, num_head, drop_rate, window_size, std_lambda, 
                 num_neighbor, temperature, use_plasticity, use_stability, num_P, P_length, num_S, S_length, matching_loss_lambda, 
                 DPA = False, device = "cpu"):
        super(CSR, self).__init__()
                        
        head_dims = hidden_dims // num_head
        assert head_dims * num_head == hidden_dims, "hidden_dims must be divisible by num_head"
        
        self.hidden_dims = hidden_dims
        self.head_dims = head_dims
        self.num_head = num_head

        self.num_user = num_user
        self.num_item = num_item
        
        self.item_emb = nn.Embedding(num_item + 1, hidden_dims, padding_idx = 0)
        self.positional_emb = nn.Embedding(window_size, hidden_dims)
        self.emb_dropout = nn.Dropout(drop_rate)
        
        self.block_id = 0
        
        self.layers = nn.ModuleList([
            CSR_Layer(num_user, hidden_dims, head_dims, num_head, drop_rate, device, self.block_id)
            for _ in range(num_layer)
        ])
        
        self.window_size = window_size
        self.std_lambda = std_lambda
        self.use_history = False
        self.dev = device
        
        # update_hisotry_new_users
        self.num_neighbor = num_neighbor
        self.temperature = temperature
        
        # weight init
        self.apply(self._init_weights) # initalziation should be called in advance before executing various prompts setup.
        
        # prompts
        self.use_plasticity = use_plasticity
        self.use_stability = use_stability
        self.DPA = DPA # Deterministic Prompt Assignment
                
        self.S_prompts = nn.Parameter(torch.randn(num_S, S_length, hidden_dims))
        self.S_keys = nn.Parameter(torch.randn(num_S, hidden_dims))
        #self.S_table = torch.zeros(num_user + 1).to(device).detach()
        self.register_buffer('S_table', torch.zeros(num_user + 1))
        self.initialize_prompts_and_keys(self.S_prompts, self.S_keys)
        
        if DPA:
            num_P = 4 # total blocks( = total tasks)
        
        self.P_prompts = nn.Parameter(torch.randn(num_P, P_length, hidden_dims))
        self.P_keys = nn.Parameter(torch.randn(num_P, hidden_dims))
        #self.P_table = torch.zeros(num_user + 1).to(device).detach()
        self.register_buffer('P_table', torch.zeros(num_user + 1))
        self.initialize_prompts_and_keys(self.P_prompts, self.P_keys)
        
        self.cos = nn.CosineSimilarity(dim = 1)
        self.matching_loss_lambda = matching_loss_lambda
        
        if not use_stability:
            del self.S_prompts, self.S_keys, self.S_table
        
        if not use_plasticity:
            del self.P_prompts, self.P_keys, self.P_table
            
    def update_block_id(self, block_id):
        print("[update_block_id]", block_id)
        
        self.block_id = block_id
        for layer in self.layers:
            layer.block_id = block_id
            
            for head in layer.attention_heads:
                head.block_id = block_id
                

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
    
    def initialize_prompts_and_keys(self, prompts, keys):
        nn.init.normal_(prompts, mean = 0.0, std = (1. / self.hidden_dims))
        nn.init.normal_(keys, mean = 0.0, std = (1. / self.hidden_dims))

    def loss(self, criterion, **batch):
                
        # forward
        pos_logits, neg_logits, std_loss, matching_loss = self.forward(**batch)
        
        # loss
        pos_labels, neg_labels  = torch.ones_like(pos_logits, device = self.dev), torch.zeros_like(neg_logits, device = self.dev)
        pos_indices = batch['pos_seq'] != 0
        neg_indices = pos_indices.unsqueeze(1).expand_as(neg_logits)
        
        pos_loss = criterion(pos_logits[pos_indices], pos_labels[pos_indices])
        neg_loss = criterion(neg_logits[neg_indices], neg_labels[neg_indices])
        total_loss = pos_loss + neg_loss
        
        # std loss for Cauchy–Schwarz Inequality
        std_loss = self.std_lambda * torch.log(std_loss + 1e-8)
        total_loss += std_loss
        
        # matching_loss for plasticity prompts
        if self.block_id >= 1 and self.use_plasticity:
            matching_loss = self.matching_loss_lambda * matching_loss
            total_loss += matching_loss
        
        return {'total_loss' : total_loss, 'pos_loss' : pos_loss, 'neg_loss' : neg_loss, 'std_loss' : std_loss, 'matching_loss' : matching_loss}
    
    def forward(self, **batch):
        
        seqs_feats, std_loss, matching_loss = self.log2feats(batch['user_id'], batch['train_seq'], self.use_history, train = True)
        pos_embs = self.item_emb(batch['pos_seq'])
        neg_embs = self.item_emb(batch['neg_seq'])
        
        pos_logits = (seqs_feats * pos_embs).sum(dim=-1)  # B x L
        neg_logits = (seqs_feats.unsqueeze(1) * neg_embs).sum(dim=-1)  # B x num_neg x L
        
        return pos_logits, neg_logits, std_loss, matching_loss
        
    def log2feats(self, user_id, seqs, use_history, update_history = False, train = True):
        Prompts = {'S' : None, 'P' : None}
                
        if self.block_id >= 1:
            if use_history and self.use_stability:
                s_prompt_ids = self.S_table[user_id].long()
                Prompts['S'] = self.S_prompts[s_prompt_ids]
                
            if self.use_plasticity:
                if self.DPA and train:
                    p_prompt_id = self.block_id - 1
                    p_prompt_ids = torch.full((seqs.size(0), ), p_prompt_id, dtype=torch.long, device=self.dev)
                    Prompts['P'] = self.P_prompts[p_prompt_id].unsqueeze(0)
                else:
                    p_prompt_ids = self.P_table[user_id].long()
                    Prompts['P'] = self.P_prompts[p_prompt_ids]
        
        seqs_emb = self.item_emb(seqs)
        seqs_emb *= self.item_emb.embedding_dim ** 0.5
        seqs_emb += self.positional_emb(torch.arange(seqs.size(1)).to(self.dev))
        timeline_mask = (seqs != 0).unsqueeze(-1)

        seqs_emb = self.emb_dropout(seqs_emb)
        seqs_emb *= timeline_mask
        
        total_std = torch.tensor(0., device=self.dev)
        for layer in self.layers:
            seqs_emb, layer_std = layer(user_id, seqs_emb, use_history, update_history, **Prompts)
            seqs_emb *= timeline_mask
            if train:
                total_std += layer_std
        
        matching_loss = torch.tensor(0., device=self.dev)
        if train and self.block_id >=1:
            if Prompts['P'] is not None:
                distances = 1 - self.cos(seqs_emb[:, -1, :], 
                                         self.P_keys[p_prompt_ids])
                matching_loss += distances.mean()
            if Prompts['S'] is not None:
                distances = 1 - self.cos(seqs_emb[:, -1, :], 
                                         self.S_keys[s_prompt_ids])
                matching_loss += distances.mean()
        
        return seqs_emb, total_std, matching_loss
        
    def predict(self, eval_mode, return_score = True, **batch):
                
        seqs_feats, _, _ = self.log2feats(batch['user_id'], batch['seq'], self.use_history, train = False)
        final_feat = seqs_feats[:, -1, :]
        
        if eval_mode == "Full":
            logits = torch.mm(final_feat, self.item_emb.weight.T) # batch_size x |I| 
        elif eval_mode == "LOO":
            logits = torch.bmm(self.item_emb(batch['target_item']), final_feat.unsqueeze(-1)).squeeze() # batch_size x (1 pos + N neg)
        
        if return_score:
            return torch.sigmoid(logits)
        return logits
    
    def get_user_emb(self, dataloder, use_history = False, update_history = False):
        
        total_user_emb = list()
        id2idx = dict()
        idx2id = dict()
        Prompts = {'S' : None, 'P' : None}
        
        self.eval()
        with torch.no_grad():
            for idx, (user_id, items) in enumerate(dataloder.dataset.User.items()):
                
                id2idx[user_id] = idx
                idx2id[idx] = user_id
                
                user_id = torch.tensor(user_id).to(self.dev).unsqueeze(0)
                seqs_emb = self.item_emb(torch.tensor(items).to(self.dev)).unsqueeze(0) # 1 X I x D
                seqs_emb *= self.item_emb.embedding_dim ** 0.5
                
                seq_len = len(items)
                if seq_len <= self.window_size:
                    seqs_emb += self.positional_emb(torch.arange(seq_len).to(self.dev))
                else:
                    positional_emb = self.positional_emb.weight.permute(1, 0).unsqueeze(0) # 1 x D x W
                    positional_emb = F.interpolate(positional_emb, size = seq_len, mode = 'linear', align_corners = True)
                    seqs_emb += positional_emb.permute(0, 2, 1) # 1 x L x D
                
                for layer in self.layers:
                    seqs_emb, _ = layer(user_id, seqs_emb, use_history, update_history, **Prompts)
                total_user_emb.append(seqs_emb.squeeze()[-1])
                
        return torch.stack(total_user_emb, dim = 0), id2idx, idx2id
    
    def update_history(self, prev_dataloader):
        print(f"\t[Update History] self.use_history: {self.use_history}", end = " ")
        _ = self.get_user_emb(prev_dataloader, use_history = False, update_history = True)
        self.use_history = True
        print(f"-> {self.use_history}")
        
    def update_hisotry_new_users(self, cur_dataloader, new_user_ids):
        print(f"\t[Update_hisotry_new_users] num_neighbor = {self.num_neighbor}, temperature = {self.temperature}")
        cur_user_emb, id2idx, idx2id = self.get_user_emb(cur_dataloader, use_history = False)
        scores = torch.mm(cur_user_emb, cur_user_emb.T)
        
        new_user_idxs = torch.tensor([id2idx[id] for id in new_user_ids])
        scores[:, new_user_idxs] = -torch.inf
        scores.fill_diagonal_(-torch.inf)
        values, neighbor_users = torch.topk(scores, k = self.num_neighbor, dim = -1)
        
        weights = torch.softmax(values / self.temperature, dim = -1)
    
        for new_user_id in new_user_ids:
            new_user_idx = id2idx[new_user_id]
            neighbor_user_ids = torch.tensor([idx2id[neighbor_user_idx.item()] 
                                              for neighbor_user_idx in neighbor_users[new_user_idx]]).to(self.dev)
            neighbor_weights = weights[new_user_idx]
            
            for layer in self.layers:
                layer.update_hisotry_new_users(new_user_id, neighbor_user_ids, neighbor_weights)
                
    def get_plasticity_prompts(self, cur_dataloader):
        print("\t[Get_plasticity_prompts]\n")
        cur_user_emb, id2idx, _ = self.get_user_emb(cur_dataloader, use_history = False)
        cos_distance = pairwise_cos_distance(cur_user_emb, self.P_keys)
        _, prompt_ids = torch.topk(cos_distance, k = 1, largest = False)
                
        for i, user_id in enumerate(id2idx.keys()):
            self.P_table[user_id] = prompt_ids[i].long()
    
    def get_stability_prompts(self, cur_dataloader):
        print("\t[Get_stability_prompts]\n")
        cur_user_emb, id2idx, _ = self.get_user_emb(cur_dataloader, use_history = True)
        cos_distance = pairwise_cos_distance(cur_user_emb, self.S_keys)
        _, prompt_ids = torch.topk(cos_distance, k = 1, largest = False)
                
        for i, user_id in enumerate(id2idx.keys()):
            self.S_table[user_id] = prompt_ids[i].long()
# https://github.com/pmixer/SASRec.pytorch/blob/master/model.py
import numpy as np
import torch
import torch.nn as nn
from models.PointWiseFeedForward import PointWiseFeedForward


class SASREC(nn.Module):
    def __init__(self, user_num, item_num, device, args):
        super(SASREC, self).__init__()
        
        self.user_num = user_num
        self.item_num = item_num
        self.hidden_dims = args.hidden_dims
        self.dev = device
        
        self.item_emb = nn.Embedding(self.item_num + 1, self.hidden_dims, padding_idx = 0)
        self.positional_emb = nn.Embedding(args.window_size, self.hidden_dims)
        self.emb_dropout = nn.Dropout(p = args.dropout_rate)
        
        self.attention_layernorms = nn.ModuleList() # to be Q for self-attention
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        
        self.last_layernorm = nn.LayerNorm(self.hidden_dims, eps=1e-8)
        
        for _ in range(args.num_layers):
            new_attn_layernorm = nn.LayerNorm(self.hidden_dims, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  nn.MultiheadAttention(self.hidden_dims,
                                                    args.num_heads,
                                                    args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = nn.LayerNorm(self.hidden_dims, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(self.hidden_dims, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)
            
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean = 0.0, std = (1. / self.hidden_dims))
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
                
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
        
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_uniform_(module.weight)
            module.bias.data.zero_()

    def log2feats(self, train_seqs):
        seqs_emb = self.item_emb(train_seqs)
        seqs_emb *= self.item_emb.embedding_dim ** 0.5
        seqs_emb += self.positional_emb(torch.arange(train_seqs.size(1)).to(self.dev))
        seqs_emb = self.emb_dropout(seqs_emb)

        timeline_mask = (train_seqs != 0).unsqueeze(-1)
        seqs_emb *= timeline_mask # broadcast in last dim

        tl = seqs_emb.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs_emb = torch.transpose(seqs_emb, 0, 1)
            Q = self.attention_layernorms[i](seqs_emb)
            mha_outputs, _ = self.attention_layers[i](Q, seqs_emb, seqs_emb, 
                                            attn_mask = attention_mask)
            
            seqs_emb = Q + mha_outputs
            seqs_emb = torch.transpose(seqs_emb, 0, 1)

            seqs_emb = self.forward_layernorms[i](seqs_emb)
            seqs_emb = self.forward_layers[i](seqs_emb)
            seqs_emb *= timeline_mask

        log_feats = self.last_layernorm(seqs_emb)

        return log_feats

    def loss(self, criterion, **batch):
        
        # forward
        pos_logits, neg_logits = self.forward(batch['train_seq'], batch['pos_seq'], batch['neg_seq'])
        
        # loss
        pos_labels, neg_labels = torch.ones_like(pos_logits, device = self.dev), torch.zeros_like(neg_logits, device = self.dev)
        pos_indices = batch['pos_seq'] != 0
        neg_indices = pos_indices.unsqueeze(1).expand_as(neg_logits)
        
        pos_loss = criterion(pos_logits[pos_indices], pos_labels[pos_indices])
        neg_loss = criterion(neg_logits[neg_indices], neg_labels[neg_indices])
        total_loss = pos_loss + neg_loss
        
        return {'total_loss' : total_loss, 'pos_loss' : pos_loss, 'neg_loss' : neg_loss}
        
    def forward(self, train_seqs, pos_seqs, neg_seqs):
        train_seqs_feats = self.log2feats(train_seqs)

        pos_embs = self.item_emb(pos_seqs)
        neg_embs = self.item_emb(neg_seqs)

        pos_logits = (train_seqs_feats * pos_embs).sum(dim=-1) # B x L
        neg_logits = (train_seqs_feats.unsqueeze(1) * neg_embs).sum(dim=-1) # B X num_neg x L
        
        return pos_logits, neg_logits
    
    def predict(self, eval_mode, return_score = False, **batch):
        
        seqs_feats = self.log2feats(batch['seq'])
        final_feat = seqs_feats[:, -1, :] # batch_size x hidden_dims
        
        if eval_mode == "Full":
            logits = torch.mm(final_feat, self.item_emb.weight.T) # batch_size x |I| 
        elif eval_mode == "LOO":
            logits = torch.bmm(self.item_emb(batch['target_item']), final_feat.unsqueeze(-1)).squeeze() # target_item: batch_size x (1 pos + N neg)
        
        if return_score:
            return torch.sigmoid(logits)
        return logits

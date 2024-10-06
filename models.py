import torch
import torch.nn as nn
import numpy as np
from InfoNCE import InfoNCE


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate,activation='relu'):
        super(PointWiseFeedForward, self).__init__()
        act = None
        if activation == 'relu':
            act = torch.nn.ReLU()
        elif activation == 'gelu':
            act = torch.nn.GELU()
        self.pwff = torch.nn.Sequential(
            torch.nn.Linear(hidden_units, hidden_units),
            act,
            torch.nn.Linear(hidden_units, hidden_units),
            torch.nn.Dropout(p=dropout_rate)
        )

    def forward(self, inputs):
        outputs = self.pwff(inputs)
        outputs = outputs + inputs
        return outputs


class Model(nn.Module):
    def __init__(self, args, word_item_emb):
        super(Model, self).__init__()
        self.config = args
        self.item_maxid = args.num_items
        self.emb_size = args.embed_dim
        self.block_num = args.n_blocks
        self.head_num = args.n_heads
        self.drop_rate = args.drop_rate
        self.max_len = args.max_len
        self.rec_loss = torch.nn.BCEWithLogitsLoss()
        self._init_model()

        self.word_item_emb = nn.Embedding.from_pretrained(word_item_emb.data, freeze=True)
        self.transfer_layer = nn.Linear(word_item_emb.data.size(-1), self.emb_size)

        self.infonce = InfoNCE(negative_mode="paired")
        self.seq_transfer_layer = nn.Linear(word_item_emb.data.size(-1), self.emb_size)

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        self.item_emb = nn.Parameter(initializer(torch.empty(self.item_maxid, self.emb_size)))
        self.pos_emb = nn.Parameter(initializer(torch.empty(self.max_len+1, self.emb_size)))
        self.attention_layer_norms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layer_norms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.emb_dropout = torch.nn.Dropout(self.drop_rate)
        self.last_layer_norm = torch.nn.LayerNorm(self.emb_size, eps=1e-8)

        for n in range(self.block_num):
            self.attention_layer_norms.append(torch.nn.LayerNorm(self.emb_size, eps=1e-8))
            new_attn_layer =  torch.nn.MultiheadAttention(self.emb_size, self.head_num, self.drop_rate)
            self.attention_layers.append(new_attn_layer)
            self.forward_layer_norms.append(torch.nn.LayerNorm(self.emb_size, eps=1e-8))
            new_fwd_layer = PointWiseFeedForward(self.emb_size, self.drop_rate)
            self.forward_layers.append(new_fwd_layer)
    def set_word_item_embed(self, word_item_embed):
        self.word_item_emb = word_item_embed.clone().detach().requires_grad_(False)

    def forward(self, seq, pos, seq_emb=None):
        if seq_emb is None:
            seq_emb = self.item_emb[seq]
            word_emb = self.transfer_layer(self.word_item_emb(seq))
            seq_emb = seq_emb + word_emb
            seq_emb *= self.emb_size ** 0.5
        pos_emb = self.pos_emb[pos]
        seq_emb = seq_emb + pos_emb
        seq_emb = self.emb_dropout(seq_emb)

        timeline_mask = torch.BoolTensor(pos.cpu() == 0).to(self.config.device_id)
        seq_emb *= ~timeline_mask.unsqueeze(-1)
        tl = seq_emb.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool).to(self.config.device_id))
        for i in range(len(self.attention_layers)):
            seq_emb = torch.transpose(seq_emb, 0, 1)
            normalized_emb = self.attention_layer_norms[i](seq_emb)
            mha_outputs, _ = self.attention_layers[i](normalized_emb, seq_emb, seq_emb, attn_mask=attention_mask)
            seq_emb = normalized_emb + mha_outputs
            seq_emb = torch.transpose(seq_emb, 0, 1)
            seq_emb = self.forward_layer_norms[i](seq_emb)
            seq_emb = self.forward_layers[i](seq_emb)
            seq_emb *=  ~timeline_mask.unsqueeze(-1)
        seq_emb = self.last_layer_norm(seq_emb)
        return seq_emb

    def u_contrastive_loss(self, seq_emb, llm_seq_emb, llm_seq_neg_emb):
        llm_seq_emb = self.seq_transfer_layer(llm_seq_emb)
        llm_seq_neg_emb = self.seq_transfer_layer(llm_seq_neg_emb)
        llm_seq_neg_emb = llm_seq_neg_emb.unsqueeze(1)
        last_item_embeddings = seq_emb[:, -1, :]
        loss = self.infonce(last_item_embeddings, llm_seq_emb, llm_seq_neg_emb)
        return loss

    def loss_function(self, seq_emb, y, neg, pos):
        y_emb = self.item_emb[y]
        neg_emb = self.item_emb[neg]
        y_emb = y_emb + self.transfer_layer(self.word_item_emb(y))
        neg_emb = neg_emb + self.transfer_layer(self.word_item_emb(neg))
        pos_logits = (seq_emb * y_emb).sum(dim=-1)
        tmp_seq_emb = seq_emb.unsqueeze(dim=2)
        neg_logits = (tmp_seq_emb * neg_emb).sum(dim=-1)
        pos_labels, neg_labels = torch.ones(pos_logits.shape).to(self.config.device_id), torch.zeros(neg_logits.shape).to(self.config.device_id)
        indices = np.where(pos.cpu() != 0)
        loss = self.rec_loss(pos_logits[indices], pos_labels[indices])
        loss += self.rec_loss(neg_logits[indices], neg_labels[indices])
        return loss
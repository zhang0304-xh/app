# coding:utf-8
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from common.model.torch_crf import CRF
from common.layers.dynamic_rnn import DynamicLSTM

import numpy as np


class Joint_model_Soft_Interaction(nn.Module):
    def __init__(self, args, hidden_dim, batch_size, max_length, n_class, n_tag, embedding_matrix):
        super(Joint_model_Soft_Interaction, self).__init__()
        self.args = args
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.max_length = max_length
        self.n_class = n_class
        self.n_tag = n_tag
        self.LayerNorm = LayerNorm(self.hidden_dim, eps=1e-12)
        self.emb_drop = nn.Dropout(self.args.emb_dorpout)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), padding_idx=0)
        self.embed.weight.requires_grad = True

        self.fc_drop = nn.Dropout(self.args.fc_dropout)

        self.biLSTM = DynamicLSTM(self.args, self.args.emb_dim, self.args.hidden_dim // 2, bidirectional=True,
                                  batch_first=True,
                                  dropout=self.args.lstm_dropout, num_layers=1)

        # self.slot_gru = DynamicLSTM(self.args.emb_dim, self.args.hidden_dim, bidirectional=True, batch_first=True,
        #                             dropout=self.args.lstm_dropout, num_layers=1, rnn_type='GRU')
        # self.biLSTM = DynamicLSTM(config.emb_dim, config.hidden_dim // 2, bidirectional=True, batch_first=True,
        #                           dropout=config.lstm_dropout, num_layers=1, rnn_type='GRU')

        # Initialize an self-attention layer.
        self.__attention = SelfAttention(
            self.args.emb_dim,
            self.args.emb_dim,
            self.args.attention_output_dim,
            self.args.attention_dropout
        )

        self.__sentattention = UnflatSelfAttention(
            self.args.hidden_dim + self.args.attention_output_dim,
            self.args.attention_dropout
        )

        self.intent_embeddings = nn.Parameter(torch.rand([self.n_class, self.args.intent_dim]))
        # nn.init.xavier_uniform_(self.intent_embeddings.data)
        self.slot_embeding = nn.Embedding(self.n_tag, self.args.slot_dim)  # (relation,dim)(19,768)
        self.relation = nn.Linear(self.args.slot_dim, self.args.slot_dim)  # (768,768)

        self.intent_fc = nn.Linear(2 * self.hidden_dim, self.n_class)
        self.intent_to_slot = nn.Linear(self.hidden_dim, 2 * self.hidden_dim)
        self.intent_slot_attention = Intent_Slot_Attention(2 * self.hidden_dim)
        self.slot_decoder = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.n_tag)
        )
        self.slot_reduce = nn.Linear(3 * self.hidden_dim, self.hidden_dim)

        self.crflayer = CRF(self.n_tag)

        self.intent_criterion = nn.CrossEntropyLoss()
        self.slot_criterion = nn.CrossEntropyLoss()

        self.down = nn.Linear(2 * self.hidden_dim, self.hidden_dim)  # (3*768,768)
        self.start_tail = nn.Linear(self.hidden_dim, 1)  # (768,1)

        self.layers = nn.ModuleList([GATLayer(self.args.hidden_dim) for _ in range(self.args.gat_layers)])

    def forward_logit(self, x, mask):
        x, x_char = x

        seq_len = torch.sum(x != 0, dim=-1)  # (batch_size,seq_len)-->(batch_size)

        x_emb = self.emb_drop(self.embed(x))  # (batch_szie,seq_len,dim)

        lstm_hiddens, (_, _) = self.biLSTM(x_emb, seq_len.cpu())  # (batch_size,seq_len,dim)

        attention_hiddens = self.__attention(x_emb, seq_len)

        share_features = torch.cat([attention_hiddens, lstm_hiddens], dim=2)

        intent_context = self.__sentattention(share_features, seq_len)  # (batch, dim)

        logits_intent = self.intent_fc(intent_context)

        intent_weights = F.softmax(logits_intent)

        intent_weights = intent_weights.unsqueeze(-1).expand(-1, -1, self.intent_embeddings.size(-1))

        intent_slot_inputs = intent_weights * self.intent_embeddings.unsqueeze(0).expand(intent_weights.size(0), -1, -1)

        intent_slot_inputs = self.intent_to_slot(self.fc_drop(intent_slot_inputs))
        intent_slot_inputs = self.intent_slot_attention(self.fc_drop(share_features), self.fc_drop(intent_slot_inputs))

        logits_slot = self.slot_decoder(intent_slot_inputs)

        return logits_intent, logits_slot  # (b,n_class), #(b,n,n_tag)

    def loss1(self, logits_intent, logits_slot, intent_label, slot_label, mask):
        mask = mask[:, 0:logits_slot.size(1)]

        slot_label = slot_label[:, 0:logits_slot.size(1)]  # 去除掩码

        logits_slot = logits_slot.transpose(1, 0)  # (n,b,n_tag)

        slot_label = slot_label.transpose(1, 0)  # (n,b)
        mask = mask.transpose(1, 0)
        loss_intent = self.intent_criterion(logits_intent, intent_label)

        loss_slot = -self.crflayer(logits_slot, slot_label, mask) / logits_intent.size()[0]  # 单个样本的均值

        return loss_intent, loss_slot

    def pred_intent_slot(self, logits_intent, logits_slot, mask):
        mask = mask[:, 0:logits_slot.size(1)]
        mask = mask.transpose(1, 0)
        logits_slot = logits_slot.transpose(1, 0)
        pred_intent = torch.max(logits_intent, 1)[1]
        pred_slot = self.crflayer.decode(logits_slot, mask=mask)
        return pred_intent, pred_slot

    def gat_layer(self, x, p, mask=None):
        '''

        :param x: (batch_size,seq_len,dim)
        :param p: (batch_size,relation_num,dim)
        :param mask: (batch,seq_len)
        :return:
        '''
        for m in self.layers:
            x, p = m(x, p, mask)
        return x, p


class Joint_model_Soft_Interaction_MultiIntent(nn.Module):
    def __init__(self, args, hidden_dim, batch_size, max_length, n_class, n_tag, embedding_matrix):
        super(Joint_model_Soft_Interaction_MultiIntent, self).__init__()
        self.args = args
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.max_length = max_length
        self.n_class = n_class
        self.n_tag = n_tag
        self.LayerNorm = LayerNorm(self.hidden_dim, eps=1e-12)
        self.emb_drop = nn.Dropout(self.args.emb_dorpout)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), padding_idx=0)
        self.embed.weight.requires_grad = True

        self.fc_drop = nn.Dropout(self.args.fc_dropout)

        self.biLSTM = DynamicLSTM(self.args, self.args.emb_dim, self.args.hidden_dim // 2, bidirectional=True,
                                  batch_first=True,
                                  dropout=self.args.lstm_dropout, num_layers=1)

        # Initialize an self-attention layer.
        self.__attention = SelfAttention(
            self.args.emb_dim,
            self.args.emb_dim,
            self.args.attention_output_dim,
            self.args.attention_dropout
        )

        self.__sentattention = UnflatSelfAttention(
            self.args.hidden_dim + self.args.attention_output_dim,
            self.args.attention_dropout
        )

        self.intent_embeddings = nn.Parameter(torch.rand([self.n_class, self.args.intent_dim]))
        # nn.init.xavier_uniform_(self.intent_embeddings.data)
        self.slot_embeding = nn.Embedding(self.n_tag, self.args.slot_dim)  # (relation,dim)(19,768)
        self.relation = nn.Linear(self.args.slot_dim, self.args.slot_dim)  # (768,768)

        self.intent_fc = nn.Linear(2 * self.hidden_dim, self.n_class)
        self.intent_to_slot = nn.Linear(self.hidden_dim, 2 * self.hidden_dim)
        self.intent_slot_attention = Intent_Slot_Attention(2 * self.hidden_dim)
        self.slot_decoder = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.n_tag)
        )
        self.slot_reduce = nn.Linear(3 * self.hidden_dim, self.hidden_dim)

        self.crflayer = CRF(self.n_tag)

        self.intent_criterion = nn.CrossEntropyLoss()
        self.slot_criterion = nn.CrossEntropyLoss()

        if self.args.multi_task:
            self.intent_criterion = nn.BCEWithLogitsLoss()

        self.down = nn.Linear(2 * self.hidden_dim, self.hidden_dim)  # (3*768,768)
        self.start_tail = nn.Linear(self.hidden_dim, 1)  # (768,1)

        self.layers = nn.ModuleList([GATLayer(self.args.hidden_dim) for _ in range(self.args.gat_layers)])

    def forward_logit(self, x, mask):
        x, x_char = x

        seq_len = torch.sum(x != 0, dim=-1)  # (batch_size,seq_len)-->(batch_size)

        x_emb = self.emb_drop(self.embed(x))  # (batch_szie,seq_len,dim)

        lstm_hiddens, (_, _) = self.biLSTM(x_emb, seq_len.cpu())  # (batch_size,seq_len,dim)

        attention_hiddens = self.__attention(x_emb, seq_len)

        share_features = torch.cat([attention_hiddens, lstm_hiddens], dim=2)

        intent_context = self.__sentattention(share_features, seq_len)  # (batch, dim)

        logits_intent = self.intent_fc(intent_context)

        intent_weights = F.softmax(logits_intent)
        # intent_weights = F.sigmoid(logits_intent)

        intent_weights = intent_weights.unsqueeze(-1).expand(-1, -1, self.intent_embeddings.size(-1))

        intent_slot_inputs = intent_weights * self.intent_embeddings.unsqueeze(0).expand(intent_weights.size(0), -1, -1)

        intent_slot_inputs = self.intent_to_slot(self.fc_drop(intent_slot_inputs))
        intent_slot_inputs = self.intent_slot_attention(self.fc_drop(share_features), self.fc_drop(intent_slot_inputs))

        logits_slot = self.slot_decoder(intent_slot_inputs)

        return logits_intent, logits_slot  # (b,n_class), #(b,n,n_tag)

    def loss1(self, logits_intent, logits_slot, intent_label, slot_label, mask):
        mask = mask[:, 0:logits_slot.size(1)]

        slot_label = slot_label[:, 0:logits_slot.size(1)]  # 去除掩码

        logits_slot = logits_slot.transpose(1, 0)  # (n,b,n_tag)

        slot_label = slot_label.transpose(1, 0)  # (n,b)
        mask = mask.transpose(1, 0)

        if not self.args.multi_task:
            loss_intent = self.intent_criterion(logits_intent, intent_label.float())
        else:
            loss_intent = self.intent_criterion(logits_intent, intent_label.float()) / logits_intent.size()[
                0]  # 单个样本的均值

        loss_slot = -self.crflayer(logits_slot, slot_label, mask) / logits_intent.size()[0]  # 单个样本的均值

        return loss_intent, loss_slot

    def pred_intent_slot(self, logits_intent, logits_slot, mask):
        mask = mask[:, 0:logits_slot.size(1)]
        mask = mask.transpose(1, 0)
        logits_slot = logits_slot.transpose(1, 0)

        if self.args.multi_task:
            intent_sig = torch.sigmoid(logits_intent)
            # print(intent_sig)
            intent_idx = (intent_sig > self.args.threshold).nonzero()
            intent_idx_ = [[] for i in range(logits_intent.size(0))]
            for item in intent_idx:
                intent_idx_[item[0]].append(item[1])
            pred_intent = intent_idx_
        else:
            pred_intent = torch.max(logits_intent, 1)[1]

        pred_slot = self.crflayer.decode(logits_slot, mask=mask)
        # print(pred_intent)
        return pred_intent, pred_slot

    def gat_layer(self, x, p, mask=None):
        '''

        :param x: (batch_size,seq_len,dim)
        :param p: (batch_size,relation_num,dim)
        :param mask: (batch,seq_len)
        :return:
        '''
        for m in self.layers:
            x, p = m(x, p, mask)
        return x, p


class Joint_model_Soft_Interaction_Without_Intent_Weight(nn.Module):
    def __init__(self, args, hidden_dim, batch_size, max_length, n_class, n_tag, embedding_matrix):
        super(Joint_model_Soft_Interaction_Without_Intent_Weight, self).__init__()
        self.args = args
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.max_length = max_length
        self.n_class = n_class
        self.n_tag = n_tag
        self.LayerNorm = LayerNorm(self.hidden_dim, eps=1e-12)
        self.emb_drop = nn.Dropout(self.args.emb_dorpout)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), padding_idx=0)
        self.embed.weight.requires_grad = True

        self.fc_drop = nn.Dropout(self.args.fc_dropout)

        self.biLSTM = DynamicLSTM(self.args, self.args.emb_dim, self.args.hidden_dim // 2, bidirectional=True,
                                  batch_first=True,
                                  dropout=self.args.lstm_dropout, num_layers=1)

        self.slot_gru = DynamicLSTM(self.args.emb_dim, self.args.hidden_dim, bidirectional=True, batch_first=True,
                                    dropout=self.args.lstm_dropout, num_layers=1, rnn_type='GRU')
        # self.biLSTM = DynamicLSTM(config.emb_dim, config.hidden_dim // 2, bidirectional=True, batch_first=True,
        #                           dropout=config.lstm_dropout, num_layers=1, rnn_type='GRU')

        # Initialize an self-attention layer.
        self.__attention = SelfAttention(
            self.args.emb_dim,
            self.args.emb_dim,
            self.args.attention_output_dim,
            self.args.attention_dropout
        )

        self.__sentattention = UnflatSelfAttention(
            self.args.hidden_dim + self.args.attention_output_dim,
            self.args.attention_dropout
        )

        self.intent_embeddings = nn.Parameter(torch.rand([self.n_class, self.args.intent_dim]))
        # nn.init.xavier_uniform_(self.intent_embeddings.data)
        self.slot_embeding = nn.Embedding(self.n_tag, self.args.slot_dim)  # (relation,dim)(19,768)
        self.relation = nn.Linear(self.args.slot_dim, self.args.slot_dim)  # (768,768)

        self.intent_fc = nn.Linear(2 * self.hidden_dim, self.n_class)
        self.intent_to_slot = nn.Linear(self.hidden_dim, 2 * self.hidden_dim)
        self.intent_slot_attention = Intent_Slot_Attention(2 * self.hidden_dim)
        self.slot_decoder = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.n_tag)
        )
        self.slot_reduce = nn.Linear(3 * self.hidden_dim, self.hidden_dim)

        self.crflayer = CRF(self.n_tag)

        self.intent_criterion = nn.CrossEntropyLoss()
        self.slot_criterion = nn.CrossEntropyLoss()

        self.down = nn.Linear(2 * self.hidden_dim, self.hidden_dim)  # (3*768,768)
        self.start_tail = nn.Linear(self.hidden_dim, 1)  # (768,1)

        self.layers = nn.ModuleList([GATLayer(self.args.hidden_dim) for _ in range(self.args.gat_layers)])

    def forward_logit(self, x, mask):
        x, x_char = x

        seq_len = torch.sum(x != 0, dim=-1)  # (batch_size,seq_len)-->(batch_size)

        x_emb = self.emb_drop(self.embed(x))  # (batch_szie,seq_len,dim)

        lstm_hiddens, (_, _) = self.biLSTM(x_emb, seq_len)  # (batch_size,seq_len,dim)

        attention_hiddens = self.__attention(x_emb, seq_len)

        share_features = torch.cat([attention_hiddens, lstm_hiddens], dim=2)

        intent_context = self.__sentattention(share_features, seq_len)  # (batch, dim)

        logits_intent = self.intent_fc(intent_context)

        # intent_weights = F.softmax(logits_intent)
        #
        # intent_weights = intent_weights.unsqueeze(-1).expand(-1, -1, self.intent_embeddings.size(-1))
        #
        # intent_slot_inputs = intent_weights * self.intent_embeddings.unsqueeze(0).expand(intent_weights.size(0), -1, -1)

        intent_slot_inputs = self.intent_embeddings.unsqueeze(0).expand(intent_context.size(0), -1, -1)

        intent_slot_inputs = self.intent_to_slot(self.fc_drop(intent_slot_inputs))
        intent_slot_inputs = self.intent_slot_attention(self.fc_drop(share_features), self.fc_drop(intent_slot_inputs))

        logits_slot = self.slot_decoder(intent_slot_inputs)

        return logits_intent, logits_slot  # (b,n_class), #(b,n,n_tag)

    def loss1(self, logits_intent, logits_slot, intent_label, slot_label, mask):
        mask = mask[:, 0:logits_slot.size(1)]

        slot_label = slot_label[:, 0:logits_slot.size(1)]  # 去除掩码

        logits_slot = logits_slot.transpose(1, 0)  # (n,b,n_tag)

        slot_label = slot_label.transpose(1, 0)  # (n,b)
        mask = mask.transpose(1, 0)
        loss_intent = self.intent_criterion(logits_intent, intent_label)

        loss_slot = -self.crflayer(logits_slot, slot_label, mask) / logits_intent.size()[0]  # 单个样本的均值

        return loss_intent, loss_slot

    def pred_intent_slot(self, logits_intent, logits_slot, mask):
        mask = mask[:, 0:logits_slot.size(1)]
        mask = mask.transpose(1, 0)
        logits_slot = logits_slot.transpose(1, 0)
        pred_intent = torch.max(logits_intent, 1)[1]

        multi_pred = F.sigmoid(logits_intent)

        # print(multi_pred[0])

        pred_slot = self.crflayer.decode(logits_slot, mask=mask)
        return pred_intent, pred_slot

    def gat_layer(self, x, p, mask=None):
        '''

        :param x: (batch_size,seq_len,dim)
        :param p: (batch_size,relation_num,dim)
        :param mask: (batch,seq_len)
        :return:
        '''
        for m in self.layers:
            x, p = m(x, p, mask)
        return x, p


class Joint_Model_Soft_Interaction_With_Single_Intent(nn.Module):
    def __init__(self, args, hidden_dim, batch_size, max_length, n_class, n_tag, embedding_matrix):
        super(Joint_Model_Soft_Interaction_With_Single_Intent, self).__init__()
        self.args = args
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.max_length = max_length
        self.n_class = n_class
        self.n_tag = n_tag
        self.LayerNorm = LayerNorm(self.hidden_dim, eps=1e-12)
        self.emb_drop = nn.Dropout(self.args.emb_dorpout)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), padding_idx=0)
        self.embed.weight.requires_grad = True

        self.fc_drop = nn.Dropout(self.args.fc_dropout)

        self.biLSTM = DynamicLSTM(self.args.emb_dim, self.args.hidden_dim // 2, bidirectional=True, batch_first=True,
                                  dropout=self.args.lstm_dropout, num_layers=1)

        self.slot_gru = DynamicLSTM(self.args.emb_dim, self.args.hidden_dim, bidirectional=True, batch_first=True,
                                    dropout=self.args.lstm_dropout, num_layers=1, rnn_type='GRU')
        # self.biLSTM = DynamicLSTM(config.emb_dim, config.hidden_dim // 2, bidirectional=True, batch_first=True,
        #                           dropout=config.lstm_dropout, num_layers=1, rnn_type='GRU')

        # Initialize an self-attention layer.
        self.__attention = SelfAttention(
            self.args.emb_dim,
            self.args.emb_dim,
            self.args.attention_output_dim,
            self.args.attention_dropout
        )

        self.__sentattention = UnflatSelfAttention(
            self.args.hidden_dim + self.args.attention_output_dim,
            self.args.attention_dropout
        )

        self._intent_embeddings = nn.Parameter(torch.rand([self.n_class, self.args.intent_dim]))
        # nn.init.xavier_uniform_(self.intent_embeddings.data)

        self.intent_embeddings = nn.Embedding(self.n_class, self.args.intent_dim).from_pretrained(
            self._intent_embeddings)

        self.slot_embeding = nn.Embedding(self.n_tag, self.args.slot_dim)  # (relation,dim)(19,768)

        self.relation = nn.Linear(self.args.slot_dim, self.args.slot_dim)  # (768,768)

        self.intent_fc = nn.Linear(2 * self.hidden_dim, self.n_class)
        self.intent_to_slot = nn.Linear(self.hidden_dim, 2 * self.hidden_dim)
        self.intent_slot_attention = Intent_Slot_Attention(2 * self.hidden_dim)
        self.slot_decoder = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.n_tag)
        )
        self.slot_reduce = nn.Linear(3 * self.hidden_dim, self.hidden_dim)

        self.crflayer = CRF(self.n_tag)

        self.intent_criterion = nn.CrossEntropyLoss()
        self.slot_criterion = nn.CrossEntropyLoss()

        self.down = nn.Linear(2 * self.hidden_dim, self.hidden_dim)  # (3*768,768)
        self.start_tail = nn.Linear(self.hidden_dim, 1)  # (768,1)

        self.layers = nn.ModuleList([GATLayer(self.args.hidden_dim) for _ in range(self.args.gat_layers)])

    def forward_logit(self, x, mask):
        x, x_char = x

        seq_len = torch.sum(x != 0, dim=-1)  # (batch_size,seq_len)-->(batch_size)

        x_emb = self.emb_drop(self.embed(x))  # (batch_szie,seq_len,dim)

        lstm_hiddens, (_, _) = self.biLSTM(x_emb, seq_len.cpu())  # (batch_size,seq_len,dim)

        attention_hiddens = self.__attention(x_emb, seq_len)

        share_features = torch.cat([attention_hiddens, lstm_hiddens], dim=2)

        intent_context = self.__sentattention(share_features, seq_len)  # (batch, dim)

        logits_intent = self.intent_fc(intent_context)

        intent_weights = F.softmax(logits_intent)
        intent_slot_inputs = self.intent_embeddings(torch.argmax(intent_weights, dim=-1)).unsqueeze(1).expand(-1,
                                                                                                              self.n_class,
                                                                                                              -1)  # (batch,seq_len,dim)

        intent_slot_inputs = self.intent_to_slot(self.fc_drop(intent_slot_inputs))
        intent_slot_inputs = self.intent_slot_attention(self.fc_drop(share_features), self.fc_drop(intent_slot_inputs))

        logits_slot = self.slot_decoder(intent_slot_inputs)

        return logits_intent, logits_slot  # (b,n_class), #(b,n,n_tag)

    def loss1(self, logits_intent, logits_slot, intent_label, slot_label, mask):
        mask = mask[:, 0:logits_slot.size(1)]

        slot_label = slot_label[:, 0:logits_slot.size(1)]  # 去除掩码

        logits_slot = logits_slot.transpose(1, 0)  # (n,b,n_tag)

        slot_label = slot_label.transpose(1, 0)  # (n,b)
        mask = mask.transpose(1, 0)
        loss_intent = self.intent_criterion(logits_intent, intent_label)

        loss_slot = -self.crflayer(logits_slot, slot_label, mask) / logits_intent.size()[0]  # 单个样本的均值

        return loss_intent, loss_slot

    def pred_intent_slot(self, logits_intent, logits_slot, mask):
        mask = mask[:, 0:logits_slot.size(1)]
        mask = mask.transpose(1, 0)
        logits_slot = logits_slot.transpose(1, 0)
        pred_intent = torch.max(logits_intent, 1)[1]
        pred_slot = self.crflayer.decode(logits_slot, mask=mask)
        return pred_intent, pred_slot

    def gat_layer(self, x, p, mask=None):
        '''

        :param x: (batch_size,seq_len,dim)
        :param p: (batch_size,relation_num,dim)
        :param mask: (batch,seq_len)
        :return:
        '''
        for m in self.layers:
            x, p = m(x, p, mask)
        return x, p


class Joint_Model_Soft_Interaction_With_Single_Intent_SlotGate(nn.Module):
    def __init__(self, args, hidden_dim, batch_size, max_length, n_class, n_tag, embedding_matrix):
        super(Joint_Model_Soft_Interaction_With_Single_Intent_SlotGate, self).__init__()
        self.args = args
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.max_length = max_length
        self.n_class = n_class
        self.n_tag = n_tag
        self.LayerNorm = LayerNorm(self.hidden_dim, eps=1e-12)
        self.emb_drop = nn.Dropout(self.args.emb_dorpout)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), padding_idx=0)
        self.embed.weight.requires_grad = True

        self.fc_drop = nn.Dropout(self.args.fc_dropout)

        self.biLSTM = DynamicLSTM(self.args.emb_dim, self.args.hidden_dim // 2, bidirectional=True, batch_first=True,
                                  dropout=self.args.lstm_dropout, num_layers=1)

        # Initialize an self-attention layer.
        self.__attention = SelfAttention(
            self.args.emb_dim,
            self.args.emb_dim,
            self.args.attention_output_dim,
            self.args.attention_dropout
        )

        self.__sentattention = UnflatSelfAttention(
            self.args.hidden_dim + self.args.attention_output_dim,
            self.args.attention_dropout
        )

        self.slotGate = SlotGate(2 * self.args.hidden_dim, 2 * self.args.hidden_dim)

        self._intent_embeddings = nn.Parameter(torch.rand([self.n_class, self.args.intent_dim]))
        # nn.init.xavier_uniform_(self.intent_embeddings.data)

        self.intent_embeddings = nn.Embedding(self.n_class, self.args.intent_dim).from_pretrained(
            self._intent_embeddings)

        self.slot_embeding = nn.Embedding(self.n_tag, self.args.slot_dim)  # (relation,dim)(19,768)

        self.relation = nn.Linear(self.args.slot_dim, self.args.slot_dim)  # (768,768)

        self.intent_fc = nn.Linear(2 * self.hidden_dim, self.n_class)
        self.intent_to_slot = nn.Linear(self.hidden_dim, 2 * self.hidden_dim)
        self.intent_slot_attention = Intent_Slot_Attention(2 * self.hidden_dim)
        self.slot_decoder = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.n_tag)
        )
        self.slot_reduce = nn.Linear(3 * self.hidden_dim, self.hidden_dim)

        self.crflayer = CRF(self.n_tag)

        self.intent_criterion = nn.CrossEntropyLoss()
        self.slot_criterion = nn.CrossEntropyLoss()

        self.down = nn.Linear(2 * self.hidden_dim, self.hidden_dim)  # (3*768,768)
        self.start_tail = nn.Linear(self.hidden_dim, 1)  # (768,1)

        self.layers = nn.ModuleList([GATLayer(self.args.hidden_dim) for _ in range(self.args.gat_layers)])

    def forward_logit(self, x, mask):
        x, x_char = x

        seq_len = torch.sum(x != 0, dim=-1)  # (batch_size,seq_len)-->(batch_size)

        x_emb = self.emb_drop(self.embed(x))  # (batch_szie,seq_len,dim)

        lstm_hiddens, (_, _) = self.biLSTM(x_emb, seq_len.cpu())  # (batch_size,seq_len,dim)

        attention_hiddens = self.__attention(x_emb, seq_len)

        share_features = torch.cat([attention_hiddens, lstm_hiddens], dim=2)

        intent_context = self.__sentattention(share_features, seq_len)  # (batch, dim)

        logits_intent = self.intent_fc(intent_context)

        intent_weights = F.softmax(logits_intent)
        intent_slot_inputs = self.intent_embeddings(torch.argmax(intent_weights, dim=-1))
        intent_slot_inputs = self.intent_to_slot(self.fc_drop(intent_slot_inputs))
        intent_slot_inputs = self.slotGate(share_features, intent_slot_inputs)

        # intent_slot_inputs = self.intent_slot_attention(self.fc_drop(share_features), self.fc_drop(intent_slot_inputs))

        logits_slot = self.slot_decoder(intent_slot_inputs)

        return logits_intent, logits_slot  # (b,n_class), #(b,n,n_tag)

    def loss1(self, logits_intent, logits_slot, intent_label, slot_label, mask):
        mask = mask[:, 0:logits_slot.size(1)]

        slot_label = slot_label[:, 0:logits_slot.size(1)]  # 去除掩码

        logits_slot = logits_slot.transpose(1, 0)  # (n,b,n_tag)

        slot_label = slot_label.transpose(1, 0)  # (n,b)
        mask = mask.transpose(1, 0)
        loss_intent = self.intent_criterion(logits_intent, intent_label)

        loss_slot = -self.crflayer(logits_slot, slot_label, mask) / logits_intent.size()[0]  # 单个样本的均值

        return loss_intent, loss_slot

    def pred_intent_slot(self, logits_intent, logits_slot, mask):
        mask = mask[:, 0:logits_slot.size(1)]
        mask = mask.transpose(1, 0)
        logits_slot = logits_slot.transpose(1, 0)
        pred_intent = torch.max(logits_intent, 1)[1]
        pred_slot = self.crflayer.decode(logits_slot, mask=mask)
        return pred_intent, pred_slot

    def gat_layer(self, x, p, mask=None):
        '''

        :param x: (batch_size,seq_len,dim)
        :param p: (batch_size,relation_num,dim)
        :param mask: (batch,seq_len)
        :return:
        '''
        for m in self.layers:
            x, p = m(x, p, mask)
        return x, p


class Joint_model_Soft_Interaction_Single_Intent_SlotGate_Without_BiLSTM(nn.Module):
    def __init__(self, args, hidden_dim, batch_size, max_length, n_class, n_tag, embedding_matrix):
        super(Joint_model_Soft_Interaction_Single_Intent_SlotGate_Without_BiLSTM, self).__init__()
        self.args = args
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.max_length = max_length
        self.n_class = n_class
        self.n_tag = n_tag
        self.LayerNorm = LayerNorm(self.hidden_dim, eps=1e-12)
        self.emb_drop = nn.Dropout(self.args.emb_dorpout)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), padding_idx=0)
        self.embed.weight.requires_grad = True

        self.fc_drop = nn.Dropout(self.args.fc_dropout)

        self.biLSTM = DynamicLSTM(self.args, self.args.emb_dim, self.args.hidden_dim // 2, bidirectional=True,
                                  batch_first=True,
                                  dropout=self.args.lstm_dropout, num_layers=1)

        # Initialize an self-attention layer.
        self.__attention = SelfAttention(
            self.args.emb_dim,
            self.args.emb_dim,
            self.args.attention_output_dim,
            self.args.attention_dropout
        )

        self.__sentattention = UnflatSelfAttention(
            self.args.attention_output_dim,
            self.args.attention_dropout
        )

        self.slotGate = SlotGate(self.args.hidden_dim, self.args.hidden_dim)

        self._intent_embeddings = nn.Parameter(torch.rand([self.n_class, self.args.intent_dim]))
        # nn.init.xavier_uniform_(self.intent_embeddings.data)

        self.intent_embeddings = nn.Embedding(self.n_class, self.args.intent_dim).from_pretrained(
            self._intent_embeddings)

        self.slot_embeding = nn.Embedding(self.n_tag, self.args.slot_dim)  # (relation,dim)(19,768)

        self.relation = nn.Linear(self.args.slot_dim, self.args.slot_dim)  # (768,768)

        self.intent_fc = nn.Linear(self.hidden_dim, self.n_class)
        self.intent_to_slot = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.intent_slot_attention = Intent_Slot_Attention(2 * self.hidden_dim)
        self.slot_decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.n_tag)
        )
        self.slot_reduce = nn.Linear(3 * self.hidden_dim, self.hidden_dim)

        self.crflayer = CRF(self.n_tag)

        self.intent_criterion = nn.CrossEntropyLoss()
        self.slot_criterion = nn.CrossEntropyLoss()

        self.down = nn.Linear(2 * self.hidden_dim, self.hidden_dim)  # (3*768,768)
        self.start_tail = nn.Linear(self.hidden_dim, 1)  # (768,1)

        self.layers = nn.ModuleList([GATLayer(self.args.hidden_dim) for _ in range(self.args.gat_layers)])

    def forward_logit(self, x, mask):
        x, x_char = x

        seq_len = torch.sum(x != 0, dim=-1)  # (batch_size,seq_len)-->(batch_size)

        x_emb = self.emb_drop(self.embed(x))  # (batch_szie,seq_len,dim)

        # lstm_hiddens, (_, _) = self.biLSTM(x_emb, seq_len)  # (batch_size,seq_len,dim)

        attention_hiddens = self.__attention(x_emb, seq_len)
        # share_features = torch.cat([attention_hiddens, lstm_hiddens], dim=2)

        share_features = attention_hiddens

        intent_context = self.__sentattention(share_features, seq_len)  # (batch, dim)

        logits_intent = self.intent_fc(intent_context)

        intent_weights = F.softmax(logits_intent)
        intent_slot_inputs = self.intent_embeddings(torch.argmax(intent_weights, dim=-1))
        intent_slot_inputs = self.intent_to_slot(self.fc_drop(intent_slot_inputs))
        intent_slot_inputs = self.slotGate(share_features, intent_slot_inputs)

        # intent_slot_inputs = self.intent_slot_attention(self.fc_drop(share_features), self.fc_drop(intent_slot_inputs))

        logits_slot = self.slot_decoder(intent_slot_inputs)

        return logits_intent, logits_slot  # (b,n_class), #(b,n,n_tag)

    def loss1(self, logits_intent, logits_slot, intent_label, slot_label, mask):
        mask = mask[:, 0:logits_slot.size(1)]

        slot_label = slot_label[:, 0:logits_slot.size(1)]  # 去除掩码

        logits_slot = logits_slot.transpose(1, 0)  # (n,b,n_tag)

        slot_label = slot_label.transpose(1, 0)  # (n,b)
        mask = mask.transpose(1, 0)
        loss_intent = self.intent_criterion(logits_intent, intent_label)

        loss_slot = -self.crflayer(logits_slot, slot_label, mask) / logits_intent.size()[0]  # 单个样本的均值

        return loss_intent, loss_slot

    def pred_intent_slot(self, logits_intent, logits_slot, mask):
        mask = mask[:, 0:logits_slot.size(1)]
        mask = mask.transpose(1, 0)
        logits_slot = logits_slot.transpose(1, 0)
        pred_intent = torch.max(logits_intent, 1)[1]
        pred_slot = self.crflayer.decode(logits_slot, mask=mask)
        return pred_intent, pred_slot

    def gat_layer(self, x, p, mask=None):
        '''

        :param x: (batch_size,seq_len,dim)
        :param p: (batch_size,relation_num,dim)
        :param mask: (batch,seq_len)
        :return:
        '''
        for m in self.layers:
            x, p = m(x, p, mask)
        return x, p


class Joint_model_Soft_Interaction_Single_Intent_SlotGate_Without_Attention(nn.Module):
    def __init__(self, args, hidden_dim, batch_size, max_length, n_class, n_tag, embedding_matrix):
        super(Joint_model_Soft_Interaction_Single_Intent_SlotGate_Without_Attention, self).__init__()
        self.args = args
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.max_length = max_length
        self.n_class = n_class
        self.n_tag = n_tag
        self.LayerNorm = LayerNorm(self.hidden_dim, eps=1e-12)
        self.emb_drop = nn.Dropout(self.args.emb_dorpout)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), padding_idx=0)
        self.embed.weight.requires_grad = True

        self.fc_drop = nn.Dropout(self.args.fc_dropout)

        self.biLSTM = DynamicLSTM(self.args, self.args.emb_dim, self.args.hidden_dim // 2, bidirectional=True,
                                  batch_first=True,
                                  dropout=self.args.lstm_dropout, num_layers=1)

        # Initialize an self-attention layer.
        self.__attention = SelfAttention(
            self.args.emb_dim,
            self.args.emb_dim,
            self.args.attention_output_dim,
            self.args.attention_dropout
        )

        self.__sentattention = UnflatSelfAttention(
            self.args.attention_output_dim,
            self.args.attention_dropout
        )

        self.slotGate = SlotGate(self.args.hidden_dim, self.args.hidden_dim)

        self._intent_embeddings = nn.Parameter(torch.rand([self.n_class, self.args.intent_dim]))
        # nn.init.xavier_uniform_(self.intent_embeddings.data)

        self.intent_embeddings = nn.Embedding(self.n_class, self.args.intent_dim).from_pretrained(
            self._intent_embeddings)

        self.slot_embeding = nn.Embedding(self.n_tag, self.args.slot_dim)  # (relation,dim)(19,768)

        self.relation = nn.Linear(self.args.slot_dim, self.args.slot_dim)  # (768,768)

        self.intent_fc = nn.Linear(self.hidden_dim, self.n_class)
        self.intent_to_slot = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.intent_slot_attention = Intent_Slot_Attention(2 * self.hidden_dim)
        self.slot_decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.n_tag)
        )
        self.slot_reduce = nn.Linear(3 * self.hidden_dim, self.hidden_dim)

        self.crflayer = CRF(self.n_tag)

        self.intent_criterion = nn.CrossEntropyLoss()
        self.slot_criterion = nn.CrossEntropyLoss()

        self.down = nn.Linear(2 * self.hidden_dim, self.hidden_dim)  # (3*768,768)
        self.start_tail = nn.Linear(self.hidden_dim, 1)  # (768,1)

        self.layers = nn.ModuleList([GATLayer(self.args.hidden_dim) for _ in range(self.args.gat_layers)])

    def forward_logit(self, x, mask):
        x, x_char = x

        seq_len = torch.sum(x != 0, dim=-1)  # (batch_size,seq_len)-->(batch_size)

        x_emb = self.emb_drop(self.embed(x))  # (batch_szie,seq_len,dim)

        lstm_hiddens, (_, _) = self.biLSTM(x_emb, seq_len.cpu())  # (batch_size,seq_len,dim)

        # attention_hiddens = self.__attention(x_emb, seq_len)
        # share_features = torch.cat([attention_hiddens, lstm_hiddens], dim=2)

        share_features = lstm_hiddens

        intent_context = self.__sentattention(share_features, seq_len)  # (batch, dim)

        logits_intent = self.intent_fc(intent_context)

        intent_weights = F.softmax(logits_intent)
        intent_slot_inputs = self.intent_embeddings(torch.argmax(intent_weights, dim=-1))
        intent_slot_inputs = self.intent_to_slot(self.fc_drop(intent_slot_inputs))
        intent_slot_inputs = self.slotGate(share_features, intent_slot_inputs)

        # intent_slot_inputs = self.intent_slot_attention(self.fc_drop(share_features), self.fc_drop(intent_slot_inputs))

        logits_slot = self.slot_decoder(intent_slot_inputs)

        return logits_intent, logits_slot  # (b,n_class), #(b,n,n_tag)

    def loss1(self, logits_intent, logits_slot, intent_label, slot_label, mask):
        mask = mask[:, 0:logits_slot.size(1)]

        slot_label = slot_label[:, 0:logits_slot.size(1)]  # 去除掩码

        logits_slot = logits_slot.transpose(1, 0)  # (n,b,n_tag)

        slot_label = slot_label.transpose(1, 0)  # (n,b)
        mask = mask.transpose(1, 0)
        loss_intent = self.intent_criterion(logits_intent, intent_label)

        loss_slot = -self.crflayer(logits_slot, slot_label, mask) / logits_intent.size()[0]  # 单个样本的均值

        return loss_intent, loss_slot

    def pred_intent_slot(self, logits_intent, logits_slot, mask):
        mask = mask[:, 0:logits_slot.size(1)]
        mask = mask.transpose(1, 0)
        logits_slot = logits_slot.transpose(1, 0)
        pred_intent = torch.max(logits_intent, 1)[1]
        pred_slot = self.crflayer.decode(logits_slot, mask=mask)
        return pred_intent, pred_slot

    def gat_layer(self, x, p, mask=None):
        '''

        :param x: (batch_size,seq_len,dim)
        :param p: (batch_size,relation_num,dim)
        :param mask: (batch,seq_len)
        :return:
        '''
        for m in self.layers:
            x, p = m(x, p, mask)
        return x, p


class Joint_Model_Soft_Interaction_Without_Intent_Attention(nn.Module):
    def __init__(self, args, hidden_dim, batch_size, max_length, n_class, n_tag, embedding_matrix):
        super(Joint_Model_Soft_Interaction_Without_Intent_Attention, self).__init__()
        self.args = args
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.max_length = max_length
        self.n_class = n_class
        self.n_tag = n_tag
        self.LayerNorm = LayerNorm(self.hidden_dim, eps=1e-12)
        self.emb_drop = nn.Dropout(self.args.emb_dorpout)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), padding_idx=0)
        self.embed.weight.requires_grad = True

        self.fc_drop = nn.Dropout(self.args.fc_dropout)

        self.biLSTM = DynamicLSTM(self.args.emb_dim, self.args.hidden_dim // 2, bidirectional=True, batch_first=True,
                                  dropout=self.args.lstm_dropout, num_layers=1)

        self.slot_gru = DynamicLSTM(self.args.emb_dim, self.args.hidden_dim, bidirectional=True, batch_first=True,
                                    dropout=self.args.lstm_dropout, num_layers=1, rnn_type='GRU')
        # self.biLSTM = DynamicLSTM(config.emb_dim, config.hidden_dim // 2, bidirectional=True, batch_first=True,
        #                           dropout=config.lstm_dropout, num_layers=1, rnn_type='GRU')

        # Initialize an self-attention layer.
        self.__attention = SelfAttention(
            self.args.emb_dim,
            self.args.emb_dim,
            self.args.attention_output_dim,
            self.args.attention_dropout
        )

        self.__sentattention = UnflatSelfAttention(
            self.args.hidden_dim + self.args.attention_output_dim,
            self.args.attention_dropout
        )

        self._intent_embeddings = nn.Parameter(torch.rand([self.n_class, self.args.intent_dim]))
        # nn.init.xavier_uniform_(self.intent_embeddings.data)

        self.intent_embeddings = nn.Embedding(self.n_class, self.args.intent_dim).from_pretrained(
            self._intent_embeddings)

        self.slot_embeding = nn.Embedding(self.n_tag, self.args.slot_dim)  # (relation,dim)(19,768)

        self.relation = nn.Linear(self.args.slot_dim, self.args.slot_dim)  # (768,768)

        self.intent_fc = nn.Linear(2 * self.hidden_dim, self.n_class)
        self.intent_to_slot = nn.Linear(self.hidden_dim, 2 * self.hidden_dim)
        self.intent_slot_attention = Intent_Slot_Attention(2 * self.hidden_dim)
        self.slot_decoder = nn.Sequential(
            nn.Linear(3 * self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.n_tag)
        )
        self.slot_reduce = nn.Linear(3 * self.hidden_dim, self.hidden_dim)

        self.crflayer = CRF(self.n_tag)

        self.intent_criterion = nn.CrossEntropyLoss()
        self.slot_criterion = nn.CrossEntropyLoss()

        self.down = nn.Linear(2 * self.hidden_dim, self.hidden_dim)  # (3*768,768)
        self.start_tail = nn.Linear(self.hidden_dim, 1)  # (768,1)

        self.layers = nn.ModuleList([GATLayer(self.args.hidden_dim) for _ in range(self.args.gat_layers)])

    def forward_logit(self, x, mask):
        x, x_char = x

        seq_len = torch.sum(x != 0, dim=-1)  # (batch_size,seq_len)-->(batch_size)

        x_emb = self.emb_drop(self.embed(x))  # (batch_szie,seq_len,dim)

        lstm_hiddens, (_, _) = self.biLSTM(x_emb, seq_len.cpu())  # (batch_size,seq_len,dim)

        attention_hiddens = self.__attention(x_emb, seq_len)

        share_features = torch.cat([attention_hiddens, lstm_hiddens], dim=2)

        intent_context = self.__sentattention(share_features, seq_len)  # (batch, dim)

        logits_intent = self.intent_fc(intent_context)

        intent_weights = F.softmax(logits_intent)
        intent_slot_inputs = self.intent_embeddings(torch.argmax(intent_weights, dim=-1)).unsqueeze(1).expand(-1,
                                                                                                              share_features.size(
                                                                                                                  1),
                                                                                                              -1)  # (batch,seq_len,dim)

        slot_inputs = torch.cat([share_features, intent_slot_inputs], dim=-1)

        logits_slot = self.slot_decoder(slot_inputs)

        return logits_intent, logits_slot  # (b,n_class), #(b,n,n_tag)

    def loss1(self, logits_intent, logits_slot, intent_label, slot_label, mask):
        mask = mask[:, 0:logits_slot.size(1)]

        slot_label = slot_label[:, 0:logits_slot.size(1)]  # 去除掩码

        logits_slot = logits_slot.transpose(1, 0)  # (n,b,n_tag)

        slot_label = slot_label.transpose(1, 0)  # (n,b)
        mask = mask.transpose(1, 0)
        loss_intent = self.intent_criterion(logits_intent, intent_label)

        loss_slot = -self.crflayer(logits_slot, slot_label, mask) / logits_intent.size()[0]  # 单个样本的均值

        return loss_intent, loss_slot

    def pred_intent_slot(self, logits_intent, logits_slot, mask):
        mask = mask[:, 0:logits_slot.size(1)]
        mask = mask.transpose(1, 0)
        logits_slot = logits_slot.transpose(1, 0)
        pred_intent = torch.max(logits_intent, 1)[1]
        pred_slot = self.crflayer.decode(logits_slot, mask=mask)
        return pred_intent, pred_slot

    def gat_layer(self, x, p, mask=None):
        '''

        :param x: (batch_size,seq_len,dim)
        :param p: (batch_size,relation_num,dim)
        :param mask: (batch,seq_len)
        :return:
        '''
        for m in self.layers:
            x, p = m(x, p, mask)
        return x, p


class Joint_model_Soft_Interaction_Without_Attention(nn.Module):
    def __init__(self, args, hidden_dim, batch_size, max_length, n_class, n_tag, embedding_matrix):
        super(Joint_model_Soft_Interaction_Without_Attention, self).__init__()

        self.args = args
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.max_length = max_length
        self.n_class = n_class
        self.n_tag = n_tag
        self.LayerNorm = LayerNorm(self.hidden_dim, eps=1e-12)
        self.emb_drop = nn.Dropout(self.args.emb_dorpout)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), padding_idx=0)
        self.embed.weight.requires_grad = True

        self.fc_drop = nn.Dropout(self.args.fc_dropout)

        self.biLSTM = DynamicLSTM(self.args.emb_dim, self.args.hidden_dim // 2, bidirectional=True, batch_first=True,
                                  dropout=self.args.lstm_dropout, num_layers=1)

        self.slot_gru = DynamicLSTM(self.args.emb_dim, self.args.hidden_dim, bidirectional=True, batch_first=True,
                                    dropout=self.args.lstm_dropout, num_layers=1, rnn_type='GRU')
        # self.biLSTM = DynamicLSTM(config.emb_dim, config.hidden_dim // 2, bidirectional=True, batch_first=True,
        #                           dropout=config.lstm_dropout, num_layers=1, rnn_type='GRU')

        # Initialize an self-attention layer.
        self.__attention = SelfAttention(
            self.args.emb_dim,
            self.args.emb_dim,
            self.args.attention_output_dim,
            self.args.attention_dropout
        )

        # self.__sentattention = UnflatSelfAttention(
        #     config.hidden_dim + config.attention_output_dim,
        #     config.attention_dropout
        # )
        self.__sentattention = UnflatSelfAttention(
            self.args.hidden_dim,
            self.args.attention_dropout
        )

        self.intent_embeddings = nn.Parameter(torch.rand([self.n_class, self.args.intent_dim]))
        self.slot_embeding = nn.Embedding(self.n_tag, self.args.slot_dim)  # (relation,dim)(19,768)
        self.relation = nn.Linear(self.args.slot_dim, self.args.slot_dim)  # (768,768)
        # ------------级联后的嵌入维度-------------------------------------------
        # self.intent_fc = nn.Linear(2 * self.hidden_dim, self.n_class)
        # self.intent_to_slot = nn.Linear(self.hidden_dim, 2 * self.hidden_dim)
        # self.intent_slot_attention = Intent_Slot_Attention(2 * self.hidden_dim)
        # self.slot_decoder = nn.Sequential(
        #     nn.Linear(2 * self.hidden_dim, self.hidden_dim),
        #     nn.Tanh(),
        #     nn.Linear(self.hidden_dim, self.n_tag)
        # )
        self.intent_fc = nn.Linear(self.hidden_dim, self.n_class)
        self.intent_to_slot = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.intent_slot_attention = Intent_Slot_Attention(self.hidden_dim)

        self.slot_decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.n_tag)
        )

        self.slot_reduce = nn.Linear(3 * self.hidden_dim, self.hidden_dim)

        self.crflayer = CRF(self.n_tag)

        self.intent_criterion = nn.CrossEntropyLoss()
        self.slot_criterion = nn.CrossEntropyLoss()

        self.down = nn.Linear(2 * self.hidden_dim, self.hidden_dim)  # (3*768,768)
        self.start_tail = nn.Linear(self.hidden_dim, 1)  # (768,1)

        self.layers = nn.ModuleList([GATLayer(self.args.hidden_dim) for _ in range(self.args.gat_layers)])

    def forward_logit(self, x, mask):
        x, x_char = x

        seq_len = torch.sum(x != 0, dim=-1)  # (batch_size,seq_len)-->(batch_size)

        x_emb = self.emb_drop(self.embed(x))  # (batch_szie,seq_len,dim)

        lstm_hiddens, (_, _) = self.biLSTM(x_emb, seq_len)  # (batch_size,seq_len,dim)

        # attention_hiddens = self.__attention(x_emb, seq_len)

        # share_features = torch.cat([attention_hiddens, lstm_hiddens], dim=2)
        share_features = lstm_hiddens
        intent_context = self.__sentattention(share_features, seq_len)  # (batch, dim)

        logits_intent = self.intent_fc(intent_context)

        intent_weights = F.softmax(logits_intent)

        intent_weights = intent_weights.unsqueeze(-1).expand(-1, -1, self.intent_embeddings.size(-1))

        intent_slot_inputs = intent_weights * self.intent_embeddings.unsqueeze(0).expand(intent_weights.size(0), -1, -1)

        intent_slot_inputs = self.intent_to_slot(self.fc_drop(intent_slot_inputs))
        intent_slot_inputs = self.intent_slot_attention(self.fc_drop(share_features), self.fc_drop(intent_slot_inputs))

        logits_slot = self.slot_decoder(intent_slot_inputs)

        return logits_intent, logits_slot  # (b,n_class), #(b,n,n_tag)

    def loss1(self, logits_intent, logits_slot, intent_label, slot_label, mask):
        mask = mask[:, 0:logits_slot.size(1)]

        slot_label = slot_label[:, 0:logits_slot.size(1)]  # 去除掩码

        logits_slot = logits_slot.transpose(1, 0)  # (n,b,n_tag)

        slot_label = slot_label.transpose(1, 0)  # (n,b)

        mask = mask.transpose(1, 0)

        loss_intent = self.intent_criterion(logits_intent, intent_label)

        loss_slot = -self.crflayer(logits_slot, slot_label, mask) / logits_intent.size()[0]  # 单个样本的均值

        return loss_intent, loss_slot

    def pred_intent_slot(self, logits_intent, logits_slot, mask):
        mask = mask[:, 0:logits_slot.size(1)]
        mask = mask.transpose(1, 0)
        logits_slot = logits_slot.transpose(1, 0)
        pred_intent = torch.max(logits_intent, 1)[1]
        pred_slot = self.crflayer.decode(logits_slot, mask=mask)
        return pred_intent, pred_slot

    def gat_layer(self, x, p, mask=None):
        '''

        :param x: (batch_size,seq_len,dim)
        :param p: (batch_size,relation_num,dim)
        :param mask: (batch,seq_len)
        :return:
        '''
        for m in self.layers:
            x, p = m(x, p, mask)
        return x, p


class Joint_model_Soft_Interaction_Without_BiLSTM(nn.Module):
    def __init__(self, args, hidden_dim, batch_size, max_length, n_class, n_tag, embedding_matrix):
        super(Joint_model_Soft_Interaction_Without_BiLSTM, self).__init__()
        self.args = args
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.max_length = max_length
        self.n_class = n_class
        self.n_tag = n_tag
        self.LayerNorm = LayerNorm(self.hidden_dim, eps=1e-12)
        self.emb_drop = nn.Dropout(self.args.emb_dorpout)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), padding_idx=0)
        self.embed.weight.requires_grad = True

        self.fc_drop = nn.Dropout(self.args.fc_dropout)

        self.biLSTM = DynamicLSTM(self.args.emb_dim, self.args.hidden_dim // 2, bidirectional=True, batch_first=True,
                                  dropout=self.args.lstm_dropout, num_layers=1)

        self.slot_gru = DynamicLSTM(self.args.emb_dim, self.args.hidden_dim, bidirectional=True, batch_first=True,
                                    dropout=self.args.lstm_dropout, num_layers=1, rnn_type='GRU')
        self.__attention = SelfAttention(
            self.args.emb_dim,
            self.args.emb_dim,
            self.args.attention_output_dim,
            self.args.attention_dropout
        )

        self.intent_embeddings = nn.Parameter(torch.rand([self.n_class, self.args.intent_dim]))
        # nn.init.xavier_uniform_(self.intent_embeddings.data)
        self.slot_embeding = nn.Embedding(self.n_tag, self.args.slot_dim)  # (relation,dim)(19,768)
        self.relation = nn.Linear(self.args.slot_dim, self.args.slot_dim)  # (768,768)

        # ------------级联后的嵌入维度-------------------------------------------
        # self.__sentattention = UnflatSelfAttention(
        #     config.hidden_dim + config.attention_output_dim,
        #     config.attention_dropout
        # )
        # self.intent_fc = nn.Linear(2 * self.hidden_dim, self.n_class)
        # self.intent_to_slot = nn.Linear(self.hidden_dim, 2 * self.hidden_dim)
        # self.intent_slot_attention = Intent_Slot_Attention(2 * self.hidden_dim)
        # self.slot_decoder = nn.Sequential(
        #     nn.Linear(2 * self.hidden_dim, self.hidden_dim),
        #     nn.Tanh(),
        #     nn.Linear(self.hidden_dim, self.n_tag)
        # )

        self.__sentattention = UnflatSelfAttention(
            self.args.hidden_dim,
            self.args.attention_dropout
        )

        self.intent_fc = nn.Linear(self.hidden_dim, self.n_class)
        self.intent_to_slot = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.intent_slot_attention = Intent_Slot_Attention(self.hidden_dim)
        self.slot_decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.n_tag)
        )

        self.slot_reduce = nn.Linear(3 * self.hidden_dim, self.hidden_dim)

        self.crflayer = CRF(self.n_tag)

        self.intent_criterion = nn.CrossEntropyLoss()
        self.slot_criterion = nn.CrossEntropyLoss()

        self.down = nn.Linear(2 * self.hidden_dim, self.hidden_dim)  # (3*768,768)
        self.start_tail = nn.Linear(self.hidden_dim, 1)  # (768,1)

        self.layers = nn.ModuleList([GATLayer(self.args.hidden_dim) for _ in range(self.args.gat_layers)])

    def forward_logit(self, x, mask):
        x, x_char = x

        seq_len = torch.sum(x != 0, dim=-1)  # (batch_size,seq_len)-->(batch_size)

        x_emb = self.emb_drop(self.embed(x))  # (batch_szie,seq_len,dim)

        # lstm_hiddens, (_, _) = self.biLSTM(x_emb, seq_len)  # (batch_size,seq_len,dim)

        attention_hiddens = self.__attention(x_emb, seq_len)
        # share_features = torch.cat([attention_hiddens, lstm_hiddens], dim=2)
        share_features = attention_hiddens
        intent_context = self.__sentattention(share_features, seq_len)  # (batch, dim)

        logits_intent = self.intent_fc(intent_context)

        intent_weights = F.softmax(logits_intent)

        intent_weights = intent_weights.unsqueeze(-1).expand(-1, -1, self.intent_embeddings.size(-1))

        intent_slot_inputs = intent_weights * self.intent_embeddings.unsqueeze(0).expand(intent_weights.size(0), -1, -1)

        intent_slot_inputs = self.intent_to_slot(self.fc_drop(intent_slot_inputs))
        intent_slot_inputs = self.intent_slot_attention(self.fc_drop(share_features), self.fc_drop(intent_slot_inputs))
        logits_slot = self.slot_decoder(intent_slot_inputs)
        return logits_intent, logits_slot  # (b,n_class), #(b,n,n_tag)

    def loss1(self, logits_intent, logits_slot, intent_label, slot_label, mask):
        mask = mask[:, 0:logits_slot.size(1)]

        slot_label = slot_label[:, 0:logits_slot.size(1)]  # 去除掩码

        logits_slot = logits_slot.transpose(1, 0)  # (n,b,n_tag)

        slot_label = slot_label.transpose(1, 0)  # (n,b)

        mask = mask.transpose(1, 0)

        loss_intent = self.intent_criterion(logits_intent, intent_label)

        loss_slot = -self.crflayer(logits_slot, slot_label, mask) / logits_intent.size()[0]  # 单个样本的均值

        return loss_intent, loss_slot

    def pred_intent_slot(self, logits_intent, logits_slot, mask):
        mask = mask[:, 0:logits_slot.size(1)]
        mask = mask.transpose(1, 0)
        logits_slot = logits_slot.transpose(1, 0)
        pred_intent = torch.max(logits_intent, 1)[1]
        pred_slot = self.crflayer.decode(logits_slot, mask=mask)
        return pred_intent, pred_slot

    def gat_layer(self, x, p, mask=None):
        '''

        :param x: (batch_size,seq_len,dim)
        :param p: (batch_size,relation_num,dim)
        :param mask: (batch,seq_len)
        :return:
        '''
        for m in self.layers:
            x, p = m(x, p, mask)
        return x, p


class SlotGate(nn.Module):
    """
    g =\sum v·tanh(c^S_i + W ·c^I)，对c^I乘以权重W进行线性变换，将维度转换和slot_c的单个step一致
    slot_c 维度：[batch_size,maxlen,2*lstm_units]
    """

    def __init__(self, S_dim, I_dim):
        super(SlotGate, self).__init__()
        self.intent_linear = nn.Linear(I_dim, S_dim, bias=True)
        self.v = nn.Parameter(torch.FloatTensor(S_dim))

    def forward(self, slot_x, intent_x):
        # W ·c^I
        intent_gate = self.intent_linear(intent_x).unsqueeze(1)  # (batch_size,dim)-->(batch_size,1, dim)

        slot_gate = self.v * F.tanh(slot_x + intent_gate)  # (batch_size,seq_len, dim)
        slot_gate = torch.sum(slot_gate, axis=2,
                              keepdims=True) + 1e-10  # K.cast指将数据类型转换为float类型。 [batch_size,seq_len,1]

        return slot_x * slot_gate


class Intent_Slot_Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Intent_Slot_Attention, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(2 * hidden_size, 1)
        self.gate = nn.Linear(hidden_size * 2, 1)
        self.__hidden_dim = hidden_size

    def forward(self, p, x, mask=None):
        '''

        :param p: (batch_size, seq_len, 2*dim)
        :param x: (batch_size,nums_intents,dim)
        :param mask:
        :return:
        '''
        q = self.query(p)  # (batch_size, seq_len,dim)

        k = self.key(x)  # (batch_size, relation_num, dim)

        scores = F.softmax(torch.matmul(
            q,
            k.transpose(-2, -1)
        ), dim=-1) / math.sqrt(self.__hidden_dim)

        v = self.value(x)  # (batch_size, relation_num, dim)
        out = torch.einsum('bcl,bld->bcd', scores, v) + p

        g = self.gate(torch.cat([out, p], 2)).sigmoid()
        out = g * out + (1 - g) * p
        return out  # (batch_size, seq_len, dim)


class QKVAttention(nn.Module):
    """
    Attention mechanism based on Query-Key-Value architecture. And
    especially, when query == key == value, it's self-attention.
    """

    def __init__(self, query_dim, key_dim, value_dim, hidden_dim, output_dim, dropout_rate):
        super(QKVAttention, self).__init__()

        # Record hyper-parameters.
        self.__query_dim = query_dim
        self.__key_dim = key_dim
        self.__value_dim = value_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate

        # Declare network structures.
        self.__query_layer = nn.Linear(self.__query_dim, self.__hidden_dim)
        self.__key_layer = nn.Linear(self.__key_dim, self.__hidden_dim)
        self.__value_layer = nn.Linear(self.__value_dim, self.__output_dim)

        self.__dropout_layer = nn.Dropout(p=self.__dropout_rate)

    def forward(self, input_query, input_key, input_value):
        """ The forward propagation of attention.

        Here we require the first dimension of input key
        and value are equal.

        :param input_query: is query tensor, (n, d_q)
        :param input_key:  is key tensor, (m, d_k)
        :param input_value:  is value tensor, (m, d_v)
        :return: attention based tensor, (n, d_h)
        """

        # Linear transform to fine-tune dimension.
        linear_query = self.__query_layer(input_query)
        linear_key = self.__key_layer(input_key)
        linear_value = self.__value_layer(input_value)

        score_tensor = F.softmax(torch.matmul(
            linear_query,
            linear_key.transpose(-2, -1)
        ), dim=-1) / math.sqrt(self.__hidden_dim)

        forced_tensor = torch.matmul(score_tensor, linear_value)
        forced_tensor = self.__dropout_layer(forced_tensor)

        return forced_tensor


class SelfAttention(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(SelfAttention, self).__init__()

        # Record parameters.
        self.__input_dim = input_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate

        # Record network parameters.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__attention_layer = QKVAttention(
            self.__input_dim, self.__input_dim, self.__input_dim,
            self.__hidden_dim, self.__output_dim, self.__dropout_rate
        )

    def forward(self, input_x, seq_lens):
        dropout_x = self.__dropout_layer(input_x)
        attention_x = self.__attention_layer(
            dropout_x, dropout_x, dropout_x
        )

        return attention_x


class UnflatSelfAttention(nn.Module):
    """
    scores each element of the sequence with a linear layer and uses the normalized scores to compute a context over the sequence.
    """

    def __init__(self, d_hid, dropout=0.):
        super().__init__()
        self.scorer = nn.Linear(d_hid, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, inp, lens):

        batch_size, seq_len, d_feat = inp.size()
        inp = self.dropout(inp)

        scores = self.scorer(inp.contiguous().view(-1, d_feat)).view(batch_size, seq_len)  # (batch_size,seq_len)
        max_len = max(lens)
        for i, l in enumerate(lens):
            if l < max_len:
                scores.data[i, l:] = -np.inf

        scores = F.softmax(scores, dim=1)
        context = scores.unsqueeze(2).expand_as(inp).mul(inp).sum(
            1)  # (batch_size,seq_len,dim)*(batch_size,seq_len,dim)--->(batch, dim)
        return context


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SelfOutput, self).__init__()

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Intermediate(nn.Module):
    def __init__(self, args, intermediate_size, hidden_size):
        super(Intermediate, self).__init__()
        self.dense_in = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = nn.ReLU()
        self.dense_out = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.attention_dropout)

    def forward(self, hidden_states_in):
        hidden_states = self.dense_in(hidden_states_in)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense_out(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + hidden_states_in)
        return hidden_states


class Intermediate_I_S(nn.Module):
    def __init__(self, args, intermediate_size, hidden_size):
        super(Intermediate_I_S, self).__init__()

        self.args = args
        self.dense_in = nn.Linear(hidden_size * 6, intermediate_size)

        self.intermediate_act_fn = nn.ReLU()

        self.dense_out = nn.Linear(intermediate_size, hidden_size)

        self.LayerNorm_I = LayerNorm(hidden_size, eps=1e-12)

        self.LayerNorm_S = LayerNorm(hidden_size, eps=1e-12)

        self.dropout = nn.Dropout(self.args.attention_dropout)

    def forward(self, hidden_states_I, hidden_states_S):
        hidden_states_in = torch.cat([hidden_states_I, hidden_states_S], dim=2)  # (batch_size,n, 2*out_size)

        batch_size, max_length, hidden_size = hidden_states_in.size()

        h_pad = torch.zeros(batch_size, 1, hidden_size)

        if self.args.use_gpu and torch.cuda.is_available():
            h_pad = h_pad.cuda()

        h_left = torch.cat([h_pad, hidden_states_in[:, :max_length - 1, :]], dim=1)

        h_right = torch.cat([hidden_states_in[:, 1:, :], h_pad], dim=1)

        hidden_states_in = torch.cat([hidden_states_in, h_left, h_right], dim=2)  # (batch_size,n, 6*out_size)

        hidden_states = self.dense_in(hidden_states_in)  # (batch_size,n, 6*out_size)*(6*out_size,out_size)
        # -->(batch_size,n,out_size)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_out(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states_I_NEW = self.LayerNorm_I(hidden_states + hidden_states_I)
        hidden_states_S_NEW = self.LayerNorm_S(hidden_states + hidden_states_S)

        return hidden_states_I_NEW, hidden_states_S_NEW


class I_S_Block(nn.Module):
    def __init__(self, args, intent_emb, slot_emb, hidden_size):
        super(I_S_Block, self).__init__()

        self.I_S_Attention = I_S_SelfAttention(hidden_size, 2 * hidden_size, hidden_size)

        self.I_Out = SelfOutput(hidden_size, args.attention_dropout)

        self.S_Out = SelfOutput(hidden_size, args.attention_dropout)

        self.I_S_Feed_forward = Intermediate_I_S(hidden_size, hidden_size)

    def forward(self, H_intent_input, H_slot_input, mask):
        '''

        :param H_intent_input: (b,n,d)
        :param H_slot_input: (b,n,d)
        :param mask:
        :return:    H_intent (b,n,d')
                    H_slot (b,n,d')
        '''

        H_slot, H_intent = self.I_S_Attention(H_intent_input, H_slot_input,
                                              mask)  # (batch_size,n, out_size) #(batch_size,n, out_size)

        H_slot = self.S_Out(H_slot, H_slot_input)
        H_intent = self.I_Out(H_intent, H_intent_input)

        H_intent, H_slot = self.I_S_Feed_forward(H_intent, H_slot)

        return H_intent, H_slot


class Label_Attention(nn.Module):
    def __init__(self, intent_emb, slot_emb):
        super(Label_Attention, self).__init__()

        self.W_intent_emb = intent_emb.weight  # (d,v_i)

        self.W_slot_emb = slot_emb.weight  # (d, v_s)

    def forward(self, input_intent, input_slot, mask):
        '''

        :param input_intent: #(batch_size,n,d)
        :param input_slot: #(batch_size,n,d)
        :param mask:
        :return:
        '''
        intent_score = torch.matmul(input_intent, self.W_intent_emb.t())  # (batch_size,n,d)*(d, v_i)-->(b,n,v_i)
        slot_score = torch.matmul(input_slot, self.W_slot_emb.t())  # (batch_size,n,d)*(d, v_s)-->(b,n,v_s)

        intent_probs = nn.Softmax(dim=-1)(intent_score)  # (batch_size,n,d)*(d, v_i)-->(b,n,v_i)
        slot_probs = nn.Softmax(dim=-1)(slot_score)  # (batch_size,n,d)*(d, v_s)-->(b,n,v_s)

        intent_res = torch.matmul(intent_probs, self.W_intent_emb)  # (b,n,v_i)*(v_i,d)-->(b,n,d)
        slot_res = torch.matmul(slot_probs, self.W_slot_emb)  # (b,n,v_s)*(v_s,d)-->(b,n,d)

        return intent_res, slot_res


class I_S_SelfAttention(nn.Module):
    def __init__(self, args, input_size, hidden_size, out_size):
        '''

        :param input_size: hidden
        :param hidden_size: 2*hidden
        :param out_size: hidden_size
        '''
        super(I_S_SelfAttention, self).__init__()

        self.num_attention_heads = 8
        self.attention_head_size = int(hidden_size / self.num_attention_heads)

        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.out_size = out_size

        self.query = nn.Linear(input_size, self.all_head_size)
        self.query_slot = nn.Linear(input_size, self.all_head_size)

        self.key = nn.Linear(input_size, self.all_head_size)
        self.key_slot = nn.Linear(input_size, self.all_head_size)

        self.value = nn.Linear(input_size, self.out_size)
        self.value_slot = nn.Linear(input_size, self.out_size)

        self.dropout = nn.Dropout(args.attention_dropout)

    def transpose_for_scores(self, x):
        last_dim = int(x.size()[-1] / self.num_attention_heads)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, last_dim)
        x = x.view(*new_x_shape)  # (batch_size,seq_len,head_num,dim)
        return x.permute(0, 2, 1, 3)  # (batch_size,head_num,seq_len,dim)

    def forward(self, intent, slot, mask):
        '''

        :param intent: (b,n,d)
        :param slot: (b,n,d)
        :param mask:
        :return:
        '''
        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        attention_mask = (1.0 - extended_attention_mask) * -10000.0

        mixed_query_layer = self.query(intent)  # (b,n,d) *(d,d')-->(b,n,d')
        mixed_key_layer = self.key(slot)  # (b,n,d) *(d,d')-->(b,n,d')
        mixed_value_layer = self.value(slot)  # (b,n,d) *(d,d')-->(b,n,d')

        mixed_query_layer_slot = self.query_slot(slot)  # (b,n,d) *(d,d')-->(b,n,d')
        mixed_key_layer_slot = self.key_slot(intent)  # (b,n,d) *(d,d')-->(b,n,d')
        mixed_value_layer_slot = self.value_slot(intent)  # (b,n,d) *(d,d')-->(b,n,d')

        query_layer = self.transpose_for_scores(mixed_query_layer)  # (batch_size,head_num,seq_len,dim)
        query_layer_slot = self.transpose_for_scores(mixed_query_layer_slot)  # (batch_size,head_num,seq_len,dim)

        key_layer = self.transpose_for_scores(mixed_key_layer)  # (batch_size,head_num,seq_len,dim)
        key_layer_slot = self.transpose_for_scores(mixed_key_layer_slot)  # (batch_size,head_num,seq_len,dim)

        value_layer = self.transpose_for_scores(mixed_value_layer)  # (batch_size,head_num,seq_len,dim)
        value_layer_slot = self.transpose_for_scores(mixed_value_layer_slot)  # (batch_size,head_num,seq_len,dim)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1,
                                                                         -2))  # (batch_size,head_num,seq_len,dim)*(batch_size,head_num,dim,seq_len)
        # ->(batch_size,head_num,seq_len,n,n)

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # attention_scores_slot = torch.matmul(query_slot, key_slot.transpose(1,0))
        attention_scores_slot = torch.matmul(query_layer_slot, key_layer_slot.transpose(-1, -2))
        attention_scores_slot = attention_scores_slot / math.sqrt(self.attention_head_size)
        attention_scores_intent = attention_scores + attention_mask

        attention_scores_slot = attention_scores_slot + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs_slot = nn.Softmax(dim=-1)(attention_scores_slot)
        attention_probs_intent = nn.Softmax(dim=-1)(attention_scores_intent)  # (batch_size,head_num,n,n)

        attention_probs_slot = self.dropout(attention_probs_slot)
        attention_probs_intent = self.dropout(attention_probs_intent)

        context_layer_slot = torch.matmul(attention_probs_slot, value_layer_slot)
        context_layer_intent = torch.matmul(attention_probs_intent,
                                            value_layer)  # (batch_size,head_num,n,n)*(batch_size,head_num,n,n)*(batch_size,head_num,n,dim)
        # -->(batch_size,head_num,n,dim)

        context_layer = context_layer_slot.permute(0, 2, 1, 3).contiguous()

        context_layer_intent = context_layer_intent.permute(0, 2, 1,
                                                            3).contiguous()  # (batch_size,head_num,n,dim)-->(batch_size,n,head_num,dim)

        new_context_layer_shape = context_layer.size()[:-2] + (self.out_size,)

        new_context_layer_shape_intent = context_layer_intent.size()[:-2] + (
            self.out_size,)  # (batch_size,n,)+(out_size)-->(batch_size,n, out_size)

        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer_intent = context_layer_intent.view(*new_context_layer_shape_intent)  # (batch_size,n, out_size)

        return context_layer, context_layer_intent  # (batch_size,n, out_size)


class HGAT(nn.Module):
    def __init__(self, config):

        super(HGAT, self).__init__()

        self.config = config
        hidden_size = config.hidden_size  # 768

        self.embeding = nn.Embedding(config.class_nums, hidden_size)  # (relation,dim)(19,768)

        self.relation = nn.Linear(hidden_size, hidden_size)  # (768,768)

        self.down = nn.Linear(3 * hidden_size, hidden_size)  # (3*768,768)
        # t
        self.start_head = nn.Linear(hidden_size, 1)  # (768,1)

        self.end_head = nn.Linear(hidden_size, 1)  # (768,1)

        self.start_tail = nn.Linear(hidden_size, 1)  # (768,1)

        self.end_tail = nn.Linear(hidden_size, 1)  # (768,1)

        self.layers = nn.ModuleList([GATLayer(hidden_size) for _ in range(config.gat_layers)])

    def forward(self, x, sub_head=None, sub_tail=None, mask=None):
        # relation
        '''

        :param x: (6,128,768)->(batch_size,seq_len,dim)
        :param sub_head: (6) (batch,)
        :param sub_tail: (7) (batch,)
        :param mask:
        :return:
        '''
        p = torch.arange(self.config.class_nums).long()  # (relation_num)

        if torch.cuda.is_available():
            p = p.cuda()

        p = self.relation(self.embeding(p))  # (19,768)->(relation_num,dim)

        p = p.unsqueeze(0).expand(x.size(0), p.size(0), p.size(1))  # bcd #(6,19,768)

        x, p = self.gat_layer(x, p, mask)  # x bcd x:(batch_size, seq_len, dim) p:(batch_size, relatioin, dim)

        ts, te = self.pre_head(x)

        if sub_head is not None and sub_tail is not None:
            e1 = self.entity_trans(x, sub_head, sub_tail)
            hs, he = self.pre_tail(x, e1, p)
            return ts, te, hs, he

        return ts, te


class GATLayer(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()

        self.ra1 = RelationAttention(hidden_size)

        self.ra2 = RelationAttention(hidden_size)

    def forward(self, x, p, mask=None):
        '''

        :param x:   (batch_size,seq_len,dim)
        :param p:   (batch_size,relation_num,dim)
        :param mask:(batch,seq_len)
        :return:
        '''
        x_ = self.ra1(x, p)  # (batch_size, seq_len, dim)
        x = x_ + x

        p_ = self.ra2(p, x, mask)  # (batch_size, relation_num, dim)
        p = p_ + p
        return x, p


class RelationAttention(nn.Module):
    def __init__(self, hidden_size):
        super(RelationAttention, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.score = nn.Linear(2 * hidden_size, 1)
        self.gate = nn.Linear(hidden_size * 2, 1)

    def forward(self, p, x, mask=None):
        '''

        :param p: (batch_size, seq_len, dim)
        :param x: (batch_size,relation_num,dim)
        :param mask:
        :return:
        '''
        q = self.query(p)  # (batch_size, seq_len,dim)

        k = self.key(x)  # (batch_size, relation_num, dim)
        score = self.fuse(q, k)  # (batch, seq_len, relation_num)
        if mask is not None:
            mask = 1 - mask[:, None, :].expand(-1, score.size(1),
                                               -1)  # 1-(batch, seq_len, relation_num) 即：掩码由0变为1，非掩码由1变为0
            score = score.masked_fill(mask == 1, -1e9)

        score = F.softmax(score, 2)  # (batch, seq_len, relation_num) 在关系维度上加权，即所有relation_num对特定token的重要程度
        v = self.value(x)  # (batch_size, relation_num, dim)
        out = torch.einsum('bcl,bld->bcd', score,
                           v) + p  # (batch, seq_len, relation_num)*(batch_size, relation_num, dim)->(batch,seq_len,dim)
        # 上一步将关系对字符的影响加入到seq_len中，得到加权后的表示

        g = self.gate(torch.cat([out, p],
                                2)).sigmoid()  # (batch_szie,seq_len,2*dim)->(batch_size,seq_len,1)-->(batch_size,seq_len,1) 得到0-1之间的概率值
        out = g * out + (1 - g) * p
        return out  # (batch_size, seq_len, dim)

    def fuse(self, x, y):
        '''

        :param x: #(batch_size, seq_len,dim)
        :param y: #(batch_size, relation_num, dim)
        :return:
        '''
        x = x.unsqueeze(2).expand(-1, -1, y.size(1),
                                  -1)  # (batch_size, seq_len,dim)->(batch_size, seq_len,1,dim)->(batch_size, seq_len, relation_num, dim)
        y = y.unsqueeze(1).expand(-1, x.size(1), -1,
                                  -1)  # (batch_size, relation, dim)->(batch_szie, 1, relation, dim)->(batch, seq_len, relation_num, dim)
        temp = torch.cat([x, y], 3)  # (batch, seq_len, relation_num, 2*dim)
        return self.score(temp).squeeze(3)  # (batch, seq_len, relation,1) -> (batch, seq_Len, relation)

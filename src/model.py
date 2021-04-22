import torch
import numpy as np

# -*- encoding:utf-8 -*-
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers.dynamic_rnn import DynamicLSTM
from layers.attention import Attention
import numpy as np
import pickle
from layers.gnn_layer import GraphAttentionLayer

max_doc_len = 75
# max_sen_len = 45

class ShareNetworks(nn.Module):
    def __init__(self, embeddings, opt):
        super().__init__()
        self.opt = opt
        self.in_dim = opt.embed_dim
        self.emb = nn.Embedding.from_pretrained(torch.tensor(embeddings, dtype=torch.float))
        self.mem_dim = 2 * opt.hidden_dim  # 200

        self.input_dropout = nn.Dropout(opt.input_dropout)

        # rnn layer
        if self.opt.no_word_rnn == False:
            self.input_W_R = nn.Linear(self.in_dim, opt.rnn_hidden)
            self.w_rnn = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=opt.rnn_layer, batch_first=True,
                                     bidirectional=True)  # (32,75,45,200)
            self.in_dim = opt.hidden_dim * 2
            self.w_rnn_drop = nn.Dropout(opt.rnn_word_dropout)  # use on last layer output

        if self.opt.no_clause_rnn == False:
            self.input_W_R = nn.Linear(self.in_dim, opt.rnn_hidden)
            self.c_rnn = DynamicLSTM(self.in_dim, opt.hidden_dim, num_layers=opt.rnn_layer, batch_first=True,
                                     bidirectional=True)  # (32,75,45,200)
            self.in_dim = opt.hidden_dim * 2
            self.c_rnn_drop = nn.Dropout(opt.rnn_clause_dropout)  # use on last layer output

        self.in_drop = nn.Dropout(opt.input_dropout)

        # self.emo_trans = nn.Linear(self.in_dim, 100)

        self.clause_encode = Attention(2 * opt.hidden_dim, 1, opt.max_sen_len, opt)  # (32,75,200)


    def pack_sen_len(self, sen_len):
        """
        :param sen_len: [32, 75]
        :return:
        """
        batch_size = sen_len.shape[0]
        up_sen_len = np.zeros([batch_size, self.opt.max_doc_len])
        for i, doc in enumerate(sen_len):
            for j, sen in enumerate(doc):
                if sen == 0:
                    up_sen_len[i][j] = 1
                else:
                    up_sen_len[i][j] = sen

        return torch.tensor(up_sen_len)


    def forward(self, inputs):
        words, sen_len, doc_len, doc_id = inputs
        up_sen_len = self.pack_sen_len(sen_len)
        word_embs = torch.reshape(self.emb(words),
                                  [-1, self.opt.max_sen_len, 2 * self.opt.hidden_dim])  # (32*75, 45, 200)
        word_embs = self.input_dropout(word_embs)
        if self.opt.no_word_rnn == False:
            up_sen_len = torch.reshape(up_sen_len, [-1])
            word_embs = self.w_rnn_drop(self.w_rnn(word_embs, up_sen_len))
        clause_encode = self.clause_encode(word_embs, sen_len)

        if self.opt.no_clause_rnn == False:
            clause_encode = self.c_rnn_drop(self.c_rnn(clause_encode, doc_len.cpu()))

        return clause_encode

class Networks(nn.Module):
    def __init__(self, embeddings, opt):
        super().__init__()
        self.opt = opt
        self.shared_networks = ShareNetworks(embeddings, opt)
        self.emotion_prediction = Emotion_Predictions(configs=opt, in_dim=2 * opt.hidden_dim)
        self.emotion_oriented_pair_prediction = Emotion_Oriented_Pair_Prediction(configs=opt)
        self.no_emotion_oriented_pair_prediction = No_Emotion_Oriented_Pair_Prediction(configs=opt)

    def pack_sen_len(self, sen_len):
        """
        :param sen_len: [32, 75]
        :return:
        """
        batch_size = sen_len.shape[0]
        up_sen_len = np.zeros([batch_size, self.opt.max_doc_len])
        for i, doc in enumerate(sen_len):
            for j, sen in enumerate(doc):
                if sen == 0:
                    up_sen_len[i][j] = 1
                else:
                    up_sen_len[i][j] = sen

        return torch.tensor(up_sen_len)


    def output_util(self, emo_cau_pos, doc_couples, y_mask, test=False):
        """
        TODO: combine this function to data_loader
        """
        batch = self.opt.batch_size
        doc_len = int(math.sqrt(len(emo_cau_pos)))

        couples_mask = np.zeros([batch, doc_len, doc_len])
        couples_true = np.zeros([batch, doc_len, doc_len])
        for i in range(batch):
            y_mask_i = y_mask[i]
            max_doc_idx = sum(y_mask_i) - 1

            doc_couples_i = doc_couples[i]
            for couple_idx, emo_cau in enumerate(emo_cau_pos):
                if emo_cau[0] > max_doc_idx or emo_cau[1] > max_doc_idx:
                    couples_mask[i][emo_cau[0]][emo_cau[1]] = 0
                    # couples_true_i.append(0)
                    couples_true[i][emo_cau[0]][emo_cau[1]] = 0
                else:
                    couples_mask[i][emo_cau[0]][emo_cau[1]] = 1
                    # couples_true_i.append(1 if emo_cau in doc_couples_i else 0)
                    if emo_cau in doc_couples_i:
                        couples_true[i][emo_cau[0]][emo_cau[1]] = 1
                    else:
                        couples_true[i][emo_cau[0]][emo_cau[1]] = 0

        return couples_true, couples_mask

    def couple_generator(self, doc_sents_h):
        ###构造真实的情感原因对,需要在限定窗口内进行的
        batch, seq_len, _ = doc_sents_h.size()

        base_idx = np.arange(0, seq_len)
        emo_pos = np.concatenate([base_idx.reshape(-1, 1)] * seq_len, axis=1).reshape(1, -1)[0]
        cau_pos = np.concatenate([base_idx] * seq_len, axis=0)

        rel_pos = cau_pos - emo_pos
        rel_pos = torch.LongTensor(rel_pos).to(self.opt.device)
        emo_pos = torch.LongTensor(emo_pos).to(self.opt.device)
        cau_pos = torch.LongTensor(cau_pos).to(self.opt.device)

        rel_pos = rel_pos.unsqueeze(0).expand(batch, -1)

        emo_cau_pos = []
        for emo, cau in zip(emo_pos.tolist(), cau_pos.tolist()):
            emo_cau_pos.append([emo, cau])
        return rel_pos, emo_cau_pos

    def prob_pair(self, couples_true, couples_mask, couple_index):
        all_labels = []
        all_masks = []
        for i, indexes in enumerate(couple_index):
            labels = []
            masks = []
            for pair in indexes:
                emo = pair[0]
                cau = pair[1]
                label = couples_true[i][emo][cau]
                labels.append(label)
                mask = couples_mask[i][emo][cau]
                masks.append(mask)

            all_labels.append(labels)
            all_masks.append(masks)

        return all_labels, all_masks

    def loss_pair(self, couples_pred, couples_true, couple_index, couples_mask):

        ###从couples_true的取出对应的couple_true
        labels, masks = self.prob_pair(couples_true, couples_mask, couple_index)

        couples_mask = torch.ByteTensor(masks).to(self.opt.device)
        couples_true = torch.FloatTensor(labels).to(self.opt.device)
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        couples_true = couples_true.masked_select(couples_mask)
        couples_pred = couples_pred.masked_select(couples_mask)
        loss_pair = criterion(couples_pred, couples_true)

        return loss_pair, labels
    def forward(self, inputs):
        clause_encode = self.shared_networks(inputs)
        rel_pos, emo_cau_pos = self.couple_generator(clause_encode)
        emo_pred, topk_index, pair_index, cand_emo_encode, context_clause, no_emotion_clause, context_no_emotion_clause, pair_no_emotion_oriented_index \
            = self.emotion_prediction(clause_encode)
        pair_pred = self.emotion_oriented_pair_prediction(cand_emo_encode, context_clause)
        pair_no_emo_pred = self.no_emotion_oriented_pair_prediction(no_emotion_clause, context_no_emotion_clause)

        ##分别计算两个不同的loss
        ##一个计算用面向emotion构造出来的pair计算LOSS, 一个用不面向emotion构造出来的pair的来计算loss.
        return topk_index, emo_pred, pair_pred, pair_index, pair_no_emo_pred, pair_no_emotion_oriented_index, emo_cau_pos


class GraphNN(nn.Module):
    def __init__(self, configs):
        super(GraphNN, self).__init__()
        in_dim = configs.feat_dim
        self.gnn_dims = [in_dim] + [int(dim) for dim in configs.gnn_dims.strip().split(',')]

        self.gnn_layers = len(self.gnn_dims) - 1
        self.att_heads = [int(att_head) for att_head in configs.att_heads.strip().split(',')]
        self.gnn_layer_stack = nn.ModuleList()
        for i in range(self.gnn_layers):
            in_dim = self.gnn_dims[i] * self.att_heads[i - 1] if i != 0 else self.gnn_dims[i]
            self.gnn_layer_stack.append(
                GraphAttentionLayer(self.att_heads[i], in_dim, self.gnn_dims[i + 1], configs.dp)
            )

    def forward(self, doc_sents_h, doc_len, adj):
        batch, max_doc_len, _ = doc_sents_h.size()
        assert max(doc_len) == max_doc_len

        for i, gnn_layer in enumerate(self.gnn_layer_stack):
            doc_sents_h = gnn_layer(doc_sents_h, adj)

        return doc_sents_h

class Emotion_Predictions(nn.Module):
    def __init__(self, configs, in_dim=None):
        super(Emotion_Predictions, self).__init__()
        if not in_dim:
            in_dim = 2 * configs.hidden_dim

        self.topk = configs.topk
        self.configs = configs
        self.window_size = configs.window_size
        self.out_e = nn.Linear(in_dim, 1)

        ####将common rep 分别映射到emotion 和 context维度来做
        self.emo_trans = nn.Linear(in_dim, in_dim)
        self.con_trans = nn.Linear(in_dim, in_dim)

        self.emo_dropout = nn.Dropout(configs.emo_dropout)
        self.con_dropout = nn.Dropout(configs.con_dropout)

    def emotion_index_select(self, doc_sents_h, emotion_index):
        # print(emotion_index.size())
        # print(doc_sents_h.size())
        if len(emotion_index.size()) == 1:
            emotion_index = emotion_index.unsqueeze(0)
        dummy = emotion_index.unsqueeze(-1).expand((doc_sents_h.size(0), emotion_index.size(1), doc_sents_h.size(2)))
        emotion_encode = doc_sents_h.gather(1, dummy)
        return emotion_encode
    def context_index_select(self, doc_sents_h, context_index):
        dummy = context_index.view(-1, context_index.size(1)*self.window_size)
        dummy = dummy.unsqueeze(-1).expand(doc_sents_h.size(0), context_index.size(1)*self.window_size, doc_sents_h.size(2))
        context_encode = doc_sents_h.gather(1, dummy).view(doc_sents_h.size(0), context_index.size(1), self.window_size, doc_sents_h.size(2))
        return context_encode

    def gen_pairs(self, cand_emotion_index, context_index):
        if len(cand_emotion_index.size()) == 1:
            cand_emotion_index = cand_emotion_index.unsqueeze(0)
        emotion_index = cand_emotion_index.unsqueeze(-1).expand(cand_emotion_index.size(0), cand_emotion_index.size(1),
                                                                  self.configs.window_size)

        pair_index_all = []
        emotion_index = torch.reshape(emotion_index,[cand_emotion_index.size(0), -1])
        context_index_tmp = torch.reshape(context_index,[cand_emotion_index.size(0), -1])

        for i in range(cand_emotion_index.size(0)):
            pair_index = []
            emotion = emotion_index[i]
            context = context_index_tmp[i]
            for emo, cau in zip(emotion.tolist(), context.tolist()):
                pair_index.append([emo, cau])
            pair_index_all.append(pair_index)
        return pair_index_all

    def gen_window(self, emotion_index, doc_len):
        if len(emotion_index.size()) == 1:
            emotion_index = emotion_index.unsqueeze(0)
        batch_size = emotion_index.size(0)
        offsets = torch.arange(-2, 3).repeat(batch_size).to(self.configs.device)

        out = emotion_index.view(batch_size, emotion_index.size(1), 1) + offsets.view(batch_size, 1, 5)
        return out.clamp(0, doc_len-1)
    def build_no_emotion_origented_index(self, emotion_oriented_index, doc_len):
        if len(emotion_oriented_index.size()) == 1:
            emotion_oriented_index = emotion_oriented_index.unsqueeze(0)
        batch_size = emotion_oriented_index.size(0)
        emotion_index = emotion_oriented_index.cpu().numpy() #(32,3)
        all_index = np.tile(np.expand_dims(np.arange(0, doc_len), axis=0), batch_size).reshape([batch_size, doc_len]) #(32,75)
        no_emotion_index = torch.LongTensor([np.setdiff1d(all_index[i], emotion_index[i]) for i in range(batch_size)]).to(self.configs.device)
        return no_emotion_index

    def forward(self, doc_sents_h, window_size=5):
        emo_rep = self.emo_trans(doc_sents_h)
        con_rep = self.con_trans(doc_sents_h)

        emo_rep = self.emo_dropout(emo_rep)
        con_rep = self.con_dropout(con_rep)

        pred_e = self.out_e(emo_rep).squeeze()##(32, doc_len, 1)
        emotion_oriented_index = torch.topk(pred_e, self.topk, dim=-1).indices ###(32, 3)
        doc_len = doc_sents_h.size(1)

        ####取对应INDEX下的句子表示
        cand_emotion_clause = self.emotion_index_select(emo_rep, emotion_oriented_index)
        ####基于topk_index 生成面向情感的的窗口范围(32, 3, 5)
        context_emotion_index = self.gen_window(emotion_oriented_index, doc_len)
        context_clause = self.context_index_select(con_rep, context_emotion_index)
        pair_emotion_oriented_index = self.gen_pairs(emotion_oriented_index, context_emotion_index)



        no_emotion_oriented_index = self.build_no_emotion_origented_index(emotion_oriented_index, doc_len)

        ###取非情感子句的表示(32, 72, 200)
        no_emotion_clause = self.emotion_index_select(emo_rep, no_emotion_oriented_index)
        context_no_emotion_index = self.gen_window(no_emotion_oriented_index, doc_len)
        context_no_emotion_clause = self.context_index_select(con_rep, context_no_emotion_index)
        pair_no_emotion_oriented_index = self.gen_pairs(no_emotion_oriented_index, context_no_emotion_index)

        return pred_e, emotion_oriented_index, pair_emotion_oriented_index, cand_emotion_clause, context_clause, \
    no_emotion_clause,context_no_emotion_clause,pair_no_emotion_oriented_index

class Emotion_Oriented_Pair_Prediction(nn.Module):
    def __init__(self, configs):
        super(Emotion_Oriented_Pair_Prediction, self).__init__()
        self.configs = configs

        self.agg_linear = nn.Linear(2*configs.hidden_dim, 2*configs.hidden_dim//configs.window_size)
        self.kl = nn.KLDivLoss()
        self.classifier = nn.Linear(2*configs.hidden_dim, 1)

    def forward(self, cand_emotion_clause, context_clause):
        #cand_emotion_clause => (32, 3, 200)
        #context_clause => (32, 3, 5, 200)

        ###生成emotion oriented pair 表示：
        ###这里可以有多种方式来做。
        if self.configs.oriented_way == 1:
            cand_eq = cand_emotion_clause.unsqueeze(dim=2).transpose(2,3)
            score = torch.matmul(context_clause, cand_eq) ## (32, 3, 5, 1)
            ###softmax
            pair_oriented = context_clause * score ###(元素乘) (32, 3, 5, 200)
            ###利用context_oriented 重新生成emotion clause representation.

        elif self.configs.oriented_way == 2:
            ###直接元素相乘
            cand_eq = cand_emotion_clause.unsqueeze(dim=2).expand(cand_emotion_clause.size(0), cand_emotion_clause.size(1),
                                                                  self.configs.window_size, cand_emotion_clause.size(-1))
            pair_oriented = cand_eq * context_clause ##(32, 3, 5, 200)
        else:
            ###按照拼接来做
            cand_eq = cand_emotion_clause.unsqueeze(dim=2).expand(cand_emotion_clause.size(0), cand_emotion_clause.size(1), self.configs.window_size, cand_emotion_clause.size(-1))
            ###(32, 3, 5, 200)
            pair_oriented = torch.cat([cand_eq, context_clause], dim=-1) ###(32, 3, 5, 400)

        # con_eq = self.agg_linear(pair_oriented).view(pair_oriented.size(0), pair_oriented.size(1), -1)
        # emotion_loss = self.kl(cand_emotion_clause, con_eq)
        pair_pred = self.classifier(pair_oriented)
        pair_pred = pair_pred.view(pair_pred.size(0), pair_oriented.size(1) * self.configs.window_size)
        return pair_pred


class No_Emotion_Oriented_Pair_Prediction(nn.Module):
    def __init__(self, configs):
        super(No_Emotion_Oriented_Pair_Prediction, self).__init__()
        self.configs = configs
        self.classifier = nn.Linear(2 * 2*configs.hidden_dim, 1)


    def forward(self, no_emotion_clause, context_no_emotion_clause):
        ###想想如何组成这种no_emotion_oriented pair的表示更有助于前面emotion representation的学
        no_eq = no_emotion_clause.unsqueeze(dim=2).expand(no_emotion_clause.size(0), no_emotion_clause.size(1),
                                                              self.configs.window_size, no_emotion_clause.size(-1))
        ###(32, 72, 5, 200)
        pair_no_oriented = torch.cat([no_eq, context_no_emotion_clause], dim=-1)  ###(32, 72, 5, 400)
        pair_no_emotion_pred = self.classifier(pair_no_oriented).view(no_eq.size(0),\
                                                                      no_emotion_clause.size(1) * self.configs.window_size)
        return pair_no_emotion_pred















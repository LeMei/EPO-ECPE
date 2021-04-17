# -*- encoding:utf-8 -*-
'''
@time: 2019/12/21 9:48 下午
@author: huguimin
@email: 718400742@qq.com
'''
import os
import random
import math
import torch
import pickle
import argparse
import numpy as np
from utils.loader_data import *
# from models.ecaggcn_no_dcn import ECClassifier
from sklearn import metrics
import torch.nn as nn
import time
# from src.model import Networks
from src.model_recon_14 import Networks

###尝试采用梯度裁剪and将预训练部分和正式训练部分分别用两个优化器来做

class Model:

    def __init__(self, opt, idx):
        self.opt = opt
        self.embedding = load_embedding(opt.embedding_path)
        self.emo_embedding = load_emo_embedding(opt.emo_embedding_path)
        self.embedding_pos = load_pos_embedding(opt.embedding_dim_pos)
        self.split_size = math.ceil(opt.data_size / opt.n_split)

        self.global_f1 = 0
        # self.train, self.test = load_data(self.split_size, idx, opt.data_size) #意味着只能从一个角度上训练，应该换几种姿势轮着训练
        if opt.dataset == 'EC':
            self.train, self.test = load_percent_train(opt.per, self.split_size, idx, opt.data_size)
        elif opt.dataset == 'EC_en':
            self.train, self.test = load_data_en()
        else:
            print('DATASET NOT EXIST')
        # self.train, self.test = load_data(self.split_size, idx, opt.data_size)
        self.sub_model = opt.model_class(self.embedding, self.opt).to(opt.device)
        with open(opt.emotion_model, 'rb') as fr:
            self.emotion_model = pickle.load(fr)


    def _reset_params(self):
        for p in self.sub_model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.opt.initializer(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _print_args(self):
        n_trainable_params, n_nontrainable_params, model_params = 0, 0, 0
        for p in self.sub_model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            model_params += n_params
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('n_trainable_params: {0}, n_nontrainable_params: {1}, model_params: {2}'.format(n_trainable_params, n_nontrainable_params, model_params))
        print('> training arguments:')
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _train(self, pre_optimizer, train_optimizer, f_out):

        ec_max_test_pre = 0
        ec_max_test_rec = 0
        ec_max_test_f1 = 0

        e_max_test_pre = 0
        e_max_test_rec = 0
        e_max_test_f1 = 0

        c_max_test_pre = 0
        c_max_test_rec = 0
        c_max_test_f1 = 0

        m_ec_max_test_pre = 0
        m_ec_max_test_rec = 0
        m_ec_max_test_f1 = 0

        m_e_max_test_pre = 0
        m_e_max_test_rec = 0
        m_e_max_test_f1 = 0

        m_c_max_test_pre = 0
        m_c_max_test_rec = 0
        m_c_max_test_f1 = 0

        global_step = 0
        continue_not_increase = 0

        # for epoch in range(self.opt.pre_num_epoch):
        #      print('pre_training' + '>' * 100)
        #      print('epoch: ', epoch)
        #
        #      for train in get_train_batch_data(self.train, self.opt.batch_size, self.opt.keep_prob1,
        #                                        self.opt.keep_prob2):
        #          global_step += 1
        #          self.sub_model.train()
        #          #pre_optimizer.zero_grad()
        #
        #          inputs = [train[col].to(self.opt.device) for col in self.opt.inputs_cols]
        #          topk_index, emo_pred, pair_pred, pair_index, pair_no_emo_pred, \
        #          pair_no_emotion_oriented_index, emo_cau_pos = self.sub_model(inputs)
        #
        #          doc_len_batch = emo_pred.size(1)
        #          y_mask = train['y_mask'][:, :doc_len_batch]
        #          emo_targets = train['y_emotion'].to(self.opt.device)[:, :doc_len_batch]
        #
        #          emo_targets = torch.argmax(emo_targets, dim=2).float()
        #
        #
        #          ##在情感上设置下mask
        #          y_mask = y_mask.bool().to(self.opt.device)
        #          emo_pred = emo_pred.masked_select(y_mask)
        #          emo_targets = emo_targets.masked_select(y_mask)
        #
        #          ###通过pos_weight来调整下准确率和召回率
        #          pos_weight = torch.where(emo_targets==1, 1.5, 1.0)
        #          criterion = nn.BCEWithLogitsLoss(reduction='mean',pos_weight=pos_weight)
        #
        #          emotion_loss = criterion(emo_pred, emo_targets)
        #          doc_couple = train['doc_couple']
        #          couples_true, couples_mask = \
        #              self.sub_model.output_util(emo_cau_pos, doc_couple, y_mask)
        #          loss_no_emo_pair, no_emo_labels = self.sub_model.loss_pair(pair_no_emo_pred, couples_true,
        #                                                                     pair_no_emotion_oriented_index,
        #                                                                     couples_mask)
        #          loss1 = emotion_loss + loss_no_emo_pair
        #          # loss1 = loss_no_emo_pair
        #
        #
        #          if global_step % 2 == 0:
        #
        #             loss1.backward()
        #             pre_optimizer.step()


        for epoch in range(self.opt.num_epoch):
            print('>' * 100)
            print('epoch: ', epoch)
            n_correct, n_total = 0, 0
            increase_flag = False

            for train in get_train_batch_data(self.train, self.opt.batch_size, self.opt.keep_prob1, self.opt.keep_prob2):
                global_step += 1
                self.sub_model.train()
                train_optimizer.zero_grad()

                inputs = [train[col].to(self.opt.device) for col in self.opt.inputs_cols]
                topk_index, emo_pred, pair_pred, pair_index, pair_no_emo_pred, \
                pair_no_emotion_oriented_index, emo_cau_pos = self.sub_model(inputs)

                doc_len_batch = emo_pred.size(1)
                doc_id_batch = train['doc_id']
                emo_targets = train['y_emotion'].to(self.opt.device)[:, :doc_len_batch]

                doc_couple = train['doc_couple']
                emo_targets = torch.argmax(emo_targets, dim=2).float()
                y_mask = train['y_mask'][:, :doc_len_batch]

                y_mask = y_mask.bool().to(self.opt.device)
                emo_pred = emo_pred.masked_select(y_mask)
                emo_targets = emo_targets.masked_select(y_mask)

                ###通过pos_weight来调整下准确率和召回率
                pos_weight = torch.where(emo_targets == 1, 1.5, 1.0)
                criterion = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)
                emotion_loss = criterion(emo_pred, emo_targets)

                ###分别计算两个通道下的loss损失
                ###面向情感对的损失
                couples_true, couples_mask = \
                    self.sub_model.output_util(emo_cau_pos, doc_couple, y_mask)

                loss_emo_pair, emo_labels = self.sub_model.loss_pair(pair_pred, couples_true, pair_index, couples_mask)
                ###非面向情感对的损失
                loss_no_emo_pair, no_emo_labels = self.sub_model.loss_pair(pair_no_emo_pred, couples_true, pair_no_emotion_oriented_index, couples_mask)
                loss = loss_emo_pair + loss_no_emo_pair + emotion_loss
                # loss = loss_emo_pair + loss_no_emo_pair

                #loss = emotion_loss + loss_emo_pair

                loss.backward()
                ##梯度裁剪部分
                # nn.utils.clip_grad_norm(self.sub_model.parameters(), max_norm=20, norm_type=2)
                train_optimizer.step()
                if global_step % self.opt.log_step == 0:
                    train_optimizer.step()
                    print('Train: loss:{:.4f}\n'.format(loss))
                    f_out.write('Train: loss:{:.4f}\n'.format(loss))
                    emo_cau_pair = self._evaluate_prf_binary(doc_id_batch, pair_pred, pair_index)
                    ec_train, e_train, c_train = self.eval_func(list(train['doc_couple']), emo_cau_pair)
                    print('Train: emotion-caus-pair: P {:.4f} R {:.4f} F {:.4f}\n'.format(ec_train[0], ec_train[1], ec_train[2]))
                    print('Train: emotion: P {:.4f} R {:.4f} F {:.4f}\n'.format(e_train[0], e_train[1], e_train[2]))
                    print('Train: cause: P {:.4f} R {:.4f} F {:.4f}\n'.format(c_train[0], c_train[1], c_train[2]))

                    f_out.write('Train: emotion-caus-pair: P {:.4f} R {:.4f} F {:.4f}\n'.format(ec_train[0], ec_train[1], ec_train[2]))
                    f_out.write('Train: emotion: P {:.4f} R {:.4f} F {:.4f}\n'.format(e_train[0], e_train[1], e_train[2]))
                    f_out.write('Train: cause: P {:.4f} R {:.4f} F {:.4f}\n'.format(c_train[0], c_train[1], c_train[2]))

                    sin_ec, sin_e, sin_c, multi_ec, multi_e, multi_c = self._evaluate_acc_f1()
                    print('Single Test: emotion-caus-pair: P {:.4f} R {:.4f} F {:.4f}\n'.format(sin_ec[0], sin_ec[1], sin_ec[2]))
                    print('Single Test: emotion: P {:.4f} R {:.4f} F {:.4f}\n'.format(sin_e[0], sin_e[1], sin_e[2]))
                    print('Single Test: cause: P {:.4f} R {:.4f} F {:.4f}\n'.format(sin_c[0], sin_c[1], sin_c[2]))

                    f_out.write('Single Test: emotion-caus-pair: P {:.4f} R {:.4f} F {:.4f}\n'.format(sin_ec[0], sin_ec[1], sin_ec[2]))
                    f_out.write('Single Test: emotion: P {:.4f} R {:.4f} F {:.4f}\n'.format(sin_e[0], sin_e[1], sin_e[2]))
                    f_out.write('Single Test: cause: P {:.4f} R {:.4f} F {:.4f}\n'.format(sin_c[0], sin_c[1], sin_c[2]))

                    print('Multi Test: emotion-caus-pair: P {:.4f} R {:.4f} F {:.4f}\n'.format(multi_ec[0], multi_ec[1],
                                                                                                multi_ec[2]))
                    print('Multi Test: emotion: P {:.4f} R {:.4f} F {:.4f}\n'.format(multi_e[0], multi_e[1], multi_e[2]))
                    print('Multi Test: cause: P {:.4f} R {:.4f} F {:.4f}\n'.format(multi_c[0], multi_c[1], multi_c[2]))

                    f_out.write(
                        'Multi Test: emotion-caus-pair: P {:.4f} R {:.4f} F {:.4f}\n'.format(multi_ec[0], multi_ec[1],
                                                                                              multi_ec[2]))
                    f_out.write(
                        'Multi Test: emotion: P {:.4f} R {:.4f} F {:.4f}\n'.format(multi_e[0], multi_e[1], multi_e[2]))
                    f_out.write('Multi Test: cause: P {:.4f} R {:.4f} F {:.4f}\n'.format(multi_c[0], multi_c[1], multi_c[2]))

                    if sin_ec[2] > ec_max_test_f1:
                        ec_max_test_f1 = sin_ec[2]
                        ec_max_test_pre = sin_ec[0]
                        ec_max_test_rec = sin_ec[1]

                    if sin_e[2] > e_max_test_f1:
                        e_max_test_f1 = sin_e[2]
                        e_max_test_pre = sin_e[0]
                        e_max_test_rec = sin_e[1]

                    if sin_c[2] > c_max_test_f1:
                        c_max_test_f1 = sin_c[2]
                        c_max_test_pre = sin_c[0]
                        c_max_test_rec = sin_c[1]


                    if multi_ec[2] > m_ec_max_test_f1:
                        m_ec_max_test_f1 = multi_ec[2]
                        m_ec_max_test_pre = multi_ec[0]
                        m_ec_max_test_rec = multi_ec[1]

                    if multi_e[2] > m_e_max_test_f1:
                        m_e_max_test_f1 = multi_e[2]
                        m_e_max_test_pre = multi_e[0]
                        m_e_max_test_rec = multi_e[1]

                    if multi_c[2] > m_c_max_test_f1:
                        m_c_max_test_f1 = multi_c[2]
                        m_c_max_test_pre = multi_c[0]
                        m_c_max_test_rec = multi_c[1]

            # if increase_flag == False:
            #      continue_not_increase += 1
            #      if continue_not_increase >= 20:
            #          print('early stop.')
            #          break
            # else:
            #      continue_not_increase = 0

        return (ec_max_test_pre, ec_max_test_rec, ec_max_test_f1), \
               (e_max_test_pre, e_max_test_rec, e_max_test_f1), \
               (c_max_test_pre, c_max_test_rec, c_max_test_f1), \
               (m_ec_max_test_pre, m_ec_max_test_rec, m_ec_max_test_f1), \
               (m_e_max_test_pre, m_e_max_test_rec, m_e_max_test_f1), \
               (m_c_max_test_pre, m_c_max_test_rec, m_c_max_test_f1)

    def _evaluate_acc_f1(self):
        # switch model to evaluation mode
        self.sub_model.eval()
        with torch.no_grad():
            all_emo_cau_pairs = None
            all_doc_pairs = None
            for test in get_test_single_batch_data(self.test, self.opt.batch_size):
                inputs = [test[col].to(self.opt.device) for col in self.opt.inputs_cols]
                topk_index, emo_pred, pair_pred, pair_index, pair_no_emo_pred, \
                pair_no_emotion_oriented_index, emo_cau_pos = self.sub_model(inputs)

                doc_id_batch = test['doc_id']
                doc_couple = list(test['doc_couple'])

                emo_cau_pair1 = self._evaluate_prf_binary(doc_id_batch, pair_pred, pair_index)
                # emo_cau_pair2 = self._evaluate_prf_binary(doc_id_batch, pair_no_emo_pred, pair_no_emotion_oriented_index)
                if all_emo_cau_pairs is None:
                    all_emo_cau_pairs = emo_cau_pair1
                    all_doc_pairs = doc_couple
                else:
                    all_emo_cau_pairs.extend(emo_cau_pair1)
                    all_doc_pairs.extend(doc_couple)

            sin_ec, sin_e, sin_c = self.eval_func(all_doc_pairs, all_emo_cau_pairs)

            all_emo_cau_pairs = None
            all_doc_pairs = None
            for test in get_test_multi_batch_data(self.test, self.opt.batch_size):
                inputs = [test[col].to(self.opt.device) for col in self.opt.inputs_cols]
                topk_index, emo_pred, pair_pred, pair_index, pair_no_emo_pred, \
                pair_no_emotion_oriented_index, emo_cau_pos = self.sub_model(inputs)

                doc_id_batch = test['doc_id']
                doc_couple = list(test['doc_couple'])

                emo_cau_pair1 = self._evaluate_prf_binary(doc_id_batch, pair_pred, pair_index)
                # emo_cau_pair2 = self._evaluate_prf_binary(doc_id_batch, pair_no_emo_pred, pair_no_emotion_oriented_index)
                if all_emo_cau_pairs is None:
                    all_emo_cau_pairs = emo_cau_pair1
                    all_doc_pairs = doc_couple
                else:
                    all_emo_cau_pairs.extend(emo_cau_pair1)
                    all_doc_pairs.extend(doc_couple)

            multi_ec, multi_e, multi_c = self.eval_func(all_doc_pairs, all_emo_cau_pairs)
            return sin_ec, sin_e, sin_c, multi_ec, multi_e, multi_c




    def _evaluate_prf_binary(self, doc_ids, pair_pred, pair_index):
        ###pair_pred (32,15)//(32, 5*)
        ###pair_index (32,15,2)//(32, 5*,2)

        top1 = torch.topk(pair_pred, 1).indices

        emo_cau_pairs = []

        for i, sample in enumerate(pair_pred):
            emo_cau_pair = []
            emo_index = pair_index[i][top1[i]][0]
            if logistic(pair_pred[i][top1[i]]) <= 0.5 and (emo_index + 1) in self.emotion_model[str(doc_ids[i].item())]:
                emo_cau_pair.append(pair_index[i][top1[i]])
            for j in range(0, sample.shape[-1]):
                if logistic(sample[j]) > 0.5 and pair_index[i][j][0] + 1 in self.emotion_model[str(doc_ids[i].item())]:
                    emo_cau_pair.append(pair_index[i][j])
            emo_cau_pairs.append(emo_cau_pair)

        return emo_cau_pairs


    def eval_func(self, doc_couples_all, doc_couples_pred_all):
        tmp_num = {'ec': 0, 'e': 0, 'c': 0}
        tmp_den_p = {'ec': 0, 'e': 0, 'c': 0}
        tmp_den_r = {'ec': 0, 'e': 0, 'c': 0}

        for doc_couples, doc_couples_pred in zip(doc_couples_all, doc_couples_pred_all):
            doc_couples = set([','.join(list(map(lambda x: str(x), doc_couple))) for doc_couple in doc_couples])
            doc_couples_pred = set(
                [','.join(list(map(lambda x: str(x), doc_couple))) for doc_couple in doc_couples_pred])

            tmp_num['ec'] += len(doc_couples & doc_couples_pred)
            tmp_den_p['ec'] += len(doc_couples_pred)
            tmp_den_r['ec'] += len(doc_couples)

            doc_emos = set([doc_couple.split(',')[0] for doc_couple in doc_couples])
            doc_emos_pred = set([doc_couple.split(',')[0] for doc_couple in doc_couples_pred])
            tmp_num['e'] += len(doc_emos & doc_emos_pred)
            tmp_den_p['e'] += len(doc_emos_pred)
            tmp_den_r['e'] += len(doc_emos)

            doc_caus = set([doc_couple.split(',')[1] for doc_couple in doc_couples])
            doc_caus_pred = set([doc_couple.split(',')[1] for doc_couple in doc_couples_pred])
            tmp_num['c'] += len(doc_caus & doc_caus_pred)
            tmp_den_p['c'] += len(doc_caus_pred)
            tmp_den_r['c'] += len(doc_caus)

        metrics = {}
        for task in ['ec', 'e', 'c']:
            p = tmp_num[task] / (tmp_den_p[task] + 1e-8)
            r = tmp_num[task] / (tmp_den_r[task] + 1e-8)
            f = 2 * p * r / (p + r + 1e-8)
            metrics[task] = (p, r, f)

        return metrics['ec'], metrics['e'], metrics['c']


    def run(self, folder, repeats=1):
        # Loss and Optimizer
        print(('-'*50 + 'Folder{}' + '-'*50).format(folder))
        # criterion = nn.CrossEntropyLoss()
        # criterion = nn.functional.nll_loss()
        pre_params = list(self.sub_model.shared_networks.parameters()) + list(self.sub_model.emotion_prediction.parameters())+\
                     (list(self.sub_model.no_emotion_oriented_pair_prediction.parameters()))

        pre_params_ = filter(lambda p: p.requires_grad, pre_params)

        pre_optimizer = self.opt.optimizer(pre_params_, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

        train_params = list(self.sub_model.shared_networks.parameters()) + list(
            self.sub_model.emotion_prediction.parameters()) + \
                     (list(self.sub_model.emotion_oriented_pair_prediction.parameters()))

        train_params_ = filter(lambda p: p.requires_grad, train_params)

        train_optimizer = self.opt.optimizer(train_params_, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

        if not os.path.exists('log/single_multi_/'):
            os.mkdir('log/single_multi_/')

        f_out = open('log/single_multi_/' + self.opt.model_name + '_' + str(folder) + '_test.txt', 'a+', encoding='utf-8')

        ec_max_test_pre_avg = 0
        ec_max_test_rec_avg = 0
        ec_max_test_f1_avg = 0

        e_max_test_pre_avg = 0
        e_max_test_rec_avg = 0
        e_max_test_f1_avg = 0

        c_max_test_pre_avg = 0
        c_max_test_rec_avg = 0
        c_max_test_f1_avg = 0
        for i in range(repeats):
            print('repeat: ', (i + 1))
            f_out.write('repeat: ' + str(i + 1))
            self._reset_params()
            ec_max_test, e_max_test, c_max_test, m_ec_max_test, m_e_max_test, m_c_max_test = self._train(pre_optimizer, train_optimizer, f_out)
            print('Single ec_max_test: {}     e_max_test: {}   c_max_test: {}\n'.format(ec_max_test, e_max_test, c_max_test))
            print('Multi ec_max_test: {}     e_max_test: {}   c_max_test: {}\n'.format(m_ec_max_test, m_e_max_test, m_c_max_test))
            f_out.write('Single ec_max_test: {}     e_max_test: {}   c_max_test: {}\n'.format(ec_max_test, e_max_test, c_max_test))
            f_out.write('Multi ec_max_test: {}     e_max_test: {}   c_max_test: {}\n'.format(m_ec_max_test, m_e_max_test, m_c_max_test))

            ec_max_test_pre_avg += ec_max_test[0]
            ec_max_test_rec_avg += ec_max_test[1]
            ec_max_test_f1_avg += ec_max_test[2]

            e_max_test_pre_avg += e_max_test[0]
            e_max_test_rec_avg += e_max_test[1]
            e_max_test_f1_avg += e_max_test[2]

            c_max_test_pre_avg += c_max_test[0]
            c_max_test_rec_avg += c_max_test[1]
            c_max_test_f1_avg += c_max_test[2]
            print('#' * 100)

        f_out.close()
        return (ec_max_test_pre_avg, ec_max_test_rec_avg, ec_max_test_f1_avg),\
               (e_max_test_pre_avg, e_max_test_rec_avg, e_max_test_f1_avg),\
               (c_max_test_pre_avg, c_max_test_rec_avg, c_max_test_f1_avg)

if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='networks', type=str)
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--input_dropout', default=0.1, type=float)
    parser.add_argument('--gcn_dropout', default=0.1, type=float)
    parser.add_argument('--head_dropout', default=0.1, type=float)
    parser.add_argument('--keep_prob2', default=0.1, type=float)
    parser.add_argument('--keep_prob1', default=0.1, type=float)
    parser.add_argument('--alpha', default=0.3, type=float)
    # parser.add_argument('--l2reg', default=0.00001, type=float)
    parser.add_argument('--l2reg', default=0.00001, type=float)

    # parser.add_argument('--l2reg', default=0.000005, type=float)
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--pre_num_epoch', default=5, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--embed_dim', default=200, type=int)
    parser.add_argument('--embedding_dim_pos', default=100, type=int)
    ###中文数据集的embedding文件
    parser.add_argument('--embedding_path', default='embedding.txt', type=str)
    parser.add_argument('--emo_embedding_path', default='emo_embedding.txt', type=str)

    ###英文数据集的embedding文件################################
    # parser.add_argument('--embedding_path', default='all_embedding_en.txt', type=str)
    #################################################

    parser.add_argument('--pos_num',default=138, type=int)
    parser.add_argument('--hidden_dim', default=100, type=int)
    parser.add_argument('--emo_num', default=700, type=int)
    parser.add_argument('--embedding_dim_emo', default=200, type=int)

    parser.add_argument('--no_word_rnn', default=False, type=bool)
    parser.add_argument('--no_clause_rnn', default=False, type=bool)
    parser.add_argument('--rnn_layer', default=1, type=int)
    parser.add_argument('--rnn_hidden', default=100, type=int)
    parser.add_argument('--rnn_word_dropout', default=0.5, type=float)
    parser.add_argument('--rnn_clause_dropout', default=0.1, type=float)
    parser.add_argument('--emo_dropout', default=0.1, type=float)
    parser.add_argument('--con_dropout', default=0.1, type=float)

    parser.add_argument('--no_pos', default=False, type=bool)
    parser.add_argument('--n_split', default=10, type=int)
    parser.add_argument('--per', default=1.0, type=float)

    parser.add_argument('--no_lexicon_emotion', default=True, type=bool)
    parser.add_argument('--use_emotion_tag', default=True, type=bool)
    parser.add_argument('--window_size', default=5, type=int)
    parser.add_argument('--topk', default=3, type=int)
    parser.add_argument('--oriented_way', default=1, type=int)

    parser.add_argument('--num_class', default=2, type=int)
    parser.add_argument('--save', default=False, type=bool)
    parser.add_argument('--seed', default=776, type=int)
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument('--infer_time', default=True, type=bool)

    ####数据集为英文数据集
    parser.add_argument('--dataset', default='EC', type=str)
    parser.add_argument('--emotion_model', default='./model/sentimental_clauses.pkl', type=str)

    ####数据集为中文数据集
    # parser.add_argument('--dataset', default='EC', type=str)

    opt = parser.parse_args()

    model_classes = {
        'networks':Networks
    }
    input_colses = {
        'networks': ['content', 'sen_len', 'doc_len', 'doc_id']
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamw': torch.optim.AdamW,
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    if opt.dataset == 'EC':
        opt.max_doc_len = 75
        opt.max_sen_len = 45
        opt.data_size = 2105
        opt.hidden_dim = 100
        opt.rnn_hidden = 100
        opt.embed_dim = 200
        opt.embedding_path = 'all_embedding.txt'
    else:
        opt.max_doc_len = 45
        opt.max_sen_len = 130
        opt.data_size = 2105
        opt.hidden_dim = 150
        opt.rnn_hidden = 150
        opt.embed_dim = 300
        opt.embedding_path = 'all_embedding_en.txt'

    opt.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    if opt.seed is not None:
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    ec_p, ec_r, ec_f1 = [], [], []
    e_p, e_r, e_f1 = [], [], []
    c_p, c_r, c_f1 = [], [], []

    for i in range(9, 10):
        model = Model(opt, i)
        ###计算模型大
        model._print_args()
        ec, e, c = model.run(i)
        ec_p.append(ec[0])
        ec_r.append(ec[1])
        ec_f1.append(ec[2])

        e_p.append(e[0])
        e_r.append(e[1])
        e_f1.append(e[2])

        c_p.append(c[0])
        c_r.append(c[1])
        c_f1.append(c[2])
    print("EC: max_test_pre_avg: {:.4f}, max_test_rec_avg: {:.4f}, max_test_f1_avg: {:.4f}".format(np.mean(ec_p), np.mean(ec_r), np.mean(ec_f1)))
    print("E: max_test_pre_avg: {:.4f}, max_test_rec_avg: {:.4f}, max_test_f1_avg: {:.4f}".format(np.mean(e_p), np.mean(e_r), np.mean(e_f1)))
    print("C: max_test_pre_avg: {:.4f}, max_test_rec_avg: {:.4f}, max_test_f1_avg: {:.4f}".format(np.mean(c_p), np.mean(c_r), np.mean(c_f1)))













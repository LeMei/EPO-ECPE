import pickle as pk
import numpy as np
import torch
import random
import math

path = './data/'

def load_percent_train(per, split_size, start_index, data_size):
    x = pk.load(open(path + 'x.txt', 'rb'))
    y_emotion = pk.load(open(path + 'y_emotion.txt', 'rb'))
    y_cause = pk.load(open(path + 'y_cause.txt', 'rb'))
    sen_len = pk.load(open(path + 'sen_len.txt', 'rb'))
    doc_len = pk.load(open(path + 'doc_len.txt', 'rb'))
    doc_id = pk.load(open(path + 'doc_id.txt', 'rb'))
    doc_couple = pk.load(open(path + 'doc_couple.txt', 'rb'))
    emotion_lexicon = pk.load(open(path + 'emotion_lexicon.txt', 'rb'))

    y_mask = pk.load(open(path + 'y_mask.txt', 'rb'))

    length = int(per * data_size)
    mid = start_index * split_size
    right = min((start_index + 1) * split_size, data_size)
    left = max((start_index - 1) * split_size, 0)
    x_test, y_emotion_test, y_cause_test = map(lambda d: d[mid:right, :, :], [x, y_emotion, y_cause])
    sen_len_test, emotion_lexicon_test, y_mask_test = map(lambda d: d[mid:right, :], [sen_len, emotion_lexicon, y_mask])
    doc_len_test, doc_id_test, doc_couple_test = \
        map(lambda d: d[mid:right], [doc_len, doc_id, doc_couple])

    x_train, y_emotion_train, y_cause_train = map(lambda d: d[:length], map(lambda d: np.vstack((d[0:mid, :, :], d[right:data_size, :, :])), [x, y_emotion, y_cause]))
    sen_len_train = np.vstack((sen_len[0:mid, :], sen_len[right:data_size, :]))[:length]
    emotion_lexicon_train = np.vstack((emotion_lexicon[0:mid, :], emotion_lexicon[right:data_size, :]))[:length]
    y_mask_train = np.vstack((y_mask[0:mid, :], y_mask[right:data_size, :]))[:length]
    doc_len_train, doc_id_train, doc_couple_train = map(lambda d: d[:length], map(lambda d: np.hstack((d[0:mid], d[right:data_size])), [doc_len, doc_id, doc_couple]))


    train = {
        'content': x_train,
        'y_emotion': y_emotion_train,
        'y_cause': y_cause_train,
        'sen_len': sen_len_train,
        'doc_len': doc_len_train,
        'doc_id': doc_id_train,
        'doc_couple': doc_couple_train,
        'emotion_lexicon': emotion_lexicon_train,
        'y_mask':y_mask_train
    }
    test = {
        'content': x_test,
        'y_emotion': y_emotion_test,
        'y_cause': y_cause_test,
        'sen_len': sen_len_test,
        'doc_len': doc_len_test,
        'doc_id': doc_id_test,
        'doc_couple': doc_couple_test,
        'emotion_lexicon': emotion_lexicon_test,
        'y_mask':y_mask_test
    }
    return train, test
def get_train_batch_data(train_data, batch_size, keep_prob1, keep_prob2):

    x_train, y_emotion_train, y_cause_train, sen_len_train, doc_len_train, doc_id_train, doc_couple_train, emotion_lexicon_train, y_mask_train = \
        train_data['content'],train_data['y_emotion'],train_data['y_cause'],train_data['sen_len'],train_data['doc_len'],train_data['doc_id'],train_data['doc_couple'], train_data['emotion_lexicon'], train_data['y_mask']
    for index in batch_index(len(y_emotion_train), batch_size):
        feed_list = {
            'content': torch.tensor(x_train[index]).long(),
            'y_emotion': torch.tensor(y_emotion_train[index]),
            'y_cause': torch.tensor(y_cause_train[index]),
            'sen_len': torch.tensor(sen_len_train[index]),
            'doc_len': torch.tensor(doc_len_train[index]),
            'doc_id': torch.tensor(doc_id_train[index]),
            'doc_couple': doc_couple_train[index],
            'y_mask': torch.tensor(y_mask_train[index]),
            'emotion_lexicon': torch.LongTensor(emotion_lexicon_train[index]),
            'keep_prob1':keep_prob1,
            'keep_prob2':keep_prob2
        }
        yield feed_list

def get_test_batch_data(test_data, batch_size):
    x_test, y_emotion_test, y_cause_test, sen_len_test, doc_len_test, doc_id_test,doc_couple_test, emotion_lexicon_test, y_mask_test = \
        test_data['content'],test_data['y_emotion'], test_data['y_cause'], test_data['sen_len'],test_data['doc_len'],test_data['doc_id'], test_data['doc_couple'], test_data['emotion_lexicon'], test_data['y_mask']
    for index in batch_index(len(y_emotion_test), batch_size, test=True):
        feed_list = {
            'content': torch.tensor(x_test[index]).long(),
            'y_emotion': torch.tensor(y_emotion_test[index]),
            'y_cause': torch.tensor(y_cause_test[index]),
            'sen_len': torch.tensor(sen_len_test[index]),
            'doc_len': torch.tensor(doc_len_test[index]),
            'doc_id': torch.tensor(doc_id_test[index]),
            'doc_couple': doc_couple_test[index],
            'y_mask': torch.tensor(y_mask_test[index]),
            'emotion_lexicon': torch.LongTensor(emotion_lexicon_test[index]),
            'keep_prob1': 1.0,
            'keep_prob2': 1.0
        }
        yield feed_list
def get_test_single_batch_data(test_data, batch_size):
    x_test, y_emotion_test, y_cause_test, sen_len_test, doc_len_test, doc_id_test,doc_couple_test, emotion_lexicon_test, y_mask_test = \
        test_data['content'],test_data['y_emotion'], test_data['y_cause'], test_data['sen_len'],test_data['doc_len'],test_data['doc_id'], test_data['doc_couple'], test_data['emotion_lexicon'], test_data['y_mask']
    single_index, multi_index = single_multi(len(y_emotion_test),doc_couple_test, test=True)
    single_index_b = batch_index_single_multi(single_index, batch_size)
    for index in single_index_b:
        feed_list = {
            'content': torch.tensor(x_test[index]).long(),
            'y_emotion': torch.tensor(y_emotion_test[index]),
            'y_cause': torch.tensor(y_cause_test[index]),
            'sen_len': torch.tensor(sen_len_test[index]),
            'doc_len': torch.tensor(doc_len_test[index]),
            'doc_id': torch.tensor(doc_id_test[index]),
            'doc_couple': doc_couple_test[index],
            'y_mask': torch.tensor(y_mask_test[index]),
            'emotion_lexicon': torch.LongTensor(emotion_lexicon_test[index]),
            'keep_prob1': 1.0,
            'keep_prob2': 1.0
        }
        yield feed_list

def get_test_multi_batch_data(test_data, batch_size):
    x_test, y_emotion_test, y_cause_test, sen_len_test, doc_len_test, doc_id_test, doc_couple_test, emotion_lexicon_test, y_mask_test = \
        test_data['content'], test_data['y_emotion'], test_data['y_cause'], test_data['sen_len'], test_data[
            'doc_len'], test_data['doc_id'], test_data['doc_couple'], test_data['emotion_lexicon'], test_data[
            'y_mask']
    single_index, multi_index = single_multi(len(y_emotion_test), doc_couple_test)
    multi_index_b = batch_index_single_multi(multi_index, batch_size)
    for index in multi_index_b:
        feed_list = {
            'content': torch.tensor(x_test[index]).long(),
            'y_emotion': torch.tensor(y_emotion_test[index]),
            'y_cause': torch.tensor(y_cause_test[index]),
            'sen_len': torch.tensor(sen_len_test[index]),
            'doc_len': torch.tensor(doc_len_test[index]),
            'doc_id': torch.tensor(doc_id_test[index]),
            'doc_couple': doc_couple_test[index],
            'y_mask': torch.tensor(y_mask_test[index]),
            'emotion_lexicon': torch.LongTensor(emotion_lexicon_test[index]),
            'keep_prob1': 1.0,
            'keep_prob2': 1.0
        }
        yield feed_list
def batch_index(length, batch_size, test=False):
    index = list(range(length))
    if not test:
        random.shuffle(index)
    for i in range(int((length + batch_size - 1) / batch_size)):
        ret = index[i * batch_size: (i + 1) * batch_size]
        if not test and len(ret) < batch_size:
            break
        yield ret

def single_multi(length, doc_couples, test=False):
    index = list(range(length))
    if not test:
        random.shuffle(index)
    multi_index, single_index = [], []
    for i in index:
        if len(doc_couples[i]) > 1:
            multi_index.append(i)
        else:
            single_index.append(i)
    return single_index, multi_index

def batch_index_single_multi(index, batch_size):
    length = len(index)
    for i in range(int((length + batch_size - 1) / batch_size)):
        ret = index[i * batch_size: (i + 1) * batch_size]
        yield ret

def load_embedding(embedding_path):
    word_embedding = pk.load(open(path + embedding_path,'rb'))
    return word_embedding

def load_emo_embedding(emo_embedding_path):
    word_embedding = pk.load(open(path + emo_embedding_path,'rb'))
    return word_embedding

def load_pos_embedding(embedding_dim_pos):
    embedding_pos = [list(np.zeros(embedding_dim_pos))]
    embedding_pos.extend([list(np.random.normal(
        loc=0.0, scale=0.1, size=embedding_dim_pos)) for i in range(-100, 100)])
    return np.array(embedding_pos)

def load_data_en():
    x_train = pk.load(open(path + 'train_x_en.txt', 'rb'))
    y_emotion_train = pk.load(open(path + 'train_y_emotion_en.txt', 'rb'))
    y_cause_train = pk.load(open(path + 'train_y_cause_en.txt', 'rb'))
    sen_len_train = pk.load(open(path + 'train_sen_len_en.txt', 'rb'))
    doc_len_train = pk.load(open(path + 'train_doc_len_en.txt', 'rb'))
    doc_id_train = pk.load(open(path + 'train_doc_id_en.txt', 'rb'))
    doc_couple_train = pk.load(open(path + 'doc_couple_train.txt', 'rb'))
    print(
        'x.shape {} \ny_emotion.shape {}\ny_cause.shape {}\nsen_len.shape {} \ndoc_len.shape {}\ndoc_id.shape {}\ndoc_couple.shape {}'
            .format(x_train.shape, y_emotion_train.shape, y_cause_train.shape, sen_len_train.shape, doc_len_train.shape, doc_id_train.shape, doc_couple_train.shape))

    train = {
        'content': x_train,
        'y_emotion': y_emotion_train,
        'y_cause': y_cause_train,
        'sen_len': sen_len_train,
        'doc_len': doc_len_train,
        'doc_id': doc_id_train,
        'doc_couple': doc_couple_train
    }

    x_test = pk.load(open(path + 'test_x_en.txt', 'rb'))
    y_emotion_test = pk.load(open(path + 'test_y_emotion_en.txt', 'rb'))
    y_cause_test = pk.load(open(path + 'test_y_cause_en.txt', 'rb'))
    sen_len_test = pk.load(open(path + 'test_sen_len_en.txt', 'rb'))
    doc_len_test = pk.load(open(path + 'test_doc_len_en.txt', 'rb'))
    doc_id_test = pk.load(open(path + 'test_doc_id_en.txt', 'rb'))
    doc_couple_test = pk.load(open(path + 'test_doc_couple_en.txt', 'rb'))

    print(
        'x.shape {} \ny_emotion.shape {}\ny_cause.shape {}\nsen_len.shape {} \ndoc_len.shape {}\ndoc_id.shape {}\ndoc_couple.shape {}\n'
            .format(x_test.shape, y_emotion_test.shape, y_cause_test.shape, sen_len_test.shape, doc_len_test.shape, doc_id_test.shape, doc_couple_test.shape))

    test = {
        'content': x_test,
        'y_emotion': y_emotion_test,
        'y_cause': y_cause_test,
        'sen_len': sen_len_test,
        'doc_len': doc_len_test,
        'doc_id': doc_id_test
    }
    return train, test

def logistic(x):
    return 1 / (1 + np.exp(-x.detach().cpu().numpy()))
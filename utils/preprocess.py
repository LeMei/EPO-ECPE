import json
import numpy as np
import codecs
import pickle as pk
import jieba

path = '../data/'
max_doc_len = 75
max_sen_len = 45

def read_lexicon(lexicon):
    f = open(path + lexicon, 'r', encoding='utf-8')
    lexicon_set = set()
    lines = f.readlines()
    for line in lines:
        lexicon_set.add(line.strip())

    return lexicon_set

def read_data_file(data_file, word_idx, emotion_idx, idx_emotion, max_doc_len=max_doc_len, max_sen_len=max_sen_len):

    doc_id_list = []
    doc_len_list = []
    sen_len_list = []
    doc_couples_list = []
    x_words_list = []
    y_emotions_list, y_causes_list = [], []
    y_mask_list = []

    lexicon_list = []

    data_list = read_json(data_file)
    for doc in data_list:
        doc_id = doc['doc_id']
        doc_len = doc['doc_len']
        doc_couples = doc['pairs']
        doc_emotions, doc_causes = zip(*doc_couples)
        doc_id_list.append(int(doc_id))
        doc_len_list.append(doc_len)
        doc_couples = list(map(lambda x: [i - 1 for i in x], doc_couples))
        # doc_couples = [doc_emotions-1, doc_causes-1]
        doc_couples_list.append(doc_couples)

        y_emotions, y_causes = np.zeros((max_doc_len, 2)), np.zeros((max_doc_len, 2))
        clause_all, sen_len_all, lexicon_all = [], [0]*max_doc_len, [0]*max_doc_len
        doc_clauses = doc['clauses']
        y_mask =  [1]*doc_len + [0]*(max_doc_len-doc_len)

        print('DOC:' + doc_id)
        for i in range(doc_len):
            emotion_label = int(i + 1 in doc_emotions)
            cause_label = int(i + 1 in doc_causes)
            y_emotions[i][emotion_label] = 1
            y_causes[i][cause_label] = 1

            clause_ = doc_clauses[i]
            clause_id = clause_['clause_id']
            clause_text = clause_['clause']

            assert int(clause_id) == i + 1
            ###生成mask

            sen_len_all[i] = len(clause_text.split(' '))

            clause = [0] * max_sen_len
            for i, word in enumerate(clause_text.split(' ')):
                if word in word_idx.keys():
                    clause[i] = int(word_idx[word])
                else:
                    clause[i] = 0
                if word in emotion_idx.keys():
                    lexicon_all[i] = int(emotion_idx[word])
                    print('clause:' + clause_text)
                    print('word:'+ word)
                    print(int(emotion_idx[word]))
            clause_all.append(np.array(clause))

        for j in range(max_doc_len - len(clause_all)):
            clause_all.append(np.zeros((max_sen_len,)))


        sen_len_list.append(sen_len_all)
        lexicon_list.append(lexicon_all)
        x_words_list.append(clause_all)
        y_emotions_list.append(y_emotions)
        y_causes_list.append(y_causes)
        y_mask_list.append(y_mask)

    x = np.array(x_words_list)
    y_emotion = np.array(y_emotions_list)
    y_cause = np.array(y_causes_list)
    sen_len = np.array(sen_len_list)
    emotion_lexicon = np.array(lexicon_list)
    doc_len = np.array(doc_len_list)
    doc_id = np.array(doc_id_list)
    doc_couple = np.array(doc_couples_list)
    y_mask = np.array(y_mask_list)
    max_len = np.argmax(doc_len)
    print(max_len)
    # pk.dump(x, open(path + 'x.txt', 'wb'))
    # pk.dump(y_emotion, open(path + 'y_emotion.txt', 'wb'))
    # pk.dump(y_cause, open(path + 'y_cause.txt', 'wb'))
    # pk.dump(sen_len, open(path + 'sen_len.txt', 'wb'))
    # pk.dump(doc_len, open(path + 'doc_len.txt', 'wb'))
    # pk.dump(doc_id, open(path + 'doc_id.txt', 'wb'))
    pk.dump(doc_couple, open(path + 'doc_couple.txt', 'wb'))
    # pk.dump(emotion_lexicon, open(path + 'emotion_lexicon.txt', 'wb'))
    # pk.dump(y_mask, open(path + 'y_mask.txt', 'wb'))



    print('doc_id.shape {}\nx.shape {} \ny_emotion.shape {}\ny_cause.shape {}\nsen_len.shape {} \ndoc_len.shape {}\ndoc_couple.shape {}\nemotion_lexicon {}'.format(
        doc_id.shape, x.shape, y_emotion.shape, y_cause.shape, sen_len.shape, doc_len.shape, doc_couple.shape, emotion_lexicon.shape
    ))
    print('y_mask.shape {}'.format(y_mask.shape))
    print('load data done!\n')


def load_w2v(embedding_dim, data_file, embedding_path, lexicon):
    print('\nload embedding...')
    words = []
    data_list = read_json(data_file)
    lexicons = read_lexicon(lexicon)

    emotions = []

    for doc in data_list:
        clauses_list = doc['clauses']
        for clause in clauses_list:
            emotion_token = clause['emotion_token']
            clause_words = clause['clause']
            tokens = clause_words.split(' ')
            words.extend([emotion_token] + tokens)
            emotions.extend([emotion_token] + list(set(tokens).intersection(set(lexicons))))

    words = set(words)  # 所有不重复词的集合
    word_idx = dict((c, k + 1) for k, c in enumerate(words))
    idx_word = dict((k+1, c) for k, c in enumerate(words))
    emotions = set(emotions)
    emotion_idx = dict((c, k + 1) for k, c in enumerate(emotions))
    idx_emotion = dict((k+1, c) for k, c in enumerate(emotions))

    w2v = {}
    emb_file = codecs.open(embedding_path, 'r', 'utf-8')
    for line in emb_file.readlines():
        line = line.strip().split(' ')
        w, ebd = line[0], line[1:]
        w2v[w] = ebd
    emb_file.close()
    embedding = [list(np.zeros(embedding_dim))] #0 padding on 0
    hit = 0
    for item in words:
        if item in w2v:
            vec = list(map(float, w2v[item]))
            hit += 1
        else:
            vec = list(np.random.rand(embedding_dim) / 5. - 0.1)  # 从均匀分布[-0.1,0.1]中随机取
        embedding.append(vec)
    print('w2v_file: {}\nall_words: {} hit_words: {}'.format(
        embedding_path, len(words), hit))

    embedding = np.array(embedding)

    emo_embedding = [list(np.zeros(embedding_dim))]  # 0 padding on 0
    hit = 0
    for item in emotions:
        if item in w2v:
            vec = list(map(float, w2v[item]))
            hit += 1
        else:
            vec = list(np.random.rand(embedding_dim) / 5. - 0.1)  # 从均匀分布[-0.1,0.1]中随机取
        emo_embedding.append(vec)
    print('w2v_file: {}\nall_words: {} hit_words: {}'.format(
        embedding_path, len(words), hit))

    emo_embedding = np.array(emo_embedding)

    # pk.dump(embedding, open(path + 'all_embedding.txt', 'wb'))
    # pk.dump(json.dumps(idx_word), open(path + 'all_idx_word.txt', 'wb'))
    # pk.dump(emo_embedding, open(path + 'emo_embedding.txt', 'wb'))


    print("embedding.shape: {}".format(
        embedding.shape))
    print("emo_embedding.shape: {}".format(
        emo_embedding.shape))
    print("load embedding done!\n")
    return word_idx, emotion_idx, idx_emotion

def read_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as fr:
        js = json.load(fr)
    return js

def read_full_json(json_file1, json_file2, json_file):
    js1 = read_json(json_file1)
    js2 = read_json(json_file2)
    data_list = js1 + js2

    for doc in data_list:
        doc_len = doc['doc_len']
        doc_clauses = doc['clauses']
        update_doc_clauses = []
        for i in range(doc_len):
            clause = doc_clauses[i]
            clause_id = clause['clause_id']
            clause_text = clause['clause']
            assert int(clause_id) == i + 1
            clause_token = list(jieba.cut(clause_text))
            clause_token = ' '.join(clause_token)
            clause['clause'] = clause_token
            update_doc_clauses.append(clause)
        doc['clauses'] = update_doc_clauses

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False)

#
# js1 = '../data/split10/fold1_test.json'
# js2 = '../data/split10/fold1_train.json'
# js = '../data/split10/full_data.json'
#
# read_full_json(js1, js2, js)

word_idx, emotion_idx, idx_emotion = load_w2v(200, path + 'full_data.json', path + 'w2v_200.txt', path+'lexicon.txt')
read_data_file(path + 'full_data.json', word_idx, emotion_idx, idx_emotion)
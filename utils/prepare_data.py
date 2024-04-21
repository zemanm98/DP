############################################ IMPORT ##########################################################
import sys, os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import librosa
from torch import nn
from torch import optim
from torch.nn import functional as F

############################################ LOAD DATA PAIR STEP ##############################################
def load_data_pair(input_file, word_idx, max_doc_len = 75, max_sen_len = 45):
    print('load data_file: {}'.format(input_file))
    pair_id_all, y_position, y_cause, y_pair, x, sen_len, doc_len, distance, audio, doc_ids = [], [], [], [], [], [], [], [], [], []
    n_cut = 0
    inputFile = open(input_file, 'r', encoding='utf-8')
    overall_length = 0
    while True:
        # takes line with document id and document length.
        # document in this context is one dialogue.
        line = inputFile.readline()
        if line == '': break
        line = line.strip().split()
        doc_id = int(line[0])
        d_len = int(line[1])
        overall_length += d_len
        ######################################## doc_len_condition ########################################
        if d_len >= max_doc_len :
          for i in range(d_len+1) :
            line = inputFile.readline().strip().split(',')
          continue
        ######################################## doc_len_condition ########################################

        # pairs of emotion cause pairs. (3, 1) meaning that emotion in third sentence is caused by 1st sentence
        pairs = eval('[' + inputFile.readline().strip() + ']')
        if len(pairs) > 0:
            pos_list, cause_list = zip(*pairs)
            pairs = [[pos_list[i], cause_list[i]] for i in range(len(pos_list))]
        else:
            pos_list = (0,)
            cause_list = (0,)
            pairs = [[0, 0]]
        pair_id_all.extend([doc_id * 10000 + p[0] * 100 + p[1] for p in pairs])
        audio_tmp = np.zeros((max_doc_len, 50))
        for i in range(0, max_doc_len):
            folder = inputFile.name[19:].split(".")[0]
            if i < d_len:
                file = "./" + folder + "/dia" + str(doc_id) + "utt" + str(i + 1) + ".wav"
                audio_x, sample_rate = librosa.load(file)
                mfcc = np.mean(librosa.feature.mfcc(y=audio_x, sr=sample_rate, n_mfcc=50).T, axis=0)
                audio_tmp[i] = mfcc
            else:
                audio_tmp[i] = np.zeros(50)
        y_position_tmp, y_cause_tmp, y_pair_tmp, sen_len_tmp, x_tmp, distance_tmp = \
        np.zeros((max_doc_len, 2)), np.zeros((max_doc_len, 2)), np.zeros((max_doc_len * max_doc_len, 2)), \
        np.zeros((max_doc_len, )), np.zeros((max_doc_len, max_sen_len)), np.zeros((max_doc_len * max_doc_len, ))

        # for cycle for word to ids in passed word_idx
        for i in range(d_len):
            line = inputFile.readline().strip().split(';')
            words = line[-1]
            sen_len_tmp[i] = min(len(words.split()), max_sen_len)
            for j, word in enumerate(words.split()):
                word = word.lower()
                if j >= max_sen_len:
                    n_cut += 1
                    break
                elif word not in word_idx : x_tmp[i][j] = 24166
                else : x_tmp[i][j] = int(word_idx[word])

        for i in range(d_len):
            for j in range(d_len):
                # Check whether i is an emotion clause
                if i+1 in pos_list :
                    y_position_tmp[i][0] = 0; y_position_tmp[i][1] = 1
                else :
                    y_position_tmp[i][0] = 1; y_position_tmp[i][1] = 0
                # Check whether j is a cause clause
                if j+1 in cause_list :
                    y_cause_tmp[j][0] = 0; y_cause_tmp[j][1] = 1
                else :
                    y_cause_tmp[j][0] = 1; y_cause_tmp[j][1] = 0
                # Check whether i, j clauses are emotion cause pairs
                pair_id_curr = doc_id*10000+(i+1)*100+(j+1)
                if pair_id_curr in pair_id_all :
                    y_pair_tmp[i*max_doc_len+j][0] = 0; y_pair_tmp[i*max_doc_len+j][1] = 1
                else :
                    y_pair_tmp[i*max_doc_len+j][0] = 1; y_pair_tmp[i*max_doc_len+j][1] = 0
                # Find the distance between the clauses, and use the same embedding beyond 10 clauses
                distance_tmp[i*max_doc_len+j] = min(max(j-i+100, 90), 110)

        y_position.append(y_position_tmp)
        y_cause.append(y_cause_tmp)
        y_pair.append(y_pair_tmp)
        x.append(x_tmp)
        sen_len.append(sen_len_tmp)
        doc_len.append(d_len)
        distance.append(distance_tmp)
        audio.append(audio_tmp)
        doc_ids.append(doc_id)

    y_position, y_cause, y_pair, x, sen_len, doc_len, distance, audio, doc_ids = map(torch.tensor, \
    [y_position, y_cause, y_pair, x, sen_len, doc_len, distance, audio, doc_ids])

    for var in ['y_position', 'y_cause', 'y_pair', 'x', 'sen_len', 'doc_len', 'distance']:
        print('{}.shape {}'.format( var, eval(var).shape ))
    print('n_cut {}'.format(n_cut))
    print('load data done!\n')
    return y_position, y_cause, y_pair, x, sen_len, doc_len, distance, audio, doc_ids

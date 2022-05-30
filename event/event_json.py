#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import json  # import json module
import sys
import argparse
import os
import gensim
import matplotlib.pyplot as plt
import operator
import matplotlib
import time
import csv
import numpy
import random
import scipy
import matplotlib.pylab as pylab
from matplotlib import gridspec
from matplotlib import font_manager, rc
from konlpy.tag import Komoran

# 사건 추출
from sklearn import cluster
from gensim.models import KeyedVectors
from gensim.models import FastText
from collections import OrderedDict
from matplotlib import font_manager, rc
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.lines import Line2D
from wordcloud import WordCloud

komoran = Komoran()
path = os.path.dirname(os.path.abspath(__file__))
font_path = '/Users/amy/Library/Fonts/Seoulnamsan_B.otf'
model_path = path + '/model/old4_2_5_min100.vec'


def FullTextProcess(input_path):
    # with statement
    with open(path + "/" + input_path, encoding='utf-8') as json_file:
        json_data = json.load(json_file)
    # 본문
    texts = json_data["texts"]
    # 문장별로 잘려서 들어가있는 본문
    splitted_text = list()
    split = False
    split_n = False
    split_sentence = ""
    for i in range(len(texts)):
        page = int(texts[i]["page"])
        text = texts[i]["text"]

        if text == " ":
            continue
        if text[-2] not in ["?", ".", "!", "\n"]:  # 문장이 끊겨 있으면
            split_n = True
            end = -1
        else:
            split_n = False
            end = 0
        text = re.sub("[^ ㄱ-ㅣ가-힣0-9a-zA-Z\.|\?|\!|\n]+", "", text)
        sents = re.split(r"[\?|\.|\!|\n]", text)
        if split:
            splitted_text.append(split_sentence + sents[0] + " : " + str(page - 1) + ", " + str(page))
            start = 1
            split = False
        else:
            start = 0
        if split_n:
            split_sentence = sents[-1]
        split = split_n
        for i in range(start, len(sents) + end):
            if sents[i] == "f ":
                pass
            elif sents[i] == None:
                pass
            elif sents[i] == "\n":
                pass
            elif sents[i] == "":
                pass
            elif sents[i] == " ":
                pass
            else:
                splitted_text.append(sents[i] + " : " + str(page))
    for i in range(len(splitted_text)):
        if splitted_text[i][0] == " ":
            splitted_text[i] = splitted_text[i][1:]
    # 페이지별로 읽기
    full_text = ""
    for lines in range(len(splitted_text)):
        if lines != (len(splitted_text) - 1):
            full_text = full_text + splitted_text[lines] + "\n"
        else:
            full_text = full_text + splitted_text[lines]
    # 이 부분은  full_text 안에 저장되어 있으니까 따로 파일로 저장 안하고 full_text로 넘기면 될 것 같습니다!
    '''
    with open(path + "/" + output, mode="w", encoding='utf-8') as fout:
        fout.writelines(full_text)
    '''
    return full_text


def set2List(NumpyArray):
    list = []
    for item in NumpyArray:
        list.append(item.tolist())
    return list


def DBSCAN(Dataset, Epsilon, MinumumPoints, DistanceMethod='euclidean'):
    m, n = Dataset.shape
    Visited = numpy.zeros(m, 'int')
    Type = numpy.zeros(m)
    #   -1 noise, outlier
    #    0 border (우린 신경 ㄴㄴ)
    #    1 core
    ClustersList = []
    Cluster = []
    PointClusterNumber = numpy.zeros(m)
    PointClusterNumberIndex = 1
    PointNeighbors = []
    DistanceMatrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(Dataset, DistanceMethod))
    for i in range(m):
        if Visited[i] == 0:
            Visited[i] = 1
            PointNeighbors = numpy.where(DistanceMatrix[i] < Epsilon)[0]
            if len(PointNeighbors) < MinumumPoints:
                Type[i] = -1
            else:
                for k in range(len(Cluster)):
                    Cluster.pop()
                Cluster.append(i)
                PointClusterNumber[i] = PointClusterNumberIndex
                PointNeighbors = set2List(PointNeighbors)
                ExpandClsuter(Dataset[i], PointNeighbors, Cluster, MinumumPoints, Epsilon, Visited, DistanceMatrix,
                              PointClusterNumber, PointClusterNumberIndex)
                Cluster.append(PointNeighbors[:])
                ClustersList.append(Cluster[:])
                PointClusterNumberIndex = PointClusterNumberIndex + 1
    return PointClusterNumber


def ExpandClsuter(PointToExapnd, PointNeighbors, Cluster, MinumumPoints, Epsilon, Visited, DistanceMatrix,
                  PointClusterNumber, PointClusterNumberIndex):
    Neighbors = []

    for i in PointNeighbors:
        if Visited[i] == 0:
            Visited[i] = 1
            Neighbors = numpy.where(DistanceMatrix[i] < Epsilon)[0]
            if len(Neighbors) >= MinumumPoints:
                for j in Neighbors:
                    try:
                        PointNeighbors.index(j)  # PointNeighbors에서 j를 이웃으로 갖는 애를 찾음
                    except ValueError:
                        PointNeighbors.append(j)
        if PointClusterNumber[i] == 0:
            Cluster.append(i)
            PointClusterNumber[i] = PointClusterNumberIndex
    return


# case2
def DBSCAN_sim(Dataset, Dataset2, Epsilon, MinumumPoints, DistanceMethod='euclidean'):
    m, n = Dataset.shape
    Visited = numpy.zeros(m, 'int')
    Type = numpy.zeros(m)
    #   -1 noise, outlier
    #    0 border (우린 신경 ㄴㄴ)
    #    1 core
    ClustersList = []
    Cluster = []
    PointClusterNumber = numpy.zeros(m)
    PointClusterNumberIndex = 1
    PointNeighbors = []
    DistanceMatrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(Dataset, DistanceMethod))
    for i in range(m):
        if Visited[i] == 0:
            Visited[i] = 1
            PointNeighbors = numpy.where(DistanceMatrix[i] < Epsilon)[0]
            # total_sum=PointNeighbors.sum()
            total_sum = 0
            for t in range(len(PointNeighbors)):
                total_sum += Dataset2[PointNeighbors[t]][1]
            if len(PointNeighbors) < MinumumPoints:
                Type[i] = -1
            else:
                if total_sum >= 3:
                    for k in range(len(Cluster)):
                        Cluster.pop()
                    Cluster.append(i)
                    PointClusterNumber[i] = PointClusterNumberIndex
                    PointNeighbors = set2List(PointNeighbors)
                    ExpandClsuter_sim(Dataset[i], Dataset2, PointNeighbors, Cluster, MinumumPoints, Epsilon, Visited,
                                      DistanceMatrix, PointClusterNumber, PointClusterNumberIndex)
                    Cluster.append(PointNeighbors[:])
                    ClustersList.append(Cluster[:])
                    PointClusterNumberIndex = PointClusterNumberIndex + 1
    return PointClusterNumber


def ExpandClsuter_sim(PointToExapnd, Dataset2, PointNeighbors, Cluster, MinumumPoints, Epsilon, Visited, DistanceMatrix,
                      PointClusterNumber, PointClusterNumberIndex):
    Neighbors = []

    for i in PointNeighbors:
        if Visited[i] == 0:
            Visited[i] = 1
            Neighbors = numpy.where(DistanceMatrix[i] < Epsilon)[0]
            # print(Neighbors)
            total_sum2 = 0
            for t in range(len(Neighbors)):
                total_sum2 += Dataset2[Neighbors[t]][1]
            if len(Neighbors) >= MinumumPoints:
                if total_sum2 >= 3:
                    for j in Neighbors:
                        try:
                            PointNeighbors.index(j)  # PointNeighbors에서 j를 이웃으로 갖는 애를 찾음
                        except ValueError:
                            PointNeighbors.append(j)
        if PointClusterNumber[i] == 0:
            Cluster.append(i)
            PointClusterNumber[i] = PointClusterNumberIndex
    return


def find_event_idx(full_text):
    model = KeyedVectors.load_word2vec_format(model_path)
    original_sentence = full_text.split("\n")
    postag_sentence = []
    for i in range(len(original_sentence)):
        original_sentence[i] = original_sentence[i].split(":")
        original_sentence[i][1] = original_sentence[i][1].replace(" ", "")
        original_sentence[i][1] = re.split(",", original_sentence[i][1])
        tmp_list = []
        try:
            morph = komoran.pos(original_sentence[i][0])
            for word, tag in morph:
                if tag in ['NNG', 'NNP', 'VV', 'VA']:
                    try:
                        tmp_list.append(word)
                    except TypeError:
                        pass
        except Exception:
            tmp_list.append(original_sentence[i][0])
        postag_sentence.append([tmp_list, original_sentence[i][1]])

    search_topic = ['죽음', '사망', '살해', '살인', '타살', '탄생', '출생', '출산', '폭력', '싸움', '다툼', '전쟁', '폭격', '참전', '피난', '이별',
                    '결별', '사별', '체포', '투옥', '수감', '해방', '독립', '만남', '이사', '병', '치료']

    death_word_list = []
    birth_word_list = []
    violence_word_list = []
    war_word_list = []
    part_word_list = []
    arrest_word_list = []
    independent_word_list = []
    meet_word_list = []
    move_word_list = []
    illness_word_list = []

    threshold = 0.7

    for i in range(len(postag_sentence)):
        death_tmp_word = []
        birth_tmp_word = []
        violence_tmp_word = []
        war_tmp_word = []
        part_tmp_word = []
        arrest_tmp_word = []
        independent_tmp_word = []
        meet_tmp_word = []
        move_tmp_word = []
        illness_tmp_word = []
        for j in range(len(postag_sentence[i][0])):
            softmax_score = []
            for k in range(len(search_topic)):
                try:
                    softmax_score.append(model.similarity(str(postag_sentence[i][0][j]), search_topic[k]))
                except TypeError:
                    softmax_score.append(0)
                except KeyError:
                    softmax_score.append(0)
            if softmax_score[0] >= 0.65 or softmax_score[1] >= 0.65 or softmax_score[2] >= 0.65 or softmax_score[
                3] >= 0.65 or softmax_score[4] >= 0.65:
                death_tmp_word.append(str(postag_sentence[i][0][j]))
            elif softmax_score[5] >= 0.65 or softmax_score[6] >= 0.65 or softmax_score[7] >= 0.65:
                birth_tmp_word.append(str(postag_sentence[i][0][j]))
            elif softmax_score[8] >= threshold or softmax_score[9] >= threshold or softmax_score[10] >= threshold:
                violence_tmp_word.append(str(postag_sentence[i][0][j]))
            elif softmax_score[11] >= 0.67 or softmax_score[12] >= threshold or softmax_score[13] >= 0.67 or \
                    softmax_score[14] >= 0.67:
                war_tmp_word.append(str(postag_sentence[i][0][j]))
            elif softmax_score[15] >= 0.65 or softmax_score[16] >= 0.65 or softmax_score[17] >= 0.65:
                part_tmp_word.append(str(postag_sentence[i][0][j]))
            elif softmax_score[18] >= threshold or softmax_score[19] >= threshold or softmax_score[20] >= threshold:
                arrest_tmp_word.append(str(postag_sentence[i][0][j]))
            elif softmax_score[21] >= threshold or softmax_score[22] >= threshold:
                independent_tmp_word.append(str(postag_sentence[i][0][j]))
            elif softmax_score[23] >= threshold:
                meet_tmp_word.append(str(postag_sentence[i][0][j]))
            elif softmax_score[24] >= threshold:
                move_tmp_word.append(str(postag_sentence[i][0][j]))
            elif softmax_score[25] >= 0.65 or softmax_score[26] >= 0.65:
                illness_tmp_word.append(str(postag_sentence[i][0][j]))

        # 문장 1개당 event 단어 저장
        death_word_list.append(death_tmp_word)
        birth_word_list.append(birth_tmp_word)
        violence_word_list.append(violence_tmp_word)
        war_word_list.append(war_tmp_word)
        part_word_list.append(part_tmp_word)
        arrest_word_list.append(arrest_tmp_word)
        independent_word_list.append(independent_tmp_word)
        meet_word_list.append(meet_tmp_word)
        move_word_list.append(move_tmp_word)
        illness_word_list.append(illness_tmp_word)

    # 모든 문장의 각 문장마다 event 단어 저장
    all_evt_word = []
    all_evt_word.append(death_word_list)
    all_evt_word.append(birth_word_list)
    all_evt_word.append(violence_word_list)
    all_evt_word.append(war_word_list)
    all_evt_word.append(part_word_list)
    all_evt_word.append(arrest_word_list)
    all_evt_word.append(independent_word_list)
    all_evt_word.append(meet_word_list)
    all_evt_word.append(move_word_list)
    all_evt_word.append(illness_word_list)

    return numpy.array(all_evt_word)


def event_detect(full_text, output, image_path):
    model = KeyedVectors.load_word2vec_format(model_path)
    original_text = full_text
    original_sentence = full_text.split("\n")
    postag_sentence = []
    for i in range(len(original_sentence)):
        original_sentence[i] = original_sentence[i].split(":")
        original_sentence[i][1] = original_sentence[i][1].replace(" ", "")
        original_sentence[i][1] = re.split(",", original_sentence[i][1])
        tmp_list = []
        try:
            morph = komoran.pos(original_sentence[i][0])
            for word, tag in morph:
                if tag in ['NNG', 'NNP', 'VV', 'VA']:
                    try:
                        tmp_list.append(word)
                    except TypeError:
                        pass
        except Exception:
            tmp_list.append(original_sentence[i][0])
        postag_sentence.append([tmp_list, original_sentence[i][1]])

    search_topic = ['죽음', '사망', '살해', '살인', '타살', '탄생', '출생', '출산', '폭력', '싸움', '다툼', '전쟁', '폭격', '참전', '피난', '이별',
                    '결별', '사별', '체포', '투옥', '수감', '해방', '독립', '만남', '이사', '병', '치료']
    death_word_list = []
    birth_word_list = []
    violence_word_list = []
    war_word_list = []
    part_word_list = []
    arrest_word_list = []
    independent_word_list = []
    meet_word_list = []
    move_word_list = []
    illness_word_list = []

    death_score_list = []
    birth_score_list = []
    violence_score_list = []
    war_score_list = []
    part_score_list = []
    arrest_score_list = []
    independent_score_list = []
    meet_score_list = []
    move_score_list = []
    illness_score_list = []

    threshold = 0.7

    for i in range(len(postag_sentence)):
        death_tmp_word = []
        death_tmp_score = []
        birth_tmp_word = []
        birth_tmp_score = []
        violence_tmp_word = []
        violence_tmp_score = []
        war_tmp_word = []
        war_tmp_score = []
        part_tmp_word = []
        part_tmp_score = []
        arrest_tmp_word = []
        arrest_tmp_score = []
        independent_tmp_word = []
        independent_tmp_score = []
        meet_tmp_word = []
        meet_tmp_score = []
        move_tmp_word = []
        move_tmp_score = []
        illness_tmp_word = []
        illness_tmp_score = []
        for j in range(len(postag_sentence[i][0])):
            softmax_score = []
            for k in range(len(search_topic)):
                try:
                    softmax_score.append(model.similarity(str(postag_sentence[i][0][j]), search_topic[k]))
                except TypeError:
                    softmax_score.append(0)
                except KeyError:
                    softmax_score.append(0)
            if softmax_score[0] >= 0.65 or softmax_score[1] >= 0.65 or softmax_score[2] >= 0.65 or softmax_score[
                3] >= 0.65 or softmax_score[4] >= 0.65:
                death_max_score = max(softmax_score[0:5])
                death_tmp_word.append(str(postag_sentence[i][0][j]))
                death_tmp_score.append(death_max_score)
            elif softmax_score[5] >= 0.65 or softmax_score[6] >= 0.65 or softmax_score[7] >= 0.65:
                birth_max_score = max(softmax_score[5:8])
                birth_tmp_word.append(str(postag_sentence[i][0][j]))
                birth_tmp_score.append(birth_max_score)
            elif softmax_score[8] >= threshold or softmax_score[9] >= threshold or softmax_score[10] >= threshold:
                violence_max_score = max(softmax_score[8:11])
                violence_tmp_word.append(str(postag_sentence[i][0][j]))
                violence_tmp_score.append(violence_max_score)
            elif softmax_score[11] >= 0.67 or softmax_score[12] >= threshold or softmax_score[13] >= 0.67 or \
                    softmax_score[14] >= 0.67:
                war_max_score = max(softmax_score[11:15])
                war_tmp_word.append(str(postag_sentence[i][0][j]))
                war_tmp_score.append(war_max_score)
            elif softmax_score[15] >= 0.65 or softmax_score[16] >= 0.65 or softmax_score[17] >= 0.65:
                part_max_score = max(softmax_score[15:18])
                part_tmp_word.append(str(postag_sentence[i][0][j]))
                part_tmp_score.append(part_max_score)
            elif softmax_score[18] >= threshold or softmax_score[19] >= threshold or softmax_score[20] >= threshold:
                arrest_max_score = max(softmax_score[18:21])
                arrest_tmp_word.append(str(postag_sentence[i][0][j]))
                arrest_tmp_score.append(arrest_max_score)
            elif softmax_score[21] >= threshold or softmax_score[22] >= threshold:
                independent_max_score = max(softmax_score[21:23])
                independent_tmp_word.append(str(postag_sentence[i][0][j]))
                independent_tmp_score.append(independent_max_score)
            elif softmax_score[23] >= threshold:
                meet_max_score = max(softmax_score[23:24])
                meet_tmp_word.append(str(postag_sentence[i][0][j]))
                meet_tmp_score.append(meet_max_score)
            elif softmax_score[24] >= threshold:
                move_max_score = max(softmax_score[24:25])
                move_tmp_word.append(str(postag_sentence[i][0][j]))
                move_tmp_score.append(move_max_score)
            elif softmax_score[25] >= 0.65 or softmax_score[26] >= 0.65:
                illness_max_score = max(softmax_score[25:27])
                illness_tmp_word.append(str(postag_sentence[i][0][j]))
                illness_tmp_score.append(illness_max_score)

        death_word_list.append(death_tmp_word)
        death_score_list.append(death_tmp_score)
        birth_word_list.append(birth_tmp_word)
        birth_score_list.append(birth_tmp_score)
        violence_word_list.append(violence_tmp_word)
        violence_score_list.append(violence_tmp_score)
        war_word_list.append(war_tmp_word)
        war_score_list.append(war_tmp_score)
        part_word_list.append(part_tmp_word)
        part_score_list.append(part_tmp_score)
        arrest_word_list.append(arrest_tmp_word)
        arrest_score_list.append(arrest_tmp_score)
        independent_word_list.append(independent_tmp_word)
        independent_score_list.append(independent_tmp_score)
        meet_word_list.append(meet_tmp_word)
        meet_score_list.append(meet_tmp_score)
        move_word_list.append(move_tmp_word)
        move_score_list.append(move_tmp_score)
        illness_word_list.append(illness_tmp_word)
        illness_score_list.append(illness_tmp_score)

    #
    death_cluster_data = []
    birth_cluster_data = []
    violence_cluster_data = []
    war_cluster_data = []
    part_cluster_data = []
    arrest_cluster_data = []
    independent_cluster_data = []
    meet_cluster_data = []
    move_cluster_data = []
    illness_cluster_data = []

    for i in range(len(death_word_list)):
        if len(death_word_list[i]) != 0:
            if len(death_word_list[i]) >= 2:
                for j in range(len(death_word_list[i])):
                    death_cluster_data.append([i, 1])
            else:
                death_cluster_data.append([i, 1])

    for i in range(len(birth_word_list)):
        if len(birth_word_list[i]) != 0:
            if len(birth_word_list[i]) >= 2:
                for j in range(len(birth_word_list[i])):
                    birth_cluster_data.append([i, 1])
            else:
                birth_cluster_data.append([i, 1])

    for i in range(len(violence_word_list)):
        if len(violence_word_list[i]) != 0:
            if len(violence_word_list[i]) >= 2:
                for j in range(len(violence_word_list[i])):
                    violence_cluster_data.append([i, 1])
            else:
                violence_cluster_data.append([i, 1])

    for i in range(len(war_word_list)):
        if len(war_word_list[i]) != 0:
            if len(war_word_list[i]) >= 2:
                for j in range(len(war_word_list[i])):
                    war_cluster_data.append([i, 1])
            else:
                war_cluster_data.append([i, 1])

    for i in range(len(part_word_list)):
        if len(part_word_list[i]) != 0:
            if len(part_word_list[i]) >= 2:
                for j in range(len(part_word_list[i])):
                    part_cluster_data.append([i, 1])
            else:
                part_cluster_data.append([i, 1])

    for i in range(len(arrest_word_list)):
        if len(arrest_word_list[i]) != 0:
            if len(arrest_word_list[i]) >= 2:
                for j in range(len(arrest_word_list[i])):
                    arrest_cluster_data.append([i, 1])
            else:
                arrest_cluster_data.append([i, 1])

    for i in range(len(independent_word_list)):
        if len(independent_word_list[i]) != 0:
            if len(independent_word_list[i]) >= 2:
                for j in range(len(independent_word_list[i])):
                    independent_cluster_data.append([i, 1])
            else:
                independent_cluster_data.append([i, 1])

    for i in range(len(meet_word_list)):
        if len(meet_word_list[i]) != 0:
            if len(meet_word_list[i]) >= 2:
                for j in range(len(meet_word_list[i])):
                    meet_cluster_data.append([i, 1])
            else:
                meet_cluster_data.append([i, 1])

    for i in range(len(move_word_list)):
        if len(move_word_list[i]) != 0:
            if len(move_word_list[i]) >= 2:
                for j in range(len(move_word_list[i])):
                    move_cluster_data.append([i, 1])
            else:
                move_cluster_data.append([i, 1])

    for i in range(len(illness_word_list)):
        if len(illness_word_list[i]) != 0:
            if len(illness_word_list[i]) >= 2:
                for j in range(len(illness_word_list[i])):
                    illness_cluster_data.append([i, 1])
            else:
                illness_cluster_data.append([i, 1])

    death_cluster_data = numpy.array(death_cluster_data)
    birth_cluster_data = numpy.array(birth_cluster_data)
    violence_cluster_data = numpy.array(violence_cluster_data)
    war_cluster_data = numpy.array(war_cluster_data)
    part_cluster_data = numpy.array(part_cluster_data)
    arrest_cluster_data = numpy.array(arrest_cluster_data)
    independent_cluster_data = numpy.array(independent_cluster_data)
    meet_cluster_data = numpy.array(meet_cluster_data)
    move_cluster_data = numpy.array(move_cluster_data)
    illness_cluster_data = numpy.array(illness_cluster_data)

    Epsilon = 5
    MinimumPoints = 2

    # case 1
    if len(death_cluster_data) > 0:
        death_result = DBSCAN(death_cluster_data, Epsilon, MinimumPoints)
    else:
        death_result = []
    if len(birth_cluster_data) > 0:
        birth_result = DBSCAN(birth_cluster_data, Epsilon, MinimumPoints)
    else:
        birth_result = []
    if len(violence_cluster_data) > 0:
        violence_result = DBSCAN(violence_cluster_data, Epsilon, MinimumPoints)
    else:
        violence_result = []
    if len(war_cluster_data) > 0:
        war_result = DBSCAN(war_cluster_data, Epsilon, MinimumPoints)
    else:
        war_result = []
    if len(part_cluster_data) > 0:
        part_result = DBSCAN(part_cluster_data, Epsilon, MinimumPoints)
    else:
        part_result = []
    if len(arrest_cluster_data) > 0:
        arrest_result = DBSCAN(arrest_cluster_data, Epsilon, MinimumPoints)
    else:
        arrest_result = []
    if len(independent_cluster_data) > 0:
        independent_result = DBSCAN(independent_cluster_data, Epsilon, MinimumPoints)
    else:
        independent_result = []
    if len(meet_cluster_data) > 0:
        meet_result = DBSCAN(meet_cluster_data, Epsilon, MinimumPoints)
    else:
        meet_result = []
    if len(move_cluster_data) > 0:
        move_result = DBSCAN(move_cluster_data, Epsilon, MinimumPoints)
    else:
        move_result = []
    if len(illness_cluster_data) > 0:
        illness_result = DBSCAN(illness_cluster_data, Epsilon, MinimumPoints)
    else:
        illness_result = []

    # 겹치는 부분은 여러개 표시 할 것
    # 1. 죽음
    death_cluster_idx = []
    num = 0
    for i in range(len(death_result)):
        num += 1
        clustering = numpy.where(death_result == num)[0]
        clustering.tolist()
        i = i + len(clustering)
        if len(clustering) != 0:
            tmp_idx = []
            for j in range(len(clustering)):
                tmp_idx.append(death_cluster_data[clustering[j]][0])
            death_cluster_idx.append(tmp_idx)

    death_final_data = []
    for i in range(len(death_cluster_idx)):
        death_cluster_idx[i] = list(set(death_cluster_idx[i]))
        death_cluster_idx[i].sort()
        start = death_cluster_idx[i][0]
        last = death_cluster_idx[i][len(death_cluster_idx[i]) - 1]
        final_tmp = []
        for j in range(start, last + 1):
            final_tmp.append(j)
        death_final_data.append(final_tmp)
    # 2. 탄생
    birth_cluster_idx = []
    num = 0
    for i in range(len(birth_result)):
        num += 1
        clustering = numpy.where(birth_result == num)[0]
        clustering.tolist()
        i = i + len(clustering)
        if len(clustering) != 0:
            tmp_idx = []
            for j in range(len(clustering)):
                tmp_idx.append(birth_cluster_data[clustering[j]][0])
            birth_cluster_idx.append(tmp_idx)

    birth_final_data = []
    for i in range(len(birth_cluster_idx)):
        birth_cluster_idx[i] = list(set(birth_cluster_idx[i]))
        birth_cluster_idx[i].sort()
        start = birth_cluster_idx[i][0]
        last = birth_cluster_idx[i][len(birth_cluster_idx[i]) - 1]
        final_tmp = []
        for j in range(start, last + 1):
            final_tmp.append(j)
        birth_final_data.append(final_tmp)
    # 3.폭력
    violence_cluster_idx = []
    num = 0
    for i in range(len(violence_result)):
        num += 1
        clustering = numpy.where(violence_result == num)[0]
        clustering.tolist()
        i = i + len(clustering)
        if len(clustering) != 0:
            tmp_idx = []
            for j in range(len(clustering)):
                tmp_idx.append(violence_cluster_data[clustering[j]][0])
            violence_cluster_idx.append(tmp_idx)

    violence_final_data = []
    for i in range(len(violence_cluster_idx)):
        violence_cluster_idx[i] = list(set(violence_cluster_idx[i]))
        violence_cluster_idx[i].sort()
        start = violence_cluster_idx[i][0]
        last = violence_cluster_idx[i][len(violence_cluster_idx[i]) - 1]
        final_tmp = []
        for j in range(start, last + 1):
            final_tmp.append(j)
        violence_final_data.append(final_tmp)
    # 4.전쟁
    war_cluster_idx = []
    num = 0
    for i in range(len(war_result)):
        num += 1
        clustering = numpy.where(war_result == num)[0]
        clustering.tolist()
        i = i + len(clustering)
        if len(clustering) != 0:
            tmp_idx = []
            for j in range(len(clustering)):
                tmp_idx.append(war_cluster_data[clustering[j]][0])
            war_cluster_idx.append(tmp_idx)

    war_final_data = []
    for i in range(len(war_cluster_idx)):
        war_cluster_idx[i] = list(set(war_cluster_idx[i]))
        war_cluster_idx[i].sort()
        start = war_cluster_idx[i][0]
        last = war_cluster_idx[i][len(war_cluster_idx[i]) - 1]
        final_tmp = []
        for j in range(start, last + 1):
            final_tmp.append(j)
        war_final_data.append(final_tmp)
    # 5.이별
    part_cluster_idx = []
    num = 0
    for i in range(len(part_result)):
        num += 1
        clustering = numpy.where(part_result == num)[0]
        clustering.tolist()
        i = i + len(clustering)
        if len(clustering) != 0:
            tmp_idx = []
            for j in range(len(clustering)):
                tmp_idx.append(part_cluster_data[clustering[j]][0])
            part_cluster_idx.append(tmp_idx)

    part_final_data = []
    for i in range(len(part_cluster_idx)):
        part_cluster_idx[i] = list(set(part_cluster_idx[i]))
        part_cluster_idx[i].sort()
        start = part_cluster_idx[i][0]
        last = part_cluster_idx[i][len(part_cluster_idx[i]) - 1]
        final_tmp = []
        for j in range(start, last + 1):
            final_tmp.append(j)
        part_final_data.append(final_tmp)
    # 6.체포
    arrest_cluster_idx = []
    num = 0
    for i in range(len(arrest_result)):
        num += 1
        clustering = numpy.where(arrest_result == num)[0]
        clustering.tolist()
        i = i + len(clustering)
        if len(clustering) != 0:
            tmp_idx = []
            for j in range(len(clustering)):
                tmp_idx.append(arrest_cluster_data[clustering[j]][0])
            arrest_cluster_idx.append(tmp_idx)

    arrest_final_data = []
    for i in range(len(arrest_cluster_idx)):
        arrest_cluster_idx[i] = list(set(arrest_cluster_idx[i]))
        arrest_cluster_idx[i].sort()
        start = arrest_cluster_idx[i][0]
        last = arrest_cluster_idx[i][len(arrest_cluster_idx[i]) - 1]
        final_tmp = []
        for j in range(start, last + 1):
            final_tmp.append(j)
        arrest_final_data.append(final_tmp)

    # 7.해방
    independent_cluster_idx = []
    num = 0
    for i in range(len(independent_result)):
        num += 1
        clustering = numpy.where(independent_result == num)[0]
        clustering.tolist()
        i = i + len(clustering)
        if len(clustering) != 0:
            tmp_idx = []
            for j in range(len(clustering)):
                tmp_idx.append(independent_cluster_data[clustering[j]][0])
            independent_cluster_idx.append(tmp_idx)

    independent_final_data = []
    for i in range(len(independent_cluster_idx)):
        independent_cluster_idx[i] = list(set(independent_cluster_idx[i]))
        independent_cluster_idx[i].sort()
        start = independent_cluster_idx[i][0]
        last = independent_cluster_idx[i][len(independent_cluster_idx[i]) - 1]
        final_tmp = []
        for j in range(start, last + 1):
            final_tmp.append(j)
        independent_final_data.append(final_tmp)
    # 8.만남
    meet_cluster_idx = []
    num = 0
    for i in range(len(meet_result)):
        num += 1
        clustering = numpy.where(meet_result == num)[0]
        clustering.tolist()
        i = i + len(clustering)
        if len(clustering) != 0:
            tmp_idx = []
            for j in range(len(clustering)):
                tmp_idx.append(meet_cluster_data[clustering[j]][0])
            meet_cluster_idx.append(tmp_idx)

    meet_final_data = []
    for i in range(len(meet_cluster_idx)):
        meet_cluster_idx[i] = list(set(meet_cluster_idx[i]))
        meet_cluster_idx[i].sort()
        start = meet_cluster_idx[i][0]
        last = meet_cluster_idx[i][len(meet_cluster_idx[i]) - 1]
        final_tmp = []
        for j in range(start, last + 1):
            final_tmp.append(j)
        meet_final_data.append(final_tmp)

    # 9.이사
    move_cluster_idx = []
    num = 0
    for i in range(len(move_result)):
        num += 1
        clustering = numpy.where(move_result == num)[0]
        clustering.tolist()
        i = i + len(clustering)
        if len(clustering) != 0:
            tmp_idx = []
            for j in range(len(clustering)):
                tmp_idx.append(move_cluster_data[clustering[j]][0])
            move_cluster_idx.append(tmp_idx)

    move_final_data = []
    for i in range(len(move_cluster_idx)):
        move_cluster_idx[i] = list(set(move_cluster_idx[i]))
        move_cluster_idx[i].sort()
        start = move_cluster_idx[i][0]
        last = move_cluster_idx[i][len(move_cluster_idx[i]) - 1]
        final_tmp = []
        for j in range(start, last + 1):
            final_tmp.append(j)
        move_final_data.append(final_tmp)

    # 10.병
    illness_cluster_idx = []
    num = 0
    for i in range(len(illness_result)):
        num += 1
        clustering = numpy.where(illness_result == num)[0]
        clustering.tolist()
        i = i + len(clustering)
        if len(clustering) != 0:
            tmp_idx = []
            for j in range(len(clustering)):
                tmp_idx.append(illness_cluster_data[clustering[j]][0])
            illness_cluster_idx.append(tmp_idx)

    illness_final_data = []
    for i in range(len(illness_cluster_idx)):
        illness_cluster_idx[i] = list(set(illness_cluster_idx[i]))
        illness_cluster_idx[i].sort()
        start = illness_cluster_idx[i][0]
        last = illness_cluster_idx[i][len(illness_cluster_idx[i]) - 1]
        final_tmp = []
        for j in range(start, last + 1):
            final_tmp.append(j)
        illness_final_data.append(final_tmp)

    # json output

    json_data = []
    death = sum(death_final_data, [])
    birth = sum(birth_final_data, [])
    violence = sum(violence_final_data, [])
    war = sum(war_final_data, [])
    part = sum(part_final_data, [])
    arrest = sum(arrest_final_data, [])
    independent = sum(independent_final_data, [])
    meet = sum(meet_final_data, [])
    move = sum(move_final_data, [])
    illness = sum(illness_final_data, [])

    sen_idx = 0
    for i in range(len(original_sentence)):
        for p in range(len(original_sentence[i][1])):
            json_data.append({"page": str(original_sentence[i][1][p]), "text": original_sentence[i][0], "tags": []})
            if i in death:
                if len(death_word_list[i]) > 0:
                    for j in range(len(death_word_list[i])):
                        json_data[sen_idx]["tags"].append(
                            {"str": death_word_list[i][j], "type": "EVT", "sub-type": "죽음"})
                else:
                    json_data[sen_idx]["tags"].append({"str": "", "type": "EVT", "sub-type": "죽음"})
            if i in birth:
                if len(birth_word_list[i]) > 0:
                    for j in range(len(birth_word_list[i])):
                        json_data[sen_idx]["tags"].append(
                            {"str": birth_word_list[i][j], "type": "EVT", "sub-type": "탄생"})
                else:
                    json_data[sen_idx]["tags"].append({"str": "", "type": "EVT", "sub-type": "탄생"})
            if i in violence:
                if len(violence_word_list[i]) > 0:
                    for j in range(len(violence_word_list[i])):
                        json_data[sen_idx]["tags"].append(
                            {"str": violence_word_list[i][j], "type": "EVT", "sub-type": "폭력"})
                else:
                    json_data[sen_idx]["tags"].append({"str": "", "type": "EVT", "sub-type": "폭력"})
            if i in war:
                if len(war_word_list[i]) > 0:
                    for j in range(len(war_word_list[i])):
                        json_data[sen_idx]["tags"].append({"str": war_word_list[i][j], "type": "EVT", "sub-type": "전쟁"})
                else:
                    json_data[sen_idx]["tags"].append({"str": "", "type": "EVT", "sub-type": "전쟁"})
            if i in part:
                if len(part_word_list[i]) > 0:
                    for j in range(len(part_word_list[i])):
                        json_data[sen_idx]["tags"].append(
                            {"str": part_word_list[i][j], "type": "EVT", "sub-type": "이별"})
                else:
                    json_data[sen_idx]["tags"].append({"str": "", "type": "EVT", "sub-type": "이별"})
            if i in arrest:
                if len(arrest_word_list[i]) > 0:
                    for j in range(len(arrest_word_list[i])):
                        json_data[sen_idx]["tags"].append(
                            {"str": arrest_word_list[i][j], "type": "EVT", "sub-type": "체포"})
                else:
                    json_data[sen_idx]["tags"].append({"str": "", "type": "EVT", "sub-type": "체포"})
            if i in independent:
                if len(independent_word_list[i]) > 0:
                    for j in range(len(independent_word_list[i])):
                        json_data[sen_idx]["tags"].append(
                            {"str": independent_word_list[i][j], "type": "EVT", "sub-type": "독립"})
                else:
                    json_data[sen_idx]["tags"].append({"str": "", "type": "EVT", "sub-type": "독립"})
            if i in meet:
                if len(meet_word_list[i]) > 0:
                    for j in range(len(meet_word_list[i])):
                        json_data[sen_idx]["tags"].append(
                            {"str": meet_word_list[i][j], "type": "EVT", "sub-type": "만남"})
                else:
                    json_data[sen_idx]["tags"].append({"str": "", "type": "EVT", "sub-type": "만남"})
            if i in move:
                if len(move_word_list[i]) > 0:
                    for j in range(len(move_word_list[i])):
                        json_data[sen_idx]["tags"].append(
                            {"str": move_word_list[i][j], "type": "EVT", "sub-type": "이사"})
                else:
                    json_data[sen_idx]["tags"].append({"str": "", "type": "EVT", "sub-type": "이사"})
            if i in illness:
                if len(illness_word_list[i]) > 0:
                    for j in range(len(illness_word_list[i])):
                        json_data[sen_idx]["tags"].append(
                            {"str": illness_word_list[i][j], "type": "EVT", "sub-type": "병"})
                else:
                    json_data[sen_idx]["tags"].append({"str": "", "type": "EVT", "sub-type": "병"})
            sen_idx += 1


    output_file_path = path + "/json/" + output
    with open(output_file_path, 'w', encoding='utf-8') as jason_output_file:
        json.dump(json_data, jason_output_file, ensure_ascii=False, indent="\t")

    # 폰트 경로 제거
    # font_name = font_manager.FontProperties(fname=font_path).get_name()

    matplotlib.rc('font', family='NanumGothic')
    fig, ax = plt.subplots(figsize=(20, 8))
    # ax.set(title="Plot Timeline",)
    # 동일한 이벤트가 두개 이상 벌어질 경우 -> score 합쳐버림
    # 한 문장에 두개 이상의 이벤트 가질 수 있음
    # %로 표현하는 것도 한 문장에 동일한 이벤트 두개라고 해도 하나로 보기로함
    total_events = []
    total_events_score = []

    death_count = 0
    birth_count = 0
    violence_count = 0
    part_count = 0
    war_count = 0
    arrest_count = 0
    independent_count = 0
    meet_count = 0
    move_count = 0
    illness_count = 0

    for i in range(len(original_sentence)):
        tmp_word = []
        tmp_score = []
        if i in death:
            if len(death_score_list[i]) > 0:
                tmp_word.append('죽음')
                tmp_score.append(sum(death_score_list[i]))
            else:
                tmp_word.append('죽음')
                tmp_score.append(0.65)
        if i in birth:
            if len(birth_score_list[i]) > 0:
                tmp_word.append('탄생')
                tmp_score.append(-sum(birth_score_list[i]))
            else:
                tmp_word.append('탄생')
                tmp_score.append(-0.65)
        if i in violence:
            if len(violence_score_list[i]) > 0:
                tmp_word.append('폭력')
                tmp_score.append(sum(violence_score_list[i]))
            else:
                tmp_word.append('폭력')
                tmp_score.append(0.65)
        if i in war:
            if len(war_score_list[i]) > 0:
                tmp_word.append('전쟁')
                tmp_score.append(sum(war_score_list[i]))
            else:
                tmp_word.append('전쟁')
                tmp_score.append(0.65)
        if i in part:
            if len(part_score_list[i]) > 0:
                tmp_word.append('이별')
                tmp_score.append(sum(part_score_list[i]))
            else:
                tmp_word.append('이별')
                tmp_score.append(0.65)
        if i in arrest:
            if len(arrest_score_list[i]) > 0:
                tmp_word.append('체포')
                tmp_score.append(-sum(arrest_score_list[i]))
            else:
                tmp_word.append('체포')
                tmp_score.append(-0.65)
        if i in independent:
            if len(independent_score_list[i]) > 0:
                tmp_word.append('독립')
                tmp_score.append(-sum(independent_score_list[i]))
            else:
                tmp_word.append('독립')
                tmp_score.append(-0.65)
        if i in meet:
            if len(meet_score_list[i]) > 0:
                tmp_word.append('만남')
                tmp_score.append(-sum(meet_score_list[i]))
            else:
                tmp_word.append('만남')
                tmp_score.append(-0.65)
        if i in move:
            if len(move_score_list[i]) > 0:
                tmp_word.append('이사')
                tmp_score.append(-sum(move_score_list[i]))
            else:
                tmp_word.append('이사')
                tmp_score.append(-0.65)
        if i in illness:
            if len(illness_score_list[i]) > 0:
                tmp_word.append('병')
                tmp_score.append(sum(illness_score_list[i]))
            else:
                tmp_word.append('병')
                tmp_score.append(0.65)
        total_events.append(tmp_word)
        total_events_score.append(tmp_score)

    for i in range(len(total_events)):
        for p in range(len(death_final_data)):
            y_death = total_events[death_final_data[p][0]].index("죽음")
            y_sdeath = total_events_score[death_final_data[p][0]][y_death]
            ax.annotate(death_final_data[p][0], xy=(death_final_data[p][0] - 2, y_sdeath + 0.1), rotation=90,
                        fontsize=10)
            if len(death_final_data[p]) > 1:
                y_death = total_events[death_final_data[p][len(death_final_data[p]) - 1]].index("죽음")
                y_sdeath = total_events_score[death_final_data[p][len(death_final_data[p]) - 1]][y_death]
                ax.annotate(death_final_data[p][len(death_final_data[p]) - 1],
                            xy=(death_final_data[p][len(death_final_data[p]) - 1] + 2, y_sdeath + 0.1), rotation=90,
                            fontsize=10)
        for p in range(len(birth_final_data)):
            y_birth = total_events[birth_final_data[p][0]].index("탄생")
            y_sbirth = total_events_score[birth_final_data[p][0]][y_birth]
            ax.annotate(birth_final_data[p][0], xy=(birth_final_data[p][0] - 2, y_sbirth - 0.1), rotation=90,
                        fontsize=10)
            if len(birth_final_data[p]) > 1:
                y_birth = total_events[birth_final_data[p][len(birth_final_data[p]) - 1]].index("탄생")
                y_sbirth = total_events_score[birth_final_data[p][len(birth_final_data[p]) - 1]][y_birth]
                ax.annotate(birth_final_data[p][len(birth_final_data[p]) - 1],
                            xy=(birth_final_data[p][len(birth_final_data[p]) - 1] + 2, y_sbirth - 0.1), rotation=90,
                            fontsize=10)
        for p in range(len(violence_final_data)):
            y_violence = total_events[violence_final_data[p][0]].index("폭력")
            y_sviolence = total_events_score[violence_final_data[p][0]][y_violence]
            ax.annotate(violence_final_data[p][0], xy=(violence_final_data[p][0] - 2, y_sviolence + 0.1), rotation=90,
                        fontsize=10)
            if len(violence_final_data[p]) > 1:
                y_violence = total_events[violence_final_data[p][len(violence_final_data[p]) - 1]].index("폭력")
                y_sviolence = total_events_score[violence_final_data[p][len(violence_final_data[p]) - 1]][y_violence]
                ax.annotate(violence_final_data[p][len(violence_final_data[p]) - 1],
                            xy=(violence_final_data[p][len(violence_final_data[p]) - 1] + 2, y_sviolence + 0.1),
                            rotation=90, fontsize=10)
        for p in range(len(war_final_data)):
            y_war = total_events[war_final_data[p][0]].index("전쟁")
            y_swar = total_events_score[war_final_data[p][0]][y_war]
            ax.annotate(war_final_data[p][0], xy=(war_final_data[p][0] - 2, y_swar + 0.1), rotation=90, fontsize=10)
            if len(war_final_data[p]) > 1:
                y_war = total_events[war_final_data[p][len(war_final_data[p]) - 1]].index("전쟁")
                y_swar = total_events_score[war_final_data[p][len(war_final_data[p]) - 1]][y_war]
                ax.annotate(war_final_data[p][len(war_final_data[p]) - 1],
                            xy=(war_final_data[p][len(war_final_data[p]) - 1] + 2, y_swar + 0.1), rotation=90,
                            fontsize=10)
        for p in range(len(part_final_data)):
            y_part = total_events[part_final_data[p][0]].index("이별")
            y_spart = total_events_score[part_final_data[p][0]][y_part]
            ax.annotate(part_final_data[p][0], xy=(part_final_data[p][0] - 2, y_spart + 0.1), rotation=90, fontsize=10)
            if len(part_final_data[p]) > 1:
                y_part = total_events[part_final_data[p][len(part_final_data[p]) - 1]].index("이별")
                y_spart = total_events_score[part_final_data[p][len(part_final_data[p]) - 1]][y_part]
                ax.annotate(part_final_data[p][len(part_final_data[p]) - 1],
                            xy=(part_final_data[p][len(part_final_data[p]) - 1] + 2, y_spart + 0.1), rotation=90,
                            fontsize=10)
        for p in range(len(illness_final_data)):
            y_illness = total_events[illness_final_data[p][0]].index("병")
            y_sillness = total_events_score[illness_final_data[p][0]][y_illness]
            ax.annotate(illness_final_data[p][0], xy=(illness_final_data[p][0] - 2, y_sillness + 0.1), rotation=90,
                        fontsize=10)
            if len(illness_final_data[p]) > 1:
                y_illness = total_events[illness_final_data[p][len(illness_final_data[p]) - 1]].index("병")
                y_sillness = total_events_score[illness_final_data[p][len(illness_final_data[p]) - 1]][y_illness]
                ax.annotate(illness_final_data[p][len(illness_final_data[p]) - 1],
                            xy=(illness_final_data[p][len(illness_final_data[p]) - 1] + 2, y_sillness + 0.1),
                            rotation=90, fontsize=10)
        for p in range(len(arrest_final_data)):
            y_arrest = total_events[arrest_final_data[p][0]].index("체포")
            y_sarrest = total_events_score[arrest_final_data[p][0]][y_arrest]
            ax.annotate(arrest_final_data[p][0], xy=(arrest_final_data[p][0] - 2, y_sarrest - 0.1), rotation=90,
                        fontsize=10)
            if len(arrest_final_data[p]) > 1:
                y_arrest = total_events[arrest_final_data[p][len(arrest_final_data[p]) - 1]].index("체포")
                y_sarrest = total_events_score[arrest_final_data[p][len(arrest_final_data[p]) - 1]][y_arrest]
                ax.annotate(arrest_final_data[p][len(arrest_final_data[p]) - 1],
                            xy=(arrest_final_data[p][len(arrest_final_data[p]) - 1] + 2, y_sarrest - 0.1), rotation=90,
                            fontsize=10)
        for p in range(len(independent_final_data)):
            y_independent = total_events[independent_final_data[p][0]].index("독립")
            y_sindependent = total_events_score[independent_final_data[p][0]][y_independent]
            ax.annotate(independent_final_data[p][0], xy=(independent_final_data[p][0] - 2, y_sindependent - 0.1),
                        rotation=90, fontsize=10)
            if len(independent_final_data[p]) > 1:
                y_independent = total_events[independent_final_data[p][len(independent_final_data[p]) - 1]].index("독립")
                y_sindependent = total_events_score[independent_final_data[p][len(independent_final_data[p]) - 1]][
                    y_independent]
                ax.annotate(independent_final_data[p][len(independent_final_data[p]) - 1], xy=(
                independent_final_data[p][len(independent_final_data[p]) - 1] + 2, y_sindependent - 0.1), rotation=90,
                            fontsize=10)
        for p in range(len(meet_final_data)):
            y_meet = total_events[meet_final_data[p][0]].index("만남")
            y_smeet = total_events_score[meet_final_data[p][0]][y_meet]
            ax.annotate(meet_final_data[p][0], xy=(meet_final_data[p][0] - 2, y_smeet - 0.1), rotation=90, fontsize=10)
            if len(meet_final_data[p]) > 1:
                y_meet = total_events[meet_final_data[p][len(meet_final_data[p]) - 1]].index("만남")
                y_smeet = total_events_score[meet_final_data[p][len(meet_final_data[p]) - 1]][y_meet]
                ax.annotate(meet_final_data[p][len(meet_final_data[p]) - 1],
                            xy=(meet_final_data[p][len(meet_final_data[p]) - 1] + 2, y_smeet - 0.1), rotation=90,
                            fontsize=10)
        for p in range(len(move_final_data)):
            y_move = total_events[move_final_data[p][0]].index("이사")
            y_smove = total_events_score[move_final_data[p][0]][y_move]
            ax.annotate(move_final_data[p][0], xy=(move_final_data[p][0] - 2, y_smove - 0.1), rotation=90, fontsize=10)
            if len(move_final_data[p]) > 1:
                y_move = total_events[move_final_data[p][len(move_final_data[p]) - 1]].index("이사")
                y_smove = total_events_score[move_final_data[p][len(move_final_data[p]) - 1]][y_move]
                ax.annotate(move_final_data[p][len(move_final_data[p]) - 1],
                            xy=(move_final_data[p][len(move_final_data[p]) - 1] + 2, y_smove - 0.1), rotation=90,
                            fontsize=10)
        if len(total_events[i]) != 0:
            for j in range(len(total_events[i])):
                if total_events[i][j] == "죽음":
                    death_count += 1
                    markerline, stemline, baseline = ax.stem([i], [total_events_score[i][j]], '#FE2E2E', markerfmt='o',
                                                             basefmt="k-", use_line_collection=True)
                    plt.setp(markerline, color='#FE2E2E', linewidth=2)
                    plt.setp(stemline, color='#FE2E2E')
                elif total_events[i][j] == "탄생":
                    birth_count += 1
                    markerline1, stemline1, baseline1 = ax.stem([i], [total_events_score[i][j]], '#FF8000',
                                                                markerfmt='o', basefmt="k-", use_line_collection=True)
                    plt.setp(markerline1, color='#FF8000', linewidth=2)
                    plt.setp(stemline1, color='#FF8000')
                elif total_events[i][j] == "폭력":
                    violence_count += 1
                    markerline2, stemline2, baseline2 = ax.stem([i], [total_events_score[i][j]], '#088A29',
                                                                markerfmt='o', basefmt="k-", use_line_collection=True)
                    plt.setp(markerline2, color='#088A29', linewidth=2)
                    plt.setp(stemline2, color='#088A29')
                elif total_events[i][j] == "전쟁":
                    war_count += 1
                    markerline3, stemline3, baseline3 = ax.stem([i], [total_events_score[i][j]], '#FFFF00',
                                                                markerfmt='o', basefmt="k-", use_line_collection=True)
                    plt.setp(markerline3, color='#FFFF00', linewidth=2)
                    plt.setp(stemline3, color='#FFFF00')
                elif total_events[i][j] == "이별":
                    part_count += 1
                    markerline4, stemline4, baseline4 = ax.stem([i], [total_events_score[i][j]], '#00FFFF',
                                                                markerfmt='o', basefmt="k-", use_line_collection=True)
                    plt.setp(markerline4, color='#00FFFF', linewidth=2)
                    plt.setp(stemline4, color='#00FFFF')
                elif total_events[i][j] == "병":
                    illness_count += 1
                    markerline5, stemline5, baseline5 = ax.stem([i], [total_events_score[i][j]], '#0040FF',
                                                                markerfmt='o', basefmt="k-", use_line_collection=True)
                    plt.setp(markerline5, color='#0040FF', linewidth=2)
                    plt.setp(stemline5, color='#0040FF')
                elif total_events[i][j] == "체포":
                    arrest_count += 1
                    markerline6, stemline6, baseline6 = ax.stem([i], [total_events_score[i][j]], '#8258FA',
                                                                markerfmt='o', basefmt="k-", use_line_collection=True)
                    plt.setp(markerline6, color='#8258FA', linewidth=2)
                    plt.setp(stemline6, color='#8258FA')
                elif total_events[i][j] == "독립":
                    independent_count += 1
                    markerline7, stemline7, baseline7 = ax.stem([i], [total_events_score[i][j]], '#FE2EF7',
                                                                markerfmt='o', basefmt="k-", use_line_collection=True)
                    plt.setp(markerline7, color='#FE2EF7', linewidth=2)
                    plt.setp(stemline7, color='#FE2EF7')
                elif total_events[i][j] == "만남":
                    meet_count += 1
                    markerline8, stemline8, baseline8 = ax.stem([i], [total_events_score[i][j]], '#E2DAB5',
                                                                markerfmt='o', basefmt="k-", use_line_collection=True)
                    plt.setp(markerline8, color='#E2DAB5', linewidth=2)
                    plt.setp(stemline8, color='#E2DAB5')
                elif total_events[i][j] == "이사":
                    move_count += 1
                    markerline9, stemline9, baseline9 = ax.stem([i], [total_events_score[i][j]], '#BD9424',
                                                                markerfmt='o', basefmt="k-", use_line_collection=True)
                    plt.setp(markerline9, color='#BD9424', linewidth=2)
                    plt.setp(stemline9, color='#BD9424')

    ax.get_yaxis().set_visible(False)
    for spine in ["left", "top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.axhline(y=0, color='#BDBDBD', linewidth=6, linestyle=':')
    custom_lines = [Line2D([0], [0], color='#FE2E2E', lw=4), Line2D([0], [0], color='#FF8000', lw=4),
                    Line2D([0], [0], color='#088A29', lw=4), Line2D([0], [0], color='#FFFF00', lw=4),
                    Line2D([0], [0], color='#00FFFF', lw=4), Line2D([0], [0], color='#0040FF', lw=4),
                    Line2D([0], [0], color='#8258FA', lw=4), Line2D([0], [0], color='#FE2EF7', lw=4),
                    Line2D([0], [0], color='#E2DAB5', lw=4), Line2D([0], [0], color='#BD9424', lw=4)]
    plt.legend(custom_lines, ["-- 죽음", "-- 탄생", "-- 폭력", "-- 전쟁", "--이별", "--병", "--체포", "--독립", "--만남", "--이사"],
               loc='lower left', bbox_to_anchor=(0, 0.975), ncol=4)
    plt.tight_layout()
    plt.savefig(path + '/visualization/' + image_path + '_events_timeline.png', dpi=300)
    # plt.show()
    plt.close(fig)

    with open(path + '/matching/' + image_path + '.txt', mode="w", encoding='utf-8') as fout:
        fout.write('죽음' + ':' + str(death_count) + '\n')
        fout.write('탄생' + ':' + str(birth_count) + '\n')
        fout.write('폭력' + ':' + str(violence_count) + '\n')
        fout.write('전쟁' + ':' + str(war_count) + '\n')
        fout.write('이별' + ':' + str(part_count) + '\n')
        fout.write('체포' + ':' + str(arrest_count) + '\n')
        fout.write('독립' + ':' + str(independent_count) + '\n')
        fout.write('만남' + ':' + str(meet_count) + '\n')
        fout.write('이사' + ':' + str(move_count) + '\n')
        fout.write('병' + ':' + str(illness_count) + '\n')
    fout.close()

    plt.figure(2)
    fig, ax1 = plt.subplots(figsize=(30, 13))

    ax1.set_xlabel('문장')
    ax1.set_ylabel('사건(스코어)')

    for i in range(len(total_events)):
        if len(total_events[i]) == 1:
            if total_events[i][0] == "죽음":
                if abs(total_events_score[i][0]) >= 1.5:
                    ax1.scatter(i, abs(total_events_score[i][0]), marker='d', color='#FE2E2E', s=6000)
                    ax1.text(i, abs(total_events_score[i][0]) - 0.02, "({})".format(i), fontsize=10.5, color="black",
                             ha='center', va='bottom', rotation=90)
                elif abs(total_events_score[i][0]) >= 1:
                    ax1.scatter(i, abs(total_events_score[i][0]), marker='d', color='#FE2E2E', s=4000)
                    ax1.text(i, abs(total_events_score[i][0]) - 0.02, "({})".format(i), fontsize=10.5, color="black",
                             ha='center', va='bottom', rotation=90)
                elif abs(total_events_score[i][0]) > 0.65:
                    ax1.scatter(i, abs(total_events_score[i][0]), marker='d', color='#FE2E2E', s=3000)
                    ax1.text(i, abs(total_events_score[i][0]) - 0.02, "({})".format(i), fontsize=10.5, color="black",
                             ha='center', va='bottom', rotation=90)
                else:
                    ax1.scatter(i, abs(total_events_score[i][0]), marker='d', color='#FE2E2E', s=1000)
                    ax1.text(i, abs(total_events_score[i][0]) - 0.02, "({})".format(i), fontsize=10.5, color="black",
                             ha='center', va='bottom', rotation=90)
            if total_events[i][0] == "탄생":
                if abs(total_events_score[i][0]) >= 1.5:
                    ax1.scatter(i, abs(total_events_score[i][0]), marker='d', color='#FF8000', s=6000)
                    ax1.text(i, abs(total_events_score[i][0]) - 0.02, "({})".format(i), fontsize=10, color="black",
                             ha='center', va='bottom', rotation=90)
                elif abs(total_events_score[i][0]) >= 1.0:
                    ax1.scatter(i, abs(total_events_score[i][0]), marker='d', color='#FF8000', s=4000)
                    ax1.text(i, abs(total_events_score[i][0]) - 0.02, "({})".format(i), fontsize=10, color="black",
                             ha='center', va='bottom', rotation=90)
                elif abs(total_events_score[i][0]) > 0.65:
                    ax1.scatter(i, abs(total_events_score[i][0]), marker='d', color='#FF8000', s=3000)
                    ax1.text(i, abs(total_events_score[i][0]) - 0.02, "({})".format(i), fontsize=10, color="black",
                             ha='center', va='bottom', rotation=90)
                else:
                    ax1.scatter(i, abs(total_events_score[i][0]), marker='d', color='#FF8000', s=1000)
                    ax1.text(i, abs(total_events_score[i][0]) - 0.02, "({})".format(i), fontsize=10, color="black",
                             ha='center', va='bottom', rotation=90)
            if total_events[i][0] == "폭력":
                if abs(total_events_score[i][0]) >= 1.5:
                    ax1.scatter(i, abs(total_events_score[i][0]), marker='d', color='#088A29', s=6000)
                    ax1.text(i, abs(total_events_score[i][0]) - 0.02, "({})".format(i), fontsize=10, color="black",
                             ha='center', va='bottom', rotation=90)
                elif abs(total_events_score[i][0]) >= 1.0:
                    ax1.scatter(i, abs(total_events_score[i][0]), marker='d', color='#088A29', s=4000)
                    ax1.text(i, abs(total_events_score[i][0]) - 0.02, "({})".format(i), fontsize=10, color="black",
                             ha='center', va='bottom', rotation=90)
                elif abs(total_events_score[i][0]) > 0.65:
                    ax1.scatter(i, abs(total_events_score[i][0]), marker='d', color='#088A29', s=3000)
                    ax1.text(i, abs(total_events_score[i][0]) - 0.02, "({})".format(i), fontsize=10, color="black",
                             ha='center', va='bottom', rotation=90)
                else:
                    ax1.scatter(i, abs(total_events_score[i][0]), marker='d', color='#088A29', s=1000)
                    ax1.text(i, abs(total_events_score[i][0]) - 0.02, "({})".format(i), fontsize=10, color="black",
                             ha='center', va='bottom', rotation=90)
            if total_events[i][0] == "전쟁":
                if abs(total_events_score[i][0]) >= 1.5:
                    ax1.scatter(i, abs(total_events_score[i][0]), marker='d', color='#FFFF00', s=6000)
                    ax1.text(i, abs(total_events_score[i][0]) - 0.02, "({})".format(i), fontsize=10, color="black",
                             ha='center', va='bottom', rotation=90)
                elif abs(total_events_score[i][0]) >= 1:
                    ax1.scatter(i, abs(total_events_score[i][0]), marker='d', color='#FFFF00', s=4000)
                    ax1.text(i, abs(total_events_score[i][0]) - 0.02, "({})".format(i), fontsize=10, color="black",
                             ha='center', va='bottom', rotation=90)
                elif abs(total_events_score[i][0]) > 0.65:
                    ax1.scatter(i, abs(total_events_score[i][0]), marker='d', color='#FFFF00', s=3000)
                    ax1.text(i, abs(total_events_score[i][0]) - 0.02, "({})".format(i), fontsize=10, color="black",
                             ha='center', va='bottom', rotation=90)
                else:
                    ax1.scatter(i, abs(total_events_score[i][0]), marker='d', color='#FFFF00', s=1000)
                    ax1.text(i, abs(total_events_score[i][0]) - 0.02, "({})".format(i), fontsize=10, color="black",
                             ha='center', va='bottom', rotation=90)
            if total_events[i][0] == "이별":
                if abs(total_events_score[i][0]) >= 1.5:
                    ax1.scatter(i, abs(total_events_score[i][0]), marker='d', color='#00FFFF', s=6000)
                    ax1.text(i, abs(total_events_score[i][0]) - 0.02, "({})".format(i), fontsize=10, color="black",
                             ha='center', va='bottom', rotation=90)
                elif abs(total_events_score[i][0]) >= 1:
                    ax1.scatter(i, abs(total_events_score[i][0]), marker='d', color='#00FFFF', s=4000)
                    ax1.text(i, abs(total_events_score[i][0]) - 0.02, "({})".format(i), fontsize=10, color="black",
                             ha='center', va='bottom', rotation=90)
                elif abs(total_events_score[i][0]) > 0.65:
                    ax1.scatter(i, abs(total_events_score[i][0]), marker='d', color='#00FFFF', s=3000)
                    ax1.text(i, abs(total_events_score[i][0]) - 0.02, "({})".format(i), fontsize=10, color="black",
                             ha='center', va='bottom', rotation=90)
                else:
                    ax1.scatter(i, abs(total_events_score[i][0]), marker='d', color='#00FFFF', s=1000)
                    ax1.text(i, abs(total_events_score[i][0]) - 0.02, "({})".format(i), fontsize=10, color="black",
                             ha='center', va='bottom', rotation=90)
            if total_events[i][0] == "병":
                if abs(total_events_score[i][0]) >= 1.5:
                    ax1.scatter(i, abs(total_events_score[i][0]), marker='d', color='#0040FF', s=6000)
                    ax1.text(i, abs(total_events_score[i][0]) - 0.02, "({})".format(i), fontsize=10.5, color="black",
                             ha='center', va='bottom', rotation=90)
                elif abs(total_events_score[i][0]) >= 1:
                    ax1.scatter(i, abs(total_events_score[i][0]), marker='d', color='#0040FF', s=4000)
                    ax1.text(i, abs(total_events_score[i][0]) - 0.02, "({})".format(i), fontsize=10.5, color="black",
                             ha='center', va='bottom', rotation=90)
                elif abs(total_events_score[i][0]) > 0.65:
                    ax1.scatter(i, abs(total_events_score[i][0]), marker='d', color='#0040FF', s=3000)
                    ax1.text(i, abs(total_events_score[i][0]) - 0.02, "({})".format(i), fontsize=10.5, color="black",
                             ha='center', va='bottom', rotation=90)
                else:
                    ax1.scatter(i, abs(total_events_score[i][0]), marker='d', color='#0040FF', s=1000)
                    ax1.text(i, abs(total_events_score[i][0]) - 0.02, "({})".format(i), fontsize=10.5, color="black",
                             ha='center', va='bottom', rotation=90)
            if total_events[i][0] == "체포":
                if abs(total_events_score[i][0]) >= 1.5:
                    ax1.scatter(i, abs(total_events_score[i][0]), marker='d', color='#8258FA', s=6000)
                    ax1.text(i, abs(total_events_score[i][0]) - 0.02, "({})".format(i), fontsize=10, color="black",
                             ha='center', va='bottom', rotation=90)
                elif abs(total_events_score[i][0]) >= 1.0:
                    ax1.scatter(i, abs(total_events_score[i][0]), marker='d', color='#8258FA', s=4000)
                    ax1.text(i, abs(total_events_score[i][0]) - 0.02, "({})".format(i), fontsize=10, color="black",
                             ha='center', va='bottom', rotation=90)
                elif abs(total_events_score[i][0]) > 0.65:
                    ax1.scatter(i, abs(total_events_score[i][0]), marker='d', color='#8258FA', s=3000)
                    ax1.text(i, abs(total_events_score[i][0]) - 0.02, "({})".format(i), fontsize=10, color="black",
                             ha='center', va='bottom', rotation=90)
                else:
                    ax1.scatter(i, abs(total_events_score[i][0]), marker='d', color='#8258FA', s=1000)
                    ax1.text(i, abs(total_events_score[i][0]) - 0.02, "({})".format(i), fontsize=10, color="black",
                             ha='center', va='bottom', rotation=90)
            if total_events[i][0] == "독립":
                if abs(total_events_score[i][0]) >= 1.5:
                    ax1.scatter(i, abs(total_events_score[i][0]), marker='d', color='#FE2EF7', s=6000)
                    ax1.text(i, abs(total_events_score[i][0]) - 0.02, "({})".format(i), fontsize=10, color="black",
                             ha='center', va='bottom', rotation=90)
                elif abs(total_events_score[i][0]) >= 1.0:
                    ax1.scatter(i, abs(total_events_score[i][0]), marker='d', color='#FE2EF7', s=4000)
                    ax1.text(i, abs(total_events_score[i][0]) - 0.02, "({})".format(i), fontsize=10, color="black",
                             ha='center', va='bottom', rotation=90)
                elif abs(total_events_score[i][0]) > 0.65:
                    ax1.scatter(i, abs(total_events_score[i][0]), marker='d', color='#FE2EF7', s=3000)
                    ax1.text(i, abs(total_events_score[i][0]) - 0.02, "({})".format(i), fontsize=10, color="black",
                             ha='center', va='bottom', rotation=90)
                else:
                    ax1.scatter(i, abs(total_events_score[i][0]), marker='d', color='#FE2EF7', s=1000)
                    ax1.text(i, abs(total_events_score[i][0]) - 0.02, "({})".format(i), fontsize=10, color="black",
                             ha='center', va='bottom', rotation=90)
            if total_events[i][0] == "만남":
                if abs(total_events_score[i][0]) >= 1.5:
                    ax1.scatter(i, abs(total_events_score[i][0]), marker='d', color='#E2DAB5', s=6000)
                    ax1.text(i, abs(total_events_score[i][0]) - 0.02, "({})".format(i), fontsize=10, color="black",
                             ha='center', va='bottom', rotation=90)
                elif abs(total_events_score[i][0]) >= 1:
                    ax1.scatter(i, abs(total_events_score[i][0]), marker='d', color='#E2DAB5', s=4000)
                    ax1.text(i, abs(total_events_score[i][0]) - 0.02, "({})".format(i), fontsize=10, color="black",
                             ha='center', va='bottom', rotation=90)
                elif abs(total_events_score[i][0]) > 0.65:
                    ax1.scatter(i, abs(total_events_score[i][0]), marker='d', color='#E2DAB5', s=3000)
                    ax1.text(i, abs(total_events_score[i][0]) - 0.02, "({})".format(i), fontsize=10, color="black",
                             ha='center', va='bottom', rotation=90)
                else:
                    ax1.scatter(i, abs(total_events_score[i][0]), marker='d', color='#E2DAB5', s=1000)
                    ax1.text(i, abs(total_events_score[i][0]) - 0.02, "({})".format(i), fontsize=10, color="black",
                             ha='center', va='bottom', rotation=90)
            if total_events[i][0] == "이사":
                if abs(total_events_score[i][0]) >= 1.5:
                    ax1.scatter(i, abs(total_events_score[i][0]), marker='d', color='#BD9424', s=6000)
                    ax1.text(i, abs(total_events_score[i][0]) - 0.02, "({})".format(i), fontsize=10, color="black",
                             ha='center', va='bottom', rotation=90)
                elif abs(total_events_score[i][0]) >= 1:
                    ax1.scatter(i, abs(total_events_score[i][0]), marker='d', color='#BD9424', s=4000)
                    ax1.text(i, abs(total_events_score[i][0]) - 0.02, "({})".format(i), fontsize=10, color="black",
                             ha='center', va='bottom', rotation=90)
                elif abs(total_events_score[i][0]) > 0.65:
                    ax1.scatter(i, abs(total_events_score[i][0]), marker='d', color='#BD9424', s=3000)
                    ax1.text(i, abs(total_events_score[i][0]) - 0.02, "({})".format(i), fontsize=10, color="black",
                             ha='center', va='bottom', rotation=90)
                else:
                    ax1.scatter(i, abs(total_events_score[i][0]), marker='d', color='#BD9424', s=1000)
                    ax1.text(i, abs(total_events_score[i][0]) - 0.02, "({})".format(i), fontsize=10, color="black",
                             ha='center', va='bottom', rotation=90)
        elif len(total_events[i]) > 1:
            for j in range(len(total_events[i])):
                if total_events[i][j] == "죽음":
                    if abs(total_events_score[i][j]) >= 1.5:
                        ax1.scatter(i, abs(total_events_score[i][j]), marker='d', color='#FE2E2E', s=6000)
                        ax1.text(i, abs(total_events_score[i][j]) - 0.02, "({})".format(i), fontsize=10.5,
                                 color="black", ha='center', va='bottom', rotation=90)
                    elif abs(total_events_score[i][j]) >= 1:
                        ax1.scatter(i, abs(total_events_score[i][j]), marker='d', color='#FE2E2E', s=4000)
                        ax1.text(i, abs(total_events_score[i][j]) - 0.02, "({})".format(i), fontsize=10.5,
                                 color="black", ha='center', va='bottom', rotation=90)
                    elif abs(total_events_score[i][j]) > 0.65:
                        ax1.scatter(i, abs(total_events_score[i][j]), marker='d', color='#FE2E2E', s=3000)
                        ax1.text(i, abs(total_events_score[i][j]) - 0.02, "({})".format(i), fontsize=10.5,
                                 color="black", ha='center', va='bottom', rotation=90)
                    else:
                        ax1.scatter(i, abs(total_events_score[i][j]), marker='d', color='#FE2E2E', s=1000)
                        ax1.text(i, abs(total_events_score[i][j]) - 0.02, "({})".format(i), fontsize=10.5,
                                 color="black", ha='center', va='bottom', rotation=90)
                if total_events[i][j] == "탄생":
                    if abs(total_events_score[i][j]) >= 1.5:
                        ax1.scatter(i, abs(total_events_score[i][j]), marker='d', color='#FF8000', s=6000)
                        ax1.text(i, abs(total_events_score[i][j]) - 0.02, "({})".format(i), fontsize=10, color="black",
                                 ha='center', va='bottom', rotation=90)
                    elif abs(total_events_score[i][j]) >= 1.0:
                        ax1.scatter(i, abs(total_events_score[i][j]), marker='d', color='#FF8000', s=4000)
                        ax1.text(i, abs(total_events_score[i][j]) - 0.02, "({})".format(i), fontsize=10, color="black",
                                 ha='center', va='bottom', rotation=90)
                    elif abs(total_events_score[i][j]) > 0.65:
                        ax1.scatter(i, abs(total_events_score[i][j]), marker='d', color='#FF8000', s=3000)
                        ax1.text(i, abs(total_events_score[i][j]) - 0.02, "({})".format(i), fontsize=10, color="black",
                                 ha='center', va='bottom', rotation=90)
                    else:
                        ax1.scatter(i, abs(total_events_score[i][j]), marker='d', color='#FF8000', s=1000)
                        ax1.text(i, abs(total_events_score[i][j]) - 0.02, "({})".format(i), fontsize=10, color="black",
                                 ha='center', va='bottom', rotation=90)
                if total_events[i][j] == "폭력":
                    if abs(total_events_score[i][j]) >= 1.5:
                        ax1.scatter(i, abs(total_events_score[i][j]), marker='d', color='#088A29', s=6000)
                        ax1.text(i, abs(total_events_score[i][j]) - 0.02, "({})".format(i), fontsize=10, color="black",
                                 ha='center', va='bottom', rotation=90)
                    elif abs(total_events_score[i][j]) >= 1.0:
                        ax1.scatter(i, abs(total_events_score[i][j]), marker='d', color='#088A29', s=4000)
                        ax1.text(i, abs(total_events_score[i][j]) - 0.02, "({})".format(i), fontsize=10, color="black",
                                 ha='center', va='bottom', rotation=90)
                    elif abs(total_events_score[i][j]) > 0.65:
                        ax1.scatter(i, abs(total_events_score[i][j]), marker='d', color='#088A29', s=3000)
                        ax1.text(i, abs(total_events_score[i][j]) - 0.02, "({})".format(i), fontsize=10, color="black",
                                 ha='center', va='bottom', rotation=90)
                    else:
                        ax1.scatter(i, abs(total_events_score[i][j]), marker='d', color='#088A29', s=1000)
                        ax1.text(i, abs(total_events_score[i][j]) - 0.02, "({})".format(i), fontsize=10, color="black",
                                 ha='center', va='bottom', rotation=90)
                if total_events[i][j] == "전쟁":
                    if abs(total_events_score[i][j]) >= 1.5:
                        ax1.scatter(i, abs(total_events_score[i][j]), marker='d', color='#FFFF00', s=6000)
                        ax1.text(i, abs(total_events_score[i][j]) - 0.02, "({})".format(i), fontsize=10, color="black",
                                 ha='center', va='bottom', rotation=90)
                    elif abs(total_events_score[i][j]) >= 1:
                        ax1.scatter(i, abs(total_events_score[i][j]), marker='d', color='#FFFF00', s=4000)
                        ax1.text(i, abs(total_events_score[i][j]) - 0.02, "({})".format(i), fontsize=10, color="black",
                                 ha='center', va='bottom', rotation=90)
                    elif abs(total_events_score[i][j]) > 0.65:
                        ax1.scatter(i, abs(total_events_score[i][j]), marker='d', color='#FFFF00', s=3000)
                        ax1.text(i, abs(total_events_score[i][j]) - 0.02, "({})".format(i), fontsize=10, color="black",
                                 ha='center', va='bottom', rotation=90)
                    else:
                        ax1.scatter(i, abs(total_events_score[i][j]), marker='d', color='#FFFF00', s=1000)
                        ax1.text(i, abs(total_events_score[i][j]) - 0.02, "({})".format(i), fontsize=10, color="black",
                                 ha='center', va='bottom', rotation=90)
                if total_events[i][j] == "이별":
                    if abs(total_events_score[i][j]) >= 1.5:
                        ax1.scatter(i, abs(total_events_score[i][j]), marker='d', color='#00FFFF', s=6000)
                        ax1.text(i, abs(total_events_score[i][j]) - 0.02, "({})".format(i), fontsize=10, color="black",
                                 ha='center', va='bottom', rotation=90)
                    elif abs(total_events_score[i][j]) >= 1:
                        ax1.scatter(i, abs(total_events_score[i][j]), marker='d', color='#00FFFF', s=4000)
                        ax1.text(i, abs(total_events_score[i][j]) - 0.02, "({})".format(i), fontsize=10, color="black",
                                 ha='center', va='bottom', rotation=90)
                    elif abs(total_events_score[i][j]) > 0.65:
                        ax1.scatter(i, abs(total_events_score[i][j]), marker='d', color='#00FFFF', s=3000)
                        ax1.text(i, abs(total_events_score[i][j]) - 0.02, "({})".format(i), fontsize=10, color="black",
                                 ha='center', va='bottom', rotation=90)
                    else:
                        ax1.scatter(i, abs(total_events_score[i][j]), marker='d', color='#00FFFF', s=1000)
                        ax1.text(i, abs(total_events_score[i][j]) - 0.02, "({})".format(i), fontsize=10, color="black",
                                 ha='center', va='bottom', rotation=90)
                if total_events[i][j] == "병":
                    if abs(total_events_score[i][j]) >= 1.5:
                        ax1.scatter(i, abs(total_events_score[i][j]), marker='d', color='#0040FF', s=6000)
                        ax1.text(i, abs(total_events_score[i][j]) - 0.02, "({})".format(i), fontsize=10.5,
                                 color="black", ha='center', va='bottom', rotation=90)
                    elif abs(total_events_score[i][j]) >= 1:
                        ax1.scatter(i, abs(total_events_score[i][j]), marker='d', color='#0040FF', s=4000)
                        ax1.text(i, abs(total_events_score[i][j]) - 0.02, "({})".format(i), fontsize=10.5,
                                 color="black", ha='center', va='bottom', rotation=90)
                    elif abs(total_events_score[i][j]) > 0.65:
                        ax1.scatter(i, abs(total_events_score[i][j]), marker='d', color='#0040FF', s=3000)
                        ax1.text(i, abs(total_events_score[i][j]) - 0.02, "({})".format(i), fontsize=10.5,
                                 color="black", ha='center', va='bottom', rotation=90)
                    else:
                        ax1.scatter(i, abs(total_events_score[i][j]), marker='d', color='#0040FF', s=1000)
                        ax1.text(i, abs(total_events_score[i][j]) - 0.02, "({})".format(i), fontsize=10.5,
                                 color="black", ha='center', va='bottom', rotation=90)
                if total_events[i][j] == "체포":
                    if abs(total_events_score[i][j]) >= 1.5:
                        ax1.scatter(i, abs(total_events_score[i][j]), marker='d', color='#8258FA', s=6000)
                        ax1.text(i, abs(total_events_score[i][j]) - 0.02, "({})".format(i), fontsize=10, color="black",
                                 ha='center', va='bottom', rotation=90)
                    elif abs(total_events_score[i][j]) >= 1.0:
                        ax1.scatter(i, abs(total_events_score[i][j]), marker='d', color='#8258FA', s=4000)
                        ax1.text(i, abs(total_events_score[i][j]) - 0.02, "({})".format(i), fontsize=10, color="black",
                                 ha='center', va='bottom', rotation=90)
                    elif abs(total_events_score[i][j]) > 0.65:
                        ax1.scatter(i, abs(total_events_score[i][j]), marker='d', color='#8258FA', s=3000)
                        ax1.text(i, abs(total_events_score[i][j]) - 0.02, "({})".format(i), fontsize=10, color="black",
                                 ha='center', va='bottom', rotation=90)
                    else:
                        ax1.scatter(i, abs(total_events_score[i][j]), marker='d', color='#8258FA', s=1000)
                        ax1.text(i, abs(total_events_score[i][j]) - 0.02, "({})".format(i), fontsize=10, color="black",
                                 ha='center', va='bottom', rotation=90)
                if total_events[i][j] == "독립":
                    if abs(total_events_score[i][j]) >= 1.5:
                        ax1.scatter(i, abs(total_events_score[i][j]), marker='d', color='#FE2EF7', s=6000)
                        ax1.text(i, abs(total_events_score[i][j]) - 0.02, "({})".format(i), fontsize=10, color="black",
                                 ha='center', va='bottom', rotation=90)
                    elif abs(total_events_score[i][j]) >= 1.0:
                        ax1.scatter(i, abs(total_events_score[i][j]), marker='d', color='#FE2EF7', s=4000)
                        ax1.text(i, abs(total_events_score[i][j]) - 0.02, "({})".format(i), fontsize=10, color="black",
                                 ha='center', va='bottom', rotation=90)
                    elif abs(total_events_score[i][j]) > 0.65:
                        ax1.scatter(i, abs(total_events_score[i][j]), marker='d', color='#FE2EF7', s=3000)
                        ax1.text(i, abs(total_events_score[i][j]) - 0.02, "({})".format(i), fontsize=10, color="black",
                                 ha='center', va='bottom', rotation=90)
                    else:
                        ax1.scatter(i, abs(total_events_score[i][j]), marker='d', color='#FE2EF7', s=1000)
                        ax1.text(i, abs(total_events_score[i][j]) - 0.02, "({})".format(i), fontsize=10, color="black",
                                 ha='center', va='bottom', rotation=90)
                if total_events[i][j] == "만남":
                    if abs(total_events_score[i][j]) >= 1.5:
                        ax1.scatter(i, abs(total_events_score[i][j]), marker='d', color='#E2DAB5', s=6000)
                        ax1.text(i, abs(total_events_score[i][j]) - 0.02, "({})".format(i), fontsize=10, color="black",
                                 ha='center', va='bottom', rotation=90)
                    elif abs(total_events_score[i][j]) >= 1:
                        ax1.scatter(i, abs(total_events_score[i][j]), marker='d', color='#E2DAB5', s=4000)
                        ax1.text(i, abs(total_events_score[i][j]) - 0.02, "({})".format(i), fontsize=10, color="black",
                                 ha='center', va='bottom', rotation=90)
                    elif abs(total_events_score[i][j]) > 0.65:
                        ax1.scatter(i, abs(total_events_score[i][j]), marker='d', color='#E2DAB5', s=3000)
                        ax1.text(i, abs(total_events_score[i][j]) - 0.02, "({})".format(i), fontsize=10, color="black",
                                 ha='center', va='bottom', rotation=90)
                    else:
                        ax1.scatter(i, abs(total_events_score[i][j]), marker='d', color='#E2DAB5', s=1000)
                        ax1.text(i, abs(total_events_score[i][j]) - 0.02, "({})".format(i), fontsize=10, color="black",
                                 ha='center', va='bottom', rotation=90)
                if total_events[i][j] == "이사":
                    if abs(total_events_score[i][j]) >= 1.5:
                        ax1.scatter(i, abs(total_events_score[i][j]), marker='d', color='#BD9424', s=6000)
                        ax1.text(i, abs(total_events_score[i][j]) - 0.02, "({})".format(i), fontsize=10, color="black",
                                 ha='center', va='bottom', rotation=90)
                    elif abs(total_events_score[i][j]) >= 1:
                        ax1.scatter(i, abs(total_events_score[i][j]), marker='d', color='#BD9424', s=4000)
                        ax1.text(i, abs(total_events_score[i][j]) - 0.02, "({})".format(i), fontsize=10, color="black",
                                 ha='center', va='bottom', rotation=90)
                    elif abs(total_events_score[i][j]) > 0.65:
                        ax1.scatter(i, abs(total_events_score[i][j]), marker='d', color='#BD9424', s=3000)
                        ax1.text(i, abs(total_events_score[i][j]) - 0.02, "({})".format(i), fontsize=10, color="black",
                                 ha='center', va='bottom', rotation=90)
                    else:
                        ax1.scatter(i, abs(total_events_score[i][j]), marker='d', color='#BD9424', s=1000)
                        ax1.text(i, abs(total_events_score[i][j]) - 0.02, "({})".format(i), fontsize=10, color="black",
                                 ha='center', va='bottom', rotation=90)

    custom_lines = [Line2D([0], [0], color='#FE2E2E', lw=4), Line2D([0], [0], color='#FF8000', lw=4),
                    Line2D([0], [0], color='#088A29', lw=4), Line2D([0], [0], color='#FFFF00', lw=4),
                    Line2D([0], [0], color='#00FFFF', lw=4), Line2D([0], [0], color='#0040FF', lw=4),
                    Line2D([0], [0], color='#8258FA', lw=4), Line2D([0], [0], color='#FE2EF7', lw=4),
                    Line2D([0], [0], color='#E2DAB5', lw=4), Line2D([0], [0], color='#BD9424', lw=4)]
    plt.legend(custom_lines, ["-- 죽음", "-- 탄생", "-- 폭력", "-- 전쟁", "--이별", "--병", "--체포", "--독립", "--만남", "--이사"],
               loc='lower left', bbox_to_anchor=(0, 1), ncol=4)
    plt.savefig(path + '/visualization/' + image_path + '_events_timeline_fig.png', dpi=300)
    # plt.show()
    plt.close(fig)

    plt.figure(3)
    ratio = [death_count, birth_count, violence_count, war_count, part_count, illness_count, arrest_count,
             independent_count, meet_count, move_count]
    labels = ['죽음', '탄생', '폭력', '전쟁', '이별', '병', '체포', '독립', '만남', '이사']
    final_ratio = []
    final_labels = []
    for i in range(len(ratio)):
        if ratio[i] != 0:
            final_ratio.append(ratio[i])
            final_labels.append(labels[i])
    colors = ['#FE2E2E', '#FF8000', '#088A29', '#FFFF00', '#00FFFF', '#0040FF', '#8258FA', '#FE2EF7', '#E2DAB5',
              '#BD9424']
    wedgeprops = {'width': 0.8, 'edgecolor': 'w', 'linewidth': 3}

    plt.title('사건 분포', fontsize=16)
    plt.pie(final_ratio, labels=final_labels, autopct='%.1f%%', startangle=260, counterclock=False, colors=colors,
            wedgeprops=wedgeprops, textprops={'fontsize': 12}, labeldistance=1.1)
    plt.tight_layout()
    plt.savefig(path + '/visualization/' + image_path + '_events_frequency.png', dpi=300)
    # plt.show()
    plt.close()

    plt.figure(4)

    for i in range(len(final_ratio)):
        final_ratio[i] += 9
    wordcloud_dict = dict(zip(final_labels, final_ratio))

    if death_count != 0:
        wordcloud_death = list(set(sum(death_word_list, [])))
        dic_death = dict()
        dic_death = make_dict(wordcloud_death)
        wordcloud_dict = merge_dic(dic_death, wordcloud_dict)
    if birth_count != 0:
        wordcloud_birth = list(set(sum(birth_word_list, [])))
        dic_birth = dict()
        dic_birth = make_dict(wordcloud_birth)
        wordcloud_dict = merge_dic(dic_birth, wordcloud_dict)
    if violence_count != 0:
        wordcloud_violence = list(set(sum(violence_word_list, [])))
        dic_violence = dict()
        dic_violence = make_dict(wordcloud_violence)
        wordcloud_dict = merge_dic(dic_violence, wordcloud_dict)
    if war_count != 0:
        wordcloud_war = list(set(sum(war_word_list, [])))
        dic_war = dict()
        dic_war = make_dict(wordcloud_war)
        wordcloud_dict = merge_dic(dic_war, wordcloud_dict)
    if part_count != 0:
        wordcloud_part = list(set(sum(part_word_list, [])))
        dic_part = dict()
        dic_part = make_dict(wordcloud_part)
        wordcloud_dict = merge_dic(dic_part, wordcloud_dict)
    if independent_count != 0:
        wordcloud_independent = list(set(sum(independent_word_list, [])))
        dic_independent = dict()
        dic_independent = make_dict(wordcloud_independent)
        wordcloud_dict = merge_dic(dic_independent, wordcloud_dict)
    if meet_count != 0:
        wordcloud_meet = list(set(sum(meet_word_list, [])))
        dic_meet = dict()
        dic_meet = make_dict(wordcloud_meet)
        wordcloud_dict = merge_dic(dic_meet, wordcloud_dict)
    if move_count != 0:
        wordcloud_move = list(set(sum(move_word_list, [])))
        dic_move = dict()
        dic_move = make_dict(wordcloud_move)
        wordcloud_dict = merge_dic(dic_move, wordcloud_dict)
    if illness_count != 0:
        wordcloud_illness = list(set(sum(illness_word_list, [])))
        dic_illness = dict()
        dic_illness = make_dict(wordcloud_illness)
        wordcloud_dict = merge_dic(dic_illness, wordcloud_dict)
    if arrest_count != 0:
        wordcloud_arrest = list(set(sum(arrest_word_list, [])))
        dic_arrest = dict()
        dic_arrest = make_dict(wordcloud_arrest)
        wordcloud_dict = merge_dic(dic_arrest, wordcloud_dict)

    # wordcloud error : 그림은 그려짐
    # wordcloud = WordCloud(background_color='white', colormap="Set1", width=2500, height=2000,
    #                       font_path=font_path).generate_from_frequencies(wordcloud_dict)
    # plt.axis('off')
    # plt.imshow(wordcloud)
    # # plt.show()
    # plt.savefig(path + '/visualization/' + image_path + '_events_wordcloud.png', dpi=300)


def merge_dic(x, y):
    z = x
    z.update(y)
    return z


def make_dict(x):
    counts = dict()
    for name in x:
        if name not in counts:
            counts[name] = 1
        else:
            counts[name] = counts[name] + 1
    return counts


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='6가지 문맥 정보 추출')
    # parser.add_argument('--input', '--input', required=True)
    # parser.add_argument('--output', '--output', required=True)
    # args = vars(parser.parse_args())
    # input = args["input"]
    # output = args["output"]  # output은 json 결과 파일
    # image_path = re.sub(".json", "", input)
    # input = '/book_list/' + input
    #
    # full_text = FullTextProcess(input)
    # event_detect(full_text, output, image_path)


    book_list = os.listdir('./book_list')

    for book_name in book_list[:1]:
        parser = argparse.ArgumentParser(description='6가지 문맥 정보 추출')
        parser.add_argument('--input', default=book_name)  # , required=True)
        parser.add_argument('--output', default=book_name)  # , required=True)
        args = vars(parser.parse_args())

        output = book_name  # output은 json 결과 파일
        image_path = re.sub(".json", "", book_name)
        input = '/book_list/' + book_name
        print(input)

        full_text = FullTextProcess(input)
        event_detect(full_text, output, image_path)



import pandas as pd
import numpy as np
import re
import csv
import os
import math
from gensim import models
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from scipy.stats import wasserstein_distance
#from hist_scatter import scatter_hist2d
import seaborn as sns
import sliced
import glob
import random
from itertools import cycle
import argparse
import json


def clean_str(text):
    pattern = "['\"!@#$%^&*()<>‘“”.]|\[UNK\]"  # 특수기호제거
    text = re.sub(pattern=pattern, repl='', string=text)
    return text


def mean_squared_error(xs, ys, xs_t, ys_t):
    total = 0

    for i in range(len(xs_t)):
        distance = 0
        for j in range(len(xs)):
            distance += np.sqrt((np.float_power((xs[j] - xs_t[i]), 2) + np.float_power((ys[j] - ys_t[i]),
                                                                                       2)))  # (int(freq[j]) * np.sqrt((np.float_power((xs[j] - xs_t[i]), 2) + np.float_power((ys[j] - ys_t[i]), 2))))
        total += (distance / len(xs))
    return total


def json_to_dic_loc(path):
    with open(path, "r", encoding="utf-8") as json_file:
        model_output_json = json.load(json_file)

    loc_freq_dict = {}  # { 'word': frequency, ... }
    tim_freq_dict = {}

    for line in model_output_json:
        if len(line['tags']) is not 0:
            if line['tags'][0]['type'] in ["LOC"]:
                cleaned_word = clean_str(line['tags'][0]['str'])  # 특수문자 제거하고
                if len(cleaned_word) == 1:  # 공간 정보가 한 글자인 경우 패스 (무의미하다고 판단)
                    continue
                if cleaned_word in loc_freq_dict:  # dictionary 안에 있으면 더하고
                    loc_freq_dict[cleaned_word] += 1
                else:
                    loc_freq_dict[cleaned_word] = 1
    # print(loc_freq_dict)
    return loc_freq_dict


def json_to_dic_tim(path):
    with open(path, "r", encoding="utf-8") as json_file:
        model_output_json = json.load(json_file)

    tim_freq_dict = {}

    for line in model_output_json:
        if len(line['tags']) is not 0:
            if line['tags'][0]['type'] in ['TIM']:  # 시간 정보
                cleaned_word = clean_str(line['tags'][0]['str'])
                # print("Group:", group_num, "Sentence_num:", result_data['sentence_num'][i], cleaned_word)
                if len(cleaned_word) == 1:  # 시간 정보가 한 글자인 경우 패스
                    continue
                if cleaned_word in tim_freq_dict:  # dictionary 안에 있으면 더하고
                    tim_freq_dict[cleaned_word] += 1
                else:
                    tim_freq_dict[cleaned_word] = 1

    # print(loc_freq_dict)
    return tim_freq_dict

def main(parser, file_dir, model_path, input_dir, matching_type):
    file_dir = glob.glob(os.path.join(file_dir, "*.json"))
    # input open
    loc_input_dic = {}
    all_loc_freq_dict = []
    all_tim_freq_dict = []
    if matching_type is "LOC":
        loc_input_dic = json_to_dic_loc(input_dir)
        for file_name in file_dir:
            all_loc_freq_dict.append(json_to_dic_loc(file_name))
    else:
        loc_input_dic = json_to_dic_tim(input_dir)
        for file_name in file_dir:
            all_loc_freq_dict.append(json_to_dic_tim(file_name))

        # print(tim_freq_dict)

    ###################################################################################3
    list_file_name = []
    words = []
    merge_words = []
    bookind = []
    for i, loc_freq_dict in enumerate(all_loc_freq_dict):  # 177
        if len(loc_freq_dict) is not 0:
            word = []
            for key, freq in loc_freq_dict.items():
                if key is not "" or key is not " ":
                    word.extend([key] * freq)
                    bookind.extend(["{}".format(i)] * freq)
            list_file_name.append(file_dir[i])
            words.append(word)
            merge_words.extend(word)

    input_words = []
    for key, freq in loc_input_dic.items():
        if key is not "" or key is not " ":
            input_words.extend([key] * freq)

    merge_words.extend(input_words)
    all_merge_words = merge_words
    pca = PCA(n_components=2)
    ko_model = models.fasttext.load_facebook_model(model_path)
    xys = pca.fit_transform([ko_model.wv.word_vec(w) for w in all_merge_words])

    xs = xys[:, 0]
    ys = xys[:, 1]

    ## maching first
    ## Sliced Wasserstain distance matrix
    top_5_index = {}
    current_num = 0
    for i, book_words in enumerate(words):  # 171
        # file_dir[i] is file_name
        # len(words) is current book words num
        current = sliced.sliced_wasserstein_distance(xys[current_num:current_num + len(book_words)],
                                                     xys[-len(input_words):], 10)
        current_num += len(book_words)
        if len(top_5_index) < 5:
            top_5_index[i] = current
            top_5_index = dict(sorted(top_5_index.items(), key=(lambda x: x[1]), reverse=True))
        else:
            for key, value in top_5_index.items():
                if current < value:
                    del top_5_index[key]
                    top_5_index[i] = current
                    top_5_index = dict(sorted(top_5_index.items(), key=(lambda x: x[1]), reverse=True))
                    break

    print("sliced_wasserstein_distance top5 book")
    for key, value in top_5_index.items():
        print(list_file_name[key].split('/')[-1].split('utf-8')[0], value)

    ##visualization later

    # normalization
    # xys = xys / np.linalg.norm(xys)
    top_5_xys = []
    top_5_bookind = []
    current_num = 0
    for i, book_words in enumerate(words):
        for key, value in top_5_index.items():
            if i is key:
                top_5_xys.extend(xys[current_num: current_num + len(book_words)].tolist())
                top_5_bookind.extend(bookind[current_num: current_num + len(book_words)])
        current_num += len(book_words)
    for i in range(len(top_5_bookind)):
        top_5_bookind[i] = file_dir[int(top_5_bookind[i])].split('/')[-1].split('utf-8')[0]
    top_5_bookind = np.array(top_5_bookind)
    top_5_xys = np.array(top_5_xys)
    plt.rc('axes', unicode_minus=False)
    plt.rc('font', family='NanumGothic')

    # jontplot : 일부 소설들만 돌아감(이상_날개 안됨)
    sns.jointplot(
        data=top_5_xys,
        x=top_5_xys[:, 0], y=top_5_xys[:, 1],
        hue=top_5_bookind,
        kind='kde'
    )
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_dir', default='./result', help="Directory containing config.json of feature_data")
    parser.add_argument('--model_dir', default='./cc.ko.300.bin',
                        help="Directory containing config.json of model")
    parser.add_argument('--input_file', default='./result/이상_날개utf-8.json', help='input file')
    parser.add_argument('--matching_type', default='TIM', help='maching type : LOC, TIM')
    args = parser.parse_args()
    file_dir = args.file_dir
    model_path = args.model_dir
    input_dir = args.input_file
    matching_type = args.matching_type
    main(parser, file_dir, model_path, input_dir, matching_type)

import re
import sys
import os

import numpy as np

path = os.path.dirname( os.path.abspath( __file__ ) )

sys.path.append(path+'/person_extraction')
from person_extraction import prediction

sys.path.append(path + '/kor_relation_extraction')
from kor_relation_extraction import relation_predict #관계추출

from collections import Counter
from konlpy.tag import *

def relation_feature_extraction(name, full_text):


    # 전체 문장에서 PER 2개있는 문장 인덱스 저장
    full_text = full_text.split("\n")
    two_person_sentence_idx = prediction.find_2person_ner(full_text, name)

    # 모든 문장에 대한 최종 relation feature shape
    fin_per_relation_feature = np.zeros((len(full_text), 768))

    if len(two_person_sentence_idx) == 0:
        return fin_per_relation_feature

    # 관계 추출하기
    # PER 2개있는 문장만 parsed_novel_ner.txt 기록
    # parsed_novel_ner.txt 각 문장을 relation 추출 모델에 넣어줌.
    # ouput : 768 bert output feature, 각 문장의 relation class index
    relation_feature, relation_class = relation_predict.extract_relation_output_feature()

    # 사람 간의 관계 class만 고려
    per_rel_class = {8: "father", 9: "mother", 12: "parent", 18: "opponent", 22: "relative",
                    31: "child", 32: "family", 42: "spouse"}

    per_rel_feature_idx=[]
    for position, class_idx in enumerate(relation_class):
        # 클래스가 사람간의 관계라면
        if int(class_idx) in per_rel_class:
            # 관계만 추출된 문장 중 사람간의 관계에 해당하는 문장의 위치 저장
            per_rel_feature_idx.append(position)

    for i in per_rel_feature_idx:
        # two_person_sentence_idx[i]  : 관계만 추출된 문장중 사람간의 관계에 해당하는 문장의 위치
        # fin_per_relation_feature[two_person_sentence_idx[i]] : 모든 문장 중 사람간 관계에 해당하는 문장 위치에
        # 관계만 추출된  feature 중 사람간 관계 해당하는 값 넣어주기
        fin_per_relation_feature[two_person_sentence_idx[i]] = relation_feature[i]

    return fin_per_relation_feature



def predict_per_rel(name, full_text):
    parsed_deleted_ner = list()
    final_rel = dict()

    # NER 찾기
    full_text = full_text.split("\n")
    file_ner, deleted_ner, f_pertag_parsed, f_pertag_original = prediction.export_main(full_text, name)

    # 책의 문장 중 사람이 2명일 경우 parsed_novel_ner.txt에 적어줌.
    f = open("parsed_novel_ner.txt", "r", encoding="utf-8")
    tmp = f.readlines()
    # 관계가 없으면 바로 return
    if len(tmp) < 1:
        os.remove(path + "/parsed_novel_ner.txt")
        with open(path + "/%s_rel_fin.txt" % name[:-5], mode="w", encoding='utf-8') as rel_fin:
            rel_fin.writelines("{}")

        return final_rel, parsed_deleted_ner, deleted_ner, f_pertag_parsed, f_pertag_original

    # 관계 추출하기
    # parsed_novel_ner.txt 각 문장을 relation 추출 모델에 넣어줌. ouput : 각 문장의 relation class index
    relations = relation_predict.export_predict()

    #사람간 관계 아닌 문장 다 지워버리기
    # 사람 간의 관계인것만 표시하기
    per_relation = {8: "father", 9: "mother", 12: "parent", 18: "opponent", 22: "relative",
                    31: "child", 32: "family", 42: "spouse"}

    rel = {}
    relations_fin = list()

    for index, lines in enumerate(relations):
        if int(lines) in per_relation:
            rel[index] = int(lines)
            relations_fin.append(lines)

    ## 파싱된 것에서 인물관계만
    keys = list(rel.keys())
    parsed_ner_fin = list()
    with open(path+"/parsed_novel_ner.txt", mode="r", encoding='utf-8') as parsed_ner:
        for index, lines in enumerate(parsed_ner):
            if index in keys:
                parsed_ner_fin.append(lines)
    parsed_ner.close()
    os.remove(path+"/parsed_novel_ner.txt")

    """
    여러번 나온 인물이 포함된 문장만 남기기
    """
    for index, lines in enumerate(parsed_ner_fin):
        tmp = 1
        for i in deleted_ner:
            word = i.replace("\n", "")
            word = word.replace(" ", "")
            if len(str(word)) < 2:
                continue
            if str(word) in str(lines):
                tmp = 0
                continue
        if tmp > 0:
            parsed_deleted_ner.append(lines)
        else:
            parsed_deleted_ner.append("")

    print(parsed_deleted_ner)
    """
    관계 정리하기
    """
    rel_dict = {}
    with open(path+"/%s_rel_fin.txt" % name[:-5], mode="w", encoding='utf-8') as rel_fin:
        pattern = re.compile('<.{0,7}:PER>')
        for index, lines in enumerate(parsed_deleted_ner):
            result = pattern.findall(lines)
            name = []
            if len(result) > 1:
                name.append(result[0][1:].split(':PER>')[0])
                name.append(result[1][1:].split(':PER>')[0])
                for i in range(len(name)):
                    han = Hannanum()
                    tmp = han.nouns(name[i])
                    if len(tmp) < 1:
                        continue
                    name[i] = tmp[0]

                if name[0] == name[1]:  # 이름이 같으면 패스
                    continue
                if (len(name[0]) < 2) or (len((name[1])) < 2):
                    continue
                name.sort()
                names = str(name[0]) + "_" + str(name[1])
                if names in rel_dict:
                    rel_list = rel_dict[names]
                    rel_list.append(per_relation[rel[keys[index]]])
                    rel_dict[names] = rel_list
                else:
                    rel_dict[names] = [per_relation[rel[keys[index]]]]


        for key, value in rel_dict.items():
            if len(value) > 0:
                cnt = Counter(value)
                mode = cnt.most_common(1)[0][0]
                final_rel[key] = mode

        rel_fin.writelines(str(final_rel))

        return final_rel, parsed_deleted_ner, deleted_ner, f_pertag_parsed, f_pertag_original

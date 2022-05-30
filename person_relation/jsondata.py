#!/usr/bin/env python
# coding: utf-8
import re
import json  # import json module
import sys
import argparse
import os

from collections import OrderedDict

# 관계추출
from konlpy.tag import *
from .predict import predict_per_rel
#path = os.path.dirname( os.path.abspath( __file__ ) )

def FullTextProcess(input_dir, book_name):
    # with statement
    with open(os.path.join(input_dir, book_name), encoding='utf-8') as json_file:
        json_data = json.load(json_file)

    # 본문
    try:
        title = json_data["title"]
    except:
        title = book_name.split('/')[1].split(".")[0]
    texts = json_data["texts"]

    # 문장별로 잘려서 들어가있는 본문
    splitted_text = list()

    split = False
    split_sentence = ""
    for i in range(len(texts)):
        page = int(texts[i]["page"])
        text = texts[i]["text"]
        # 공백인 경우 ex) 황순원 너와 나만의 시간
        if text in [' ']:
            continue
        
        if text[-2] not in ["?", ".", "!", "\n"]: # 문장이 끊겨 있으면
            split_n = True
            end = -1
        else :
            split_n = False
            end = 0
            
        text = re.sub("[^ ㄱ-ㅣ가-힣0-9a-zA-Z\.|\?|\!|\n]+", "", text)
        sents = re.split(r"[\?|\.|\!|\n]", text)
        
        if split :
            splitted_text.append(split_sentence + sents[0] + " : " + str(page-1) + ", " + str(page))
            start = 1
            split = False
        else : start = 0
        
        if split_n :
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
        if lines!=(len(splitted_text)-1):
            full_text = full_text + splitted_text[lines] + "\n"
        else:
            full_text = full_text + splitted_text[lines]

    return title, full_text



def make_output(pred_config):


    book_list = os.listdir(pred_config.input_dir)

    for book_name in book_list:

        title, full_text = FullTextProcess(pred_config.input_dir, book_name)
        rel_dict, _, deleted, parsed, original = predict_per_rel(book_name, full_text)  # person, relation 2개 다 predict

        rel_list = list() # [[['기애', '죠오'], 'spouse'], [['근수', '기애'], 'parent']]
        for key, value in rel_dict.items():
            tmp = list()
            perlist = key.split("_")
            tmp.append(perlist)
            tmp.append(value)
            rel_list.append(tmp)


        # 본문
        full_json = OrderedDict()
        full_json["title"] = title
        texts = list() # 페이지 별 json 태그 저장 위한 리스트

        for index, line in enumerate(parsed):
            flag = False
            sentence = original[index]
            pages = sentence.split(": ")[1].split(", ")

            for i, person in enumerate(rel_list):
                # 관계 존재하는 경우
                if (person[0][0] in sentence) and (person[0][1] in sentence):
                    # 문장이 페이지에 걸치는 경우
                    flag = True
                    rel_ppl = person[0]
                    relation = person[1]

            # 관계가 있는 경우
            if flag:
                # 페이지가 걸치면
                if len(pages) > 1:
                    for i in range(len(pages)):
                        result = OrderedDict()
                        result["page"] = int(pages[i])
                        result["text"] = str(sentence.split(" : ")[0])

                        # 인물 + 관계
                        taglist = list()

                        tmp = OrderedDict()
                        # 인물
                        tmp["str"] = rel_ppl[0] + "_" + rel_ppl[1]
                        # 태그 타입
                        tmp["type"] = "REL"
                        tmp["sub-type"] = relation

                        taglist.append(tmp)

                        result["tag"] = taglist
                        texts.append(result)

                else:
                    result = OrderedDict()
                    result["page"] = int(pages[0])
                    result["text"] = str(sentence.split(" : ")[0])

                    # 인물 + 관계
                    taglist = list()

                    tmp = OrderedDict()
                    # 인물
                    tmp["str"] = rel_ppl[0] + "_" + rel_ppl[1]
                    # 태그 타입
                    tmp["type"] = "REL"
                    tmp["sub-type"] = relation

                    taglist.append(tmp)

                    result["tag"] = taglist
                    texts.append(result)
            else:
                # 페이지가 걸치면
                if len(pages) > 1:
                    for i in range(len(pages)):
                        result = OrderedDict()
                        result["page"] = int(pages[i])
                        result["text"] = str(sentence.split(" : ")[0])
                        taglist = list()
                        result["tag"] = taglist
                        texts.append(result)

                else:
                    result = OrderedDict()
                    result["page"] = int(pages[0])
                    result["text"] = str(sentence.split(" : ")[0])
                    taglist = list()
                    result["tag"] = taglist
                    texts.append(result)

        full_json["texts"] = texts
        with open("rel_" + book_name, 'w', encoding='utf-8') as outfile:
            json.dump(full_json, outfile, ensure_ascii=False, indent="\t")

        # 본문
        full_json = OrderedDict()
        full_json["title"] = title
        texts = list()

        for index, line in enumerate(parsed):
            sentence = original[index]
            pages = sentence.split(": ")[1].split(", ")
            if line == "none":  # 태그가 존재하지 않는 경우
                # 페이지가 걸치면
                if len(pages) > 1:
                    for i in range(len(pages)):
                        result = OrderedDict()
                        result["page"] = int(pages[i])
                        result["text"] = str(sentence.split(" : ")[0])
                        taglist = list()
                        result["tag"] = taglist
                        texts.append(result)

                else:
                    result = OrderedDict()
                    result["page"] = int(pages[0])
                    result["text"] = str(sentence.split(" : ")[0])
                    taglist = list()
                    result["tag"] = taglist
                    texts.append(result)

            else:

                # 문장 인물 추출
                perlist = list()
                regex = re.compile("\w+:PER>")  # PER 태그 추출
                pertags = regex.findall(line)
                for pers in pertags:
                    pers = re.sub(":PER>", "", pers)
                    perlist.append(pers)

                # 문장 내 중복 없애기
                perlist = list(set(perlist))

                finper = list()
                for ner in perlist:
                    if ner not in deleted:
                        finper.append(ner)

                # 인물 존재 하는 경우
                if len(finper) > 0:
                    # 문장이 페이지에 걸치는 경우
                    if len(pages) > 1:
                        for i in range(len(pages)):
                            result = OrderedDict()
                            result["page"] = int(pages[i])
                            result["text"] = str(sentence.split(" : ")[0])

                            # 인물 + 관계
                            taglist = list()

                            for i in range(len(finper)):
                                tmp = OrderedDict()
                                # 인물
                                tmp["str"] = finper[i]
                                # 태그 타입
                                tmp["type"] = "PER"

                                taglist.append(tmp)

                            result["tag"] = taglist
                            texts.append(result)
                    else:
                        result = OrderedDict()
                        result["page"] = int(pages[0])
                        result["text"] = str(sentence.split(" : ")[0])
                        tmp = OrderedDict()

                        # 인물 + 관계
                        taglist = list()

                        for i in range(len(finper)):
                            tmp = OrderedDict()
                            tmp["str"] = finper[i]
                            tmp["type"] = "PER"
                            taglist.append(tmp)

                        result["tag"] = taglist
                        texts.append(result)

                else:  # 만약 인물들이 다 사라진 경우
                    pages = sentence.split(": ")[1].split(", ")
                    # 페이지가 걸치면
                    if len(pages) > 1:
                        for i in range(len(pages)):
                            result = OrderedDict()
                            result["page"] = int(pages[i])
                            result["text"] = str(sentence.split(" : ")[0])
                            taglist = list()
                            result["tag"] = taglist
                            texts.append(result)

                    else:
                        result = OrderedDict()
                        result["page"] = int(pages[0])
                        result["text"] = str(sentence.split(" : ")[0])
                        taglist = list()
                        result["tag"] = taglist
                        texts.append(result)

        full_json["texts"] = texts
        with open(os.path.join(pred_config.output_dir, "per_" + book_name), 'w', encoding='utf-8') as outfile:
            json.dump(full_json, outfile, ensure_ascii=False, indent="\t")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='6가지 문맥 정보 추출')
    # parser.add_argument('--input', typs=str, default="오헨리_크리스마스선물.json")  #required=True )
    # parser.add_argument('--output', default="오헨리_크리스마스선물.json") #required=True )
    #
    # args = vars(parser.parse_args())
    # input = args["input"]
    # output = args["output"]  # output은 json 결과 파일
    #
    # make_output(input, output)

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", default="../event/book_list", type=str, help="Path of input novels")
    parser.add_argument("--output_dir", default="./output_json", type=str, help="Path of results for novels")

    pred_config = parser.parse_args()
    make_output(pred_config)




#json
# merge KIE
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys

import numpy as np

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

import json
import pickle
import argparse
import torch
import re

from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert.utils import get_tokenizer
from model.net import KobertSequenceFeatureExtractor, KobertCRF
from gluonnlp.data import SentencepieceTokenizer
from data_utils.utils import Config
from data_utils.vocab_tokenizer import Tokenizer
from data_utils.pad_sequence import keras_pad_fn
from pathlib import Path

import nltk
import csv
import glob
import os

def line(data):
    sentences = []
    data = re.sub("[^ ㄱ-ㅣ가-힣0-9a-zA-Z\.|\?|\!|\n]+", "", data)
    sents = re.split(r"[\?|\.|\!|\n]", data)
    for i in range(len(sents)):
        if sents[i] == "f ":
            pass
        elif sents[i] == None:
            pass
        elif sents[i] == "\n":
            pass
        elif sents[i] == "":
            pass
        elif sents[i] == " \n" or sents[i] == "  \n" or sents[i] == " ":
            pass
        else:
            sentences.append(sents[i])

    return sentences

def FullTextProcess(data):
    texts = data["texts"]

    splitted_text = list()
    split = False
    split_n = False
    split_sentence = ""
    for i in range(len(texts)):
        page = int(texts[i]["page"])
        text = texts[i]["text"]

        if text is " ":
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

    return splitted_text

def ner_feature_extraction(file_name, evt_word):

    model_dir = Path(os.path.dirname(__file__)+'/experiments/bert_model')
    model_config = Config(json_path=model_dir / 'config.json')

    # Vocab & Tokenizer
    tok_path = os.path.dirname(__file__)+"/tokenizer_78b3253a26.model"
    ptr_tokenizer = SentencepieceTokenizer(tok_path)

    # load vocab & tokenizer
    with open(model_dir / "vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)

    tokenizer = Tokenizer(vocab=vocab,
                          split_fn=ptr_tokenizer, pad_fn=keras_pad_fn, maxlen=model_config.maxlen)

    # load ner_to_index.json
    with open(model_dir / "ner_to_index.json", 'rb') as f:
        ner_to_index = json.load(f)
        index_to_ner = {v: k for k, v in ner_to_index.items()}

    # Model
    model = KobertCRF(config=model_config, num_classes=len(ner_to_index), vocab=vocab)

    # load
    model_dict = model.state_dict()
    model_bin = os.path.dirname(__file__)+"/experiments/bert_model/best-epoch-16-step-1500-acc-0.993.bin"
    checkpoint = torch.load(model_bin, map_location=torch.device('cuda'))
    convert_keys = {}
    for k, v in checkpoint['model_state_dict'].items():
        new_key_name = k.replace("module.", '')
        if new_key_name not in model_dict:
            print("{} is not int model_dict".format(new_key_name))
            continue
        convert_keys[new_key_name] = v

    model.load_state_dict(convert_keys)
    model.eval()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = model.to(device)

    # 25개 NER class
    decoder_from_res = DecoderFromNamedEntitySequence(tokenizer=tokenizer, index_to_ner=index_to_ner)


    # file input
    with open(file_name, encoding='utf-8') as json_file:
        json_data = json.load(json_file)

    #print(file_name)

    sentences = FullTextProcess(json_data)
    print(len(sentences))


    ner_feature=[]

    # json output
    for j in range(len(sentences)):
        sentence_feature = []
        if sentences[j] and sentences[j].strip():  # spacebar, null x
            #print(sentences[j])

            list_of_input_ids = tokenizer.list_of_string_to_list_of_cls_sep_token_ids([sentences[j].split(":")[0]])
            x_input = torch.tensor(list_of_input_ids).long()
            x_input = x_input.to(device)
            list_of_pred_ids, last_encoder_layer = model(x_input, merge=True)



            # event feature --------------------------------------------------------------------------------------------
            input_token = tokenizer.decode_token_ids(list_of_input_ids)[0]
            evt = ''
            evt_idx = []
            word_list = evt_word[:, j]  # j번째 문장에 해당하는 event 리스트

            for i in word_list:
                # 문장에 event가 여러개 있을 경우, 최초 한개의 event 단어만 가져옴
                if len(i) != 0:
                    evt = i[0]
                    break

            # 문장에 event가 있을 경우
            if evt != '':
                for t in input_token:
                    org_t = t

                    # '▁' remove
                    if '▁' in t:
                        t=t.replace('▁','')

                    # bert의 input token에 event 단어가 있으면 인덱스 기록
                    if (evt in t) or (t in evt):
                        evt_idx.append(input_token.index(org_t))

                        # 한 문장에 해당 evt가 여러개 있을 경우 처음 단어 feature만 가져옴
                        if len(evt_idx) >= len(evt):
                            break

                mean_ner_feature=[]
                # 문장에 event가 있지만 토큰에 없을 경우 제외함 ex) 저지르 != 저지른
                if len(evt_idx) != 0:
                    for k in evt_idx:
                        # event 단어의 input token 위치에 해당하는 bert output 위치 feature를 가져옴
                        mean_ner_feature.append(last_encoder_layer[0][k].cpu().detach().numpy())

                    # 같은 tag  768 feature 평균내기
                    mean_ner_feature = np.array(mean_ner_feature)
                    mean_ner_feature = mean_ner_feature.mean(axis=0)
                    sentence_feature.append(mean_ner_feature)

                else:
                    # 없으면 768 크기의 0 벡터 만들어 주기
                    zero_vector = np.zeros(768)
                    sentence_feature.append(zero_vector)


            else:
                # 없으면 768 크기의 0 벡터 만들어 주기
                zero_vector = np.zeros(768)
                sentence_feature.append(zero_vector)



            # NER condition --------------------------------------------------------------------------------------------
            # per index : 13, 14
            # time index(TIM, DAT, DUR) : 15, 16 / 11, 12 / 23, 24
            # location index : 17, 18
            tmp = list_of_pred_ids[0]
            ner_index = [13, 13, 17, 11, 15, 23]  # B index
            time_flag = 0

            for l in ner_index:
                if time_flag != 1:
                    mean_ner_feature = []

                    try:
                        # ner 태그에 해당되는 값이 있으면, 동일한 feature vector 위치에서 추출
                        # B-ner tag
                        idx = tmp.index(l)

                        # 시간정보 DAT, TIM, DUR 중 1개만 사용. 시간에 해당되는 인덱스 처음 1개만 사용
                        if l in [11, 15, 23]:
                            time_flag = 1

                        mean_ner_feature.append(last_encoder_layer[0][idx].cpu().detach().numpy())
                        tmp[idx] = 0
                        idx += 1

                        # i-ner tag
                        # B tag 위치 다음으로 I tag가 있으면 추출
                        while tmp[idx] in [14, 18, 12, 16, 24]:
                            mean_ner_feature.append(last_encoder_layer[0][idx].cpu().detach().numpy())
                            tmp[idx] = 0
                            idx += 1

                            if idx==len(tmp):
                                break

                        # 같은 tag(B-i)에 해당되는 768 feature 평균내기
                        mean_ner_feature = np.array(mean_ner_feature)
                        mean_ner_feature = mean_ner_feature.mean(axis=0)
                        sentence_feature.append(mean_ner_feature)

                    except:
                        # 문맥정보 없으면 768 크기의 벡터 만들어 주기
                        # 시간 정보는 3개 중 모두 없을 때 768 크기 0 백터 만들어줌
                        if l not in [11, 15]:
                            zero_vector = np.zeros(768)
                            sentence_feature.append(zero_vector)





        sentence_feature = np.array(sentence_feature).flatten()
        ner_feature.append(sentence_feature)
    ner_feature=np.array(ner_feature)
    return ner_feature





def main(parser, file_dir, file_encoding):
    args = parser.parse_args()
    model_dir = Path(args.model_dir)
    model_config = Config(json_path=model_dir / 'config.json')

    # Vocab & Tokenizer
    tok_path = "./tokenizer_78b3253a26.model"
    ptr_tokenizer = SentencepieceTokenizer(tok_path)

    # load vocab & tokenizer
    with open(model_dir / "vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)

    tokenizer = Tokenizer(vocab=vocab,
                          split_fn=ptr_tokenizer, pad_fn=keras_pad_fn, maxlen=model_config.maxlen)

    # load ner_to_index.json
    with open(model_dir / "ner_to_index.json", 'rb') as f:
        ner_to_index = json.load(f)
        index_to_ner = {v: k for k, v in ner_to_index.items()}

    # Model
    model = KobertCRF(config=model_config, num_classes=len(ner_to_index), vocab=vocab)

    # load
    model_dict = model.state_dict()
    model_bin = "./experiments/bert_model/best-epoch-16-step-1500-acc-0.993.bin"
    checkpoint = torch.load(model_bin, map_location=torch.device('cuda'))
    convert_keys = {}
    for k, v in checkpoint['model_state_dict'].items():
        new_key_name = k.replace("module.", '')
        if new_key_name not in model_dict:
            print("{} is not int model_dict".format(new_key_name))
            continue
        convert_keys[new_key_name] = v

    model.load_state_dict(convert_keys)
    model.eval()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = model.to(device)

    # 25개 NER class
    decoder_from_res = DecoderFromNamedEntitySequence(tokenizer=tokenizer, index_to_ner=index_to_ner)

    file_dir = glob.glob(os.path.join(file_dir,"*.json"))
    for file_name in file_dir:
        # file input
        with open(file_name, encoding='utf-8') as json_file:
            json_data = json.load(json_file)


        #file = open(file_name, 'r', encoding=file_encoding)
        # split text file into sentence
        #feature_data = file.read()
        # print(feature_data)
        #sentences = nltk.tokenize.sent_tokenize(feature_data)
        print(file_name)
        sentences = FullTextProcess(json_data)

        # sentences = koreanTokenizer(feature_data)
        # print(sentences)
        print(len(sentences))
        #file.close()

        # json output
        json_data = []
        pagesnum = 0
        for j in range(len(sentences)):
            # print(line)
            page = int(sentences[j].split(" ")[-1])
            if sentences[j] and sentences[j].strip():  # spacebar, null x
                print(sentences[j])
                # json feature_data
                if ',' in sentences[j].split(":")[-1]:
                    json_data.append({
                        "page": int(sentences[j].split(":")[-1].split(",")[0]),  # temporarily page num, should change to page_num later
                        "text": sentences[j].split(":")[0],
                        "tags": []
                    })
                    page = int(sentences[j].split(",")[-1])
                    list_of_input_ids = tokenizer.list_of_string_to_list_of_cls_sep_token_ids(
                        [sentences[j].split(":")[0]])

                    x_input = torch.tensor(list_of_input_ids).long()
                    x_input = x_input.to(device)
                    list_of_pred_ids = model(x_input)

                    list_of_ner_word, decoding_ner_sentence = decoder_from_res(list_of_input_ids=list_of_input_ids,
                                                                               list_of_pred_ids=list_of_pred_ids)
                    print(j)
                    # if len(list_of_ner_word) > 0:
                    for ner_word in list_of_ner_word:
                        json_data[pagesnum]["tags"].append({"str": ner_word['word'], "type": ner_word['tag'], 'sub-type': ''})
                    print("list_of_ner_word:", list_of_ner_word)
                    print("Output: ", decoding_ner_sentence)
                    print("-" * 100)

                    pagesnum += 1

                json_data.append({
                    "page": page,  # temporarily page num, should change to page_num later
                    "text": sentences[j].split(":")[0],
                    "tags": []
                })
                list_of_input_ids = tokenizer.list_of_string_to_list_of_cls_sep_token_ids([sentences[j].split(":")[0]])

                x_input = torch.tensor(list_of_input_ids).long()
                x_input = x_input.to(device)
                list_of_pred_ids = model(x_input)

                list_of_ner_word, decoding_ner_sentence = decoder_from_res(list_of_input_ids=list_of_input_ids,
                                                                           list_of_pred_ids=list_of_pred_ids)
                # if len(list_of_ner_word) > 0:
                for ner_word in list_of_ner_word:
                    json_data[pagesnum]["tags"].append({"str": ner_word['word'], "type": ner_word['tag'], 'sub-type': ''})
                print("list_of_ner_word:", list_of_ner_word)
                print("Output: ", decoding_ner_sentence)
                print("-" * 100)

                pagesnum +=1
        file_name = file_name.split('/')[2].split(".")[0]
        with open('result/' + file_name + file_encoding + '.json', 'w', encoding=file_encoding) as make_file:
            json.dump(json_data, make_file, ensure_ascii=False, indent="\t")


class DecoderFromNamedEntitySequence():
    def __init__(self, tokenizer, index_to_ner):
        self.tokenizer = tokenizer
        self.index_to_ner = index_to_ner

    def __call__(self, list_of_input_ids, list_of_pred_ids):
        input_token = self.tokenizer.decode_token_ids(list_of_input_ids)[0]
        pred_ner_tag = [self.index_to_ner[pred_id] for pred_id in list_of_pred_ids[0]]

        # print("len: {}, input_token:{}".format(len(input_token), input_token))
        # print("len: {}, pred_ner_tag:{}".format(len(pred_ner_tag), pred_ner_tag))

        # Parsing list_of_ner_word
        list_of_ner_word = []
        entity_word, entity_tag, prev_entity_tag = "", "", ""
        for i, pred_ner_tag_str in enumerate(pred_ner_tag):
            if "B-" in pred_ner_tag_str:
                entity_tag = pred_ner_tag_str[-3:]
                # if prev_entity_tag != "" and (prev_entity_tag in["TIM","DAT","LOC","DUR"]) and entity_word !="▁":
                if prev_entity_tag != "" and (prev_entity_tag in ["TIM","DAT","LOC","DUR"]) and entity_word != "▁":
                    if prev_entity_tag in ["LOC"]:
                        list_of_ner_word.append({"word": entity_word.replace("▁", ""), "tag": prev_entity_tag})
                    else:
                        list_of_ner_word.append({"word": entity_word.replace("▁", ""), "tag": "TIM"})

                entity_word = input_token[i]
                # print(entity_word)
                # print(input_token[i+1])
                prev_entity_tag = entity_tag
            elif "I-" + entity_tag in pred_ner_tag_str:
                entity_word += input_token[i]
                # print(entity_word)
            else:
                # if entity_word != "" and (entity_tag in["TIM","DAT","LOC","DUR"]) and entity_word !="▁":
                if entity_word != "" and (entity_tag in ["TIM","DAT","LOC","DUR"]) and entity_word != "▁":
                    if prev_entity_tag in ["LOC"]:
                        list_of_ner_word.append({"word": entity_word.replace("▁", ""), "tag": entity_tag})
                    else:
                        list_of_ner_word.append({"word": entity_word.replace("▁", ""), "tag": "TIM"})
                entity_word, entity_tag, prev_entity_tag = "", "", ""

        # parsing decoding_ner_sentence
        decoding_ner_sentence = ""
        is_prev_entity = False
        prev_entity_tag = ""
        is_there_B_before_I = False

        for i, (token_str, pred_ner_tag_str) in enumerate(zip(input_token, pred_ner_tag)):
            if i == 0 or i == len(pred_ner_tag) - 1:
                continue

            # print(pred_ner_tag)
            token_str = token_str.replace(chr(9601), ' ')
            token_str = token_str.replace('_', ' ')

            if 'B-' in pred_ner_tag_str:
                if is_prev_entity is True:
                    decoding_ner_sentence += ':' + prev_entity_tag + '>'

                if token_str[0] == ' ':
                    token_str = list(token_str)
                    token_str[0] = ' <'
                    token_str = ''.join(token_str)
                    decoding_ner_sentence += token_str
                else:
                    decoding_ner_sentence += '<' + token_str
                is_prev_entity = True
                prev_entity_tag = pred_ner_tag_str[-3:]
                is_there_B_before_I = True

            elif 'I-' in pred_ner_tag_str:
                decoding_ner_sentence += token_str

                if is_there_B_before_I is True:
                    is_prev_entity = True
            else:
                if is_prev_entity is True:
                    decoding_ner_sentence += ':' + prev_entity_tag + '>' + token_str
                    is_prev_entity = False
                    is_there_B_before_I = False
                else:
                    decoding_ner_sentence += token_str

        return list_of_ner_word, decoding_ner_sentence


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='./experiments/bert_model',
                        help="Directory containing config.json of model")
    parser.add_argument('--file_dir',  default='./book-json',required=False, help='text file dir')
    parser.add_argument('--e', default='utf-8', required=False, help='text file encoding')
    args = parser.parse_args()
    file_dir = args.file_dir
    file_encoding = args.e
    main(parser, file_dir, file_encoding)  # original



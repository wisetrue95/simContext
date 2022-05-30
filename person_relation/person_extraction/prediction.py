from __future__ import absolute_import, division, print_function, unicode_literals

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import os
import json
import pickle
import argparse
import torch
from konlpy.tag import *
from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert.utils import get_tokenizer
from model.net import KobertSequenceFeatureExtractor, KobertCRF
from gluonnlp.data import SentencepieceTokenizer
from data_utils.utils import Config
from data_utils.vocab_tokenizer import Tokenizer
from data_utils.pad_sequence import keras_pad_fn
from pathlib import Path
import operator

# for feature merge
def find_2person_ner(full_text, name):

    path = os.path.dirname(os.path.abspath(__file__))
    model_dir = Path(path+'/experiments/bert_model')
    model_config = Config(json_path=model_dir / 'config.json')

    # Vocab & Tokenizer
    tok_path = path + "/tokenizer_78b3253a26.model"
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
    model_bin = path + "/experiments/bert_model/best-epoch-16-step-1500-acc-0.993.bin"
    checkpoint = torch.load(model_bin, map_location=torch.device('cpu'))
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
    model.to(device)

    decoder_from_res = DecoderFromNamedEntitySequence(tokenizer=tokenizer, index_to_ner=index_to_ner)

    # 전체 문장에서 PER 2개있는 문장 인덱스 저장
    two_person_sentence_idx=[]
    # path:  context/parsed_novel_ner.txt
    with open("parsed_novel_ner.txt", mode="w", encoding='utf-8') as parsed_novel_ner:
        for index, lines in enumerate(full_text):
            input_text = str(lines)

            #인물추출하는 부분
            list_of_input_ids = tokenizer.list_of_string_to_list_of_cls_sep_token_ids([input_text])
            x_input = torch.tensor(list_of_input_ids).long().to(device)  # to(device) 추가함
            list_of_pred_ids = model(x_input)
            list_of_ner_word, decoding_ner_sentence = decoder_from_res(list_of_input_ids=list_of_input_ids, list_of_pred_ids=list_of_pred_ids)

            # for relation
            # 문장 중 사람이 2명일 경우 parsed_novel_ner.txt에 적어줌.
            if decoding_ner_sentence.count("PER") == 2:
                # 전체 문장에서 PER 2개있는 문장 인덱스 저장
                two_person_sentence_idx.append(index)
                if len(list_of_ner_word) != 0:
                    parsed_novel_ner.writelines(decoding_ner_sentence + "\n")

    return two_person_sentence_idx



def main(parser, full_text, name):
    path = os.path.dirname(os.path.abspath(__file__))
    args = parser.parse_args()
    model_dir = Path(args.model_dir)
    model_config = Config(json_path=model_dir / 'config.json')
    export = True
    file_name = name[:-5]

    # Vocab & Tokenizer
    if export:
        tok_path = path + "/tokenizer_78b3253a26.model"
    else:
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

    if export:
        model_bin = path + "/experiments/bert_model/best-epoch-16-step-1500-acc-0.993.bin"
    else:
        model_bin = "./experiments/bert_model/best-epoch-16-step-1500-acc-0.993.bin"

    checkpoint = torch.load(model_bin, map_location=torch.device('cpu'))
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

    model.to(device)

    decoder_from_res = DecoderFromNamedEntitySequence(tokenizer=tokenizer, index_to_ner=index_to_ner)

    count = {} # 등장인물이 몇번 등장하는지 센 딕셔너리
    f_pertag_parsed = list()
    f_pertag_original = list()
    novel_ner = list() # PER 태그가 존재하는 문장 원본

    if export:
        with open("parsed_novel_ner.txt", mode="w", encoding='utf-8') as parsed_novel_ner:
            ppl_ner = list() # PER 엔티티 리스트
            for index, lines in enumerate(full_text):
                input_text = str(lines)

                #인물추출하는 부분
                list_of_input_ids = tokenizer.list_of_string_to_list_of_cls_sep_token_ids([input_text])
                x_input = torch.tensor(list_of_input_ids).long().to(device)  # to(device) 추가함
                list_of_pred_ids = model(x_input)
                list_of_ner_word, decoding_ner_sentence = decoder_from_res(list_of_input_ids=list_of_input_ids, list_of_pred_ids=list_of_pred_ids)

                for i in range(len(list_of_ner_word)):
                    dict = list_of_ner_word[i]
                    if dict['tag'] == 'PER':
                        # 조사 제거
                        name = dict['word'][1:]
                        name = name.replace("_", "")
                        name = name.replace(" ", "")

                        han = Hannanum()
                        tmp = han.nouns(name)
                        if len(tmp) > 0:
                            # 나타난 빈도수 계산
                            if tmp[0] in ppl_ner:
                                val = count[str(tmp[0])]
                                val += 1
                                count[str(tmp[0])] = val
                            else:
                                count[str(tmp[0])] = 1
                                ppl_ner.append(tmp[0])

                        else:
                            # 나타난 빈도수 계산
                            if name in ppl_ner:
                                val = count[str(name)]
                                val += 1
                                count[str(name)] = val
                            else:
                                count[str(name)] = 1
                                ppl_ner.append(name)

                # for person
                if decoding_ner_sentence.count("PER") >= 1:
                    f_pertag_parsed.append(decoding_ner_sentence)
                    f_pertag_original.append(input_text)
                else:
                    f_pertag_parsed.append("none")
                    f_pertag_original.append(input_text)

                # for relation
                # 문장 중 사람이 2명일 경우 parsed_novel_ner.txt에 적어줌.
                if decoding_ner_sentence.count("PER") == 2:
                    if len(list_of_ner_word) != 0:
                        parsed_novel_ner.writelines(decoding_ner_sentence + "\n")



        file_ner = list()
        deleted_ner = list()

        # ner.txt 작성
        for i in ppl_ner:
            if len(str(i)) > 1:
                if count[str(i)] > 3:
                    file_ner.append(str(i))
                else:

                    deleted_ner.append(str(i))

        if len(file_ner) < 1:
            for i in ppl_ner:
                if count[str(i)] > 3:
                    file_ner.append(str(i))
                else:
                    deleted_ner.append(str(i))

    with open("%s_file_ner.txt" % file_name, mode="w", encoding='utf-8') as f:
        for i in file_ner:
            f.writelines(i+"\n")



    return file_ner, deleted_ner, f_pertag_parsed, f_pertag_original




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

                if prev_entity_tag != entity_tag and prev_entity_tag != "":
                    list_of_ner_word.append(
                        {"word": entity_word.replace("_", " "), "tag": prev_entity_tag, "prob": None})

                entity_word = input_token[i]
                prev_entity_tag = entity_tag
            elif "I-" + entity_tag in pred_ner_tag_str:
                entity_word += input_token[i]
            else:
                if entity_word != "" and entity_tag != "":
                    list_of_ner_word.append({"word": entity_word.replace("_", " "), "tag": entity_tag, "prob": None})
                entity_word, entity_tag, prev_entity_tag = "", "", ""

        # parsing decoding_ner_sentence
        decoding_ner_sentence = ""
        is_prev_entity = False
        prev_entity_tag = ""
        is_there_B_before_I = False

        for i, (token_str, pred_ner_tag_str) in enumerate(zip(input_token, pred_ner_tag)):
            if i == 0 or i == len(pred_ner_tag) - 1:
                continue

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

def export_main(full_text, name):
    parser = argparse.ArgumentParser()
    #parser.add_argument('--input', '--input', required=True)
    #parser.add_argument('--output', '--output', required=True)
    parser.add_argument('--data_dir', default='./person_extraction/data_in',
                        help="Directory containing config.json of feature_data")
    parser.add_argument('--model_dir', default='./person_extraction/experiments/bert_model',
                        help="Directory containing config.json of model")
    file_ner, deleted_ner, f_pertag_parsed, f_pertag_original = main(parser, full_text, name)

    return file_ner, deleted_ner, f_pertag_parsed, f_pertag_original



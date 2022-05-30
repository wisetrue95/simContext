#!/usr/bin/env python
#-*- coding: utf-8 -*-
import os
import logging
import argparse
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AutoModelForSequenceClassification

from utils import init_logger, load_tokenizer

import json
import re

logger = logging.getLogger(__name__)

def get_device(pred_config):
    return "cuda" if torch.cuda.is_available() and not pred_config.no_cuda else "cpu"


def get_args(pred_config):
    return torch.load(os.path.join(pred_config.model_dir, 'training_args.bin'))


def load_model(pred_config, args, device):
    # Check whether model exists
    if not os.path.exists(pred_config.model_dir):
        raise Exception("Model doesn't exists! Train first!")

    try:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)  # Config will be automatically loaded from model_dir
        model.to(device)
        model.eval()
        logger.info("***** Model Loaded *****")
    except:
        raise Exception("Some model files might be missing...")

    return model

def FullTextProcess(data):
### [json input -> list of {line:page_num}] code from KIE Lab
    texts = data["texts"]

    splitted_text = list()
    split = False
    split_n = False
    split_sentence = ""
    for i in range(len(texts)):
        page = int(texts[i]["page"])
        text = texts[i]["text"]

        if text in [' ']:
            continue

        if text[-2] not in ["?", ".", "!", "\n"]:
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
    
def convert_input_file_to_tensor_dataset(pred_config,
                                         args,
                                         original_file,
                                         cls_token_segment_id=0,
                                         pad_token_segment_id=0,
                                         sequence_a_segment_id=0,
                                         mask_padding_with_zero=True):
    tokenizer = load_tokenizer(args)

    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token_id = tokenizer.pad_token_id

    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []

    all_sentences = []
    all_pages = []
    with open(original_file, "r", encoding="utf-8") as input_json_file:
        json_data = json.load(input_json_file)
        sentences = FullTextProcess(json_data)

        for j in range(len(sentences)):
            # for line in sentences:
            line = sentences[j].split(":")[0]
            pages = sentences[j].split(":")[-1] # it can be " 13" or " 13, 14"
            if sentences[j] and sentences[j].strip(): # spacebar, null x
                if ',' in pages: # if it has two page nums
                    temp = int(pages.split(",")[0].split(" ")[1])
                    all_pages.append(temp)
                    all_sentences.append(line)
                all_pages.append(int(pages.split(" ")[-1]))
                all_sentences.append(line)
                line = line.strip()
                tokens = tokenizer.tokenize(line)
                # Account for [CLS] and [SEP]
                special_tokens_count = 2
                if len(tokens) > args.max_seq_len - special_tokens_count:
                    tokens = tokens[:(args.max_seq_len - special_tokens_count)]

                # Add [SEP] token
                tokens += [sep_token]
                token_type_ids = [sequence_a_segment_id] * len(tokens)

                # Add [CLS] token
                tokens = [cls_token] + tokens
                token_type_ids = [cls_token_segment_id] + token_type_ids

                input_ids = tokenizer.convert_tokens_to_ids(tokens)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
                attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

                # Zero-pad up to the sequence length.
                padding_length = args.max_seq_len - len(input_ids)
                input_ids = input_ids + ([pad_token_id] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

                all_input_ids.append(input_ids)
                all_attention_mask.append(attention_mask)
                all_token_type_ids.append(token_type_ids)

        # Change to Tensor
        all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
        all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)

        return dataset, all_sentences, all_pages

def predict(pred_config):
    # load model and args
    args = get_args(pred_config)
    device = get_device(pred_config)
    model = load_model(pred_config, args, device)
    logger.info(args)

    original_file = pred_config.input_file

    # Convert input file to TensorDataset
    dataset, all_sentences, all_pages = convert_input_file_to_tensor_dataset(pred_config, args, original_file)

    # Predict
    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=pred_config.batch_size)

    preds = None

    for batch in tqdm(data_loader, desc="Predicting"):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": None}
            if args.model_type != "distilkobert":
                inputs["token_type_ids"] = batch[2]
            outputs = model(**inputs)
            logits = outputs[0]

            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

    json_data = []
    if preds is not None:
        preds = np.argmax(preds, axis=1)
        j = 0
        for pred in preds:
            if pred == 0:
                pred = 'happy'
            elif pred == 1:
                pred = 'sad'
            elif pred == 2:
                pred = 'angry'
            elif pred == 3:
                pred = 'others'

            json_data.append({"page": all_pages[j],
                              "text": all_sentences[j],
                              "tags": []})
            json_data[j]["tags"].append({"str": all_sentences[j],
                                         "type": "EMT",
                                         "sub-type": pred})
            j += 1
    else:
         json_data.append({"page": all_pages,
                           "text": all_sentences,
                           "tags": []})
         json_data[0]["tags"].append({"str": all_sentences,
                                         "type": None,
                                         "sub-type": None})
        

    # json output
    only_file_name, file_extension = os.path.splitext(original_file)
    output_file_path = only_file_name + '_out.json'

    with open(output_file_path, 'w', encoding='utf-8') as jason_output_file:
        json.dump(json_data, jason_output_file, ensure_ascii=False, indent="\t")

    logger.info("Prediction Done for {}".format(original_file))


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", type=str, help="Input file for prediction")
    parser.add_argument("--model_dir", default="./model", type=str, help="Path to save, load model")

    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for prediction")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    pred_config = parser.parse_args()
    predict(pred_config)
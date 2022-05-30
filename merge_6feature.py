import os
import glob
import argparse
import numpy as np

from time_location.prediction_v6 import ner_feature_extraction
from emotion.extraction.predict import emotion_feature_extraction
from person_relation.jsondata import FullTextProcess
from person_relation.predict import relation_feature_extraction
from event.event_json import find_event_idx


def makedirs(path):
   try:
        os.makedirs(path)
   except OSError:
       if not os.path.isdir(path):
           raise


def main(book_dir, feature_dir):

    book_dir_list = glob.glob(os.path.join(book_dir,"*.json"))
    makedirs(feature_dir)

    # 책 한권 당
    for book in book_dir_list:

        # create book feature folder
        book_name = book.split('/')[2].split(".")[0]
        feature_name = os.path.join(feature_dir, book_name)

        print('\n##########################################################')
        print(book_name)

        # preprocessing
        _, full_text = FullTextProcess(os.path.dirname(__file__), book)

        # event
        # shape ( 10 : search event topic, 문장개수)
        evt_word = find_event_idx(full_text)

        # ner_feature_extraction : person, time, location
        ner_feature = ner_feature_extraction(book, evt_word)
        print('ner feature extraction done')

        # emotion
        emotion_feature = emotion_feature_extraction(book)
        print('emotion feature extraction done')

        # relation
        relation_feature = relation_feature_extraction(book, full_text)
        print('relation feature extraction done')

        # 문장 당 각 6종 feature 합치기
        # per, per, loc, tim, evt, emt, rel
        merged_6feature = np.hstack((ner_feature, emotion_feature, relation_feature))
        means = np.nanmean(merged_6feature, axis=0)
        print(means.shape)  # 5376

        # save
        # np.save(feature_name, means)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--book_dir', default='book_dataset/book-json', required=False, help='book dir')
    parser.add_argument('--feature_dir', default='book_dataset/book-feature', required=False, help='feature dir')
    args = parser.parse_args()
    book_dir = args.book_dir
    feature_dir = args.feature_dir
    main(book_dir, feature_dir)


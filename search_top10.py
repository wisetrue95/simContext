import os
import glob
import argparse
import torch
import torch.nn as nn
import numpy as np


def cos_similiarity(v1, v2):
    dot_product=np.dot(v1, v2)
    l2_norm = (np.sqrt(sum(np.square(v1))))*np.sqrt(sum(np.square(v2)))
    similarity = dot_product/l2_norm

    return similarity


# top10 검색 함수
def cosine_top10_query(query, book_feature_dir):

    # 모든 책 feature tensor 로딩
    all_book_list = glob.glob(os.path.join(book_feature_dir,"*.npy"))
    all_book_feature={}
    for i in all_book_list:
        book_number = i.split('/')[-1].replace('.npy', '')
        # book_number : [featrue, title]
        all_book_feature[book_number]=[torch.Tensor(np.load(i)).cuda()]

    # similarity
    cos = nn.CosineSimilarity(dim=0).cuda()

    score_list=[]
    for target in all_book_feature:
        score = cos(all_book_feature[query][0], all_book_feature[target][0])
        score_list.append([score, target])

    score_list.sort(reverse=True)

    # top10 결과 출력
    for result in score_list[:11]:   # 11
        print(result[1], result[0].item())   # id title score


def main(query, book_feature_dir):

    # 검색 함수
    # query = '알퐁스도데_마지막수업'
    cosine_top10_query(query, book_feature_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--book_feature_dir', default='book_dataset/book-feature')
    parser.add_argument('--query', default='알퐁스도데_마지막수업')
    args = parser.parse_args()
    main(args.query, args.book_feature_dir)








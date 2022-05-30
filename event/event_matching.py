#-*- coding: utf-8 -*-
import re
import json # import json module
import sys
import argparse
import os
import numpy as np
import math
from numpy import dot
from numpy.linalg import norm
from unicodedata import normalize
# 사건 추출

path = os.path.dirname(os.path.abspath(__file__))


def find_similar_book_cosine(input_path,book_name):
    
    f=open(path+input_path,'r')
    lines=f.readlines()
    f.close()
    
    find_path = path+'/matching/'
    total_books = os.listdir(find_path)
    #한글 자모 분리됨
    for filename in total_books:
        before_filename = os.path.join(find_path, filename)
        after_filename = normalize('NFC', before_filename)
        os.rename(before_filename, after_filename)

    find_book=book_name+'.txt'
    total_books.remove(find_book)
    try:
        total_books.remove('.DS_Store')
    except ValueError:
        pass
    find_list=[]
    find_de=0
    for i in range(len(lines)):
        find_events=lines[i].split(":")
        find_events[1]=int(find_events[1])
        find_list.append(find_events[1])
    
    find_list = np.array(find_list)
    
    total_list=[]
    cosine_list=[]
    for i in range(len(total_books)):
        t=open(path+'/matching/'+total_books[i],'r')
        tmp_lines=t.readlines()
        t.close()
        tmp_list=[]
        for j in range(len(tmp_lines)):
            events=tmp_lines[j].split(":")
            events[1]=int(events[1])
            tmp_list.append(events[1])
        tmp_list=np.array(tmp_list)
        cosine_list.append(cos_sim(find_list,tmp_list))
    
    final_list=[]
    a=[]
    for i in range(len(cosine_list)):
        if math.isnan(float(cosine_list[i]))==True:
            a.append(cosine_list[i])
    
    if math.isnan(float(max(cosine_list)))==False:
        max_value=max(cosine_list)
    else:
        max_value=0
        print("\n"+book_name+"(와)과 가장 유사한 소설(들)은 없습니다.\n")
    

    if max_value>0:
        for i in range(len(cosine_list)):
            if cosine_list[i]==max_value:
               final_list.append([total_books[i],cosine_list[i]])
        
        print("\n"+book_name+"(와)과 가장 유사한 소설(들)은 다음과 같습니다. (Cosine Similarity)\n")
        for i in range(len(final_list)):
            book_print=re.sub(".txt","",final_list[i][0])
            print("-- "+book_print+' ('+str(final_list[i][1])+')')
    
    ranked_list= { name:value for name, value in zip(total_books, cosine_list) }
    final_ranking=sorted(ranked_list.items(), key=lambda x: x[1], reverse=True)
    final_books=[]
    for i in range(len(final_ranking)):
        tmp_str=final_ranking[i][0]
        tmp_str=re.sub(".txt","",tmp_str)
        final_books.append(tmp_str)

    return final_books
    
def cos_sim(A, B):
    if norm(A)!=0 and norm(B)!=0:
        return dot(A, B)/(norm(A)*norm(B))
    else:
        return 0
       
if __name__ == "__main__" :
    

    parser=argparse.ArgumentParser(description='사건 소설 매칭')
    parser.add_argument('--query','--query',required=True)
    args=vars(parser.parse_args())
    input=args["query"]
    book_name=input
    input_text='/matching/'+input+'.txt'
    find_similar_book_cosine(input_text,book_name)


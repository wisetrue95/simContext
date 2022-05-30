import os
import argparse
path = os.path.dirname( os.path.abspath( __file__ ) )

def character_similarity(target_character, compare_character) :
    s1 = set(target_character) 
    s2 = set(compare_character)
    score = len(s1 & s2) / len(s1 | s2)
    return score

def char_top_rank(novel_title) :
    with open(path + "/" + novel_title, mode="r", encoding='utf-8') as target:
        target_character = list()
        for character in target :
            target_character.append(character.strip("\n"))
        
    file_list = os.listdir(path)
    per_fin_file_list = [file for file in file_list if file.endswith("_file_ner.txt")]
    file_num = len(per_fin_file_list)        
    per_fin_file_list.remove(novel_title)
            
    score = [] #비교 점수
    for file_name in per_fin_file_list :
        with open(path + "/" + file_name, mode = "r", encoding= 'utf-8') as compare:
            compare_character = list()
            for character in compare :
                compare_character.append(character.strip("\n"))
            score.append((file_name, character_similarity(target_character, compare_character)))
    score.sort(key = lambda x : x[1], reverse = True)
    score_list = []
    for i in range(file_num - 1) :
        score_list.append(score[i][0].split("_file_ner.txt")[0])
    return score_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='등장인물 유사도 계산')
    parser.add_argument('--novel_title', '--novel_title', required=True)

    args = parser.parse_args()
    
    _ = char_top_rank( args.novel_title + "_file_ner.txt")
   

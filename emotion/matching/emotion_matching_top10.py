import numpy as np
import glob
import json
import argparse
import os
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

### Emotion matching code for KIE

def model_result_to_emotion_vec_all(): # all model results to the emotion vectors in a certain dir
    model_result_files = glob.glob('./dataset/model_result_json/*')
    # print(model_result_files)
    for result_file in model_result_files:
        model_result_file_name = result_file.split('/')[-1]

        with open(result_file, "r", encoding="utf-8") as json_file:
            model_output_json = json.load(json_file)

        pred_results = []
        for line in model_output_json:
            pred_results.append(line["tags"][0]["sub-type"]) ### ------- This line should be changed later !!! ----- ###

        total_sentence_num = len(model_output_json)

        h_20 = [0] * 20
        s_20 = [0] * 20
        a_20 = [0] * 20

        for i in range(20): # divide pred_results into 20 groups
            for j in range(len(pred_results)):
                precent = int(j / total_sentence_num * 20)
                if precent == i:
                    if pred_results[j] == 'happy':
                        h_20[i] += 1
                    if pred_results[j] == 'sad':
                        s_20[i] += 1
                    if pred_results[j] == 'angry':
                        a_20[i] += 1

        ## saving vectors for matching
        output = [h_20, s_20, a_20]
        output_fname = model_result_file_name.split('.')[0] + "_emotion_vectors"
        output_path = os.path.join('dataset/emotion_vec', output_fname)
        np.save(output_path, output)

def emotion_matching(query_text_title): # distance between query_text and the others
    matching_input = './dataset/emotion_vec/' + query_text_title + '_emotion_vectors.npy'
    query = np.load(matching_input)
    result_files = glob.glob('./dataset/emotion_vec/*')
    # print(result_files)

    sad_dist = []
    hap_dist = []
    ang_dist = []

    ### Distance between three classes in two novels
    novel_titles = []
    for j in range(len(result_files)):
        file_name = result_files[j].split('/')[-1]
        file_name = file_name.split('_')[0] + '_' + file_name.split('_')[1]
        novel_titles.append(file_name)

        emt_vecs = np.load(result_files[j]) # emotion vectors
        for i in range(3): # emotion classes
            if i == 0:
                hap_dist.append(wasserstein_distance(emt_vecs[i], query[i]))
            elif i == 1:
                sad_dist.append(wasserstein_distance(emt_vecs[i], query[i]))
            elif i == 2:
                ang_dist.append(wasserstein_distance(emt_vecs[i], query[i]))

    ### Compute distances for all classes
    total_dist = []
    _dict = {}
    for i in range(len(result_files)):
        total = hap_dist[i] + sad_dist[i] + ang_dist[i]
        total_dist.append(total)
        _dict[novel_titles[i]] = total

    sorted_dict = dict(sorted(_dict.items(), key=(lambda x:x[1]))) # reverse=True
    sorted_list = []
    for key, value in sorted_dict.items():
        sorted_list.append(key)
    
    #plot(len(result_files), hap_dist, sad_dist, ang_dist, total_dist, novel_titles)


    ### Sort by distance
    def f(x):
        return float(x[1])
    sort_dict = sorted(_dict.items(), key=f) # reverse=True

    ### Print Top 10
    for i in range(1, 11):
        print("Top", i, sort_dict[i], end="\n")
    # print("Worst 1:", sort_dict[-1])
    print("Worst 1", sort_dict[-2])

    return sorted_list[1:]

def plot(num_files, hap_dist, sad_dist, ang_dist, total_dist, novel_titles):
    # to plot Korean text
    import matplotlib.font_manager as fm
    font_fname = './NanumGothic.ttf' ### You should fits your dir
    font_family = fm.FontProperties(fname=font_fname).get_name()

    plt.rcParams["font.family"] = font_family
    plt.rcParams["font.size"] = 30

    x = list(range(1, num_files + 1))

    # happy
    fig, ax = plt.subplots(1, 1, figsize=(100, 45))
    ax.scatter(x, hap_dist)
    ax.set_title("Wasserstein Distance in Happy class")
    for i, txt in enumerate(novel_titles):
        ax.annotate(txt, xy=(x[i], hap_dist[i]))
    fig.savefig(args.matching_output_plot_dir + 'scatter_dist_hap.png')
    plt.close()

    # sad
    fig, ax = plt.subplots(1, 1, figsize=(100, 45))
    ax.scatter(x, sad_dist)
    ax.set_title("Wasserstein Distance in Sad class")
    for i, txt in enumerate(novel_titles):
        ax.annotate(txt, xy=(x[i], sad_dist[i]))
    fig.savefig(args.matching_output_plot_dir + 'scatter_dist_sad.png')
    plt.close()

    # ang
    fig, ax = plt.subplots(1, 1, figsize=(100, 45))
    ax.scatter(x, ang_dist)
    ax.set_title("Wasserstein Distance in Angry class")
    for i, txt in enumerate(novel_titles):
        ax.annotate(txt, xy=(x[i], ang_dist[i]))
    fig.savefig(args.matching_output_plot_dir + 'scatter_dist_ang.png')
    plt.close()

    # total
    fig, ax = plt.subplots(1, 1, figsize=(100, 45))
    ax.scatter(x, total_dist)
    ax.set_title("Wasserstein Distance in Total classes")
    for i, txt in enumerate(novel_titles):
        ax.annotate(txt, xy=(x[i], total_dist[i]))
    fig.savefig(args.matching_output_plot_dir + 'scatter_dist_total.png')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_text_title', help='text file name to match',default='이상_날개')
    parser.add_argument('--matching_output_plot_dir', default='./result/',
                        help="Directory to save matching result images")

    args = parser.parse_args()

    model_result_to_emotion_vec_all() ### This line can be done only once
    _ = emotion_matching(args.query_text_title)



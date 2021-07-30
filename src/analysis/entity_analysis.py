import spacy
import math
from collections import Counter

def LCS(seq1, seq2):
    dp = [[0 for j in range(len(seq2) + 1)] for i in range(len(seq1) + 1)]
    result = 0
    for i in range(len(seq1)+1):
        for j in range(len(seq2)+1):
            if i == 0 or j == 0:
                dp[i][j] = 0
            elif seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                result = max(result, dp[i][j])
            else:
                dp[i][j] = 0
    return result

def LCS_metric(seq1, seq2):
    max_len = max(len(seq1), len(seq2))
    # target_len = len(seq1)
    return LCS(seq1,seq2)/max_len


# calculate bag similarity
def counter_cosine_similarity(c1, c2):
    terms = set(c1).union(c2)
    # print(terms)
    dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
    magA = math.sqrt(sum(c1.get(k, 0)**2 for k in terms))
    magB = math.sqrt(sum(c2.get(k, 0)**2 for k in terms))
    return dotprod / (magA * magB)

def Bag_metric(seq1, seq2):
    counter1 = Counter(seq1)
    counter2 = Counter(seq2)
    return counter_cosine_similarity(counter1, counter2)


def get_sim_scores(nlp, target_string, predict_string):
    target = nlp(target_string)
    predict = nlp(predict_string)

    ents_coarse_target = []
    ents_coarse_predict = []
    
    for ent in target.ents:
        ents_coarse_target.append(ent.label_)
    #     print(ent.text, ent.start_char, ent.end_char, ent.label_)
    # print('#####################')
    # print()
    for ent in predict.ents:
        ents_coarse_predict.append(ent.label_)
    #     print(ent.text, ent.start_char, ent.end_char, ent.label_)
    # print('#####################')

    if len(ents_coarse_target) == 0:
        return None, None
    
    if len(ents_coarse_target) != 0 and len(ents_coarse_predict) == 0:
        return 0, 0

    # print('LCS length:',LCS(ents_coarse_target,ents_coarse_predict))
    LCS_sim = LCS_metric(ents_coarse_target,ents_coarse_predict)
    # print('LCS sim score:',LCS_sim)

    Bag_sim = Bag_metric(ents_coarse_target, ents_coarse_predict)
    # print('Bag sim score:',Bag_sim)

    return LCS_sim, Bag_sim

def show_exmaple():
    nlp = spacy.load("en_core_web_sm")
    target_string = "He is a prominent member of the Bush family, the younger brother of President George W. Bush and the second son of former President George H. W. Bush and Barbara Bush."
    predict_string = "was a great member of the American family. and first brother of President George W. Bush. the father son of President President Ronald H. W. Bush. his Bush."
    LCS_sim, Bag_sim = get_sim_scores(nlp, target_string, predict_string)


def main(tasks):
    nlp = spacy.load("en_core_web_sm")
    for task in tasks:
        if task == 'task1':
            generation_results_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/results/task1-BrainTranslator_skipstep1-all_generation_results-7_22.txt'
        elif task == 'task1+task2':
            generation_results_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/results/task1_task2-BrainTranslator_skipstep1-all_generation_results-7_22.txt'
        elif task == 'task1+task2+taskNRv2':
            generation_results_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/results/task1_task2_taskNRv2-BrainTranslator_skipstep1-all_generation_results-7_22.txt'
        
        with open(generation_results_path) as f:
            running_LCS = 0.0
            running_BagCos = 0.0
            count = 0
            target_string = ''
            predict_string = ''
            for line in f:
                if line.startswith('target string'):
                    target_string = line.strip()[15:]
                elif line.startswith('predicted string'):
                    predict_string = line.strip()[18:]
                    LCS_sim, Bag_sim = get_sim_scores(nlp, target_string, predict_string)
                    if (LCS_sim is not None) and (Bag_sim is not None):
                        running_LCS += LCS_sim
                        running_BagCos += Bag_sim
                        count += 1
        print('task:',task)
        print('count:',count)
        mean_LCS = running_LCS/count
        mean_BagCos = running_BagCos/count
        print('mean LCS:', mean_LCS)
        print('mean BagCos:', mean_BagCos)


if __name__ == '__main__':
    tasks = ['task1','task1+task2','task1+task2+taskNRv2']
    main(tasks)
    
    # scores
        # task: task1
        # count: 202
        # mean LCS: 0.1485148514851485
        # mean BagCos: 0.17841997626236716
        
        # task: task1+task2
        # count: 552
        # mean LCS: 0.2920721187025537
        # mean BagCos: 0.5016455506518832
        
        # task: task1+task2+taskNRv2
        # count: 1153
        # mean LCS: 0.35599255471328317
        # mean BagCos: 0.5571035598880302

        
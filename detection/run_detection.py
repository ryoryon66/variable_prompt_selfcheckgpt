import sys 
import pandas as pd
import logging
import numpy as np
from tqdm import tqdm 

sys.path.append('../')

import utils.llm 
import utils.scoring

logger = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
logger.addHandler(handler)
logger.setLevel(logging.INFO)



DATA_DIR = '../data/'
# DATA_SAMPLES_PATH = '../data_out/random_query_temperature_0.01_sample_n_30_random_str_length_60_11_23_14_57_54.csv'
# DATA_SAMPLES_PATH = '../data_out/random_query_temperature_0.01_sample_n_30_random_str_length_20_11_25_00_30_30.csv'
DATA_SAMPLES_PATH = '../data_out/random_query_temperature_0.01_sample_n_30_random_str_length_40_11_24_10_24_05.csv'
DATA_SAMPLES_PATH = '../data_out/random_query_temperature_0.01_sample_n_30_random_str_length_200_12_22_20_56_39.csv'
DATA_SAMPLES_PATH = '../data_out/random_query_temperature_0.01_sample_n_30_random_str_length_280_12_23_10_22_50.csv'
DATA_SAMPLES_PATH = '../data_out/random_query_temperature_0.01_sample_n_30_random_str_length_400_12_27_20_36_48.csv'
DATA_SAMPLES_PATH = '../data_out/random_query_temperature_0.01_sample_n_30_random_str_length_120_hiragana_12_28_09_29_11.csv'
DATA_SAMPLES_PATH = '../data_out/random_query_temperature_0.01_sample_n_30_random_str_length_280_hiragana_12_28_19_56_16.csv'
DATA_SAMPLES_PATH = '../data_out/random_query_temperature_0.01_sample_n_30_random_str_length_600_12_28_21_24_57.csv'
# DATA_SAMPLES_PATH = '../data_out/same_query_temperature_0.9_sample_n_30_11_20_08_16_25.csv'
# DATA_SAMPLES_PATH = '../data_out/same_query_temperature_0.01_sample_n_30_11_22_02_12_28.csv'
NUM_SAMPLES = 30




annotated_df = pd.read_csv(DATA_DIR + "df_JAQKET_qa_annot.csv")
sampled_df = pd.read_csv(DATA_SAMPLES_PATH)


annotated_df = annotated_df[:189]
num_questions = len(annotated_df)




score_df = pd.DataFrame(columns=['query', 'score_pr','score_re','score_f1'])
for i in tqdm(range(num_questions)): 

    # logger.info("processing {} / {}".format(i,num_questions))
    
    qid = annotated_df.iloc[i]['qid']
    response  : str= annotated_df.iloc[i]['generated_answer']
    samples = list(sampled_df[sampled_df['qid'] == qid]["sample"])[:NUM_SAMPLES]
    
    
    pr_scores, re_scores,f1_scores = utils.scoring.get_hallucination_score(response,samples)
    
    
    # score_df = score_df.append({
    #         'qid':qid,
    #         'pr_score':",".join([str(x) for x in pr_scores]), 
    #         're_score':",".join([str(x) for x in re_scores]), 
    #         'f1_score':",".join([str(x) for x in f1_scores]),
    #     },
    #     ignore_index=True)
    
    # use concat instead of append
    
    score_df = pd.concat([
        score_df,
        pd.DataFrame({
            'qid':qid,
            'pr_score':",".join([str(x) for x in pr_scores]), 
            're_score':",".join([str(x) for x in re_scores]), 
            'f1_score':",".join([str(x) for x in f1_scores]),
        }, index=[0])
    ])
    

logger.info("done") 





result_df = pd.merge(annotated_df,score_df,on='qid')

result_df.to_csv(DATA_DIR + 'df_JAQKE_selfcheck.csv',index=False)

result_df = pd.read_csv(DATA_DIR + 'df_JAQKE_selfcheck.csv')

def extract_label_and_scores():
    
    df = result_df.copy()    

    df  = df[['qid','factuality','pr_score','re_score','f1_score','question']]
    
    label_and_scores = {
        'label':[],
        'pr':[],
        're':[],
        'f1':[]
    }
    
    # set data type
    df['factuality'] = df['factuality'].astype(str)
    
    for i in range(df.shape[0]):
        labels = df.iloc[i]['factuality'].split(',')
        pr_scores = df.iloc[i]['pr_score'].split(',')
        re_scores = df.iloc[i]['re_score'].split(',')
        f1_scores = df.iloc[i]['f1_score'].split(',')
        
        labels = [1 - int(x) for x in labels]
        pr_scores = [float(x) for x in pr_scores]
        re_scores = [float(x) for x in re_scores]
        f1_scores = [float(x) for x in f1_scores]
        
        assert len(labels) == len(pr_scores), f"labels: {labels}, pr_scores: {pr_scores},i: {i},q: {df.iloc[i]['question']}"
        
        label_and_scores['label'].extend(labels)
        label_and_scores['pr'].extend(pr_scores)
        label_and_scores['re'].extend(re_scores)
        label_and_scores['f1'].extend(f1_scores)
    
    import numpy as np
    label_and_scores["label"] = np.array(label_and_scores["label"])
    label_and_scores["pr"] = np.array(label_and_scores["pr"])
    label_and_scores["re"] = np.array(label_and_scores["re"])
    label_and_scores["f1"] = np.array(label_and_scores["f1"])
    

    return label_and_scores



# AUC scoreを計算   
from sklearn.metrics import average_precision_score
import numpy as np
label_and_scores = extract_label_and_scores()

pr_aucs = {
    'pr':average_precision_score(label_and_scores['label'],label_and_scores['pr']),
    're':average_precision_score(label_and_scores['label'],label_and_scores['re']),
    'f1':average_precision_score(label_and_scores['label'],label_and_scores['f1']),
    "ave":average_precision_score(label_and_scores['label'],(label_and_scores['pr'] + label_and_scores['re'] + label_and_scores['f1'])/3),
    'random' : average_precision_score(label_and_scores['label'],np.random.rand(len(label_and_scores['label']))),
}

print(f"the auc-pr (nonfact positive) is {pr_aucs}")

pr_aucs = {
    'pr':average_precision_score(1-label_and_scores['label'],-label_and_scores['pr']),
    're':average_precision_score(1-label_and_scores['label'],-label_and_scores['re']),
    'f1':average_precision_score(1-label_and_scores['label'],-label_and_scores['f1']),
    "ave":average_precision_score(1-label_and_scores['label'],-(label_and_scores['pr'] + label_and_scores['re'] + label_and_scores['f1'])/3),
    'random' : average_precision_score(1-label_and_scores['label'],-np.random.rand(len(label_and_scores['label']))),
}

print(f"the auc-pr (fact positive)  is {pr_aucs}")

from sklearn.metrics import roc_auc_score

roc_aucs = {
    'pr':roc_auc_score(label_and_scores['label'],label_and_scores['pr']),
    're':roc_auc_score(label_and_scores['label'],label_and_scores['re']),
    'f1':roc_auc_score(label_and_scores['label'],label_and_scores['f1']),
    "ave":roc_auc_score(label_and_scores['label'],(label_and_scores['pr'] + label_and_scores['re'] + label_and_scores['f1'])/3),
    'random' : roc_auc_score(label_and_scores['label'],np.random.rand(len(label_and_scores['label']))),
}

print(f"the auc-roc (nonfact positive) is {roc_aucs}")

roc_aucs = {
    'pr':roc_auc_score(1-label_and_scores['label'],-label_and_scores['pr']),
    're':roc_auc_score(1-label_and_scores['label'],-label_and_scores['re']),
    'f1':roc_auc_score(1-label_and_scores['label'],-label_and_scores['f1']),
    "ave":roc_auc_score(1-label_and_scores['label'],-(label_and_scores['pr'] + label_and_scores['re'] + label_and_scores['f1'])/3),
    'random' : roc_auc_score(1-label_and_scores['label'],-np.random.rand(len(label_and_scores['label']))),
}

print(f"the auc-roc (fact positive)  is {roc_aucs}")



# PR curveを計算
from sklearn.metrics import precision_recall_curve,roc_auc_score
import matplotlib.pyplot as plt

label_and_scores = extract_label_and_scores()

pr_curve = precision_recall_curve(label_and_scores['label'],label_and_scores['pr'])
re_curve = precision_recall_curve(label_and_scores['label'],label_and_scores['re'])
f1_curve = precision_recall_curve(label_and_scores['label'],label_and_scores['f1'])



plt.plot(pr_curve[1],pr_curve[0],label='pr')
plt.plot(re_curve[1],re_curve[0],label='re')
plt.plot(f1_curve[1],f1_curve[0],label='f1')
plt.legend()
plt.xlabel('recall')
plt.ylabel('precision')
plt.savefig('pr_curve.png')



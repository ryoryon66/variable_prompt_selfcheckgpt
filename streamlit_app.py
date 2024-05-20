import token
import streamlit as st
import pandas as pd 

import utils.scoring 
import utils.llm 

import random
import multiprocessing as mp
from typing import List,Literal
random.seed(42)

from textwrap import dedent




def get_random_string(length):
    letters = "abcdefghijklmnopqrstuvwxyz"
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

@st.cache_resource
def get_llm():
    print("loading llm")
    llm = utils.llm.LLM()
    print("loaded llm")
    return llm


llm_instance = get_llm()

st.title('Hallucination Detection Demosite')



st.write(dedent("""\
    Hallucination Detectionのデモサイトです。
    LLM出力文ごとにHallucinationスコアを算出し、Hallucinationリスクを警告します。

    ## 設定
    - Number of samples: サンプリングフェーズでサンプリングする回数
    - Temperature: サンプリング時のtemperature
    - Prefix length: サンプリング時に利用する文頭random文字列の長さ
    
    """
    )
)


# ============= ユーザー入力 =============


# text input 
query = st.text_input('Please input your text', 'スタックプロテクターについて教えてください。')
# number of samples 
num_of_samples = st.slider('Number of samples', 1, 10, 3)
# temperature
temperature = st.slider('Temperature', 0.0, 1.0, 0.01)

# mode selection "クエリ言い換え" or "そのまま"
prefix_length = st.slider('Prefix length', 0, 1000, 0)

# =========== ボタンが押されたときの処理 ==============

# button 
if st.button('Submit'):
    # output 
    st.write('Input text: ', query)
    
    response = llm_instance.generate(query)
    samples : List[str] = []
    
    st.write('Response: ', response)
    
    st.write("======= sampling pahse =======")
    

    # sampling phase
    for i in range(num_of_samples):
        random_str = get_random_string(prefix_length)
        paraphrased_query = f"{random_str} {query}"
        
        sampled = llm_instance.generate(paraphrased_query, temperature)
        st.write(f"sample{i+1}: ", sampled)
        samples.append(sampled)

    st.write("======= scoring pahse =======")
    
    # scoring phase
    PR_scores,_, _ = utils.scoring.get_hallucination_score(response,samples)
    
    split_response = utils.scoring.sentence_splitter(response)
    
    # forでわかりやすく出力する
    for i in range(len(PR_scores)):
        st.write(f"{i+1}文目のhallucinationスコア")
        st.write(f"response: {split_response[i]}")
        
        st.write(f"hallucination score: {PR_scores[i]}")
        st.write("")
    
    all_token_level_scores = utils.scoring.get_token_level_score(response,samples)
    
    st.write("token level score")
    
    for sentence_token_level_scores in all_token_level_scores:
        for elem in sentence_token_level_scores:
            token, score = elem
            box_count = score / 0.05
            print (len(token), token, box_count) 
            st.write(token +  "　" * (18 - len(token)) + '■' * int(box_count) + f" {score:.2f}")
        st.write("-" * 10)
            

    
    
    
        







### うまくいってそうな質問例
# バッファオーバフロー攻撃について教えてください。
# スタックプロテクターについて教えてください。
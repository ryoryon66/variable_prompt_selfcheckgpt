
import pandas
import threading
from threading import Thread
from typing import List, Optional,Tuple
from pprint import pprint
import bert_score
import re
import numpy as np
import bert_score
import matplotlib.pyplot as plt
import japanize_matplotlib
from bert_score.utils import (
    get_bert_embedding,
    sent_encode,
)
import torch
from collections import defaultdict

class BERTScorer(bert_score.BERTScorer):

    def __init__(self, *args, **kwargs):
        
        
        super().__init__(*args, **kwargs)
        self.tokenizer = self._tokenizer
        

        return

    def get_tokenwise_similarity(self, candidate:str, reference:str) -> Tuple[np.ndarray,List[str],List[str]]:
        """類似度行列とtokenizeされた文章を返す"""
        
        assert isinstance(candidate, str)
        assert isinstance(reference, str)

        idf_dict = defaultdict(lambda: 1.0)
        idf_dict[self._tokenizer.sep_token_id] = 0
        idf_dict[self._tokenizer.cls_token_id] = 0

        hyp_embedding, masks, padded_idf = get_bert_embedding(
            [candidate],
            self._model,
            self._tokenizer,
            idf_dict,
            device=self.device,
            all_layers=False,
        )
        ref_embedding, masks, padded_idf = get_bert_embedding(
            [reference],
            self._model,
            self._tokenizer,
            idf_dict,
            device=self.device,
            all_layers=False,
        )
        ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
        hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))
        sim = torch.bmm(hyp_embedding, ref_embedding.transpose(1, 2))
        sim = sim.squeeze(0).cpu()

        r_tokens = [
            self._tokenizer.decode([i]) for i in sent_encode(self._tokenizer, reference)
        ][1:-1]
        h_tokens = [
            self._tokenizer.decode([i]) for i in sent_encode(self._tokenizer, candidate)
        ][1:-1]
        sim = sim[1:-1, 1:-1]

        if self.rescale_with_baseline:
            sim = (sim - self.baseline_vals[2].item()) / (
                1 - self.baseline_vals[2].item()
            )


        
        cand_tokens = h_tokens
        ref_tokens = r_tokens
        sim: np.ndarray = sim.numpy()

        return sim, cand_tokens, ref_tokens
    


scorer : BERTScorer = BERTScorer(lang="ja",device="cuda:0")


    
def sentence_splitter(s:str):
        
    # 文末表現一覧
    end_pattern : str = r'[。?！？!]'
    
    # 文末表現で分割
    split_sentense = re.split(end_pattern, s)
    
    # 単なる改行をすべて削除
    split_sentense = list(map(lambda x: x.replace("\n",""),split_sentense))
    
    # 空文字列を削除
    split_sentense = list(filter(lambda x: x != "",split_sentense))
    
    # 文末表現をすべて削除
    for char in end_pattern:
        split_sentense = list(map(lambda x: x.replace(char,""),split_sentense))
    
    
    return split_sentense



def calc_bert_score(cands:List[str], refs:List[str]) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """ BERTスコアの算出

    Args:
        cands ([List[str]]): [比較元の文]
        refs ([List[str]]): [比較対象の文]

    Returns:
        [(List[float], List[float], List[float])]: [(Precision, Recall, F1スコア)]
    """

    
    
    
    P, R, F = scorer.score(cands, refs, verbose=False)
    
    return P.numpy(), R.numpy(), F.numpy()

def get_hallucination_score(response:str,samples:List[str]) -> Tuple[List[float],List[float],List[float]]:
    """
    Args:
        response (str): _description_
        samples (List[str]): _description_

    Returns:
        Tuple[List[float]]: pr_scores,re_scores,f1_scores
    """
    
    r_sentences = sentence_splitter(response)
    # print("r_sentences")
    # print(r_sentences)
    
    N = len(samples)
    
    PR_scores = []
    RE_scores = []
    F1_scores = []
    
    for r in r_sentences:
        
        assert r != "", f"r is empty"
        
        # print(r)
        
        PR_scores.append(1.0)
        RE_scores.append(1.0)
        F1_scores.append(1.0)
        
        for sample in samples:
            
            sample_sentences = sentence_splitter(sample)
        

            PR_score, RE_score, F1_score = calc_bert_score([r] * len(sample_sentences),sample_sentences)
        
            assert PR_score.shape == (len(sample_sentences),), f"PR_score.shape is {PR_score.shape}"
            
            pr_score = max(PR_score) / N
            re_score = max(RE_score) / N
            f1_score = max(F1_score) / N
            
            PR_scores[-1] -= pr_score
            RE_scores[-1] -= re_score
            F1_scores[-1] -= f1_score
            
    assert len(PR_scores) == len(RE_scores) == len(F1_scores)
    assert len(PR_scores) == len(r_sentences), f"len(PR_scores) is {len(PR_scores)}, len(r_sentences) is {len(r_sentences)}"
    
    return PR_scores, RE_scores, F1_scores

def get_token_level_score(response:str,samples:List[str]) -> list[list[Tuple[str,float]]]:
    
    assert len(samples) > 0, f"len(samples) is {len(samples)}"
    
    r_sentences = sentence_splitter(response)
    

    
    

    
    all_token_level_scores: list[list[Tuple[str,float]]] = [] 
    
    for r in r_sentences:
        
        assert r != "", f"r is empty"
        
        r_token_level_scores: list[Tuple[str,float]] = []
        r_tokens = [
            scorer.tokenizer.decode([i]) for i in sent_encode(scorer.tokenizer, r)
        ][1:-1]
        
        for r_token in r_tokens:
            r_token_level_scores.append((r_token,0.0))
        
        for sample in samples:
            
            sample_sentences = sentence_splitter(sample)
        
            sim, cand_tokens, ref_tokens = scorer.get_tokenwise_similarity(r,sample)
            assert sim.shape == (len(cand_tokens),len(ref_tokens)), f"sim.shape is {sim.shape}"
            
            max_sim = sim.max(axis=1)
            
            for i,cand_token in enumerate(cand_tokens):
                assert cand_token == r_token_level_scores[i][0], f"cand_token is {cand_token}, r_token_level_scores[i][0] is {r_token_level_scores[i][0]}"
                r_token_level_scores[i] = (r_token_level_scores[i][0],r_token_level_scores[i][1] + max_sim[i])
            
            # divide by number of samples
            r_token_level_scores = list(map(lambda x: (x[0],x[1] / len(samples)),r_token_level_scores))
    
    
        all_token_level_scores.append(r_token_level_scores)
    
    # tokenから#を消す
    all_token_level_scores = list(map(lambda x: list(map(lambda y: (y[0].replace("#",""),y[1]),x)),all_token_level_scores))
    
    return all_token_level_scores
            
            



if __name__ == "__main__":
    

    
    
    response = ""
    samples = ["","",""]
    
    
    PR_scores, RE_scores, F1_scores = get_hallucination_score(response,samples)

    print(PR_scores)
    print(RE_scores)
    print(F1_scores)
    
    import japanize_matplotlib
    
    
    
    sim, cand_tokens, ref_tokens = scorer.get_tokenwise_similarity(response,samples[0])
    
    from pprint import pprint
    

    
    all_token_level_scores = get_token_level_score(response,samples)
    
    pprint(all_token_level_scores)
    

    
    # scorer.plot_example(response,samples[0])
    
    
    




import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pprint import pprint 



class LLM:
    
    def __init__(self):
        
        self.tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt-neox-3.6b-instruction-ppo", use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            "rinna/japanese-gpt-neox-3.6b-instruction-ppo",
            revision = "2140541486bfb31269acd035edd51208da40185b",
            load_in_8bit = True,
            )
        
    def generate(self, query:str,temperture=0.01):
        
        prompt = "ユーザー:" + query + "\nシステム:"
        

        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        

        with torch.no_grad():
            output_ids = self.model.generate(
                token_ids.to(self.model.device),
                do_sample=True,
                max_new_tokens=128,
                temperature=temperture,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
 

            
        output = self.tokenizer.decode(output_ids.tolist()[0][token_ids.size(1):])
        # print(len(output_ids.tolist()))
        output = output.replace("<NL>", "\n").replace("</s>","")

        
        return output
    
if __name__ == "__main__":
    
    llm = LLM()
    res = llm.generate("わたしは京都府に住んでいる大学生です。祇園祭りって何ですか？")
    
    pprint(res)
import sys
import os
import time
import torch

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm import tqdm
from vllm import LLM, SamplingParams
from huggingface_hub import login

def main():
    file_path = './sys_prompt3.txt'

    with open(file_path, 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    login(token='')

    llm = LLM(
        model="google/gemma-3-27b-it",
        dtype="bfloat16",
        tensor_parallel_size=4,
        gpu_memory_utilization=0.9
    )
    sampling_params = SamplingParams(
        temperature=0.2,
        top_p=0.95,
        max_tokens=2048,
        repetition_penalty=1.0
    )
    batch_size = 128
    
    df_mimic4 = pd.read_csv('./mimic4/sepsis_final_data_withTimes.csv')
    df_mimic4.rename(columns={'m:text': 't:text'}, inplace=True)
    df_mimic4['t:LLM_clinical_note'] = pd.NA
    mask_nan = df_mimic4["t:text"].isna()
    df_mimic4.loc[mask_nan, "t:LLM_clinical_note"] = "no clinical note"

    valid_idx = df_mimic4.index[~mask_nan]  

    for start in tqdm(range(0, len(valid_idx), batch_size)):
       batch_idx = valid_idx[start : start + batch_size]
       user_prompts = df_mimic4.loc[batch_idx, "t:text"].tolist()
       conversations = [
           [
               {"role": "system", "content": system_prompt},
               {"role": "user",   "content": note}
           ]
           for note in user_prompts
       ]

       outputs = llm.chat(
           messages=conversations,
           sampling_params=sampling_params,
           use_tqdm=True
       )

       completions = [out.outputs[0].text for out in outputs]
       df_mimic4.loc[batch_idx, "t:LLM_clinical_note"] = completions
    df_mimic4.to_csv("./mimic4/df_mimic4_prompt3_llama8B.csv")



if __name__ == '__main__':
    main()
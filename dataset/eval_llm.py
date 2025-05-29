from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json
from med_benchmark import *
from huggingface_hub import login


login(token='')

def test_medical_QA_eng(model_name):
    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        tensor_parallel_size=4,
        gpu_memory_utilization=0.9
    )

    sampling_params = SamplingParams(
    temperature=0.1,      
    top_p=1.0,            
    top_k=1,               
    min_p=0.0,             
    repetition_penalty=1.0,
    max_tokens=10           
    )
    
    pubmed = evaluate_pubmedqa(llm, sampling_params, use_manual=False)
    with open(f'./metric/{model_name.split("/")[-1]}_pubmedQA.json', 'w', encoding='utf-8') as f:
        json.dump(pubmed, f, ensure_ascii=False, indent=2)

    mcqa = evaluate_medmcqa(llm, sampling_params)
    with open(f'./metric/{model_name.split("/")[-1]}_mcqa.json', 'w', encoding='utf-8') as f:
        json.dump(mcqa, f, ensure_ascii=False, indent=2)

    medqa = evaluate_medqa(llm, sampling_params)
    with open(f'./metric/{model_name.split("/")[-1]}_medqa.json', 'w', encoding='utf-8') as f:
        json.dump(medqa, f, ensure_ascii=False, indent=2)
    
    print("________________________________________")

    
def test_medical_QA_kor(model_name):
    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        tensor_parallel_size=4,
        gpu_memory_utilization=0.8
    )

    sampling_params = SamplingParams(
    temperature=0.1,      
    top_p=1.0,            
    top_k=1,               
    min_p=0.0,             
    repetition_penalty=1.0,
    max_tokens=10           
    )

    kor_mcqa = evaluate_kormedmcqa(llm, sampling_params)
    with open(f'./metric/{model_name.split("/")[-1]}_kor_mcqa.json', 'w', encoding='utf-8') as f:
        json.dump(kor_mcqa, f, ensure_ascii=False, indent=2)

    kor_medconcept = evaluate_atc_easy(llm, sampling_params)
    with open(f'./metric/{model_name.split("/")[-1]}_kor_medconcept.json', 'w', encoding='utf-8') as f:
        json.dump(kor_medconcept, f, ensure_ascii=False, indent=2)

    
    
def test_medical_summerization(model_name):
    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        tensor_parallel_size=4,
        gpu_memory_utilization=0.8
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.95,
        max_tokens=256,
        repetition_penalty=1.0
    )
    pubmed_summer = evaluate_pubmed_summarization(llm, tokenizer, sampling_params)
    with open(f'./metric/{model_name.split("/")[-1]}_pubmed_summer.json', 'w', encoding='utf-8') as f:
        json.dump(pubmed_summer, f, ensure_ascii=False, indent=2)

    mimic_cxr_summer = evaluate_mimic_cxr_summarization(llm, tokenizer, sampling_params)
    with open(f'./metric/{model_name.split("/")[-1]}_mimic_cxr_summer.json', 'w', encoding='utf-8') as f:
        json.dump(mimic_cxr_summer, f, ensure_ascii=False, indent=2)

    



if __name__ =="__main__":
    model_name = 'YBXL/Med-LLaMA3-8B'
    test_medical_QA_eng(model_name)

    model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    test_medical_QA_eng(model_name)

    model_name = 'ContactDoctor/Bio-Medical-Llama-3-8B-CoT-012025'
    test_medical_QA_eng(model_name)

    model_name = 'google/gemma-3-27b-it'
    test_medical_QA_eng(model_name)

    model_name = 'meta-llama/Llama-3.2-3B-Instruct'
    test_medical_QA_eng(model_name)
    ##############################################################
    model_name = 'YBXL/Med-LLaMA3-8B'
    test_medical_summerization(model_name)

    model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    test_medical_summerization(model_name)

    model_name = 'ContactDoctor/Bio-Medical-Llama-3-8B-CoT-012025'
    test_medical_summerization(model_name)

    model_name = 'google/gemma-3-27b-it'
    test_medical_summerization(model_name)

    model_name = 'Qwen/Qwen3-32B'
    test_medical_summerization(model_name)

    model_name = 'meta-llama/Llama-3.2-3B-Instruct'
    test_medical_summerization(model_name)

    ##############################################################
    model_name = 'meta-llama/Llama-3.2-3B-Instruct'
    test_medical_QA_kor(model_name)

    model_name = 'google/gemma-3-27b-it'
    test_medical_QA_kor(model_name)

    model_name = 'ContactDoctor/Bio-Medical-Llama-3-8B-CoT-012025'
    test_medical_QA_kor(model_name)

    model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    test_medical_QA_kor(model_name)

    model_name = 'YBXL/Med-LLaMA3-8B'
    test_medical_QA_kor(model_name)

    model_name = 'Qwen/Qwen3-32B'
    test_medical_QA_kor(model_name)

import requests
import re
from datasets import load_dataset, DownloadConfig
import evaluate
from transformers import AutoTokenizer, PreTrainedTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import f1_score
from evaluate import load as load_metric



def evaluate_atc_easy(llm,
                      sampling_params,
                      batch_size: int = 32,
                      debug_samples: int = 5):

    accuracy_metric = evaluate.load("accuracy")
    f1_metric       = evaluate.load("f1")

    ds = load_dataset("ChuGyouk/KorMedConceptsQA", "atc_easy", split="test")

    label_map = {"A": 0, "B": 1, "C": 2, "D": 3}

    prompts, references = [], []
    for ex in ds:
        q    = ex["question"]
        opts = [ex[f"option{i}"] for i in range(1, 5)]
        prompt = (
            q + "\n\n" +
            "\n".join(f"{lbl}. {opt}"
                      for lbl, opt in zip(["A","B","C","D"], opts)) +
            "\n\nAnswer:"
        )
        prompts.append(prompt)
        references.append(label_map.get(ex["answer_id"], -1))

    predictions = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Evaluating ATC Easy"):
        batch = prompts[i : i + batch_size]
        outputs = llm.generate(batch, sampling_params)

        for idx, out in enumerate(outputs):
            txt = out.text.lstrip() if hasattr(out, "text") else str(out).lstrip()
            m = re.search(r"\b([A-D])\b", txt)
            letter = m.group(1) if m else None
            predictions.append(label_map.get(letter, -1))

            if i + idx < debug_samples:
                print(f"\n[DEBUG] Prompt:\n{batch[idx]}")
                print(f"[DEBUG] Output:\n{txt}")
                print(f"[DEBUG] Parsed letter: {letter}")

    acc = accuracy_metric.compute(predictions=predictions, references=references)["accuracy"]
    f1  = f1_metric.compute(predictions=predictions,
                             references=references,
                             average="macro")["f1"]

    return {"accuracy": acc, "macro_f1": f1}


def evaluate_kormedmcqa(
    llm: LLM,
    sampling_params: SamplingParams,
    subjects=("doctor", "nurse", "pharm", "dentist"),
    batch_size: int = 32,
):

    acc_metric = evaluate.load("accuracy")
    f1_metric  = evaluate.load("f1")
    results = {}

    # A→0, B→1, C→2, D→3, E→4
    letter2idx = {chr(ord("A") + i): i for i in range(5)}

    for subj in subjects:
        ds = load_dataset("sean0042/KorMedMCQA", name=subj, split="test")

        prompts = []
        references = []
        for ex in ds:
            question = ex["question"]
            opts = [
                f"A. {ex['A']}",
                f"B. {ex['B']}",
                f"C. {ex['C']}",
                f"D. {ex['D']}",
                f"E. {ex['E']}",
            ]
            prompt = (
                f"문제: {question}\n"
                + "\n".join(opts)
                + "\n정답을 알파벳 하나로 답해주세요."
            )
            prompts.append(prompt)
            references.append(ex["answer"] - 1)

        raw_preds = []
        for i in tqdm(range(0, len(prompts), batch_size), desc=f"Gen {subj}"):
            batch = prompts[i : i + batch_size]
            outputs = llm.generate(batch, sampling_params=sampling_params)
            raw_preds.extend([o.outputs[0].text.strip() for o in outputs])

        pred_indices = []
        for p in raw_preds:
            text = p.strip()
            if not text:
                idx = 0
            else:
                letter = text[0].upper()
                if letter.endswith("."):
                    letter = letter[:-1]
                idx = letter2idx.get(letter, 0)
            pred_indices.append(idx)

        acc = acc_metric.compute(predictions=pred_indices, references=references)["accuracy"]
        f1  = f1_metric.compute(
            predictions=pred_indices,
            references=references,
            average="macro"
        )["f1"]

        results[subj] = {"accuracy": acc, "f1_macro": f1}

    return results

import json
def evaluate_mimic_cxr_summarization(
    llm,              
    tokenizer,        
    sampling_params,   
    batch_size: int = 16,
    buffer: int = 50,
):

    rouge = evaluate.load("rouge", keep_in_memory=True, use_stemmer=True)
    bertscore = evaluate.load("bertscore")

    ds = load_dataset("tgrex6/mimic-cxr-reports-summarization", split="validation")

    prompt_template = (
        "Summarize the following radiology report:\n\n"
        "Background: {background}\n"
        "Findings: {findings}\n\n"
        "Impression:"
    )

    preds, refs = [], []

    raw_max_len = getattr(tokenizer, "model_max_length", None)
    if not isinstance(raw_max_len, int) or raw_max_len <= 0:
        model_max_len = 1024
    else:
        model_max_len = min(raw_max_len, 1024)
    input_max = model_max_len - buffer
    if input_max <= 0:
        raise ValueError(f"buffer ({buffer}) must be smaller than model_max_length ({model_max_len})")

    total = len(ds)
    for start in tqdm(range(0, total, batch_size), desc="Generating summaries"):
        end = min(start + batch_size, total)
        prompts = []
        batch_refs = []

        for idx in range(start, end):
            ex = ds[idx] 
            background = ex["background"]
            findings   = ex["findings"]
            impression = ex["impression"]

            raw = prompt_template.format(
                background=background,
                findings=findings
            )
            tokens = tokenizer(
                raw,
                truncation=True,
                max_length=input_max,
                return_tensors="pt"
            ).input_ids[0]
            truncated = tokenizer.decode(tokens, skip_special_tokens=True)
            prompts.append(truncated)
            batch_refs.append(impression.strip())

        outputs = llm.generate(prompts, sampling_params=sampling_params)

        preds.extend(out.outputs[0].text.strip() for out in outputs)
        refs.extend(batch_refs)

    rouge_results = rouge.compute(predictions=preds, references=refs)
    bert_results = bertscore.compute(predictions=preds, references=refs, lang="en")

    f1_scores = bert_results.get("f1", [])
    mean_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    return {
        "rouge": rouge_results,
        "bertscore": mean_f1
    }


def evaluate_pubmed_summarization(
    llm: LLM,
    tokenizer: PreTrainedTokenizer,
    sampling_params: SamplingParams,
    batch_size: int = 256,
    buffer: int = 10,
):

    if tokenizer.model_max_length is None or tokenizer.model_max_length > 10_000:
        tokenizer.model_max_length = 1024

    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")

    ds = load_dataset("ccdv/pubmed-summarization", split="test")

    all_preds = []
    all_refs  = []

    for i in tqdm(range(0, len(ds), batch_size), desc="Generating summaries"):
        batch = ds[i : i + batch_size]
        prompts = []

        for art in batch["article"]:
            prefix = "Summarize the following article:\n\n"
            prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids[0]
            max_input = tokenizer.model_max_length - len(prefix_ids) - buffer
            max_input = max(max_input, 1)

            input_ids = tokenizer(
                art,
                truncation=True,
                max_length=int(max_input),
                return_tensors="pt"
            ).input_ids[0]
            truncated = tokenizer.decode(input_ids, skip_special_tokens=True)
            prompts.append(f"{prefix}{truncated}\n\nSummary:")

        outputs = llm.generate(prompts, sampling_params=sampling_params)
        preds = [out.outputs[0].text.strip() for out in outputs]

        all_preds.extend(preds)
        all_refs.extend(batch["abstract"])

    filtered_preds = []
    filtered_refs  = []
    for pred, ref in zip(all_preds, all_refs):
        if pred.strip():
            filtered_preds.append(pred)
            filtered_refs.append(ref)

    rouge_scores = rouge.compute(predictions=filtered_preds, references=filtered_refs)
    bertscore_scores = bertscore.compute(
        predictions=filtered_preds, references=filtered_refs, lang="en"
    )
    f1_scores = bertscore_scores.get("f1", [])
    mean_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    print("ROUGE scores:", rouge_scores)
    print(f"Average BERTScore F1: {mean_f1:.6f}")

    return {
        "rouge": rouge_scores,
        "average_bertscore_f1": mean_f1,
        "num_filtered_out": len(all_preds) - len(filtered_preds)
    }

def evaluate_pubmedqa(
    llm,
    sampling_params,
    batch_size: int = 32,
    use_manual: bool = False,
    debug: bool = True
):
    label_map = {"yes": 0, "no": 1, "maybe": 2}

    url = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/test_ground_truth.json"
    resp = requests.get(url); resp.raise_for_status()
    ground_truth = resp.json()
    test_pids = set(map(int, ground_truth.keys()))

    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    test_ds = ds.filter(lambda ex: ex["pubid"] in test_pids)
    prompts, references = [], []
    for ex in test_ds:
        abstract = " ".join(ex["context"]["contexts"])
        prompts.append(
            f"Abstract: {abstract}\n"
            f"Question: {ex['question']}\n\n"
            "Answer only with yes, no, or maybe.\n"
            "Answer:"
        )
        references.append(ex["final_decision"].lower())

    if debug:
        print(f"> total examples: {len(prompts)}")
        print("  distribution of references:", Counter(references))

    predictions = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        outputs = llm.generate(batch, sampling_params)
        for out in outputs:
            text = out.outputs[0].text.strip()
            m = re.search(r"\b(yes|no|maybe)\b", text, flags=re.IGNORECASE)
            if m:
                predictions.append(m.group(1).lower())
            else:
                predictions.append("")

    if debug:
        print(f">predictions: {len(predictions)}")
        print("  sample (pred, ref):", list(zip(predictions, references))[:5])

    pred_ids = [label_map.get(p, -1) for p in predictions]
    ref_ids  = [label_map[r]          for r in references]

    results = {}

    if use_manual:
        correct = sum(p == r for p, r in zip(predictions, references))
        acc = correct / len(references)
        results["accuracy"] = acc
        print(f"▶ PubMedQA Accuracy (manual): {acc:.4f} ({correct}/{len(references)})")

        if f1_score is not None:
            f1 = f1_score(ref_ids, pred_ids, average="macro", labels=[0,1,2])
            results["macro_f1"] = f1
            print(f"▶ PubMedQA Macro-F1 (manual): {f1:.4f}")
        else:
            print("sklearn not installed: Macro-F1 (manual) 계산 불가")
            results["macro_f1"] = None

    else:
        metric_acc = evaluate.load("accuracy")
        acc = metric_acc.compute(predictions=pred_ids, references=ref_ids)["accuracy"]
        results["accuracy"] = acc
        print(f"▶ PubMedQA Accuracy (mapped): {acc:.4f}")

        metric_f1 = evaluate.load("f1")
        f1 = metric_f1.compute(
            predictions=pred_ids,
            references=ref_ids,
            average="macro"
        )["f1"]
        results["macro_f1"] = f1
        print(f"▶ PubMedQA Macro-F1 (mapped): {f1:.4f}")

    return results

def evaluate_medmcqa(
    llm,                    
    sampling_params,       
    num_samples: int = None   
):

    split_str = (
        "validation" if num_samples is None
        else f"validation[:{num_samples}]"
    )
    medmcqa = load_dataset("openlifescienceai/medmcqa", split=split_str)
    accuracy = load_metric("accuracy")
    f1_macro = load_metric("f1")

    labels = ["a", "b", "c", "d"]
    predictions = []
    references  = []

    for ex in medmcqa:
        opts     = [ex["opa"], ex["opb"], ex["opc"], ex["opd"]]
        true_idx = int(ex["cop"])

        prompt = ex["question"] + "\n"
        for lbl, opt in zip(labels, opts):
            prompt += f"{lbl.upper()}. {opt}\n"
        prompt += "Answer (letter):"

        outputs = llm.generate([prompt], sampling_params=sampling_params)
        raw     = outputs[0].outputs[0].text.strip()
        token   = raw.split()[0].lower() if raw.split() else ""

        if token in labels:
            pred_idx = labels.index(token)
        else:
            pred_idx = (true_idx + 1) % 4

        predictions.append(pred_idx)
        references.append(true_idx)

    acc_result = accuracy.compute(predictions=predictions, references=references)
    f1_result  = f1_macro.compute(
        predictions=predictions,
        references=references,
        average="macro"
    )

    print(f"MEDMCQA Accuracy: {acc_result['accuracy']:.4f}")
    print(f"MEDMCQA Macro-F1: {f1_result['f1']:.4f}")

    return {"accuracy": acc_result['accuracy'], "macro_f1": f1_result['f1']}


def evaluate_medqa(llm, sampling_params):
    acc_metric = evaluate.load("accuracy")
    f1_metric  = evaluate.load("f1")

    medqa = load_dataset(
        "bigbio/med_qa",
        "med_qa_en_4options_source",
        split="test"
    )

    prompts     = []
    references  = []
    for ex in medqa:
        opt_str = "\n".join(f"{opt['key']}. {opt['value']}" for opt in ex["options"])
        prompts.append(
            f"Question: {ex['question']}\n"
            f"Options:\n{opt_str}\n"
            "Answer (letter):"
        )
        answer_letter = ex["answer_idx"].upper()
        references.append(ord(answer_letter) - ord("A"))
    outputs = llm.generate(prompts, sampling_params)
    predictions = []
    for out in outputs:
        text = out.outputs[0].text.strip()
        m = re.search(r"\b[A-D]\b", text, flags=re.IGNORECASE)
        if m:
            pred_letter = m.group(0).upper()
        else:
            tok = text.split()
            pred_letter = tok[0][0].upper() if tok else "A"
        predictions.append(ord(pred_letter) - ord("A"))
    acc_res = acc_metric.compute(predictions=predictions, references=references)
    acc = acc_res["accuracy"]
    print(f"▶ MedQA Accuracy: {acc:.4f}")
    f1_res = f1_metric.compute(
        predictions=predictions,
        references=references,
        average="macro"
    )
    macro_f1 = f1_res["f1"]
    print(f"▶ MedQA Macro-F1: {macro_f1:.4f}")

    return {"accuracy": acc, "macro_f1": macro_f1}

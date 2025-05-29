import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
from huggingface_hub import login
from pathlib import Path


def process_dataset(
    input_path: str,
    rename_map: dict,
    text_col: str,
    output_path: str,
    system_prompt: str,
    llm: LLM,
    sampling_params: SamplingParams,
    batch_size: int
):

    df = pd.read_csv(input_path)
    df.rename(columns=rename_map, inplace=True)
    llm_col = 't:LLM_clinical_note'
    df[llm_col] = pd.NA

    mask_valid = df[text_col].notna()
    df.loc[~mask_valid, llm_col] = 'no clinical note'
    valid_indices = df.index[mask_valid]

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    for start in tqdm(
        range(0, len(valid_indices), batch_size),
        desc=f"Processing {Path(input_path).stem}", unit="batch"
    ):
        batch_idx = valid_indices[start : start + batch_size]
        prompts = df.loc[batch_idx, text_col].tolist()
        conversations = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": prompt},
            ]
            for prompt in prompts
        ]

        outputs = llm.chat(
            messages=conversations,
            sampling_params=sampling_params,
            use_tqdm=False
        )
        completions = [out.outputs[0].text for out in outputs]
        df.loc[batch_idx, llm_col] = completions

    df.to_csv(output_path, index=False)


def main():
    system_prompt_path = Path('./sys_prompt1.txt')
    system_prompt = system_prompt_path.read_text(encoding='utf-8')

    login(token='')

    # Initialize LLM
    llm = LLM(
        model="google/gemma-3-27b-it",
        dtype="bfloat16",
        tensor_parallel_size=4,
        gpu_memory_utilization=0.9
    )

    # Define sampling parameters
    sampling_params = SamplingParams(
        temperature=0.2,
        top_p=0.95,
        max_tokens=2048,
        repetition_penalty=1.0
    )
    batch_size = 128

    # Dataset configurations
    configs = [
        {
            "input_path": "./mimic4/sepsis_final_data_withTimes.csv",
            "rename_map": {"m:text": "t:text"},
            "text_col": "t:text",
            "output_path": "./mimic4/df_mimic4_prompt1_gemma27b.csv",
        },
        {
            "input_path": "./mimic3/sepsis_final_data_withTimes.csv",
            "rename_map": {"m:clinical_note": "t:clinical_note"},
            "text_col": "t:clinical_note",
            "output_path": "./mimic3/df_mimic3_prompt1_gemma27B.csv",
        },
        {
            "input_path": "./pd/sepsis_final_data_withTimes.csv",
            "rename_map": {"m:clinical_note": "t:clinical_note"},
            "text_col": "t:clinical_note",
            "output_path": "./pd/df_pd_prompt1_gemma27B.csv",
        },
    ]

    for cfg in configs:
        process_dataset(
            input_path=cfg["input_path"],
            rename_map=cfg["rename_map"],
            text_col=cfg["text_col"],
            output_path=cfg["output_path"],
            system_prompt=system_prompt,
            llm=llm,
            sampling_params=sampling_params,
            batch_size=batch_size
        )


if __name__ == "__main__":
    main()

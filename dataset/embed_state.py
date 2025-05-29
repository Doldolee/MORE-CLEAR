import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForImageTextToText, AutoProcessor
from tqdm import tqdm
from huggingface_hub import login
from util import impute_notes, impute_next_notes, stack_notes, stack_next_notes, impute_notes_with_background, impute_next_notes_with_background, bg_stack_note, bg_stack_next_note,extract_background_next_notes, extract_background_states 


os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

login(token='')

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
BATCH_SIZE = 16
MAX_LENGTH = 2048

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModel.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

def embed_with_masked_mean_pool(
    texts: list[str],
    batch_size: int,
    max_length: int,
    save_path: str
) -> np.ndarray:
    
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Feature-extraction"):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state  # (B, L, D)

        mask = inputs.attention_mask.unsqueeze(-1)  # (B, L, 1)
        sum_emb = (last_hidden * mask).sum(dim=1)   # (B, D)
        lengths = mask.sum(dim=1)                  # (B, 1)
        mean_pool = sum_emb / lengths              # (B, D)

        all_embs.extend(mean_pool.cpu().numpy())

    embs = np.vstack(all_embs)  # (N, D)
    np.save(save_path, embs)
    print(f"Saved masked avg-pooled embeddings to {save_path}, shape = {embs.shape}")
    return embs

if __name__ == '__main__':
    TARGET_DATA = 'mimic3'
    TARGET_EMBEDDING = 'buffer_llama'

    test_note = np.load(f'./{TARGET_DATA}/{TARGET_EMBEDDING}/test_note.npy', allow_pickle=True)
    test_next_note = np.load(f'./{TARGET_DATA}/{TARGET_EMBEDDING}/test_next_note.npy', allow_pickle=True)
    train_note = np.load(f'./{TARGET_DATA}/{TARGET_EMBEDDING}/train_val_note.npy', allow_pickle=True)
    train_next_note = np.load(f'./{TARGET_DATA}/{TARGET_EMBEDDING}/train_val_next_note.npy', allow_pickle=True)
    test_done = np.load(f"./{TARGET_DATA}/{TARGET_EMBEDDING}/test_done.npy")
    train_done = np.load(f"./{TARGET_DATA}/{TARGET_EMBEDDING}/train_val_done.npy")

    test_note = test_note.squeeze(axis=1).tolist()  
    test_next_note = test_next_note.squeeze(axis=1).tolist() 
    train_note = train_note.squeeze(axis=1).tolist()  
    train_next_note = train_next_note.squeeze(axis=1).tolist()
    test_done = test_done.squeeze(axis=1).tolist()
    train_done = train_done.squeeze(axis=1).tolist()

    test_note_only_bg      = extract_background_states(test_note, test_done)
    test_next_note_only_bg = extract_background_next_notes(test_next_note, test_note, test_done)
    train_note_bg_only_bg = extract_background_states(train_note, train_done)
    train_next_note_only_bg = extract_background_next_notes(train_next_note, train_note, train_done)

    dummy_dict = {
               f"./{TARGET_DATA}/{TARGET_EMBEDDING}/test_impute_bg_only_note_embedding.npy": test_note_only_bg,
               f"./{TARGET_DATA}/{TARGET_EMBEDDING}/test_impute_bg_only_next_note_embedding.npy": test_next_note_only_bg,
               f"./{TARGET_DATA}/{TARGET_EMBEDDING}/train_val_impute_bg_only_note_embedding.npy": train_note_bg_only_bg,
              f"./{TARGET_DATA}/{TARGET_EMBEDDING}/train_val_impute_bg_only_next_note_embedding.npy": train_next_note_only_bg
              }
    
    for path, texts in dummy_dict.items():
        embed_with_masked_mean_pool(texts, BATCH_SIZE, MAX_LENGTH, path)

    test_note_sacled_mimic4 = np.load(f'./{TARGET_DATA}/{TARGET_EMBEDDING}/scaled_mimic4_test_note.npy', allow_pickle=True)
    test_next_note_scaled_mimic4 = np.load(f'./{TARGET_DATA}/{TARGET_EMBEDDING}/scaled_mimic4_test_next_note.npy', allow_pickle=True)
    test_note_scaled_pd = np.load(f'./{TARGET_DATA}/{TARGET_EMBEDDING}/scaled_pd_test_note.npy', allow_pickle=True)
    test_next_note_scaled_pd = np.load(f'./{TARGET_DATA}/{TARGET_EMBEDDING}/scaled_pd_test_next_note.npy', allow_pickle=True)
    test_done = np.load(f"./{TARGET_DATA}/{TARGET_EMBEDDING}/scaled_mimic4_test_done.npy")

    test_note_sacled_mimic4 = test_note_sacled_mimic4.squeeze(axis=1).tolist()  
    test_next_note_scaled_mimic4 = test_next_note_scaled_mimic4.squeeze(axis=1).tolist()  
    test_note_scaled_pd = test_note_scaled_pd.squeeze(axis=1).tolist()  
    test_next_note_scaled_pd = test_next_note_scaled_pd.squeeze(axis=1).tolist() 

    test_note_sacled_mimic4_bg_only      = extract_background_states(test_note_sacled_mimic4, test_done)
    test_next_note_scaled_mimic4_bg_only = extract_background_next_notes(test_next_note_scaled_mimic4, test_note_sacled_mimic4, test_done)
    test_note_scaled_pd_bg_only = extract_background_states(test_note_scaled_pd, test_done)
    test_next_note_scaled_pd_bg_only = extract_background_next_notes(test_next_note_scaled_pd, test_note_scaled_pd, test_done)


    dummy_dict = {
                    f"./{TARGET_DATA}/{TARGET_EMBEDDING}/scaled_mimic4_test_impute_bg_only_note_embedding.npy": test_note_sacled_mimic4_bg_only,
                    f"./{TARGET_DATA}/{TARGET_EMBEDDING}/scaled_mimic4_test_impute_bg_only_next_note_embedding.npy": test_next_note_scaled_mimic4_bg_only,
                    f"./{TARGET_DATA}/{TARGET_EMBEDDING}/scaled_pd_test_impute_bg_only_note_embedding.npy": test_note_scaled_pd_bg_only,
                    f"./{TARGET_DATA}/{TARGET_EMBEDDING}/scaled_pd_test_impute_bg_only_next_note_embedding.npy": test_next_note_scaled_pd_bg_only
                    }

    for path, texts in dummy_dict.items():
        embed_with_masked_mean_pool(texts, BATCH_SIZE, MAX_LENGTH, path)
    
    dummy_dict = {
                f"./{TARGET_DATA}/{TARGET_EMBEDDING}/scaled_mimic4_test_note_embedding.npy": test_note_sacled_mimic4,
                f"./{TARGET_DATA}/{TARGET_EMBEDDING}/scaled_mimic4_test_next_note_embedding.npy": test_next_note_scaled_mimic4,
                f"./{TARGET_DATA}/{TARGET_EMBEDDING}/scaled_pd_test_note_embedding.npy": test_note_scaled_pd,
                f"./{TARGET_DATA}/{TARGET_EMBEDDING}/scaled_pd_test_next_note_embedding.npy": test_next_note_scaled_pd}

    for path, texts in dummy_dict.items():
        embed_with_masked_mean_pool(texts, BATCH_SIZE, MAX_LENGTH, path)

    #######################################################################################################################################
    TARGET_DATA = 'mimic4'

    test_note = np.load(f'./{TARGET_DATA}/{TARGET_EMBEDDING}/test_note.npy', allow_pickle=True)
    test_next_note = np.load(f'./{TARGET_DATA}/{TARGET_EMBEDDING}/test_next_note.npy', allow_pickle=True)
    train_note = np.load(f'./{TARGET_DATA}/{TARGET_EMBEDDING}/train_val_note.npy', allow_pickle=True)
    train_next_note = np.load(f'./{TARGET_DATA}/{TARGET_EMBEDDING}/train_val_next_note.npy', allow_pickle=True)
    test_done = np.load(f"./{TARGET_DATA}/{TARGET_EMBEDDING}/test_done.npy")
    train_done = np.load(f"./{TARGET_DATA}/{TARGET_EMBEDDING}/train_val_done.npy")

    test_note = test_note.squeeze(axis=1).tolist()  
    test_next_note = test_next_note.squeeze(axis=1).tolist()  
    train_note = train_note.squeeze(axis=1).tolist()  
    train_next_note = train_next_note.squeeze(axis=1).tolist() 
    test_done = test_done.squeeze(axis=1).tolist()
    train_done = train_done.squeeze(axis=1).tolist()


    test_note_only_bg      = extract_background_states(test_note, test_done)
    test_next_note_only_bg = extract_background_next_notes(test_next_note, test_note, test_done)
    train_note_bg_only_bg = extract_background_states(train_note, train_done)
    train_next_note_only_bg = extract_background_next_notes(train_next_note, train_note, train_done)

    dummy_dict = {
               f"./{TARGET_DATA}/{TARGET_EMBEDDING}/test_impute_bg_only_note_embedding.npy": test_note_only_bg,
               f"./{TARGET_DATA}/{TARGET_EMBEDDING}/test_impute_bg_only_next_note_embedding.npy": test_next_note_only_bg,
               f"./{TARGET_DATA}/{TARGET_EMBEDDING}/train_val_impute_bg_only_note_embedding.npy": train_note_bg_only_bg,
              f"./{TARGET_DATA}/{TARGET_EMBEDDING}/train_val_impute_bg_only_next_note_embedding.npy": train_next_note_only_bg
              }
    for path, texts in dummy_dict.items():
        embed_with_masked_mean_pool(texts, BATCH_SIZE, MAX_LENGTH, path)
    
    ################################################################################################
    test_note_sacled_mimic4 = np.load(f'./{TARGET_DATA}/{TARGET_EMBEDDING}/scaled_mimic4_test_note.npy', allow_pickle=True)
    test_next_note_scaled_mimic4 = np.load(f'./{TARGET_DATA}/{TARGET_EMBEDDING}/scaled_mimic4_test_next_note.npy', allow_pickle=True)
    test_note_scaled_pd = np.load(f'./{TARGET_DATA}/{TARGET_EMBEDDING}/scaled_pd_test_note.npy', allow_pickle=True)
    test_next_note_scaled_pd = np.load(f'./{TARGET_DATA}/{TARGET_EMBEDDING}/scaled_pd_test_next_note.npy', allow_pickle=True)
    test_done = np.load(f"./{TARGET_DATA}/{TARGET_EMBEDDING}/scaled_mimic4_test_done.npy")

    test_note_sacled_mimic4 = test_note_sacled_mimic4.squeeze(axis=1).tolist()  
    test_next_note_scaled_mimic4 = test_next_note_scaled_mimic4.squeeze(axis=1).tolist()  
    test_note_scaled_pd = test_note_scaled_pd.squeeze(axis=1).tolist()  
    test_next_note_scaled_pd = test_next_note_scaled_pd.squeeze(axis=1).tolist() 

    test_note_sacled_mimic4_bg_only      = extract_background_states(test_note_sacled_mimic4, test_done)
    test_next_note_scaled_mimic4_bg_only = extract_background_next_notes(test_next_note_scaled_mimic4, test_note_sacled_mimic4, test_done)
    test_note_scaled_pd_bg_only = extract_background_states(test_note_scaled_pd, test_done)
    test_next_note_scaled_pd_bg_only = extract_background_next_notes(test_next_note_scaled_pd, test_note_scaled_pd, test_done)

    dummy_dict = {
                    f"./{TARGET_DATA}/{TARGET_EMBEDDING}/scaled_mimic4_test_impute_bg_only_note_embedding.npy": test_note_sacled_mimic4_bg_only,
                    f"./{TARGET_DATA}/{TARGET_EMBEDDING}/scaled_mimic4_test_impute_bg_only_next_note_embedding.npy": test_next_note_scaled_mimic4_bg_only,
                    f"./{TARGET_DATA}/{TARGET_EMBEDDING}/scaled_pd_test_impute_bg_only_note_embedding.npy": test_note_scaled_pd_bg_only,
                    f"./{TARGET_DATA}/{TARGET_EMBEDDING}/scaled_pd_test_impute_bg_only_next_note_embedding.npy": test_next_note_scaled_pd_bg_only
                    }
    for path, texts in dummy_dict.items():
        embed_with_masked_mean_pool(texts, BATCH_SIZE, MAX_LENGTH, path)
    
    dummy_dict = {
                f"./{TARGET_DATA}/{TARGET_EMBEDDING}/scaled_mimic4_test_note_embedding.npy": test_note_sacled_mimic4,
                f"./{TARGET_DATA}/{TARGET_EMBEDDING}/scaled_mimic4_test_next_note_embedding.npy": test_next_note_scaled_mimic4,
                f"./{TARGET_DATA}/{TARGET_EMBEDDING}/scaled_pd_test_note_embedding.npy": test_note_scaled_pd,
                f"./{TARGET_DATA}/{TARGET_EMBEDDING}/scaled_pd_test_next_note_embedding.npy": test_next_note_scaled_pd}

    for path, texts in dummy_dict.items():
        embed_with_masked_mean_pool(texts, BATCH_SIZE, MAX_LENGTH, path)

    
    TARGET_DATA = 'pd'
 

    test_note = np.load(f'./{TARGET_DATA}/{TARGET_EMBEDDING}/test_note.npy', allow_pickle=True)
    test_next_note = np.load(f'./{TARGET_DATA}/{TARGET_EMBEDDING}/test_next_note.npy', allow_pickle=True)
    train_note = np.load(f'./{TARGET_DATA}/{TARGET_EMBEDDING}/train_val_note.npy', allow_pickle=True)
    train_next_note = np.load(f'./{TARGET_DATA}/{TARGET_EMBEDDING}/train_val_next_note.npy', allow_pickle=True)
    test_done = np.load(f"./{TARGET_DATA}/{TARGET_EMBEDDING}/test_done.npy")
    train_done = np.load(f"./{TARGET_DATA}/{TARGET_EMBEDDING}/train_val_done.npy")

    test_note = test_note.squeeze(axis=1).tolist() 
    test_next_note = test_next_note.squeeze(axis=1).tolist()  
    train_note = train_note.squeeze(axis=1).tolist() 
    train_next_note = train_next_note.squeeze(axis=1).tolist()  
    test_done = test_done.squeeze(axis=1).tolist()
    train_done = train_done.squeeze(axis=1).tolist()


    test_note_only_bg      = extract_background_states(test_note, test_done)
    test_next_note_only_bg = extract_background_next_notes(test_next_note, test_note, test_done)
    train_note_bg_only_bg = extract_background_states(train_note, train_done)
    train_next_note_only_bg = extract_background_next_notes(train_next_note, train_note, train_done)

    dummy_dict = {
               f"./{TARGET_DATA}/{TARGET_EMBEDDING}/test_impute_bg_only_note_embedding.npy": test_note_only_bg,
               f"./{TARGET_DATA}/{TARGET_EMBEDDING}/test_impute_bg_only_next_note_embedding.npy": test_next_note_only_bg,
               f"./{TARGET_DATA}/{TARGET_EMBEDDING}/train_val_impute_bg_only_note_embedding.npy": train_note_bg_only_bg,
              f"./{TARGET_DATA}/{TARGET_EMBEDDING}/train_val_impute_bg_only_next_note_embedding.npy": train_next_note_only_bg
              }

    for path, texts in dummy_dict.items():
        embed_with_masked_mean_pool(texts, BATCH_SIZE, MAX_LENGTH, path)

    ################################################################################################
    test_note_sacled_mimic4 = np.load(f'./{TARGET_DATA}/{TARGET_EMBEDDING}/scaled_mimic4_test_note.npy', allow_pickle=True)
    test_next_note_scaled_mimic4 = np.load(f'./{TARGET_DATA}/{TARGET_EMBEDDING}/scaled_mimic4_test_next_note.npy', allow_pickle=True)
    test_note_scaled_pd = np.load(f'./{TARGET_DATA}/{TARGET_EMBEDDING}/scaled_pd_test_note.npy', allow_pickle=True)
    test_next_note_scaled_pd = np.load(f'./{TARGET_DATA}/{TARGET_EMBEDDING}/scaled_pd_test_next_note.npy', allow_pickle=True)
    test_done = np.load(f"./{TARGET_DATA}/{TARGET_EMBEDDING}/scaled_mimic4_test_done.npy")

    test_note_sacled_mimic4 = test_note_sacled_mimic4.squeeze(axis=1).tolist()  
    test_next_note_scaled_mimic4 = test_next_note_scaled_mimic4.squeeze(axis=1).tolist()  
    test_note_scaled_pd = test_note_scaled_pd.squeeze(axis=1).tolist()  
    test_next_note_scaled_pd = test_next_note_scaled_pd.squeeze(axis=1).tolist() 

    test_note_sacled_mimic4_bg_only      = extract_background_states(test_note_sacled_mimic4, test_done)
    test_next_note_scaled_mimic4_bg_only = extract_background_next_notes(test_next_note_scaled_mimic4, test_note_sacled_mimic4, test_done)
    test_note_scaled_pd_bg_only = extract_background_states(test_note_scaled_pd, test_done)
    test_next_note_scaled_pd_bg_only = extract_background_next_notes(test_next_note_scaled_pd, test_note_scaled_pd, test_done)

    dummy_dict = {
                    f"./{TARGET_DATA}/{TARGET_EMBEDDING}/scaled_mimic4_test_impute_bg_only_note_embedding.npy": test_note_sacled_mimic4_bg_only,
                    f"./{TARGET_DATA}/{TARGET_EMBEDDING}/scaled_mimic4_test_impute_bg_only_next_note_embedding.npy": test_next_note_scaled_mimic4_bg_only,
                    f"./{TARGET_DATA}/{TARGET_EMBEDDING}/scaled_pd_test_impute_bg_only_note_embedding.npy": test_note_scaled_pd_bg_only,
                    f"./{TARGET_DATA}/{TARGET_EMBEDDING}/scaled_pd_test_impute_bg_only_next_note_embedding.npy": test_next_note_scaled_pd_bg_only
                    }

    for path, texts in dummy_dict.items():
        embed_with_masked_mean_pool(texts, BATCH_SIZE, MAX_LENGTH, path)
    
    dummy_dict = {
                f"./{TARGET_DATA}/{TARGET_EMBEDDING}/scaled_mimic4_test_note_embedding.npy": test_note_sacled_mimic4,
                f"./{TARGET_DATA}/{TARGET_EMBEDDING}/scaled_mimic4_test_next_note_embedding.npy": test_next_note_scaled_mimic4,
                f"./{TARGET_DATA}/{TARGET_EMBEDDING}/scaled_pd_test_note_embedding.npy": test_note_scaled_pd,
                f"./{TARGET_DATA}/{TARGET_EMBEDDING}/scaled_pd_test_next_note_embedding.npy": test_next_note_scaled_pd
                }

    for path, texts in dummy_dict.items():
        embed_with_masked_mean_pool(texts, BATCH_SIZE, MAX_LENGTH, path)

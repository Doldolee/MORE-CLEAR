# MORE-CLEAR

**Multimodal Offline Reinforcement Learning for Clinical Notes Leveraged Enhanced State Representation**

This repository contains the implementation of the **MORE-CLEAR** framework. It provides:

- Offline RL training and evaluation code  
- Scripts for clinical note summarization using LLMs  
- State embedding utilities for clinical note data 


## Data Availability
Public datasets (excluding private data) used to develop and validate MORE-CELAR are available through [PhysioNet](https://physionet.org/about/database).
Preprocessing scripts for MIMIC data can be found in :

- [AI Clinician (MIMIC-III)](https://github.com/matthieukomorowski/AI_Clinician)  
- [AI Clinician (MIMIC-IV)](https://github.com/cmudig/AI-Clinician-MIMICIV)  

Clone those repositories and save the generated datasets into the project's 'dataset/mimic3' and 'dataset/mimic4' folders

## Setup

1. **Clone the repository**  
```bash
git clone https://github.com/yourusername/MORE-CLEAR.git
cd MORE-CLEAR
```

2. **Data Preprocessing**
Run the scripts in these repositories to preprocess MIMIC data.

- [AI Clinician (MIMIC-III)](https://github.com/matthieukomorowski/AI_Clinician)  
- [AI Clinician (MIMIC-IV)](https://github.com/cmudig/AI-Clinician-MIMICIV)  

2. **Clinical Note Summarization**
Use the summarization script to generate LLM-based note summaries:

```python
python ./dataset/summarize_llm.py
```

3. **State Embedding**
Generate clinical note state embeddings:
```python
python ./dataset/embed_state.py
```

4. **Training**
Train the MORE-CLEAR RL agent:
```python
python main.py
```
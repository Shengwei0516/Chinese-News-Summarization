# Chinese News Summarization

## Overview

This repository contains the code and resources for the Natural Language Generation project as part of the Applied Deep Learning course at National Taiwan University (NTU) in the Fall semester of 2023. The project focuses on Chinese News Summarization using deep learning techniques.

## Description

The task involves generating concise and coherent summaries of Chinese news articles using advanced deep learning techniques. The mT5 model, which is pre-trained on a large corpus of multilingual text, is utilized to accomplish this task.

## Requirements

- **Python 3.9 and Python Standard Library**
- **PyTorch 2.1.0**
- **transformers (<=4.35.0.dev0)**
- **datasets==2.14.5**
- **accelerate==0.23.0**
- **sentencepiece==0.1.99**
- **evaluate==0.4.0**
- **rouge==1.0.1**
- **spacy==3.7.1**
- **nltk==3.8.1**
- **ckiptagger==0.2.1**
- **tqdm==4.66.1**
- **pandas==2.1.1**
- **jsonlines==4.0.0**
- **protobuf**
- **matplotlib**
- **tw_rouge**

## Usage
1. Clone the repository to your local machine.

2. Download the pre-trained google/mt5-small model and place it in the appropriate directory.

3. Train, Predict, and Plot figures:
```bash
python main.py \
  --train_file ./data/train.jsonl \
  --validation_file ./data/public.jsonl \
  --test_file ./data/sample_test.jsonl \
  --model_name_or_path google/mt5-small \
  --max_length 2048 \
  --batch_size 1 \
  --learning_rate 5e-5 \
  --epochs 30 \
  --gradient_accumulation_steps 8 \
  --stratgy greedy \
  --output_dir /tmp/mt5-small \
  --seed 11151033 \
  --prediction_path ./output.jsonl \
  --gen_max_length 128
```

**Only Train**
```bash
python main.py \
  --train_file ./data/train.jsonl \
  --model_name_or_path google/mt5-small \
  --max_length 2048 \
  --batch_size 1 \
  --learning_rate 5e-5 \
  --epochs 30 \
  --gradient_accumulation_steps 8 \
  --output_dir /tmp/mt5-small \
  --seed 11151033 
```

## Example
To provide a quick start, here's an example of how to generate summaries using the provided code:
```bash
python main.py \
  --test_file ./data/public.jsonl \
  --model_name_or_path mt5-small \
  --max_length 2048 \
  --batch_size 8 \
  --stratgy greedy \
  --seed 11151033 \
  --prediction_path ./output.jsonl \
  --gen_max_length 128
```

## Notes
- Ensure your input file contains Chinese news articles in a readable format.
- The quality of the generated summaries may vary based on the input data and the complexity of the news articles.

Feel free to explore the code and customize the parameters according to your specific use case.

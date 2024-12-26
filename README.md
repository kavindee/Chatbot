# Fine-Tuning FLAN-T5 for Fitness Question-Answering Chatbot

This repository showcases the fine-tuning of the `google/flan-t5-large` model using LoRA (Low-Rank Adaptation) to create a fitness-related question-answering chatbot. The project leverages Hugging Face's `transformers` library and parameter-efficient training techniques for efficient and scalable model customization.

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Setup](#setup)
- [Training Pipeline](#training-pipeline)
- [Features](#features)
- [Results](#results)
- [Deployment](#deployment)
- [Acknowledgments](#acknowledgments)

---

## Overview

This project demonstrates how to:
- Preprocess a fitness Q&A dataset.
- Implement tokenization and data preparation.
- Fine-tune the FLAN-T5 model with LoRA for memory-efficient training.
- Deploy the fine-tuned model to the Hugging Face Hub for use as a chatbot API.

---

## Dataset

The dataset used is [its-myrto/fitness-question-answers](https://huggingface.co/datasets/its-myrto/fitness-question-answers). It consists of fitness-related questions and their corresponding answers, divided into training and testing sets.

---

## Setup

### Requirements

Install the required Python libraries:
```bash
pip install transformers datasets peft pandas scikit-learn huggingface_hub
```

---

## Training Pipeline

### Steps:
1. **Dataset Loading and Splitting**:
   - Load the dataset using the `datasets` library.
   - Split into training and testing sets with an 80-20 ratio.

2. **Preprocessing**:
   - Tokenize questions and answers.
   - Pad and truncate sequences to a fixed length.

3. **Model Configuration**:
   - Use `google/flan-t5-large` as the base model.
   - Integrate LoRA for efficient parameter fine-tuning.

4. **Training**:
   - Use `Seq2SeqTrainer` from Hugging Face's `transformers` library.
   - Train the model with custom arguments, including batch size, learning rate, and epochs.

5. **Model Saving and Deployment**:
   - Save the fine-tuned model and tokenizer locally.
   - Push the model to the Hugging Face Hub for API-based deployment.

---

## Features

- **Base Model**: FLAN-T5 Large.
- **Fine-Tuning**: Low-rank adaptation (LoRA) for memory and computational efficiency.
- **Tokenization**: Handles sequence padding and truncation effectively.
- **Deployment**: Easily deployable via the Hugging Face Hub.

---

## Results

The fine-tuned model achieves robust performance in generating contextually accurate and coherent answers to fitness-related queries.

---

## Deployment

The model is pushed to the Hugging Face Hub for easy access. You can load and test the model using the following snippet:

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_id = "your-huggingface-username/lora-flan-t5-large-chat"
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

def get_answer(question):
    inputs = tokenizer(question, return_tensors="pt")
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example Usage
question = "What is the best exercise for weight loss?"
answer = get_answer(question)
print("Answer:", answer)
```

---

## Acknowledgments

- Hugging Face for the `transformers` and `datasets` libraries.
- Dataset creator for the [Fitness Question-Answers dataset](https://huggingface.co/datasets/its-myrto/fitness-question-answers).
- Inspiration from parameter-efficient fine-tuning methods like LoRA.

---

# Multi-label-Emotion-Detection
<p align="center">
  <img src="assets/bridge-logo.png" alt="BRIGHTER Logo" style="width: 100%; max-width: 800px;"/>
</p>


This repository contains the code and resources for the **Track A** of the task 11, Bridging the Gap in Text-based  Emotion Detection in the competition of SemEval 2025. The goal of this task is to predict the emotions perceived in a given text snippet. Specifically, the objective is to classify the presence of the following emotions:

- Joy
- Sadness
- Fear
- Anger
- Surprise
- Disgust (for certain languages)

For each text, you need to predict a binary label for each emotion. A label of `1` indicates that the emotion is present, while `0` indicates it is absent.

## Table of Contents

- [Data](#data)
- [Model](#model)
- [Training](#training)
- [Evaluation](#evaluation)
- [Requirements](#requirements)
- [License](#license)

## 📚 Dataset

The BRIGHTER dataset supports emotion detection in multiple languages. For this project, we focus on the **English** language.
You can find the official dataset and task details on the [BRIGHTER dataset page](https://brighter-dataset.github.io/).

## Data

The dataset for this task is provided in a CSV format. The training data consists of text snippets labeled with one or more of the emotions listed above. The format of the data is as follows:

- `id`: Unique identifier for each sample
- `text`: Text snippet (e.g., a sentence or a short paragraph)
- `joy`, `sadness`, `fear`, `anger`, `surprise`, `disgust`: Binary labels indicating the presence of each emotion

Example:

| id      | text                          | joy | sadness | fear | anger | surprise | disgust |
|---------|-------------------------------|-----|---------|------|-------|----------|---------|
| sample_01 | "I am so happy today!"         | 1   | 0       | 0    | 0     | 0        | 0       |
| sample_02 | "This is such a sad day."      | 0   | 1       | 0    | 0     | 0        | 0       |
| sample_03 | "I'm terrified!"               | 0   | 0       | 1    | 0     | 0        | 0       |

## Model

The model is built using [Hugging Face Transformers](https://huggingface.co/transformers/), specifically fine-tuned on pre-trained models like **DistilBERT** or **DeBERTa**. The model architecture is configured to handle multi-label classification, predicting each emotion individually.

### Steps to fine-tune the model:

1. **Data Preprocessing**: The text data is tokenized using the Hugging Face tokenizer, and labels are binarized.
2. **Model Training**: A model for multi-label classification is trained using binary cross-entropy loss.
3. **Evaluation**: The model's performance is evaluated using metrics like F1 score.

## Training

To train the model, follow these steps:

1. Install the dependencies using the provided `requirements.txt`.
2. Prepare the dataset by running the preprocessing script or use the notebook for guidance.
3. Run the training script to fine-tune the model on the training data.
   
##  Evaluation

The performance of the submitted systems will be evaluated based on the following metric:

- **F1-Macro**: Computed over the predicted and gold labels for multi-label emotion classification.
# Install dependencies
pip install -r requirements.txt

# Preprocess the data (tokenize and prepare labels)
python data_preprocessing.py

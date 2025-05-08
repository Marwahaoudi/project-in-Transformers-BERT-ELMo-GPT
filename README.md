# Project: Transformers — BERT, ELMo, GPT, etc.

## 🎬 Project Presentation

Click the image below to watch the project demo video:

[![Watch the video](https://drive.google.com/file/d/1d-zuojdT7XRtypyiHstQ5ywt59qY8L6h/view)


## Overview

This project focuses on the study and experimentation of Transformer-based language models such as  **BERT**, **ELMo**, and **GPT**. These models are widely used in Natural Language Processing (NLP) tasks like:

- Sentiment analysis  
- Text classification  
- Text generation  
- Contextual word embeddings

## Project Members

- **Hanane Saidi**
- **Marwa Haoudi**

## Technologies Used

- Python 3.12 
- Jupyter Notebook (via Anaconda)  
- transformers library by Hugging Face  
- torch (PyTorch)  
- git for project cloning and version control  

## Environment Setup

### 1. Install Anaconda

Download and install Anaconda from [https://www.anaconda.com](https://www.anaconda.com)  
Launch **Jupyter Notebook** via Anaconda Navigator.

### 2. Clone the Project from GitHub

Make sure git is installed.  
In your terminal or Anaconda Prompt:

```bash
git clone https://github.com/Marwahaoudi/project-in-Transformers-BERT-ELMo-GPT-etc.git
cd project-in-Transformers-BERT-ELMo-GPT-etc.git
```
### 3. Create and Activate a Conda Environment
 
```bash
conda create -nd2l_env python= 3.12.7
conda activate d2l_env
```


### 4. Install Required Libraries
 ```bash
!pip install transformers scikit-learn matplotlib
```
### 5. Launch Jupyter Notebook
 ```bash
jupyter notebook
```
## Learning Objectives
- Understand the differences between BERT, ELMo, and GPT
- Apply pre-trained models to real-world NLP tasks
- Work with professional NLP tools like Hugging Face
- Ensure reproducibility using Git, Anaconda, and Jupyter Notebook

## Implementation Details

### BERT (Bidirectional Encoder Representations from Transformers)
- Task: Sentiment Analysis (Binary Classification: positive / negative)
- Model Used: bert-base-uncased (via Hugging Face Transformers)
#### Steps
1. Tokenization
Loaded the pre-trained BERT tokenizer (bert-base-uncased) to convert raw text into input tokens.

2. Dataset Preparation
Built a custom PyTorch dataset for handling labeled input text data (sentiment labels).

3. Model Architecture
Used the output of the [CLS] token from BERT to feed into a fully connected layer for classification.

4. Fine-tuning
Fine-tuned the model on labeled data using the Adam optimizer and a cross-entropy loss function.

5. Evaluation
Tested the model on a small test dataset and evaluated the performance using:
    - Accuracy
    - Confusion Matrix
    
#### Libraries Used
The following libraries were used throughout the project for training, evaluation, and visualization of Transformer-based models (especially BERT):

- **PyTorch (torch, torch.nn, torch.utils.data)** — for building the model architecture, training, and data handling.

- **Transformers (transformers)** — provides access to pre-trained models like BertTokenizer and BertForSequenceClassification, along with utilities such as get_linear_schedule_with_warmup.

- **torch.optim (AdamW)** —optimizer used for fine-tuning the BERT model.

- **scikit-learn (sklearn)** — used for dataset splitting (train_test_split) and computing evaluation metrics like accuracy, precision, recall, and F1-score.

- **matplotlib (matplotlib.pyplot)** — for plotting training curves and visualizing performance metrics.

- **json**  — for saving and loading configuration files and performance metrics.
----
### GPT-2 (Generative Pretrained Transformer)

**Tasks:**
- **Question Answering** (“C’est quoi X ?”) using Wikipedia summaries
- **Text Generation** based on user-provided prompts

**Model Used:**  
- asi/gpt-fr-cased-small



#### Steps

1. **Wikipedia Query Extraction**  
   Used regular expressions to extract relevant keywords from French questions like “C’est quoi X ?” or “Qu’est-ce que X ?”.

2. **Question Answering with Wikipedia API**  
   Queried French Wikipedia using the extracted term to retrieve a short summary using the wikipedia Python package.

3. **Text Generation**  
   Loaded a pre-trained French GPT-2 model to generate text based on a custom prompt.

4. **Dynamic Length Control**  
   Adjusted the maximum generated text length based on the number of words in the input prompt.

5. **Interactive Interface**  
   Created a command-line interface that offers:
   - Option 1: Ask a “C’est quoi X ?” question
   - Option 2: Generate a text continuation



#### Libraries Used
The following libraries were used specifically for the GPT-2 tasks (question answering and text generation):

- **transformers** — to load and use the pre-trained French GPT-2 model and tokenizer.
- **wikipedia** — to retrieve short summaries from French Wikipedia based on user questions.
- **re** — for regular expression-based keyword extraction from user input (e.g., “C’est quoi X ?”).
- **Python Standard Libraries** — for implementing the command-line interface and handling basic program logic.

----

### ELMo (Embeddings from Language Models)

**Task:**
- **Automatic Subject Detection** for sentences

**Model Used:**  
- [ELMo - TensorFlow Hub](https://tfhub.dev/google/elmo/3)


#### Steps

1. **Loading Pre-trained ELMo Model**  
   Utilized the ELMo model from TensorFlow Hub to generate word embeddings from input sentences.

2. **Sentence Embedding Extraction**  
   Extracted sentence embeddings by averaging the token embeddings generated by ELMo for each word in the sentence.

3. **Dataset Preparation**  
   Used a labeled dataset containing sentences from various categories (technology, sport, politics, etc.). The labels are the categories that represent the subject of the sentence.

4. **Model Architecture**  
   Built a simple classifier using a feed-forward neural network. The sentence embeddings obtained from ELMo were used as the input to this model.

5. **Training**  
   Trained the classifier using a supervised learning approach with the following components:
   - Optimizer: Adam
   - Loss Function: Cross-Entropy Loss

6. **Evaluation**  
   Evaluated the model performance using metrics like:
   - Accuracy
   - Confusion Matrix


#### Libraries Used
- **TensorFlow** — for model building and training
- **TensorFlow Hub (ELMo)** — for pre-trained word embeddings
- **scikit-learn** — for data preprocessing, model evaluation, and metrics
- **numpy** — for numerical operations
---
## Acknowledgments

- Our module instructor, Abdelhak Mahmoudi, for his guidance and support throughout the project.




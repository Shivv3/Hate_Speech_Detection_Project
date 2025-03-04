# Hate_Speech_Detection_Project
### Repository Contents:
This repository contains the following files:
1. Hate Speech Detection_Final.ipynb - Jupyter Notebook with the complete implementation of the project.
2. Covid-hate Dataset.csv - Dataset used for training and testing models.
3. README.md - Project documentation.

## Project Overview
Hate speech detection is a critical task in natural language processing (NLP) to identify and mitigate harmful content online. This project explores various machine learning and deep learning techniques to classify text as hate speech or non-hate speech. The dataset used is the Covid-hate Dataset, which contains labeled text data related to hate speech during the pandemic.

## Methodology
The project follows a structured pipeline for processing and modeling:

### 1. Data Preprocessing:

- **Tokenization:** Splitting text into individual words. 
- **Lemmatization:** Converting words to their base form. 
- **Stopword Removal:** Eliminating common words that do not contribute to meaning. 
- **Vectorization:** Representing text using TF-IDF (Term Frequency-Inverse Document Frequency). 

### 2. Data Visualization:

- Word Cloud representations to understand word distributions.
- Distribution plots for class balance analysis.

### 3. Model Training

Several models were trained to compare performance:

- **Machine Learning Models**

  - **Logistic Regression** - A linear model for binary classification.
  - **Random Forest Classifier** - An ensemble learning method using decision trees.
  - **Gradient Boosting Classifier** - A boosting method that improves weak classifiers.

- **Deep Learning Models**

  - **Neural Network (Basic)** - A simple feed-forward neural network.
  - **Neural Network + Multi-Head Attention (MHA)** - An advanced architecture incorporating attention mechanisms.
  - **BERT Model** - A pre-trained transformer-based deep learning model for NLP tasks.

### 4. Performance Evaluation

Each model was evaluated using the following metrics:

- **Accuracy** - Overall correctness of predictions.
- **Precision, Recall, F1-score** - Measures of classification quality.
- **Confusion Matrix** - Visual representation of misclassifications.
- **Graphs & Plots** - ROC curves, accuracy trends, and training loss curves.

### Results

The comparison of model performances showed that:

- Logistic Regression performed well with a balanced dataset but lacked deep feature extraction.
- Random Forest improved classification by handling complex patterns better.
- Gradient Boosting provided slight improvements with better generalization.
- Neural Networks, especially with Multi-Head Attention, enhanced contextual understanding.
- BERT outperformed other models by leveraging pre-trained contextual embeddings.

### Conclusion

Hate speech detection is an evolving field requiring robust models to handle linguistic nuances. The results indicate that deep learning approaches, particularly transformer-based models like BERT, significantly improve classification accuracy. Future work may involve fine-tuning transformer models further and exploring real-time detection implementations.

### How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/hate-speech-detection.git
   cd hate-speech-detection
- Install dependencies:
  ```bash
  pip install -r requirements.txt
- Open the Jupyter Notebook and run the cells sequentially.

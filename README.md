# Sentiment Analysis Project

## Introduction

Sentiment analysis, also known as opinion mining, is a key technique in Natural Language Processing (NLP) that identifies and extracts the sentiment or emotional tone within text. This project explores sentiment analysis using the IMDB dataset, comparing classical machine learning models with deep learning architectures and transformer-based models. The main goal is to evaluate the effectiveness of different models in classifying movie reviews as positive or negative sentiments.

## Project Overview

This project involves a comprehensive analysis using the IMDB dataset containing 50,000 movie reviews. The primary objectives include:

- **Text Preprocessing:** Implementing various text preprocessing techniques tailored for classical and deep learning models.
- **Classical Machine Learning Models:** Evaluating Logistic Regression, Support Vector Machines (SVM), and Naive Bayes models using Bag of Words (BoW) and Term Frequency-Inverse Document Frequency (TF-IDF) features.
- **Deep Learning Models:** Implementing Bidirectional Recurrent Neural Networks (RNNs), Long Short-Term Memory networks (LSTMs), and Gated Recurrent Units (GRUs) to capture sequential dependencies in text.
- **Transformer-Based Models:** Utilizing the BERT (Bidirectional Encoder Representations from Transformers) model to explore its state-of-the-art capabilities in NLP.

## Objectives

1. **Perform Text Preprocessing:** Develop robust text preprocessing pipelines for classical and deep learning models.
2. **Evaluate Classical Models:** Assess the performance of Logistic Regression, SVM, and Naive Bayes with BoW and TF-IDF features.
3. **Implement Deep Learning Models:** Train Bidirectional RNN, LSTM, and GRU models to capture sequential text dependencies.
4. **Fine-tune Transformer Models:** Apply and evaluate the BERT model to understand its advanced capabilities in sentiment analysis.
5. **Conduct Model Comparison:** Analyze and compare the results of classical, deep learning, and transformer-based models.

## Data Preprocessing

- **Text Cleaning:** Remove HTML tags, URLs, punctuation, and emojis.
- **Normalization and Translation:** Translate Gen Z slang and expand contractions.
- **Text Preprocessing for Classical Models:** Remove stopwords and perform general cleaning.
- **Text Preprocessing for Deep Learning Models:** Add lemmatization using SpaCy for more normalized text.
- **Tokenization and Lemmatization:** Split text into individual words and reduce words to their base forms.

## Model Performance

### Classical Machine Learning Models

1. **Logistic Regression with Bag of Words (BoW):** 88.89% accuracy with balanced precision and recall.
2. **SVM with Bag of Words (BoW):** 88.69% accuracy, slightly lower recall for the negative class.
3. **Naive Bayes with Bag of Words (BoW):** 86.19% accuracy, slightly reduced performance due to feature independence assumptions.
4. **Logistic Regression with TF-IDF:** 49% accuracy, indicating poor utilization of TF-IDF.
5. **SVM with TF-IDF:** Similar poor performance as Logistic Regression with TF-IDF.
6. **Naive Bayes with TF-IDF:** 73.37% accuracy, handling TF-IDF better than other models.

### Deep Learning Models

1. **Bidirectional RNN:** 69.71% accuracy, lowest among deep learning models.
2. **LSTM:** 86.06% accuracy, better handling of long-term dependencies.
3. **GRU:** 85.88% accuracy, highlighting its efficiency.

### Transformer-Based Models

- **BERT:** Achieved approximately 87.54% overall accuracy with balanced precision and recall, demonstrating its capability to handle complex language patterns.

## Conclusion

This project provides valuable insights into the trade-offs between classical machine learning models, deep learning architectures, and transformer-based models for sentiment analysis. The BERT model, while complex, offers robust performance, making it a solid choice for sentiment classification tasks. The results indicate that model selection should be based on specific use cases, computational resources, and desired accuracy levels.

## Future Work

Future improvements could include further fine-tuning of models, exploring additional datasets, and implementing more advanced NLP techniques to enhance model performance.
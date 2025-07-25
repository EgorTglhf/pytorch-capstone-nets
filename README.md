# pytorch-capstone-nets
This repository contains my capstone project for the Codecademy PyTorch course, demonstrating proficiency in building neural networks for diverse tasks. It includes three subprojects, each implemented as a Jupyter Notebook using PyTorch to address distinct datasets and machine learning problems in the healthcare domain.
## Project Overview
The project showcases three neural network models designed for the following tasks:
 - [**Analyzing Health Factors**](health_factors_prediction.ipynb): Predicting diabetes likelihood and patient age using tabular health data.
 - [**Classifying Medical Text**](medical_text_classification.ipynb): Classifying medical text data for diagnostic or sentiment analysis.
 - [**Classifying Retinal Images**](retinal_image_classification.ipynb): Detecting diabetic retinopathy from retinal images.

Each subproject highlights different aspects of neural network design, including data preprocessing, model architecture, training, and evaluation, using PyTorch.

## Subprojects
  1. [**Analyzing Health Factors - Predicting Diabetes and Age**](health_factors_prediction.ipynb)
     - **Objective**: Predict diabetes likelihood (binary classification) and patient age (regression) using tabular health data.
     - **Dataset**: Сleaned version of a [CDC dataset](https://www.kaggle.com/datasets/cdc/behavioral-risk-factor-surveillance-system) available within the [UCI Machine Learning Repo](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators).
     - **Model**: Two FeedForward neural networks with architectures to handle different tasks learning (classification and regression).
     - **Key Features**:
        - Preprocessing: Undersampling major class, feature scaling, and feature selection.
        - Two networks for simultaneous prediction of diabetes and age.
        - Evaluation metrics: Accuracy and AUC for diabetes prediction; MSE for age prediction.
     - **Notebook**: health_factors_prediction.ipynb
  2. [**Classifying Medical Text**](medical_text_classification.ipynb)
     - **Objective**: Classify medical text (e.g., clinical notes or reports) into diagnostic categories or sentiment labels.
     - **Dataset**: [MedQuAD dataset](https://github.com/abachaa/MedQuAD/tree/master) from the research paper [A Question-Entailment Approach to Question Answering](https://arxiv.org/pdf/1901.08079).
     - **Model**: A FeedForward neural network and Transformer.
     - **Key Features**:
        - Text preprocessing: Tokenization, embedding (e.g., GloVe or custom embeddings), and padding.
        - Model training with attention mechanisms for improved performance.
        - Evaluation metrics: Accuracy, F1-score, and confusion matrix.
     - **Notebook**: medical_text_classification.ipynb

  3. [**Classifying Retinal Images for Diabetic Retinopathy**](retinal_image_classification.ipynb)
     - **Objective**: Detect diabetic retinopathy from retinal images (binary classification).
     - **Dataset**: [IDRiD dataset](https://idrid.grand-challenge.org/Data/).
     - **Model**: CNN.
     - **Key Features**:
        - Image preprocessing: Resizing, augmentation (e.g., rotation, flipping), and normalization.
        - Evaluation metrics: Accuracy, precision, recall, F1-score.
     - **Notebook**: retinal_image_classification.ipynb



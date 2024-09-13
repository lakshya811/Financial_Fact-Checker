# Financial_Fact-Checker

Welcome to the FinBERT-X project repository! This project aims to enhance financial fact-checking by leveraging multi-modal machine learning techniques that integrate both textual and visual data for more accurate verification of financial claims.

### Project Overview
Problem Statement
In the financial domain, misinformation can have serious consequences, leading to market instability and economic losses. Traditional fact-checking methods rely mostly on textual data, overlooking the valuable insights provided by visual content such as charts and images. This project proposes a multi-modal approach to financial fact-checking, integrating both text and visual data for a more reliable and accurate verification system.

### Motivation
To mitigate the spread of financial misinformation, this project leverages the Fin-Fact dataset and state-of-the-art multi-modal learning techniques. By integrating both textual claims and corresponding visual data, we aim to build a robust financial fact-checking system capable of providing accurate, contextually rich verifications.

### Objectives
Develop a multi-modal fact-checking pipeline using financial text and images.
Improve the accuracy and reliability of fact-checking models by integrating visual and textual modalities.
Generate interpretable and explainable verifications for financial claims.
Deploy the model in a scalable, production-ready environment.


## Dataset

### Name: Fin-Fact

### Description: The dataset contains 3562 claims from various financial sectors, annotated with text, images, evidence, justifications, and more. The dataset includes:

### Claims and associated justifications.
Images with captions and visualization bias labels.
Various financial sectors represented in the claims.
Download: The dataset can be downloaded from here.

### Dataset Structure:

Claim: The financial claim to be verified.
Evidence: Supporting evidence for or against the claim.
Image href & Caption: The image associated with the claim.
Justification: Explanation of why the claim is true or false.
Claim Label: True or False label for each claim.

## Model Architecture :

The project uses a multi-modal approach to fact-checking, integrating visual and textual data through a transformer-based model. Below are the main components:

Textual Feature Extraction: Utilizes pre-trained models like BERT or RoBERTa for text processing.
Visual Feature Extraction: Pre-trained models like ResNet or ViT are used to extract features from associated images.
Feature Fusion: The textual and visual features are combined using a transformer-based architecture, allowing for cross-modal attention.
Classification: The fused features are passed through a classifier to predict whether the financial claim is true or false.

# DistilBERT Review Issue Classification

## Overview

This project fine-tunes DistilBERT to automatically classify customer reviews into predefined issue categories. The model learns to identify the underlying issue from review text and assign it to one of 20 issue classes.

The project includes data preprocessing, synthetic data generation for class balancing, DistilBERT fine-tuning, model evaluation, and inference.

---

## Problem Statement

E-commerce platforms receive thousands of customer reviews every day. Manually categorizing these reviews into issue categories is time-consuming and inefficient.

This project automates the process by predicting the issue category directly from review text using a fine-tuned DistilBERT model.

Example:

**Input Review**

The package arrived crushed and the box was torn.

**Predicted Issue**

Shipping_Damage

---

## Dataset

### Input Feature

* review_text

### Target Feature

* issue

### Issue Categories

* Comfort_Softness
* Material_Quality
* Durability
* Return_Refund
* Pilling
* Size_Fit
* Shipping_Damage
* Chemical_Odor
* Wrinkle_Resistance
* Delivery_Delay
* Customer_Service
* Packaging
* Positive
* Shrinkage
* Defect
* Color_Appearance
* Design_Issue
* Missing_Parts
* Wrong_Item
* Other

---

## Data Preprocessing

The following preprocessing steps were performed:

* Removed unnecessary columns
* Selected only relevant reviews
* Cleaned missing values
* Performed label encoding
* Generated synthetic reviews for underrepresented classes
* Balanced issue distribution
* Created train and test datasets using stratified splitting

---

## Synthetic Data Augmentation

The original dataset contained significant class imbalance.

Example:

| Issue            | Before |
| ---------------- | ------ |
| Comfort_Softness | 1060   |
| Return_Refund    | 5      |

Synthetic reviews were generated for minority classes to improve class balance and model performance.

Final dataset size: **5500+ reviews**

---

## Model Architecture

Review Text
↓
Tokenizer
↓
DistilBERT
↓
Classification Head
↓
Issue Category

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* PyTorch
* Hugging Face Transformers
* DistilBERT

---

## Training Configuration

| Parameter     | Value                      |
| ------------- | -------------------------- |
| Model         | DistilBERT                 |
| Task          | Multi-Class Classification |
| Classes       | 20                         |
| Epochs        | 5                          |
| Learning Rate | 2e-5                       |
| Batch Size    | 16                         |
| Optimizer     | AdamW                      |
| Framework     | Hugging Face Trainer       |

---

## Evaluation Metrics

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Classification Report
* Confusion Matrix

F1 Score was used as the primary metric because the dataset contains multiple classes with different frequencies.

---

## Project Workflow

1. Data Collection
2. Data Cleaning
3. Class Balancing
4. Synthetic Data Generation
5. Label Encoding
6. Train-Test Split
7. Tokenization
8. DistilBERT Fine-Tuning
9. Model Evaluation
10. Inference

---

## Sample Prediction

Input:

The sheets shrunk significantly after the first wash.

Output:

Shrinkage

---

Input:

Customer support never replied to my emails.

Output:

Customer_Service

---

Input:

The package arrived damaged and partially open.

Output:

Shipping_Damage

---

## Repository Structure

```text
DistilBERT-Review-Issue-Classification/
│
├── data/
│   ├── original_dataset.xlsx
│   ├── balanced_dataset.xlsx
│
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── model_training.ipynb
│   ├── inference.ipynb
│
├── models/
│   ├── fine_tuned_distilbert/
│   ├── label_encoder.pkl
│
├── images/
│   ├── class_distribution.png
│   ├── confusion_matrix.png
│
├── requirements.txt
├── README.md
└── app.py
```

---

## Installation

```bash
git clone https://github.com/your-username/distilbert-review-issue-classification.git

cd distilbert-review-issue-classification

pip install -r requirements.txt
```

---

## Future Improvements

* Experiment with BERT and RoBERTa
* Real-time API deployment
* Model monitoring and feedback loop

---

## Author
Shivam singh
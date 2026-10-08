# Email Spam Detection System (Naive Bayes & Comparative ML)

An end-to-end Machine Learning pipeline and interactive web application for classifying email and SMS messages into legitimate communication (**Ham**) or malicious/unsolicited messages (**Spam**).

Aligned with **Course Outcome CO5**: *"Solve real-life problems using appropriate machine learning models and evaluate performance measures."*

---

## 📁 Project Structure

```
email_spam_detection/
├── data/
│   └── spam.csv                          # UCI SMS/Email benchmark dataset (5,160 clean rows)
├── models/
│   ├── best_spam_classifier.joblib       # Serialized winning pipeline (Logistic Regression)
│   ├── naive_bayes_pipeline.joblib       # Serialized Multinomial Naive Bayes pipeline
│   ├── cv_metrics_summary.csv            # 5-Fold Stratified Cross-Validation metrics
│   └── test_metrics_summary.csv          # Holdout test set metrics
├── plots/
│   ├── class_distribution.png            # Ham vs Spam distribution chart
│   ├── correlation_heatmap.png           # Feature correlation heatmap
│   ├── confusion_matrices.png            # Multi-model confusion matrices
│   ├── roc_curves.png                    # Multi-model ROC-AUC comparison curves
│   └── model_comparison_metrics.png      # 5 metrics comparative bar chart
├── reports/
│   └── Email_Spam_Detection_Report.md     # Comprehensive 4–6 page academic report
├── src/
│   └── train.py                          # Full training and evaluation pipeline
├── streamlit_app.py                      # Interactive Streamlit Web Application
├── requirements.txt                      # Project dependencies
└── README.md                             # Documentation & user guide
```

---

## 🚀 Setup & Execution Guide

### 1. Prerequisites
Ensure Python 3.9+ is installed. Install required packages:
```bash
pip install -r requirements.txt
```

### 2. Run Training Pipeline
To retrain models, perform 5-fold cross validation, and regenerate all analytical plots:
```bash
python src/train.py
```

### 3. Launch Streamlit Web Application
Run the interactive user interface:
```bash
streamlit run streamlit_app.py
```

The application provides:
- Live classification of custom email or SMS content
- Preset buttons for lottery scam, phishing attempt, team work update, and casual chat
- Real-time decision threshold adjustment slider
- Confidence scores and TF-IDF token breakdown

---

## 📊 Summary of Experimental Results

| Model | 5-Fold CV F1 | 5-Fold CV ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Winner)** | **0.9166** | **0.9865** | **97.97%** | **95.73%** | 87.50% | **0.9143** | **0.9858** |
| **Linear SVM** | 0.9132 | 0.9890 | 97.77% | 92.68% | 89.06% | 0.9084 | 0.9878 |
| **Multinomial Naive Bayes** | 0.8850 | 0.9753 | 96.90% | 83.80% | **92.97%** | 0.8815 | 0.9758 |
| **Random Forest** | 0.8216 | 0.9816 | 96.71% | **98.96%** | 74.22% | 0.8482 | 0.9922 |

> **Key Takeaway**: While Naive Bayes achieved the highest spam recall (92.97%), it generated 23 False Positives. Logistic Regression was selected via 5-Fold CV as it maintains exceptional precision (95.73% with only 5 False Positives) and balanced F1-score (0.9143).

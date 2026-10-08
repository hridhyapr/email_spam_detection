"""
Email Spam Detection - Complete Machine Learning Pipeline
Domain: Email Spam Detection (Naive Bayes & Comparative Models)

Step 3 Implementation Order:
1. Load data and print shape, column names, class counts.
2. Preprocess: check missing values, duplicates, encode target.
3. Feature pipeline with TfidfVectorizer and StandardScaler(with_mean=False) to avoid data leakage.
4. Stratified 80/20 train-test split (stratify=y, random_state=42).
5. Define 4 algorithms: Multinomial Naive Bayes, Logistic Regression, Linear SVM, Random Forest.
6. 5-Fold Stratified Cross-Validation on training data to select the optimal model.
7. Train and evaluate on hold-out test set (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix).
8. Generate and save 5 analytical plots with plt.savefig().
9. Serialize winning pipeline with joblib.dump().
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report
)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "spam.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def load_dataset(filepath: str) -> pd.DataFrame:
    print("=" * 70)
    print("STEP 3.1: LOADING DATASET")
    print("=" * 70)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    df = pd.read_csv(filepath, encoding='latin-1')
    
    if 'type' in df.columns and 'text' in df.columns:
        df = df.rename(columns={'type': 'label', 'text': 'message'})
    elif 'v1' in df.columns and 'v2' in df.columns:
        df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})
    
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Column Names: {list(df.columns)}")
    print("\nClass Value Counts:")
    print(df['label'].value_counts())
    print("\nClass Distribution (%):")
    print((df['label'].value_counts(normalize=True) * 100).round(2).to_string())
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("STEP 3.2: PREPROCESSING (MISSING VALUES, DUPLICATES & ENCODING)")
    print("=" * 70)
    
    missing_count = df.isnull().sum().to_dict()
    print(f"Missing values per column: {missing_count}")
    
    df = df.dropna(subset=['label', 'message']).copy()
    
    duplicate_count = df.duplicated(subset=['message']).sum()
    print(f"Duplicate messages detected: {duplicate_count}")
    if duplicate_count > 0:
        df = df.drop_duplicates(subset=['message']).reset_index(drop=True)
        print(f"Dropped {duplicate_count} duplicates. Remaining records: {len(df)}")
    
    # Encode target: 'ham' -> 0, 'spam' -> 1
    df['target'] = df['label'].map({'ham': 0, 'spam': 1})
    if df['target'].isnull().any():
        raise ValueError("Target encoding encountered unmapped labels.")
        
    print("Encoded label into binary target: 0 = ham, 1 = spam")
    print(f"Cleaned Dataset Shape: {df.shape}")
    return df


def plot_class_distribution(df: pd.DataFrame, output_path: str):
    print("\n[Plot 1/5] Generating Class Distribution Plot...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    counts = df['label'].value_counts()
    colors = ['#2b5c8f', '#d9534f']
    bars = axes[0].bar(counts.index, counts.values, color=colors, edgecolor='black', width=0.5)
    axes[0].set_title("Class Frequency Count (Ham vs. Spam)", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Email / Message Class", fontsize=11)
    axes[0].set_ylabel("Number of Samples", fontsize=11)
    for bar in bars:
        h = bar.get_height()
        axes[0].annotate(f"{h} ({h/len(df)*100:.1f}%)",
                         xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 4), textcoords="offset points",
                         ha='center', va='bottom', fontweight='bold')
    
    axes[1].pie(counts.values, labels=['Legitimate (Ham)', 'Spam'], autopct='%1.1f%%',
                startangle=140, colors=colors, explode=(0, 0.08),
                wedgeprops=dict(width=0.6, edgecolor='white', linewidth=2),
                textprops={'fontsize': 12, 'fontweight': 'bold'})
    axes[1].set_title("Proportional Class Breakdown", fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def plot_correlation_heatmap(X_train: pd.Series, y_train: pd.Series, output_path: str):
    print("\n[Plot 2/5] Generating Top Keywords Correlation Heatmap...")
    tfidf = TfidfVectorizer(stop_words='english', max_features=16, ngram_range=(1, 1))
    X_tfidf = tfidf.fit_transform(X_train).toarray()
    feature_names = tfidf.get_feature_names_out()
    
    feat_df = pd.DataFrame(X_tfidf, columns=feature_names)
    feat_df['TARGET_SPAM'] = y_train.values
    
    corr_matrix = feat_df.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True,
                linewidths=0.5, linecolor='gray', square=True,
                annot_kws={'size': 9})
    plt.title("Correlation Heatmap: Top Informative Vocabulary Features & Spam Target",
              fontsize=13, fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def build_models():
    """
    Constructs pipelines comparing 4 algorithms:
    1. Multinomial Naive Bayes: Probabilistic generative model designed for discrete word frequencies.
    2. Logistic Regression: Linear discriminative baseline with calibrated probabilities.
    3. Support Vector Machine (LinearSVC): Maximum-margin classifier robust in high-dimensional text space.
    4. Random Forest Classifier: Non-linear tree ensemble capturing token interactions.
    StandardScaler(with_mean=False) scales features while preserving sparsity and avoiding data leakage.
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=4000,
        sublinear_tf=True,
        stop_words='english'
    )
    scaler = StandardScaler(with_mean=False)
    
    models = {
        'Multinomial Naive Bayes': Pipeline([
            ('tfidf', vectorizer),
            ('scaler', scaler),
            ('clf', MultinomialNB(alpha=0.5))
        ]),
        'Logistic Regression': Pipeline([
            ('tfidf', vectorizer),
            ('scaler', scaler),
            ('clf', LogisticRegression(max_iter=1000, C=1.0, random_state=42))
        ]),
        'Support Vector Machine (LinearSVC)': Pipeline([
            ('tfidf', vectorizer),
            ('scaler', scaler),
            ('clf', CalibratedClassifierCV(LinearSVC(C=1.0, dual=False, max_iter=3000, random_state=42), cv=3))
        ]),
        'Random Forest Classifier': Pipeline([
            ('tfidf', vectorizer),
            ('scaler', scaler),
            ('clf', RandomForestClassifier(n_estimators=100, max_depth=30, random_state=42, n_jobs=-1))
        ])
    }
    return models


def evaluate_cross_validation(models: dict, X_train: pd.Series, y_train: pd.Series):
    print("\n" + "=" * 70)
    print("STEP 3.3: 5-FOLD STRATIFIED CROSS-VALIDATION (MODEL SELECTION)")
    print("=" * 70)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    
    cv_summary = []
    
    for name, pipeline in models.items():
        print(f"Evaluating 5-Fold CV on: {name}...")
        scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        
        row = {
            'Model': name,
            'CV Accuracy': f"{np.mean(scores['test_accuracy']):.4f} (+/- {np.std(scores['test_accuracy']):.4f})",
            'CV Precision': f"{np.mean(scores['test_precision']):.4f} (+/- {np.std(scores['test_precision']):.4f})",
            'CV Recall': f"{np.mean(scores['test_recall']):.4f} (+/- {np.std(scores['test_recall']):.4f})",
            'CV F1': f"{np.mean(scores['test_f1']):.4f} (+/- {np.std(scores['test_f1']):.4f})",
            'CV ROC-AUC': f"{np.mean(scores['test_roc_auc']):.4f} (+/- {np.std(scores['test_roc_auc']):.4f})",
            'raw_accuracy': np.mean(scores['test_accuracy']),
            'raw_precision': np.mean(scores['test_precision']),
            'raw_recall': np.mean(scores['test_recall']),
            'raw_f1': np.mean(scores['test_f1']),
            'raw_roc_auc': np.mean(scores['test_roc_auc'])
        }
        cv_summary.append(row)
        
    cv_df = pd.DataFrame(cv_summary)
    print("\n--- 5-Fold Stratified Cross-Validation Summary Table ---")
    print(cv_df[['Model', 'CV Accuracy', 'CV Precision', 'CV Recall', 'CV F1', 'CV ROC-AUC']].to_string(index=False))
    return cv_df


def train_and_test(models: dict, X_train: pd.Series, y_train: pd.Series,
                   X_test: pd.Series, y_test: pd.Series):
    print("\n" + "=" * 70)
    print("STEP 3.4: TRAINING ON 80% TRAIN SET & TESTING ON 20% TEST SET")
    print("=" * 70)
    
    test_results = {}
    
    for name, pipeline in models.items():
        print(f"\nTraining [{name}]...")
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        
        if hasattr(pipeline, 'predict_proba'):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
        elif hasattr(pipeline, 'decision_function'):
            decision = pipeline.decision_function(X_test)
            y_proba = 1 / (1 + np.exp(-decision))
        else:
            y_proba = y_pred
            
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)
        
        test_results[name] = {
            'pipeline': pipeline,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'roc_auc': auc,
            'confusion_matrix': cm
        }
        
        print(f"Metrics for {name}:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {auc:.4f}")
        print(f"  Confusion Matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
        
    return test_results


def plot_confusion_matrices(test_results: dict, output_path: str):
    print("\n[Plot 3/5] Generating Confusion Matrices Plot...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    labels = ['Ham (0)', 'Spam (1)']
    
    for idx, (name, res) in enumerate(test_results.items()):
        cm = res['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    cbar=False, annot_kws={'size': 13, 'weight': 'bold'},
                    xticklabels=labels, yticklabels=labels)
        axes[idx].set_title(f"{name}\nAcc: {res['accuracy']:.4f} | Prec: {res['precision']:.4f} | Rec: {res['recall']:.4f}",
                            fontsize=11, fontweight='bold')
        axes[idx].set_xlabel("Predicted Label", fontsize=10)
        axes[idx].set_ylabel("True Ground Truth Label", fontsize=10)
        
    plt.suptitle("Confusion Matrix Comparison Across Models (Holdout Test Set)", fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def plot_roc_curves(test_results: dict, y_test: pd.Series, output_path: str):
    print("\n[Plot 4/5] Generating ROC Curves Plot...")
    plt.figure(figsize=(9, 7))
    
    colors = ['#2b5c8f', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, (name, res) in enumerate(test_results.items()):
        fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
        plt.plot(fpr, tpr, color=colors[idx % len(colors)], lw=2.2,
                 label=f"{name} (AUC = {res['roc_auc']:.4f})")
                 
    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random Chance Baseline (AUC = 0.50)')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12, fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC) Curves Comparison', fontsize=14, fontweight='bold', pad=12)
    plt.legend(loc='lower right', fontsize=10, frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def plot_model_comparison(cv_df: pd.DataFrame, test_results: dict, output_path: str):
    print("\n[Plot 5/5] Generating Model Comparison Bar Chart...")
    model_names = list(test_results.keys())
    data = []
    
    for name in model_names:
        res = test_results[name]
        data.append({
            'Model': name,
            'Accuracy': res['accuracy'],
            'Precision': res['precision'],
            'Recall': res['recall'],
            'F1-Score': res['f1'],
            'ROC-AUC': res['roc_auc']
        })
        
    df_metrics = pd.DataFrame(data)
    df_melted = df_metrics.melt(id_vars=['Model'], var_name='Metric', value_name='Score')
    
    plt.figure(figsize=(13, 6))
    
    ax = sns.barplot(data=df_melted, x='Metric', y='Score', hue='Model', palette='tab10', edgecolor='black')
    plt.title('Performance Metric Comparison Across Evaluated Algorithms (Holdout Test Set)', fontsize=14, fontweight='bold', pad=12)
    plt.ylim([0.7, 1.03])
    plt.ylabel('Performance Score (0.0 - 1.0)', fontsize=12)
    plt.xlabel('Evaluation Metric', fontsize=12)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=10)
    
    for p in ax.patches:
        h = p.get_height()
        if h > 0.01:
            ax.annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='bottom', fontsize=7.5, rotation=90,
                        xytext=(0, 3), textcoords='offset points', fontweight='bold')
                        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print("=" * 80)
    print("EMAIL SPAM DETECTION: MACHINE LEARNING WORKFLOW EXECUTION")
    print("=" * 80)
    
    # 1. Load Data
    df = load_dataset(DATA_PATH)
    
    # 2. Preprocess Data
    df_clean = preprocess_data(df)
    
    # Plot Class Distribution
    plot_class_distribution(df_clean, os.path.join(PLOTS_DIR, "class_distribution.png"))
    
    # 3. Train-Test Split (80/20 Stratified)
    X = df_clean['message']
    y = df_clean['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"\nTrain set: {len(X_train)} samples ({y_train.sum()} spam, {len(y_train) - y_train.sum()} ham)")
    print(f"Test set:  {len(X_test)} samples ({y_test.sum()} spam, {len(y_test) - y_test.sum()} ham)")
    
    # Plot Correlation Heatmap on training features
    plot_correlation_heatmap(X_train, y_train, os.path.join(PLOTS_DIR, "correlation_heatmap.png"))
    
    # 4. Construct Models
    models = build_models()
    
    # 5. 5-Fold Cross-Validation for Model Selection
    cv_df = evaluate_cross_validation(models, X_train, y_train)
    
    # 6. Fit on Train, Test on Test Set
    test_results = train_and_test(models, X_train, y_train, X_test, y_test)
    
    # 7. Generate Visualizations
    plot_confusion_matrices(test_results, os.path.join(PLOTS_DIR, "confusion_matrices.png"))
    plot_roc_curves(test_results, y_test, os.path.join(PLOTS_DIR, "roc_curves.png"))
    plot_model_comparison(cv_df, test_results, os.path.join(PLOTS_DIR, "model_comparison_metrics.png"))
    
    # 8. Model Selection: Choose based on 5-Fold CV (F1 and ROC-AUC)
    best_cv_row = cv_df.sort_values(by=['raw_f1', 'raw_roc_auc'], ascending=False).iloc[0]
    best_model_name = best_cv_row['Model']
    print("\n" + "=" * 70)
    print(f"FINAL MODEL SELECTION (BASED ON 5-FOLD CV): {best_model_name}")
    print(f"Mean CV F1: {best_cv_row['raw_f1']:.4f} | Mean CV ROC-AUC: {best_cv_row['raw_roc_auc']:.4f}")
    print("=" * 70)
    
    best_pipeline = test_results[best_model_name]['pipeline']
    
    model_save_path = os.path.join(MODELS_DIR, "best_spam_classifier.joblib")
    joblib.dump(best_pipeline, model_save_path)
    print(f"Saved Best Model Pipeline to: {model_save_path}")
    
    nb_save_path = os.path.join(MODELS_DIR, "naive_bayes_pipeline.joblib")
    joblib.dump(test_results['Multinomial Naive Bayes']['pipeline'], nb_save_path)
    print(f"Saved Naive Bayes Model Pipeline to: {nb_save_path}")
    
    # Save metrics summary CSV for reporting
    summary_records = []
    for name, res in test_results.items():
        summary_records.append({
            'Model': name,
            'Accuracy': res['accuracy'],
            'Precision': res['precision'],
            'Recall': res['recall'],
            'F1_Score': res['f1'],
            'ROC_AUC': res['roc_auc'],
            'TN': res['confusion_matrix'][0, 0],
            'FP': res['confusion_matrix'][0, 1],
            'FN': res['confusion_matrix'][1, 0],
            'TP': res['confusion_matrix'][1, 1],
        })
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(os.path.join(MODELS_DIR, "test_metrics_summary.csv"), index=False)
    cv_df.to_csv(os.path.join(MODELS_DIR, "cv_metrics_summary.csv"), index=False)
    print("Saved Metrics Summaries to models directory.")
    print("\nPipeline execution completed successfully!")


if __name__ == '__main__':
    main()

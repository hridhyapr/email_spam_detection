"""
Interactive Streamlit Web Application: Email Spam Detection
Domain: Email Spam Detection (Naive Bayes & Comparative Models)
Aligned with Course Outcome: CO5
"""

import os
import io
import joblib
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Email Spam Shield | ML Classifier",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Base Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
DATA_DIR = os.path.join(BASE_DIR, "data")

BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_spam_classifier.joblib")
NB_MODEL_PATH = os.path.join(MODELS_DIR, "naive_bayes_pipeline.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "test_metrics_summary.csv")
CV_METRICS_PATH = os.path.join(MODELS_DIR, "cv_metrics_summary.csv")
DATASET_PATH = os.path.join(DATA_DIR, "spam.csv")


@st.cache_resource
def load_models():
    best_clf = joblib.load(BEST_MODEL_PATH)
    nb_clf = joblib.load(NB_MODEL_PATH)
    return best_clf, nb_clf


@st.cache_data
def load_dataset_samples():
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH, encoding="latin-1")
        if "type" in df.columns:
            df = df.rename(columns={"type": "label", "text": "message"})
        elif "v1" in df.columns:
            df = df.rename(columns={"v1": "label", "v2": "message"})
        return df[["label", "message"]]
    return None


best_model, nb_model = load_models()

# Load summary CSVs
test_metrics = pd.read_csv(METRICS_PATH) if os.path.exists(METRICS_PATH) else None
cv_metrics = pd.read_csv(CV_METRICS_PATH) if os.path.exists(CV_METRICS_PATH) else None

# --- SIDEBAR CONFIGURATION ---
st.sidebar.title("🛡️ Spam Shield Controls")
st.sidebar.markdown("Configure classifier parameters and model architecture.")

model_choice = st.sidebar.selectbox(
    "Active Classification Model:",
    ("Logistic Regression (CV Optimal - High Precision)", "Multinomial Naive Bayes (Primary - High Recall)")
)

active_pipeline = best_model if "Logistic" in model_choice else nb_model

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Sensitivity Sliders")

threshold = st.sidebar.slider(
    "Spam Decision Threshold:",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05,
    help="Messages with calculated spam probability >= threshold are flagged as SPAM. Lower values catch more spam but increase false alarms."
)

max_char_len = st.sidebar.slider(
    "Max Input Preview Length (Characters):",
    min_value=200,
    max_value=2000,
    value=1000,
    step=100
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Project Metadata**:
    - **Focus**: Email & SMS Spam Detection
    - **Primary Model**: Naive Bayes (MultinomialNB)
    - **Comparison**: Logistic Regression, SVM, Random Forest
    - **Outcome**: CO5 Real-Life ML Evaluation
    """
)

# --- HEADER ---
st.title("📧 Intelligent Email & SMS Spam Detection System")
st.caption("Powered by Scikit-Learn Probabilistic NLP Pipelines | Zero Data Leakage Architecture")

# Tabs for comprehensive UI
tab_live, tab_batch, tab_analytics, tab_data = st.tabs([
    "🔍 Live Message Detector",
    "📁 Batch CSV / File Classifier",
    "📊 Model Analytics & Plots",
    "📚 Dataset & Knowledge Base (CO5)"
])

# ==============================================================================
# TAB 1: LIVE MESSAGE DETECTOR
# ==============================================================================
with tab_live:
    st.subheader("Interactive Message Classification")
    st.write("Test incoming email bodies or SMS texts against the trained pipeline.")

    # Preset scenarios
    st.markdown("**⚡ Quick Load Presets:**")
    p_cols = st.columns(4)

    preset_scenarios = {
        "🎁 Lottery Scam (Spam)": "URGENT! You have won a £2,000 cash prize or a holiday to Hawaii! Claim your reward now by texting WIN to 87121 immediately.",
        "⚠️ Phishing Alert (Spam)": "Important notice from your bank: Suspicious login attempt detected on your checking account. Verify your credentials here: http://bit.ly/secure-login-392",
        "💼 Team Standup (Ham)": "Hi Alex, please find attached the revised presentation deck for tomorrow's quarterly product review. Let me know if you need any adjustments before 9 AM.",
        "🍕 Casual Dinner (Ham)": "Hey, are we still meeting up for dinner tonight around 7:30 PM? Let me know if the Italian place works for you."
    }

    if "current_text" not in st.session_state:
        st.session_state["current_text"] = preset_scenarios["🎁 Lottery Scam (Spam)"]

    for idx, (p_name, p_text) in enumerate(preset_scenarios.items()):
        with p_cols[idx]:
            if st.button(p_name):
                st.session_state["current_text"] = p_text

    # Main text input
    user_text = st.text_area(
        "Enter Email / SMS Message Content:",
        value=st.session_state["current_text"],
        height=140,
        max_chars=max_char_len,
        placeholder="Type or paste the email header and message body..."
    )

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        predict_clicked = st.button("🚀 Classify Message", type="primary")

    if predict_clicked or user_text:
        if not user_text.strip():
            st.warning("Please enter some text to analyze.")
        else:
            # Probability computation
            probas = active_pipeline.predict_proba([user_text])[0]
            ham_prob = float(probas[0])
            spam_prob = float(probas[1])
            is_spam = spam_prob >= threshold

            st.markdown("---")
            res_left, res_right = st.columns([1.3, 1])

            with res_left:
                if is_spam:
                    st.error(
                        f"### 🛑 SPAM FLAGGED\n"
                        f"**Classification:** Malicious / Unsolicited Spam  \n"
                        f"**Spam Confidence:** `{spam_prob * 100:.2f}%` (Decision Threshold: `{threshold * 100:.1f}%`)"
                    )
                else:
                    st.success(
                        f"### ✅ LEGITIMATE EMAIL (HAM)\n"
                        f"**Classification:** Safe Communication  \n"
                        f"**Ham Confidence:** `{ham_prob * 100:.2f}%` (Decision Threshold: `{threshold * 100:.1f}%`)"
                    )

                st.write(f"**Computed Spam Probability:** `{spam_prob:.4f}`")
                st.progress(min(max(spam_prob, 0.0), 1.0))

            with res_right:
                st.info(
                    f"""
                    **Inspection Metrics**:
                    - **Selected Model**: {model_choice.split('(')[0].strip()}
                    - **Character Count**: {len(user_text)}
                    - **Word Count**: {len(user_text.split())}
                    - **Decision Rule**: Flag as Spam if $P(\\text{{Spam}}) \\ge {threshold:.2f}$
                    - **Risk Analysis**: {'Potential False Positive risk (Low threshold)' if threshold < 0.4 else 'Balanced filtering'}
                    """
                )

            # Feature / Keyword inspection
            st.markdown("#### 🔎 High-Salience Vocabulary Tokens Detected")
            try:
                tfidf_step = active_pipeline.named_steps["tfidf"]
                transformed = tfidf_step.transform([user_text])
                non_zero_cols = transformed.nonzero()[1]
                feature_names = np.array(tfidf_step.get_feature_names_out())

                if len(non_zero_cols) > 0:
                    matched_features = feature_names[non_zero_cols]
                    weights = transformed.data
                    kw_df = pd.DataFrame({
                        "Detected Keyword / N-Gram": matched_features,
                        "TF-IDF Salience Weight": weights
                    }).sort_values(by="TF-IDF Salience Weight", ascending=False)

                    st.dataframe(kw_df.head(10))
                else:
                    st.caption("No prominent vocabulary n-grams matched the pre-trained dictionary.")
            except Exception as e:
                st.caption(f"Token breakdown notice: {e}")

# ==============================================================================
# TAB 2: BATCH CSV / FILE CLASSIFIER
# ==============================================================================
with tab_batch:
    st.subheader("Batch Message Processing")
    st.write("Upload a CSV file containing multiple messages to process them in bulk.")

    uploaded_file = st.file_uploader("Upload CSV file (must contain a column with messages):", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(batch_df)} rows. Select message column:")
            
            text_col = st.selectbox("Select Message Column:", batch_df.columns)
            
            if st.button("⚡ Run Batch Classification"):
                with st.spinner("Classifying messages..."):
                    messages = batch_df[text_col].astype(str).tolist()
                    batch_probas = active_pipeline.predict_proba(messages)[:, 1]
                    batch_preds = ["Spam" if p >= threshold else "Ham" for p in batch_probas]
                    
                    batch_df["Spam_Probability"] = np.round(batch_probas, 4)
                    batch_df["Predicted_Class"] = batch_preds
                    
                    st.success("Batch classification completed!")
                    
                    # Metrics summary
                    spam_count = sum(1 for p in batch_preds if p == "Spam")
                    ham_count = len(batch_preds) - spam_count
                    
                    s_col1, s_col2, s_col3 = st.columns(3)
                    s_col1.metric("Total Messages", len(batch_df))
                    s_col2.metric("Spam Flagged", f"{spam_count} ({spam_count/len(batch_df)*100:.1f}%)")
                    s_col3.metric("Legitimate (Ham)", f"{ham_count} ({ham_count/len(batch_df)*100:.1f}%)")
                    
                    st.dataframe(batch_df.head(20))
                    
                    # Download button
                    csv_buffer = io.StringIO()
                    batch_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="📥 Download Annotated CSV",
                        data=csv_buffer.getvalue(),
                        file_name="classified_spam_results.csv",
                        mime="text/csv"
                    )
        except Exception as ex:
            st.error(f"Error processing batch file: {ex}")
    else:
        st.info("Tip: You can upload any CSV containing raw email or SMS messages to test the automated pipeline.")

# ==============================================================================
# TAB 3: MODEL ANALYTICS & PLOTS
# ==============================================================================
with tab_analytics:
    st.subheader("Model Evaluation & Comparative Analysis")

    if test_metrics is not None and cv_metrics is not None:
        st.markdown("#### 📋 5-Fold Stratified Cross-Validation Benchmark (Training Set)")
        st.dataframe(cv_metrics[["Model", "CV Accuracy", "CV Precision", "CV Recall", "CV F1", "CV ROC-AUC"]])

        st.markdown("#### 🎯 Holdout Test Set Performance (20% Split, N=1,032)")
        st.dataframe(test_metrics[["Model", "Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC", "TN", "FP", "FN", "TP"]])

    st.markdown("---")
    st.subheader("📈 Publication-Ready Evaluation Charts")

    plot_paths = {
        "Class Distribution": os.path.join(PLOTS_DIR, "class_distribution.png"),
        "Top Keyword Correlation Heatmap": os.path.join(PLOTS_DIR, "correlation_heatmap.png"),
        "Confusion Matrices": os.path.join(PLOTS_DIR, "confusion_matrices.png"),
        "ROC-AUC Curves": os.path.join(PLOTS_DIR, "roc_curves.png"),
        "Model Comparison Bar Chart": os.path.join(PLOTS_DIR, "model_comparison_metrics.png")
    }

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        if os.path.exists(plot_paths["Class Distribution"]):
            st.image(plot_paths["Class Distribution"], caption="Figure 1: Dataset Class Distribution (Ham vs Spam)")
        if os.path.exists(plot_paths["Confusion Matrices"]):
            st.image(plot_paths["Confusion Matrices"], caption="Figure 3: Confusion Matrices Across All Models")
        if os.path.exists(plot_paths["Model Comparison Bar Chart"]):
            st.image(plot_paths["Model Comparison Bar Chart"], caption="Figure 5: Five-Metric Comparative Performance Bar Chart")

    with col_p2:
        if os.path.exists(plot_paths["Top Keyword Correlation Heatmap"]):
            st.image(plot_paths["Top Keyword Correlation Heatmap"], caption="Figure 2: Informative Token Correlation Heatmap")
        if os.path.exists(plot_paths["ROC-AUC Curves"]):
            st.image(plot_paths["ROC-AUC Curves"], caption="Figure 4: Receiver Operating Characteristic (ROC) Comparison")

# ==============================================================================
# TAB 4: DATASET & KNOWLEDGE BASE (CO5)
# ==============================================================================
with tab_data:
    st.subheader("Corpus Explorer & Academic Course Outcome Alignment")

    df_samples = load_dataset_samples()
    if df_samples is not None:
        st.markdown("#### 🔎 Sample Records from Benchmark Corpus")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.markdown("**Sample Legitimate Messages (Ham)**")
            st.dataframe(df_samples[df_samples["label"] == "ham"].head(5))
        with col_f2:
            st.markdown("**Sample Malicious Messages (Spam)**")
            st.dataframe(df_samples[df_samples["label"] == "spam"].head(5))

    st.markdown("---")
    st.markdown(
        """
        ### 🎓 Course Outcome Mapping: CO5
        > *"Solve real life problems using appropriate machine learning models and evaluate the performance measures"*

        - **Real-Life Problem**: Mitigating unsolicited, fraudulent, and malicious electronic mail and SMS communications.
        - **Appropriate Models**: Evaluated four distinct mathematical frameworks:
          1. **Multinomial Naive Bayes**: Generative probabilistic model based on conditional token independence and Laplace smoothing.
          2. **Logistic Regression**: Regularized discriminative linear baseline optimizing log-odds.
          3. **Linear Support Vector Machine**: Maximum-margin hyperplane classification.
          4. **Random Forest Classifier**: Non-linear tree ensemble.
        - **Data Leakage Prevention**: Integrated feature standardization (`StandardScaler(with_mean=False)`) and vectorization within a scikit-learn `Pipeline`.
        - **Performance Measures**: Rigorously scored using Accuracy, Precision, Recall, F1-Score, and ROC-AUC via 5-Fold Stratified Cross-Validation.
        - **Error Asymmetry**: Explains why **Precision** is paramount to avoid devastating False Positives (quarantining critical personal/business emails).
        """
    )

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 13px;'>"
    "Email Spam Detection Project | Scikit-Learn, Streamlit & Python | Academic Outcome CO5"
    "</div>",
    unsafe_allow_html=True
)

import streamlit as st
import torch
import joblib
import pandas as pd
import plotly.express as px

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# ------------------------
# PAGE CONFIG
# ------------------------

st.set_page_config(
    page_title="Review Issue Classifier",
    page_icon="🤖",
    layout="wide"
)

# ------------------------
# LOAD MODEL
# ------------------------

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        "models/issue_classifier"
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        "models/issue_classifier"
    )

    encoder = joblib.load(
        "models/label_encoder.pkl"
    )

    return tokenizer, model, encoder

tokenizer, model, encoder = load_model()

# ------------------------
# CUSTOM CSS
# ------------------------

st.markdown("""
<style>

.big-font {
    font-size:40px !important;
    font-weight:bold;
}

.prediction-box {
    padding:20px;
    border-radius:10px;
    background:#0e1117;
    border:1px solid #2a2a2a;
}

</style>
""", unsafe_allow_html=True)

# ------------------------
# SIDEBAR
# ------------------------

with st.sidebar:

    st.title("🤖 Model Info")

    st.metric(
        "Classes",
        "20"
    )

    st.metric(
        "Dataset Size",
        "5500+"
    )

    st.metric(
        "Architecture",
        "DistilBERT"
    )

    st.markdown("---")

    st.subheader("Issue Categories")

    for issue in encoder.classes_:
        st.write(issue)

# ------------------------
# HEADER
# ------------------------

st.markdown(
    "<p class='big-font'>Customer Review Issue Classification</p>",
    unsafe_allow_html=True
)

st.write(
    "Predict issue categories from customer reviews using a fine-tuned DistilBERT model."
)

# ------------------------
# INPUT
# ------------------------

review = st.text_area(
    "Enter Customer Review",
    height=200
)

col1, col2 = st.columns(2)

predict_btn = col1.button(
    "🔍 Predict"
)

clear_btn = col2.button(
    "🗑 Clear"
)

# ------------------------
# PREDICTION
# ------------------------

if predict_btn and review:

    inputs = tokenizer(
        review,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():

        outputs = model(**inputs)

        probs = torch.softmax(
            outputs.logits,
            dim=1
        )[0]

    pred_idx = probs.argmax().item()

    confidence = probs.max().item()

    issue = encoder.inverse_transform(
        [pred_idx]
    )[0]

    st.markdown("---")

    st.subheader("Prediction")

    st.success(
        f"Issue: {issue}"
    )

    st.info(
        f"Confidence: {confidence*100:.2f}%"
    )

    # --------------------
    # TOP 3 PREDICTIONS
    # --------------------

    top_probs, top_idx = torch.topk(
        probs,
        k=3
    )

    rows = []

    for p, i in zip(top_probs, top_idx):

        rows.append(
            {
                "Issue":
                encoder.inverse_transform(
                    [i.item()]
                )[0],

                "Confidence":
                round(
                    p.item()*100,
                    2
                )
            }
        )

    df = pd.DataFrame(rows)

    st.subheader(
        "Top Predictions"
    )

    st.dataframe(
        df,
        use_container_width=True
    )

    # --------------------
    # CHART
    # --------------------

    fig = px.bar(
        df,
        x="Issue",
        y="Confidence",
        title="Prediction Confidence"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ------------------------
# FOOTER
# ------------------------

st.markdown("---")

st.caption(
    "Built with DistilBERT, Hugging Face, PyTorch and Streamlit"
)
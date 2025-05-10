import streamlit as st
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from transformers import RobertaTokenizer, RobertaForSequenceClassification

st.set_page_config(page_title="Comment Toxicity Checker", layout="wide")

st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Oswald&display=swap" rel="stylesheet">
    <style>
        body {
            background-color: #000000;
            color: white;
        }
        .main {
            background-color: #000000;
            border-left: 60px solid #FF0000;
            border-right: 60px solid #FF0000;
            padding: 2rem;
        }
        h1, h2, h3 {
            font-family: 'Oswald', sans-serif;
            text-align: center;
            color: white;
        }
        .stTextArea label {
            font-size: 18px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

#Loading Model
@st.cache_resource
def load_model():
    tokenizer = RobertaTokenizer.from_pretrained('s-nlp/roberta_toxicity_classifier')
    model = RobertaForSequenceClassification.from_pretrained('s-nlp/roberta_toxicity_classifier')
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# Load dataset
@st.cache_data
def load_data():
    url = "https://github.com/DavidMembreno/Data-Science/raw/refs/heads/main/Predicted_YouTube_Comments.csv"
    return pd.read_csv(url)

df = load_data()

#Check comments for toxicity using model
st.title("Comment Toxicity Checker")
st.write("This tool uses a transformer model to classify YouTube-style comments as toxic or not.")
st.warning("⚠️ Some examples may contain offensive or inappropriate language.")


comment = st.text_area("Paste your comment here:", placeholder="e.g. That was the worst take I've ever heard.")
threshold = st.slider("Set toxicity threshold", 0.0, 1.0, 0.5)

if st.button("Analyze Comment"):
    if comment.strip():
        inputs = tokenizer.encode(comment, return_tensors='pt', truncation=True)
        with torch.no_grad():
            logits = model(inputs).logits
            probs = torch.softmax(logits, dim=1)
            toxic_prob = probs[0][1].item()

        label = "Toxic" if toxic_prob > threshold else "Not Toxic"
        confidence = toxic_prob if label == "Toxic" else 1 - toxic_prob

        st.markdown(f"### Result: **{label}**")
        st.progress(confidence)
        st.caption(f"Confidence: {confidence:.2%}")
    else:
        st.warning("Please enter a comment first.")
        
#Basic Visuals
st.markdown("---")
st.header("Dataset Overview")
st.write("""
We combined three YouTube comment datasets to explore toxicity at scale:

- [General YouTube Comments](https://www.kaggle.com/datasets/atifaliak/youtube-comments-dataset)
- [Spam-focused Comments](https://www.kaggle.com/datasets/ahsenwaheed/youtube-comments-spam-dataset)
- [Labeled Toxicity Comments](https://www.kaggle.com/datasets/reihanenamdari/youtube-toxicity-data)
""")


with st.expander("📈 Show Toxic vs Non-Toxic Predictions Chart"):
    fig, ax = plt.subplots(figsize=(2.5, 1.5), dpi=100)
    sns.countplot(data=df, x='prediction', palette='Reds', ax=ax)
    ax.set_title("Label Distribution", fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    st.pyplot(fig, use_container_width=False)
#Wordclouds for the highest model confidence in Toxic and Non-Toxic
with st.expander("🤬 Show Most Toxic Comments"):
    filtered = df[df['prediction'] == 'toxicity'].copy()
    filtered = filtered.sort_values(by='confidence', ascending=False).head(5)

    filtered = filtered[['Comment']].reset_index(drop=True)
    filtered.index = range(1, len(filtered) + 1)

    st.table(filtered)


with st.expander("😎 Show Least Toxic Comments"):
    filtered = df[df['prediction'] == 'non-toxic'].copy()
    filtered = filtered.sort_values(by='confidence', ascending=False).head(5)

    filtered = filtered[['Comment']].reset_index(drop=True)
    filtered.index = range(1, len(filtered) + 1)

    st.table(filtered)




with st.expander("☁️ Show Word Clouds by Prediction Category"):
    spacer_l, left_col, right_col, spacer_r = st.columns([2, 6, 6, 2])

    with left_col:
        st.markdown("**Toxic Comments**")
        toxic_text = " ".join(df[df['prediction'] == 'toxicity']['Comment'].dropna().astype(str))
        wc_toxic = WordCloud(width=400, height=300, background_color='white').generate(toxic_text)
        st.image(wc_toxic.to_array(), use_container_width=True)

    with right_col:
        st.markdown("**Non-Toxic Comments**")
        safe_text = " ".join(df[df['prediction'] == 'non-toxic']['Comment'].dropna().astype(str))
        wc_safe = WordCloud(width=400, height=300, background_color='white').generate(safe_text)
        st.image(wc_safe.to_array(), use_container_width=True)

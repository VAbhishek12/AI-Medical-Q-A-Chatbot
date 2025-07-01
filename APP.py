
import streamlit as st
import wikipediaapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from deep_translator import GoogleTranslator

# Load Sentence Transformer
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Wikipedia API with User-Agent
wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='AI-Medical-QA-Bot (https://github.com/your-username)'
)

# Get content of a Wikipedia section
def get_section_text(page, title):
    for section in page.sections:
        if title.lower() in section.title.lower():
            return section.text
    return ""

# Fetch structured info from Wikipedia
def fetch_disease_info(disease):
    page = wiki.page(disease)
    if not page.exists():
        return None

    info = {
        "Symptoms": get_section_text(page, "Symptoms"),
        "Causes": get_section_text(page, "Cause"),
        "Treatments": get_section_text(page, "Treatment"),
        "Medical Findings": get_section_text(page, "Diagnosis")
    }

    for k in info:
        if not info[k]:
            info[k] = "Unknown"
    return info

# Convert info into Q&A
def generate_qa_pairs(info):
    return [
        (f"What are the {k.lower()} of this disease?", v)
        for k, v in info.items()
    ]

# Answer the user's question using semantic similarity
def answer_question(user_query, qa_pairs):
    questions = [q for q, _ in qa_pairs]
    q_embeddings = model.encode(questions)
    user_emb = model.encode([user_query])[0]
    sims = cosine_similarity([user_emb], q_embeddings)[0]
    best_idx = np.argmax(sims)
    return qa_pairs[best_idx][1], questions[best_idx], round(sims[best_idx], 2)

# Translate non-English to English
def translate_to_en(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except:
        return text

# ---- STREAMLIT APP ----
st.title("💬 AI Medical Q&A Chatbot")
st.markdown("Ask about any disease — we fetch and answer using Wikipedia + AI")

disease = st.text_input("Enter disease name (e.g. diabetes, malaria)", "")
user_question = st.text_input("Ask your medical question", "")

if st.button("Get Answer"):
    if not disease or not user_question:
        st.warning("Please enter both disease name and your question.")
    else:
        with st.spinner("Searching Wikipedia and generating answer..."):
            disease_info = fetch_disease_info(disease)
            if not disease_info:
                st.error("❌ Could not find information for that disease.")
            else:
                qa_pairs = generate_qa_pairs(disease_info)
                translated_query = translate_to_en(user_question)
                answer, matched_q, score = answer_question(translated_query, qa_pairs)
                st.success(f"**Answer:** {answer}")
                st.caption(f"Matched Q: _{matched_q}_ (Score: {score})")

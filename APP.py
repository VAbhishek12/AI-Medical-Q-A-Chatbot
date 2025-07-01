import streamlit as st
import wikipediaapi
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator

# ----------------------------
# Set up Wikipedia API
# ----------------------------
wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='AI-Medical-QA-Chatbot/1.0 (https://github.com/yourname)'
)

TITLE_MAP = {
    "Symptoms": ["Symptoms", "Signs and symptoms"],
    "Causes": ["Cause", "Causes", "Etiology"],
    "Treatment": ["Treatment", "Management"],
    "Diagnosis": ["Diagnosis", "Diagnosis and screening", "Medical testing"],
    "Prevention": ["Prevention", "Prevention and control"]
}

def get_structured_info(disease_name):
    page = wiki.page(disease_name)
    if not page.exists():
        return None

    info_dict = {}
    def recurse(sections):
        for sec in sections:
            for key, titles in TITLE_MAP.items():
                if any(title.lower() in sec.title.lower() for title in titles):
                    info_dict[key] = sec.text.strip()
            recurse(sec.sections)
    recurse(page.sections)
    return info_dict or None

def generate_qa_pairs(info_dict):
    if not info_dict:
        return []

    qa_pairs = []
    for key, value in info_dict.items():
        question = f"What are the {key.lower()} of this disease?"
        qa_pairs.append((question, value.strip()))
    return qa_pairs

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="AI Medical Q&A Chatbot", layout="centered")
st.markdown("## 💬 AI Medical Q&A Chatbot")
st.markdown("Ask about any disease — we fetch and answer using Wikipedia + AI")

# Inputs
disease = st.text_input("Enter disease name (e.g. diabetes, malaria)", "")
user_query = st.text_input("Ask your medical question", "")
get_answer = st.button("Get Answer")

# Load Model
model = SentenceTransformer("all-MiniLM-L6-v2")

if get_answer and disease and user_query:
    try:
        translated_query = GoogleTranslator(source='auto', target='en').translate(user_query)
        disease_info = get_structured_info(disease)

        if not disease_info:
            st.error("⚠️ Couldn't fetch info for this disease.")
        else:
            qa_pairs = generate_qa_pairs(disease_info)

            # DEBUG: Show extracted info
            st.markdown("#### 🧠 Extracted Medical Info")
            st.json(disease_info)

            questions = [q for q, _ in qa_pairs]
            embeddings = model.encode(questions, convert_to_tensor=True)
            query_embedding = model.encode(translated_query, convert_to_tensor=True)

            scores = util.cos_sim(query_embedding, embeddings)[0]
            best_idx = scores.argmax().item()
            best_score = scores[best_idx].item()

            matched_question, answer = qa_pairs[best_idx]
            if best_score >= 0.65:
                st.success(f"**Answer:** {answer}")
                st.markdown(f"**Matched Q:** *{matched_question}*  \n**(Score: {best_score:.2f})*")
            else:
                st.warning("Answer: Unknown")
                st.markdown(f"**Matched Q:** *{matched_question}*  \n**(Score: {best_score:.2f})*")
    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")

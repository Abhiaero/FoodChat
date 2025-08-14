import os
import sqlite3
import streamlit as st
# from dotenv import load_dotenv
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
import bcrypt
import zipfile

FAISS_ZIP_PATH = "faiss_index.zip"
FAISS_FOLDER = "faiss_index"

# Unzip only if folder doesn't exist
if not os.path.exists(FAISS_FOLDER):
    with zipfile.ZipFile(FAISS_ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(FAISS_FOLDER)


# Load environment variables
# load_dotenv()

# ------------------------------
# ---------- DB Setup ----------
# ------------------------------
DB_PATH = "chat_history.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT,
            query TEXT,
            response TEXT,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_chat(user: str, query: str, response: str, confidence: float = 1.0):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO chats (user, query, response, confidence) VALUES (?, ?, ?, ?)",
              (user, query, response, confidence))
    conn.commit()
    conn.close()

def get_chat_history(user: str, limit: int = 50):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT query, response, confidence, timestamp FROM chats WHERE user = ? ORDER BY timestamp DESC LIMIT ?",
              (user, limit))
    rows = c.fetchall()
    conn.close()
    return rows

init_db()

# ------------------------------
# --------- Guardrail ----------
# ------------------------------
def is_safe_query(query: str) -> bool:
    bad_keywords = ["bomb", "kill", "terror", "attack", "suicide", "porn", "drug"]
    q = query.lower()
    return not any(b in q for b in bad_keywords)

# ------------------------------
# ---- Fallback / Confidence ---
# ------------------------------
def is_confident_answer(ans: str, min_len: int = 25) -> bool:
    if not ans:
        return False
    low_confidence_phrases = ["i don't know", "can't find", "no information", "unable to", "not sure", "sorry"]
    a = ans.lower()
    if any(p in a for p in low_confidence_phrases):
        return False
    if len(ans.strip()) < min_len:
        return False
    return True

# ------------------------------
# ------- Load Credentials -----
# ------------------------------
with open("credentials.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# ------------------------------
# -------- Login/Register ------
# ------------------------------
auth_mode = st.sidebar.radio("Select Mode", ["Login", "Register"])

if auth_mode == "Register":
    st.subheader("🆕 Register New Account")
    reg_email = st.text_input("Email")
    reg_name = st.text_input("Full Name")
    reg_username = st.text_input("Username")
    reg_password = st.text_input("Password", type="password")

    if st.button("Register"):
        if not (reg_email and reg_name and reg_username and reg_password):
            st.warning("Please fill all fields.")
        elif reg_username in config['credentials']['usernames']:
            st.error("Username already exists.")
        else:
            hashed_pw = bcrypt.hashpw(reg_password.encode(), bcrypt.gensalt()).decode()
            config['credentials']['usernames'][reg_username] = {
                "email": reg_email,
                "name": reg_name,
                "password": hashed_pw
            }
            with open("credentials.yaml", "w") as file:
                yaml.dump(config, file)
            st.success("Account created successfully! Please switch to Login mode.")

elif auth_mode == "Login":
    authenticator.login(location="main", key="Login")
    auth_status = st.session_state.get("authentication_status")
    name = st.session_state.get("name")
    logged_username = st.session_state.get("username")

    if auth_status is False:
        st.error("Username/password is incorrect")
    elif auth_status is None:
        st.warning("Please enter your username and password")
    else:
        # ------------------------------
        # -------- Streamlit UI --------
        # ------------------------------
        st.set_page_config(page_title="FoodChat 🍔🧠", page_icon="🍽️", layout="centered")

        st.markdown(f"<h1 style='text-align:center;'>FoodChat 🍔🧠</h1>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align:center;color:gray;'>Your intelligent assistant for food reviews</h4>", unsafe_allow_html=True)
        st.markdown("---")

        with st.sidebar:
            authenticator.logout("Logout", "sidebar")
            st.sidebar.success(f"Logged in as {name}")
            # st.image("https://upload.wikimedia.org/wikipedia/en/thumb/1/12/Swiggy_logo.svg/1200px-Swiggy_logo.svg.png", width=50)
            st.image("214-2140676_illustration-of-food-on-a-plate-food.png", width=100)
            st.markdown("### 🤖 Powered by:")
            st.markdown("- Gemini 1.5 Flash")
            st.markdown("- FAISS Vector DB")
            st.markdown("- MiniLM Embeddings")
            st.markdown("---")
            st.markdown("#### 💡 Instructions:")
            st.info("Ask a question about restaurant reviews. Advanced options below.")

        query = st.text_area("Your question:", height=120)

        with st.expander("⚙️ Advanced Options"):
            top_k = st.number_input("Retriever top_k", min_value=1, max_value=20, value=10, step=1)
            rerank_top_n = st.number_input("Rerank top_n", min_value=1, max_value=10, value=5, step=1)
            temperature = st.slider("LLM temperature", min_value=0.0, max_value=1.0, value=0.3)

        if st.button("🚀 Submit"):
            if not query.strip():
                st.warning("Please enter a question.")
            elif not is_safe_query(query):
                st.error("🚫 Your query contains inappropriate content.")
            else:
                with st.spinner("Loading models & searching..."):
                    try:
                        from langchain_google_genai import ChatGoogleGenerativeAI
                        from langchain.vectorstores import FAISS
                        from langchain.embeddings import HuggingFaceEmbeddings
                        from langchain_core.prompts import PromptTemplate
                        from langchain.chains import LLMChain
                        import google.generativeai as genai

                        co = None
                        COHERE_API_KEY = os.getenv("COHERE_API_KEY")
                        if COHERE_API_KEY:
                            try:
                                import cohere
                                co = cohere.Client(COHERE_API_KEY)
                            except:
                                pass

                        def rerank_docs(query, docs):
                            if not co:
                                return docs[:rerank_top_n]
                            try:
                                rerank_response = co.rerank(query=query, documents=docs, top_n=rerank_top_n, model="rerank-english-v2.0")
                                results = getattr(rerank_response, "results", rerank_response)
                                ranked_texts = []
                                for r in results:
                                    if hasattr(r, "document"):
                                        ranked_texts.append(r.document)
                                    elif isinstance(r, dict) and "document" in r:
                                        ranked_texts.append(r["document"])
                                return ranked_texts[:rerank_top_n]
                            except:
                                return docs[:rerank_top_n]

                        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                        vectordb = FAISS.load_local(FAISS_FOLDER, embeddings=embedding, allow_dangerous_deserialization=True)
                        # vectordb = FAISS.load_local("faiss_index", embeddings=embedding, allow_dangerous_deserialization=True)

                        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=float(temperature), google_api_key=os.getenv("GEMINI_API_KEY"))
                        prompt = PromptTemplate(
                            input_variables=["context", "question"],
                            template="Context: {context}\nQuestion: {question}\nAnswer:"
                        )
                        llm_chain = LLMChain(llm=llm, prompt=prompt)

                        retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
                        retrieved_docs = retriever.get_relevant_documents(query)
                        docs_texts = [d.page_content for d in retrieved_docs]
                        if not docs_texts:
                            answer = "No relevant reviews found."
                            confidence_score = 0.0
                        else:
                            ranked_texts = rerank_docs(query, docs_texts)
                            context_text = "\n".join(ranked_texts)
                            answer = llm_chain.run({"context": context_text, "question": query})
                            confidence_score = 1.0 if is_confident_answer(answer) else 0.2

                        save_chat(logged_username, query, answer, confidence_score)

                        if not is_confident_answer(answer):
                            st.warning("⚠️ Low confidence in answer.")
                            st.info(answer)
                        else:
                            st.success(answer)

                    except Exception as e:
                        st.error(f"Error: {e}")

        with st.expander("📜 View Your Past Chats"):
            history = get_chat_history(logged_username, limit=100)
            if not history:
                st.info("No chat history found.")
            else:
                for q, r, conf, ts in history:
                    st.markdown(f"**🕒 {ts}** — Confidence: {conf:.2f}")
                    st.markdown(f"**You:** {q}")
                    st.markdown(f"**Bot:** {r}")
                    st.markdown("---")

import streamlit as st
from utils.indexing import process_pdf
from utils.chat import get_vector_store, ask_question
from uuid import uuid4

st.set_page_config(page_title="Chat with PDF", layout="wide")

# 🔶 Orange Header
st.markdown("""
    <h1 style='text-align: center; color: orange; font-size: 48px;'>
        🧠 Chat with your PDF
    </h1>
""", unsafe_allow_html=True)

# 📂 Upload Section
st.markdown("### 📂 Upload a PDF File")
uploaded_file = st.file_uploader("Choose your PDF", type="pdf", label_visibility="collapsed")

if uploaded_file:
    # Initialize Session State
    if 'vector_ready' not in st.session_state:
        st.session_state.vector_ready = False
        st.session_state.collection_name = f"pdf_{uuid4().hex[:8]}"

    # 🔧 Prepare PDF
    st.markdown("### Prepare the PDF")
    col1, col2 = st.columns([1, 5])
    with col2:
        if st.button("🔍 Prepare this PDF", use_container_width=True):
            with st.spinner("Processing PDF..."):
                process_pdf(uploaded_file, st.session_state.collection_name)
                st.session_state.vector_ready = True
            st.success("✅ PDF is ready for chat!")

# 💬 Chat Section
if st.session_state.get("vector_ready", False):
    st.markdown("---")
    st.markdown("### Chat with your PDF")
    
    query = st.text_input("Type your question 👇")
    if query:
        vector_db = get_vector_store(st.session_state.collection_name)
        with st.spinner("Searching for answer..."):
            answer = ask_question(vector_db, query)
        st.markdown(f"""
            <div style="
                background-color: #f5f5f5;
                border-left: 6px solid orange;
                border-radius: 8px;
                padding: 1rem;
                margin-top: 1rem;
                color: #333;
                font-size: 16px;
            ">
                <strong>🤖 Answer:</strong><br>{answer}
            </div>
            """, unsafe_allow_html=True
        )
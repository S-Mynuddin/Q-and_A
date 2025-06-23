import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db

# ---------- Page Configuration ----------
st.set_page_config(
    page_title="Smart Q&A System",
    page_icon="💡",
    layout="centered"
)

# ---------- Sidebar: Knowledgebase Setup ----------
with st.sidebar:
    st.title("🧠 Knowledgebase Manager")
    st.markdown("Use this section to build your Q&A knowledgebase from documents.")

    if st.button("🔄 Generate Vector DB"):
        create_vector_db()
        st.success("✅ Vector database created successfully.")

    st.markdown("---")
    st.markdown("Need help? Contact support@example.com")

# ---------- Main Area ----------
st.markdown(
    "<h1 style='text-align: center;'>🤖 Ask Your Assistant</h1>",
    unsafe_allow_html=True
)

st.markdown("Type your question below and get instant answers from your organization's knowledgebase.")

# Input
question = st.text_input("💬 Enter your question here:")

# Output
if question:
    with st.spinner("Generating answer..."):
        chain = get_qa_chain()
        response = chain(question)

    st.markdown("### 📢 Answer")
    st.success(response["result"])

    with st.expander("📄 View Source Documents"):
        for doc in response["source_documents"]:
            st.markdown(f"**Source:** {doc.metadata.get('source', 'N/A')}")
            st.write(doc.page_content)

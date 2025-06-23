from langchain_groq import ChatGroq
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os





load_dotenv()
llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile")


instructor_embeddings = HuggingFaceEmbeddings()
vectordb_file_path = "faiss_index"

def create_vector_db():
    loader = CSVLoader(file_path="codebasics_faqs.csv",source_column="prompt")
    docs = loader.load()
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings, allow_dangerous_deserialization=True)
    vectordb.save_local(vectordb_file_path)



def get_qa_chain():
    # Load the vector database from the local folder with safe override
    vectordb = FAISS.load_local(
        vectordb_file_path,
        instructor_embeddings,
        allow_dangerous_deserialization=True  # ✅ Important fix here
    )

    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return chain

if __name__ == "__main__":
    #create_vector_db()
    chain = get_qa_chain()
    print(chain("Do you have javascript course?"))
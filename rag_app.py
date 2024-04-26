import streamlit as st
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
import PyPDF2
import io




chat_model = ChatGoogleGenerativeAI(google_api_key="api_key",
                                    model="gemini-1.5-pro-latest")



st.title("Chat with your PDF") 
#st.subheader("Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention")

upload_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if upload_file is not None:
    st.text("PDF File Uploaded Successfully!")

    pdf_data = upload_file.read()
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_data))
    pdf_pages = pdf_reader.pages

    #creating the context
    context = "\n\n".join(page.extract_text() for page in pdf_pages)

    # Split Texts
    text_split = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
    texts = text_split.split_text(context)

    embedding_model = GoogleGenerativeAIEmbeddings(google_api_key="AIzaSyD1-xcd9ypJmJMmIsBpxh--lwEQSZtKrKE", 
                                               model="models/embedding-001", temperature=1)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    chat_template = ChatPromptTemplate.from_messages([
        # System Message Prompt Template
        SystemMessage(content="""You are a Helpful AI Bot. 
        You take the context and question from user. Your answer should be based on the specific context."""),
        # Human Message Prompt Template
        HumanMessagePromptTemplate.from_template("""Answer the question based on the given context.
        Context:
        {context}
    
        Question: 
        {question}
    
        Answer: """)
    ])


    output_parser = StrOutputParser()

    vector_index = Chroma.from_texts(texts, embedding_model).as_retriever()

    

    rag_chain = (
        {"context": vector_index | format_docs, "question": RunnablePassthrough()}
        | chat_template
        | chat_model
        | output_parser
    )

    user_question = st.text_area("Please enter your question:")
    if st.button("Click to display the answer"):
        response = rag_chain.invoke(user_question)
        st.write(response)
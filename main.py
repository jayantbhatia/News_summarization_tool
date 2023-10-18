import os
import streamlit as st
import pickle
import time
import langchain
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS


os.environ['OPENAI_API_KEY'] = 'YOUR_API_KEY_HERE'

st.title("News research tool 📈")
st.sidebar.title("news article urls")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL{i+1}")
    urls.append(url)


process_url_clicked = st.sidebar.button("process Urls")
file_path = "vector_idx.pkl"

main_placefolder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:
    #load data from the urls provided above
    loader = UnstructuredURLLoader(urls = urls)
    main_placefolder.text("DATA Loading started...✅")
    data = loader.load()

    #split data using RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placefolder.text("Splitting DATA started...✅")
    docs = text_splitter.split_documents(data)

    #create embeddings

    embeddings = OpenAIEmbeddings()
    vectorindex_openai = FAISS.from_documents(docs, embeddings)
    main_placefolder.text("Vector embedding started...✅")
    time.sleep(2)


    #saving the vectorindex_openai as file in pickle format

    with open(file_path, "wb") as f:
        pickle.dump(vectorindex_openai, f)

query = main_placefolder.text_input("Question: ")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorIndex = pickle.load(f)

            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorIndex.as_retriever())

            results = chain({"question": query}, return_only_outputs=True)
            st.header("Answer")

            st.write(results["answer"])

            # Display sources, if available
            sources = results.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)


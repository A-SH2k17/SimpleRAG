from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
#from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_ollama import ChatOllama
#from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter


@st.cache_data
def get_image_as_base64(file):
    import base64
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_image_as_base64("./im2.webp")
##just css background in streamlit
page_bg_img = '''
<style>
.stHeading,p,li,.stFileUploaderFileName{
    color: #000000;
}
[data-testid="stAppViewContainer"] {
    background-image: url("data:image/jpg;base64,'''+img+'''");
    background-size: cover;
    background-position: center;
[data-testid="stHeader"] {
    background-color: rgba(0, 0,0,0);
}
[data-testid="stVerticalBlock"] {
    background-color: rgba(204,204, 204, 0.8);
    border-radius: 10px;
    padding: 15px;
}
</style>
'''
def main():
    st.set_page_config(
        page_title="Ask your PDF"
    )
    st.markdown(page_bg_img, unsafe_allow_html=True)
    load_dotenv()
    if "GEMINI_API_KEY" in os.environ:
        del os.environ["GEMINI_API_KEY"]
    gemini_api_key = os.getenv("GOOGLE_API_KEY")
    
    st.header("RAG PDF Mini Project")

    st.divider()
    pdf = st.file_uploader("Upload your pdf",type="pdf")

    # llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",
    #                              google_api_key = gemini_api_key,
    #                              streaming=True)

    llm = ChatOllama(model="mistral")
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # text_splitter = CharacterTextSplitter(
        #     separator="\n",
        #     chunk_size = 1000,
        #     chunk_overlap = 200,
        #     length_function = len
        # )

        # text_splitter = SemanticChunker(HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,   # Large enough to hold complete thoughts
            chunk_overlap=80, # Crucial: ensures context isn't lost at the "cut"
            separators=["\n\n", "\n", ".", " ", ""] # Priority list for where to split
        )


        chunks = text_splitter.split_text(text)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        knowledge_base = FAISS.from_texts(chunks,embeddings)

        retriever = knowledge_base.as_retriever(
            search_kwargs={"k":8}
        )

        user_input = st.text_input("Ask your PDF is anything")

        if user_input:
            #docs = knowledge_base.search(user_input,search_type="similarity")
            docs = retriever.invoke(user_input) 

            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful cooking assistant. Answer the question using ONLY the context provided. If the context contains multiple recipes, ensure you select the one that specifically matches the user's request. If the context does not contain the specific recipe, state that clearly. If the text does not context the context expliocitly, answer that you don't know."),
                HumanMessagePromptTemplate.from_template("<context>\n{context}\n</context>\n\nQuestion: {input}")
            ])

            combine_docs_chain = create_stuff_documents_chain(llm,prompt)

            response = combine_docs_chain.stream(
                {
                    "input":user_input,
                    "context":docs
                }
            )
            
            st.write_stream(response)


if __name__ == '__main__':
    main()
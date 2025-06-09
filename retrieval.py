from typing import List, Union
from dotenv import find_dotenv, load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import init_chat_model
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


def get_default_splitter() -> RecursiveCharacterTextSplitter:
    """Returns a pre-configured text splitter."""
    return RecursiveCharacterTextSplitter(
        # Using markdown headers as separators is a good strategy
        separators=["\n### ", "\n## ", "\n# ", "\n\n", "\n", " "],
        chunk_size=1000,
        chunk_overlap=200,
    )

def get_default_embeddings() -> HuggingFaceEmbeddings:
    """Returns a pre-configured embedding model."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )


def build_retriever(
        data: Union[str, List[Document]],
        splitter: RecursiveCharacterTextSplitter = None,
        embeddings: HuggingFaceEmbeddings = None,
        top_k: int = 5):
    """Builds a retriever from either a raw text string or a list of documents.

    Args:
        Args:
        data (Union[str, List[Document]]): The source data to build the retriever from.
        splitter (RecursiveCharacterTextSplitter, optional): The text splitter to use.
                                                            Defaults to get_default_splitter().
        embeddings (HuggingFaceEmbeddings, optional): The embedding model to use.
                                                     Defaults to get_default_embeddings().
        top_k (int, optional): The number of top results to return. Defaults to 5.
    """
    splitter = splitter or get_default_splitter()
    embeddings = embeddings or get_default_embeddings()
    if isinstance(data, str):
        # If the input is a raw string, split it into chunks first
        chunks = splitter.split_text(data)
        # Then convert those chunks into Document objects
        docs = [Document(page_content=chunk) for chunk in chunks]
    elif isinstance(data, list):
        # If the input is already a list of documents, split them directly
        docs = splitter.split_documents(data)
    else:
        raise ValueError(f"Unsupported data type: {type(data)}. Must be str or List[Document].")

    index = FAISS.from_documents(docs, embeddings)
    return index.as_retriever(search_kwargs={"k": top_k})


def create_retrieval_qa(
        retriever,
        llm=None
    ) -> RetrievalQA:
    """Creates a RetrievalQA instance from a given retriever and LLM.

    Args:
        retriever (BaseRetriever): The retriever to be used by the QA chain.
        llm (LLM, optional): The language model to use. If not provided,
                                a default model will be initialized.
    """
    if llm is None:
        load_dotenv(find_dotenv())
        llm = init_chat_model("groq:meta-llama/llama-4-scout-17b-16e-instruct")
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

import base64
import os
from typing import Optional

import pandas as pd
import requests
import whisper

from bs4 import BeautifulSoup
from datetime import datetime
from dotenv import find_dotenv, load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import (
    UnstructuredPDFLoader, UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader, WebBaseLoader)
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from markdownify import markdownify as md
from youtube_transcript_api import YouTubeTranscriptApi
from yt_dlp import YoutubeDL


UNWANTED_SECTIONS = {
    "references",
    "external links",
    "further reading",
    "see also",
    "notes",
}

@tool
def get_weather_info(location: str) -> str:
    """Fetches weather information for a given location.

    Usage:
    ```
    # Initialize the tool
    weather_info_tool = Tool(
        name="get_weather_info",
        func=get_weather_info,
        description="Fetches weather information for a given location.")
    ```
    """
    load_dotenv(find_dotenv())
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    url = (
        f"https://api.openweathermap.org/data/2.5/"
        f"weather?q={location}&appid={api_key}&units=metric"
    )

    res = requests.get(url, timeout=15)
    data = res.json()
    humidity = data["main"]["humidity"]
    pressure = data["main"]["pressure"]
    wind = data["wind"]["speed"]
    description = data["weather"][0]["description"]
    temp = data["main"]["temp"]
    min_temp = data["main"]["temp_min"]
    max_temp = data["main"]["temp_max"]
    return (
        f"Weather in {location}: {description}, "
        f"Temperature: {temp}°C, Min: {min_temp}°C, Max: {max_temp}°C, "
        f"Humidity: {humidity}%, Pressure: {pressure} hPa, "
        f"Wind Speed: {wind} m/s"
    )


@tool
def add(a: int, b: int) -> int:
    """Adds two numbers together.

    Args:
        a (int): The first number.
        b (int): The second number.
    """
    return a + b


@tool
def get_sum(list_of_numbers: list[int]) -> int:
    """Sums a list of numbers.

    Args:
        list_of_numbers (list[int]): The list of numbers to sum.
    """
    return sum(list_of_numbers)


@tool
def subtract(a: int, b: int) -> int:
    """Subtracts the second number from the first.

    Args:
        a (int): The first number.
        b (int): The second number.
    """
    return a - b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers together.

    Args:
        a (int): The first number.
        b (int): The second number.
    """
    return a * b


@tool
def divide(a: int, b: int) -> float:
    """Divides the first number by the second.

    Args:
        a (int): The first number.
        b (int): The second number.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


@tool
def get_current_time_and_date() -> str:
    """Returns the current time and date in ISO format."""
    return datetime.now().isoformat()


@tool
def reverse_text(text: str) -> str:
    """Reverses the given text.

    Args:
        text (str): The text to reverse.
    """
    return text[::-1]


def build_retriever(text: str):
    """Builds a retriever from the given text.

    Args:
        text (str): The text to be used for retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n### ", "\n## ", "\n# "],
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_text(text)
    docs = [
        Document(page_content=chunk)
        for chunk in chunks
    ]
    hf_embed = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    index = FAISS.from_documents(docs, hf_embed)
    return index.as_retriever(search_kwargs={"k": 3})


def get_retrieval_qa(text: str):
    """Creates a RetrievalQA instance for the given text.
    Args:
        text (str): The text to be used for retrieval.
    """
    retriever = build_retriever(text)
    llm = init_chat_model("groq:meta-llama/llama-4-scout-17b-16e-instruct")
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )


def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # 1. Remove <script> & <style>
    for tag in soup(["script", "style"]):
        tag.decompose()

    # 2. Drop whole <section> blocks whose first heading is unwanted
    for sec in soup.find_all("section"):
        h = sec.find(["h1","h2","h3","h4","h5","h6"])
        if h and any(h.get_text(strip=True).lower().startswith(u) for u in UNWANTED_SECTIONS):
            sec.decompose()

    # 3. Additional filtering by CSS selector
    for selector in [".toc", ".navbox", ".vertical-navbox", ".hatnote", ".reflist", ".mw-references-wrap"]:
        for el in soup.select(selector):
            el.decompose()

    # 4. Isolate the main content container if present
    main = soup.find("div", class_="mw-parser-output")
    return str(main or soup)


def get_wikipedia_article(query: str, lang: str = "en") -> str:
    """Fetches a Wikipedia article for a given query and returns its content in Markdown format.

    Args:
        query (str): The search query.
        lang (str): The language code for the search. Default is "en".
    """
    headers = {
        'User-Agent': 'MyLLMAgent (llm_agent@example.com)'
    }

    # Step 1: Search
    search_url = f"https://api.wikimedia.org/core/v1/wikipedia/{lang}/search/page"
    search_params = {'q': query, 'limit': 1}
    search_response = requests.get(search_url, headers=headers, params=search_params, timeout=15)

    if search_response.status_code != 200:
        return f"Search error: {search_response.status_code}"

    results = search_response.json().get("pages", [])
    if not results:
        return "No results found."

    page = results[0]
    page_key = page["key"]

    # Step 2: Get the wiki page, only keep relevant content and convert to Markdown
    content_url = f"https://api.wikimedia.org/core/v1/wikipedia/{lang}/page/{page_key}/html"
    content_response = requests.get(content_url, timeout=15)

    if content_response.status_code != 200:
        return f"Content fetch error: {content_response.status_code}"

    html = clean_html(content_response.text)

    markdown = md(
        html,
        heading_style="ATX",
        bullets="*+-",
        table_infer_header=True,
        strip=['a', 'span']
    )
    return markdown


@tool
def wiki_search(query: str, question: str, lang: str="en") -> str:
    """Searches Wikipedia for a specific article and answers a question based on its content.

    The function retrieves a Wikipedia article based on the provided query, converts it to Markdown,
    and uses a retrieval-based QA system to answer the specified question.

    Args:
        query (str): A concise topic name with optional keywords, ideally matching the relevant Wikipedia page title.
        question (str): The question to answer using the article.
        lang (str): Language code for the Wikipedia edition to search (default: "en").
    """
    markdown = get_wikipedia_article(query, lang)
    qa = get_retrieval_qa(markdown)
    return qa.invoke(question)


@tool
def web_search(query: str) -> str:
    """Searches the web for a given query and returns the first result.

    Args:
        query (str): The search query.
    """
    search_tool = DuckDuckGoSearchRun()
    results = search_tool.invoke(query)
    if results:
        return results
    else:
        return "No results found."


@tool
def visit_website(url: str) -> str:
    """Visits a website and returns the content.

    Args:
        url (str): The URL of the website to visit.
    """
    loader = WebBaseLoader(url)
    documents = loader.load()
    if documents:
        return documents[0].page_content
    else:
        return "No content found."


@tool
def get_youtube_transcript(video_url: str, return_timestamps: bool = False) -> str:
    """Fetches the transcript of a YouTube video.

    Args:
        video_url (str): The URL of the YouTube video.
        return_timestamps (bool): If True, returns timestamps with the transcript. Otherwise, returns only the text.
    """
    try:
        video_id = video_url.split("v=")[-1]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        if return_timestamps:
            sentences = []
            for t in transcript:
                start = t["start"]
                end = start + t["duration"]
                sentences.append(f"{start:.2f} - {end:.2f}: {t['text']}")
            return "\n".join(sentences)
        else:
            return "\n".join([t["text"] for t in transcript])
    except Exception as e:
        return f"Error fetching transcript: {e}"


@tool
def get_youtube_video_info(video_url: str) -> str:
    """Fetches information about a YouTube video.

    Args:
        video_url (str): The URL of the YouTube video.
    """
    try:
        ydl_opts = {
            "quiet": True,
            "skip_download": True,
        }
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
        video_info = {
            "Title": info.get("title"),
            "Description": info.get("description"),
            "Uploader": info.get("uploader"),
            "Upload date": info.get("upload_date"),
            "Duration": info.get("duration"),
            "View count": info.get("view_count"),
            "Like count": info.get("like_count"),
        }
        video_info_filtered = {k: v for k, v in video_info.items() if v is not None}
        video_info_str = "\n".join(
            [f"{k}: {v}" for k, v in video_info_filtered.items()]
        )
        return video_info_str
    except Exception as e:
        return f"Error fetching video info: {e}"


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


@tool
def ask_about_image(image_path: str, question: str) -> str:
    """Performs vision-based question answering on an image.

    Args:
        image_path (str): The path to the image file.
        question (str): Your question about the image, as a natural language sentence. Provide as much context as possible.
    """
    load_dotenv(find_dotenv())
    llm = init_chat_model("groq:meta-llama/llama-4-maverick-17b-128e-instruct")
    prompt = ChatPromptTemplate(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please write a concise caption for the image that helps answer the following question: {question}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/{image_format};base64,{base64_image}",
                        },
                    },
                ],
            }
        ]
    )
    file_suffix = os.path.splitext(image_path)[-1]
    if file_suffix == ".png":
        image_format = "png"
    else:
        # We could handle other formats explicitly, but for simplicity we assume JPEG
        image_format = "jpeg"
    chain = prompt | llm
    response = chain.invoke(
        {
            "question": question,
            "base64_image": encode_image(image_path),
            "image_format": image_format,
        }
    )
    return response.text()


def transcribe_audio(audio_path: str) -> str:
    """Transcribes audio to text.

    Args:
        audio_path (str): The path to the audio file.
    """
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    text = result.text
    return text


def get_table_description(table: pd.DataFrame) -> str:
    """Generates a description of the table. If applicable, calculates sum and mean of numeric
    columns.

    Args:
        table (pd.DataFrame): The table to describe.
    """
    if table.empty:
        return "The table is empty."
    description = []
    total_sum = 0
    for column in table.select_dtypes(include=[int, float]).columns:
        column_sum = table[column].sum()
        column_mean = table[column].mean()
        description.append(
            f"Column '{column}': Sum = {column_sum}, Mean = {column_mean:.2f}"
        )
        total_sum += column_sum
    if total_sum:
        description.append(f"Total Sum of all numeric columns: {total_sum}")
    if description:
        description = "\n".join(description)
    else:
        description = "No numeric columns to summarize."
    # Add the number of rows and columns
    description += f"\n\nTable has {table.shape[0]} rows and {table.shape[1]} columns."
    df_as_markdown = table.to_markdown()
    description += f"\n\nTable:\n{df_as_markdown}"
    return description


@tool
def inspect_file_as_text(file_path: str) -> str:
    """This tool reads a file as markdown text. It handles [".csv", ".xlsx", ".pptx", ".wav",
    ".mp3", ".m4a", ".flac", ".pdf", ".docx"], and all other types of text files. IT DOES NOT
    HANDLE IMAGES.

    Args:
        file_path (str): The path to the file you want to read as text. If it is an image, use `vision_qa` tool.
    """
    # TODO we could also pass the file content to a retrieval chain
    try:
        suffix = os.path.splitext(file_path)[-1]
        if suffix in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"]:
            raise Exception(
                "Cannot use inspect_file_as_text tool with images: use `vision_qa` tool instead!"
            )
        if suffix in [".csv", ".tsv", ".xlsx"]:
            if suffix == ".csv":
                df = pd.read_csv(file_path)
            elif suffix == ".tsv":
                df = pd.read_csv(file_path, sep="\t")
            elif suffix == ".xlsx":
                df = pd.read_excel(file_path)
            else:
                raise Exception(f"Unsupported file type: {suffix}")
            table_description = get_table_description(df)
            return table_description
        elif suffix == ".pptx":
            doc = UnstructuredPowerPointLoader(file_path)
            return doc.load()[0].page_content
        elif suffix == ".pdf":
            doc = UnstructuredPDFLoader(file_path)
            return doc.load()[0].page_content
        elif suffix == ".docx":
            doc = UnstructuredWordDocumentLoader(file_path)
            return doc.load()[0].page_content
        elif suffix in [".wav", ".mp3", ".m4a", ".flac"]:
            return transcribe_audio(file_path)
        else:
            # All other text files
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
            return content
    except Exception as e:
        return f"Error file: {e}"

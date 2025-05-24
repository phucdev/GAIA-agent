import base64
import os
from datetime import datetime

import pandas as pd
import requests
import whisper
import wikipedia
from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import (
    UnstructuredPDFLoader, UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader, WebBaseLoader)
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from youtube_transcript_api import YouTubeTranscriptApi
from yt_dlp import YoutubeDL


@tool
def get_weather_info(location: str) -> str:
    """Fetches dummy weather information for a given location.

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


@tool
def wiki_search(query: str) -> str:
    """Searches Wikipedia for a given query and returns the summary.

    Args:
        query (str): The search query.
    """
    search_results = wikipedia.search(query)
    if not search_results:
        return "No results found."
    page_title = search_results[0]
    summary = wikipedia.summary(page_title)
    # Alternatively wikipedia.page(page_title).content[:max_length]
    return f"Title: {page_title}\n\nSummary: {summary}"


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
    llm = init_chat_model("groq:meta-llama/llama-4-scout-17b-16e-instruct")
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
                            "url": "data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ]
    )
    chain = prompt | llm
    response = chain.invoke(
        {"question": question, "base64_image": encode_image(image_path)}
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

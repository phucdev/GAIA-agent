---
title: Template Final Assignment
emoji: 🕵🏻‍♂️
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: false
hf_oauth: true
# optional, default duration is 8 hours/480 minutes. Max duration is 30 days/43200 minutes.
hf_oauth_expiration_minutes: 480
---

# 🧠 GAIA-Agent

A general-purpose **LLM agent** built to solve the final assignment of the [Hugging Face Agents course](https://huggingface.co/agents-course), which consists of 20 Level 1 questions from the [GAIA benchmark](https://huggingface.co/datasets/andrewrreed/GAIA).

The agent uses a variety of tools—web search, Wikipedia extraction, file parsing, audio transcription, and more—to gather evidence and reason through answers. It is built with [LangGraph](https://www.langchain.com/langgraph) and [LangChain](https://www.langchain.com/), and all interactions are tracked with [LangFuse](https://www.langfuse.com/).


## 🛠️ Technical Details

| Component               | Technology                                                                                                                                                                  |
|-------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Agent Framework**     | [LangGraph](https://www.langchain.com/langgraph) + [LangChain](https://www.langchain.com/)                                                                                  |
| **LLM**                 | [Meta LLaMA 4 Maverick 17B 128E Instruct](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct) via [Groq](https://groq.com/)                               |
| **Web Search**          | [SerperAPI](https://serper.dev/) (Google Search), [requests](https://docs.python-requests.org/), [Playwright](https://playwright.dev/python/) for dynamic content rendering |
| **HTML Parsing**        | [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/), [markdownify](https://github.com/matthewwithanm/python-markdownify) to convert HTML to Markdown            |
| **Wikipedia**           | [Wikimedia API](https://api.wikimedia.org/wiki/Core_REST_API)                                                                                                               |
| **File Parsing**        | [Unstructured](https://unstructured.io/) (PDF, DOCX, PPTX), [pandas](https://pandas.pydata.org/) (CSV, TSV, XLSX)                                                           |
| **Audio Transcription** | [OpenAI Whisper (base)](https://github.com/openai/whisper)                                                                                                                  |
| **Monitoring**          | [LangFuse](https://www.langfuse.com/)                                                                                                                                       |
| **Frontend UI**         | [Gradio](https://www.gradio.app/), based on the [Final Assignment Template](https://huggingface.co/spaces/agents-course/First_agent_template)                               |


## 🔐 Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/phucdev/GAIA-agent.git
    cd GAIA-agent
    ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Create API keys and set-up environment variables**:
   Use `env.example` to create a `.env` file in the root directory and replace the placeholders with your actual API keys:
   ```plaintext
   GROQ_API_KEY=your_groq_api_key
   SERPER_API_KEY=your_serper_api_key
   ... (other API keys as needed)
   ```
4. Optional: For rendering JS with Playwright, you may need some additional setup:
   ```bash
   playwright install
   sudo apt-get install libgtk-3-0
   ```
5. Run the app with:
    ```bash
    python app.py
    ```
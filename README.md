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

A **LLM agent** designed to solve the final assignment of the [HuggingFace Agents course](https://huggingface.co/agents-course).
The assignment consists of 20 level 1 questions of the [GAIA benchmark](https://huggingface.co/datasets/andrewrreed/GAIA).
This agent is built with [LangGraph](https://www.langchain.com/langgraph), [LangChain](https://www.langchain.com/), 
and is tracked using [LangFuse](https://www.langfuse.com/). 
It uses real-time web search, Wikipedia lookups, file parsing, image analysis, audio transcription to provide accurate answers.

## 🛠️ Technical Details

| Component/ Tool     | Tech Used                                                                                                                                                            |
|---------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Agent framework     | [LangGraph](https://www.langchain.com/langgraph) + [LangChain](https://www.langchain.com/)                                                                           |
| LLM                 | [Meta LLaMA 4 Maverick 17B 128e](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct) via [Groq](https://groq.com/)                                 |
| Web search          | [SerperAPI](https://serper.dev/), [requests](https://requests.readthedocs.io/en/latest/), [Playwright](https://playwright.dev/python/) for rendering JSON            |
| HTML parsing        | [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/), [markdownify](https://github.com/matthewwithanm/python-markdownify) for converting HTML to Markdown |
| Wikipedia           | [Wikimedia API](https://api.wikimedia.org/wiki/Core_REST_API)                                                                                                        |
| File parsing        | [Unstructured](https://unstructured.io/) (for PDFs, PowerPoint, Word), [pandas](https://pandas.pydata.org/) (for CSV, TSV, Excel)                                    |
| Audio transcription | [OpenAI Whisper base](https://github.com/openai/whisper)                                                                                                             |
| Agent monitoring    | [LangFuse](https://www.langfuse.com/)                                                                                                                                |
| UI / API            | [Gradio](https://www.gradio.app/) based on the [Final Assignment Template](https://huggingface.co/spaces/agents-course/First_agent_template)                         |


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
5. Then you can run the app with:
    ```bash
    python app.py
    ```
from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langfuse.callback import CallbackHandler
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated, TypedDict

from tools import (add, ask_about_image, divide, get_current_time_and_date,
                   get_sum, get_weather_info, get_youtube_video_info,
                   inspect_file_as_text, multiply, reverse_text, subtract, visit_website,
                   web_search, wiki_search_article, wiki_get_section, transcribe_audio)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


class BasicAgent:
    def __init__(self):
        load_dotenv(find_dotenv())
        llm = init_chat_model("groq:meta-llama/llama-4-maverick-17b-128e-instruct")
        system_prompt = (
            "You are a powerful general AI assistant designed to answer challenging questions using reasoning and tools.\n"
            "Each question has a single correct answer. Use clear, step-by-step reasoning and the available tools to "
            "find and verify that answer.\n"
            "Choose the appropriate tool:\n"
            "- \n"
            "- For text files, use `inspect_file_as_text` to read the file and extract relevant information.\n"
            "- For audio files, use `transcribe_audio` to transcribe the audio and extract relevant information.\n"
            "- For images, use `ask_about_image` to analyze the image and answer questions about it.\n"
            "- You can search the web using `web_search` to find information not available in your training data and"
            "use `visit_website` to retrieve the website content.\n"
            "- If you need to search for a specific wikipedia article, call `wiki_search_article`, parse `page_key` and `toc`, "
            "then only after this step call `wiki_get_section` to fetch exactly the section/subsection you need for answering. "
            "**Never** guess `page_key` or section names—always derive them from the previous tool's output.\n"
            "- For YouTube links you can use `get_youtube_video_info` to retrieve information and the transcript about a YouTube video.\n"
            "If the user supplies a file path or URL, **always** call the appropriate tool first—do not guess.  "
            "Think through the problem step by step, explain your reasoning and then clearly state your final answer using this format:\n"
            "FINAL ANSWER: [YOUR FINAL ANSWER]\n\n"
            "Your final answer must follow these rules:\n"
            "- If the answer is a number, do not use  or units (e.g. '$' or '%') unless the question explicitly asks for the unit.\n"
            "- If the answer is a string, use as few words as possible and do not use articles, abbreviations, or numeric digits.\n"
            "- If the answer is a comma-separated list, follow the above rules for each element. Separate elements with a comma and a single space.\n"
            "- If the answer is a single string, capitalize the first letter of the final answer and do NOT add punctuation unless the question requires it.\n"
            "Do not say “I cannot answer” or “no answer found”. Success is mandatory. "
            "Only apply criteria the question specifies—no extra assumptions. "
            "You have access to everything you need to solve this."
        )
        tools = [
            get_weather_info,
            add,
            get_sum,
            subtract,
            multiply,
            divide,
            get_current_time_and_date,
            wiki_get_section,
            wiki_search_article,
            web_search,
            visit_website,
            inspect_file_as_text,
            transcribe_audio,
            ask_about_image,
            reverse_text,
            get_youtube_video_info,
        ]
        llm_with_tools = llm.bind_tools(tools)

        def assistant(state: AgentState):
            sys_msg = SystemMessage(content=system_prompt)
            return {"messages": llm_with_tools.invoke([sys_msg] + state["messages"])}

        graph_builder = StateGraph(AgentState)

        graph_builder.add_node("assistant", assistant)
        graph_builder.add_node("tools", ToolNode(tools))

        graph_builder.add_edge(START, "assistant")
        graph_builder.add_conditional_edges(
            "assistant",
            tools_condition,
        )
        graph_builder.add_edge("tools", "assistant")

        self.agent = graph_builder.compile()
        self.langfuse_handler = CallbackHandler()
        print("BasicAgent initialized.")

    def __call__(self, question: str) -> str:
        print(f"Agent received question (first 50 chars): {question[:50]}...")
        messages = [HumanMessage(content=question)]
        state = self.agent.invoke({"messages": messages}, config={"callbacks": [self.langfuse_handler]})
        response_string = state["messages"][-1].content
        print(f"Agent's response: {response_string}")
        return response_string

from typing import Annotated, Optional, TypedDict

from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent

from tools import (add, ask_about_image, divide, get_current_time_and_date,
                   get_sum, get_weather_info, get_youtube_transcript,
                   get_youtube_video_info, inspect_file_as_text, multiply,
                   reverse_text, subtract, visit_website, web_search,
                   wiki_search)


class AgentState(TypedDict):
    input_file: Optional[str]  # Contains file path
    messages: Annotated[list[AnyMessage], add_messages]


class BasicAgent:
    def __init__(self):
        load_dotenv(find_dotenv())
        model = init_chat_model("groq:meta-llama/llama-4-scout-17b-16e-instruct")
        system_prompt = (
            "You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer "
            "with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR "
            "as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a "
            "number, don't use comma to write your number neither use units such as $ or percent sign unless specified "
            "otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), "
            "and write the digits in plain text unless specified otherwise. If you are asked for a comma separated "
            "list, apply the above rules depending of whether the element to be put in the list is a number or a string."
            "Give it all you can: I know for a fact that you have access to all the relevant tools to solve it and find "
            "the correct answer (the answer does exist). Failure or 'I cannot answer' or 'None found' will not be "
            "tolerated, success will be rewarded. Run verification steps if that's needed, you must make sure you find "
            "the correct answer! "
        )
        tools = [
            get_weather_info,
            add,
            get_sum,
            subtract,
            multiply,
            divide,
            get_current_time_and_date,
            wiki_search,
            web_search,
            visit_website,
            inspect_file_as_text,
            ask_about_image,
            reverse_text,
            get_youtube_video_info,
            get_youtube_transcript,
        ]

        self.agent = create_react_agent(model=model, tools=tools, prompt=system_prompt)
        print("BasicAgent initialized.")

    def __call__(self, question: str) -> str:
        print(f"Agent received question (first 50 chars): {question[:50]}...")
        messages = [HumanMessage(content=question)]
        response = self.agent.invoke({"messages": messages})
        response_string = response["messages"][-1].content
        print(f"Agent's response: {response_string}")
        return response_string

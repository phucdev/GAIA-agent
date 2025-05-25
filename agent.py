from typing import Annotated, TypedDict

from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langfuse.callback import CallbackHandler
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from tools import (add, ask_about_image, divide, get_current_time_and_date,
                   get_sum, get_weather_info, get_youtube_transcript,
                   get_youtube_video_info, inspect_file_as_text, multiply,
                   reverse_text, subtract, visit_website, web_search,
                   wiki_search)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


class BasicAgent:
    def __init__(self):
        load_dotenv(find_dotenv())
        llm = init_chat_model("groq:meta-llama/llama-4-scout-17b-16e-instruct")
        system_prompt = (
            "You are a powerful general AI assistant designed to answer challenging questions using reasoning and tools.\n"
            "Each question has a correct answer, and you are expected to find it.\n"
            "Use all available tools — including calculator, search, or other domain-specific utilities — to verify your work or retrieve information.\n"
            "If a question requires computation or external data, you must call the appropriate tool.\n"
            "Think through the problem step by step, then clearly state your final answer using this format:\n"
            "FINAL ANSWER: [YOUR FINAL ANSWER]\n\n"
            "Your final answer must follow these rules:\n"
            "- If the answer is a number, do not use commas or units (unless explicitly requested).\n"
            "- If the answer is a string, use as few words as possible and do not use articles, abbreviations, or numeric digits.\n"
            "- If the answer is a comma-separated list, follow the above rules for each element.\n"
            "- If the answer is a string and unless you are asked to provide a list, capitalize the first letter of the final answer.\n"
            "Do not say “I cannot answer” or “no answer found”. Success is mandatory. "
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
            wiki_search,
            web_search,
            visit_website,
            inspect_file_as_text,
            ask_about_image,
            reverse_text,
            get_youtube_video_info,
            get_youtube_transcript,
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

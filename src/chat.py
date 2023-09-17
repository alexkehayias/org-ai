import cmd
import sys
from typing import List

from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder, PromptTemplate
from langchain.tools import format_tool_to_openai_function
from langchain.utilities import SerpAPIWrapper
from langchain.vectorstores.faiss import FAISS
from langchain.agents.agent_toolkits import PlayWrightBrowserToolkit
from langchain.tools.playwright.utils import (
    create_sync_playwright_browser,
)

from config import PROJECT_ROOT_DIR, OPENAI_API_KEY, SERP_API_KEY


BROWSER = create_sync_playwright_browser()
BROWSER_TOOLKIT = PlayWrightBrowserToolkit.from_browser(sync_browser=BROWSER)
BROWSER_TOOLS = BROWSER_TOOLKIT.get_tools()


NOTES_TEMPLATE = """
You are a helpful personal assistant helping to answer the question ("QUESTION") by looking through related notes I've written ("SUMMARIES").
You are really good at finding connections between notes and finding the key ideas to answer questions.
Given the related notes ("SUMMARIES"), choose the notes that are most relevant to the question ("QUESTION") and write an answer ("ANSWER") with a list of sources ("SOURCES") you drew from.
The list of sources ("SOURCES") should include the title of the note and the ID.
ALWAYS return a "SOURCES" part in your answer unless there were none.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.

QUESTION: {question}
=========
SUMMARIES:
{summaries}
=========
ANSWER:
"""
NOTES_PROMPT = PromptTemplate(
    template=NOTES_TEMPLATE, input_variables=["summaries", "question"]
)


def search_index():
    return FAISS.load_local(
        folder_path=f"{PROJECT_ROOT_DIR}/index",
        index_name="notes_search",
        embeddings=OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
        ),
    )


def task_index():
    return FAISS.load_local(
        folder_path=f"{PROJECT_ROOT_DIR}/index",
        index_name="tasks_search",
        embeddings=OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
        ),
    )


def _gpt_answer(index, question, **kwargs):
    print(f"_gpt_answer got additional kwargs: {kwargs}")
    result = NOTES_CHAIN(
        {
            "input_documents": index.similarity_search(question, k=4),
            "question": question,
            **kwargs
        },
        return_only_outputs=True,
    )

    return result["output_text"]


def gpt_answer_notes(question, **kwargs):
    index = search_index()
    return _gpt_answer(index, question, **kwargs)


def gpt_answer_tasks(question, **kwargs):
    index = task_index()
    return _gpt_answer(index, question, **kwargs)


LLM = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name="gpt-3.5-turbo-0613",
    temperature=0,
)


SEARCH = SerpAPIWrapper(
    serpapi_api_key = SERP_API_KEY
)


TOOLS = [
    Tool(
        name = "Search",
        func=SEARCH.run,
        description="Useful for when you need to answer questions about current events or the current state of the world. The input to this should be a single search term.",
    ),
    Tool(
        name = "Notes",
        func=gpt_answer_notes,
        description="Useful for when you need to respond to a question about my notes or something I've written about before. The input to this should be a question or a phrase. If the input is a filename, only return content for the note that matches the filename.",
    ),
    Tool(
        name = "Tasks",
        func=gpt_answer_tasks,
        description="Useful for when you need to respond to a question about tasks or todo lists or projects or meetings.",
    ),
]
TOOLS += BROWSER_TOOLS


FUNCTIONS = [format_tool_to_openai_function(t) for t in TOOLS]


MEMORY = ConversationBufferMemory(memory_key="memory", return_messages=True)


AGENT = initialize_agent(
    tools=TOOLS,
    functions=FUNCTIONS,
    llm=LLM,
    agent=AgentType.OPENAI_FUNCTIONS,
    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    },
    memory=MEMORY,
    verbose=True
)


NOTES_CHAIN = load_qa_with_sources_chain(
    ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-3.5-turbo-0613",
        temperature=0,
    ),
    chain_type="stuff",
    prompt=NOTES_PROMPT,
)


class ChatCmd(cmd.Cmd):
    prompt = '> '
    commands: List[str] = []

    def do_list(self, line):
        print(self.commands)

    def default(self, line):
        answer = AGENT.run(input=line)
        print(answer)
        # Write your code here by handling the input entered
        self.commands.append(line)

    def do_exit(self, line):
        return True


if __name__ == '__main__':
    args = sys.argv
    ChatCmd().cmdloop()

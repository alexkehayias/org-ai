import cmd
import sys
from typing import List

from langchain.agents import AgentType, Tool, initialize_agent
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder, PromptTemplate
from langchain.tools import format_tool_to_openai_function, BaseTool
from langchain.utilities import SerpAPIWrapper
from langchain.agents.agent_toolkits import PlayWrightBrowserToolkit
from langchain.tools.playwright.utils import (
    create_sync_playwright_browser,
)

from config import OPENAI_API_KEY, SERP_API_KEY
from index import search_index, task_index


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


NOTES_CHAIN = load_qa_with_sources_chain(
    ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-4-1106-preview",
        temperature=0,
    ),
    chain_type="stuff",
    prompt=NOTES_PROMPT,
)


def gpt_answer_notes(question: str) -> str:
    index = search_index()
    result = NOTES_CHAIN(
        {
            "input_documents": index.similarity_search(question, k=4),
            "question": question,
        },
        return_only_outputs=True,
    )
    result_str: str = result["output_text"]
    return result_str


TASK_METADATA = [
    AttributeInfo(
        name="title",
        description="The title of the task, meeting, or heading",
        type="string",
    ),
    AttributeInfo(
        name="status",
        description="The status of the task if available",
        type="string",
    ),
    AttributeInfo(
        name="is_task",
        description="Whether this is a task or not",
        type="bool",
    ),
    AttributeInfo(
        name="is_meeting",
        description="Whether this is a meeting or not",
        type="bool",
    ),
    AttributeInfo(
        name="created_date",
        description="The date the task or meeting was created as an ISO formatted date string",
        type="string",
    ),
    AttributeInfo(
        name="deadline",
        description="The date the task is due as an ISO formatted date string",
        type="string",
    ),
    AttributeInfo(
        name="scheduled",
        description="The date the task is scheduled to be done as an ISO formatted date string",
        type="string",
    ),
    AttributeInfo(
        name="tags",
        description="A list of tags as a comma separated string that categorize the item if available",
        type="string",
    ),
]


AGENT_LLM = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name="gpt-4-1106-preview",
    temperature=0,
)


TASKS_LLM = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name="gpt-4-1106-preview",
    temperature=0,
)


def gpt_answer_tasks(question: str) -> List[Document]:
    index = task_index()
    document_content_description = "Tasks"
    retriever = SelfQueryRetriever.from_llm(
        TASKS_LLM,
        index,
        document_content_description,
        TASK_METADATA,
        verbose=True,
    )

    return retriever.get_relevant_documents(question) or []


SEARCH = SerpAPIWrapper(
    serpapi_api_key=SERP_API_KEY,
    search_engine="google",
)


TOOLS: List[Tool | BaseTool] = [
    Tool(
        name="Search",
        func=SEARCH.run,
        description="Useful for when you need to answer questions about current events or the current state of the world. The input to this should be a single search term.",
    ),
    Tool(
        name="Tasks",
        func=gpt_answer_tasks,
        description="Useful for when you need to respond to a question about tasks or todo lists or projects.",
    ),
    Tool(
        name="Meetings",
        func=gpt_answer_tasks,
        description="Useful for when you need to respond to a question about meetings.",
    ),
    Tool(
        name="Notes",
        func=gpt_answer_notes,
        description="Useful for when you need to respond to a question about my notes or something I've written about before.",
    ),
]
TOOLS += BROWSER_TOOLS


FUNCTIONS = [format_tool_to_openai_function(t) for t in TOOLS]


MEMORY = ConversationBufferMemory(memory_key="memory", return_messages=True)


AGENT = initialize_agent(
    tools=TOOLS,
    functions=FUNCTIONS,
    llm=AGENT_LLM,
    agent=AgentType.OPENAI_FUNCTIONS,
    agent_kwargs={
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    },
    memory=MEMORY,
    verbose=True,
    max_iterations=4,
)


class ChatCmd(cmd.Cmd):
    prompt = "> "
    commands: List[str] = []

    def do_list(self, line: str) -> None:
        print(self.commands)

    def default(self, line: str) -> None:
        answer = AGENT.run(input=line)
        print(answer)
        # Write your code here by handling the input entered
        self.commands.append(line)

    def do_exit(self, line: str) -> bool:
        return True


if __name__ == "__main__":
    args = sys.argv
    ChatCmd().cmdloop()

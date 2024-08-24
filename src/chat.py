import cmd
import sys
import subprocess
from typing import List

from langchain.globals import set_verbose
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import (
    create_sync_playwright_browser,
)

from config import OPENAI_API_KEY, SERP_API_KEY
from index import search_index
from langchain_core.tools import BaseTool, Tool


set_verbose(True)


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
        model_name="gpt-4o",
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


ORGQL_TEMPLATE = """
You are an AI assistant designed to help convert natural language questions into org-ql queries for use with Org mode in Emacs. org-ql queries are used to filter and search for specific entries in Org files based on various criteria like tags, properties, and timestamps.

Instructions:

	1.	Read the user's question carefully.
	2.	Identify the key criteria mentioned in the question (e.g., tags, properties, deadlines, scheduled dates, priorities, etc.).
	3.	Construct an appropriate org-ql query that reflects the user's criteria following the examples (EXAMPLES) closely.
        4.      Only return the org-ql query with no other text or explanation

EXAMPLES

User Question:

“Find all tasks that are due this week.”

Converted org-ql Query:

(org-ql-select (org-agenda-files) '(and (todo) (deadline :from -7 :to 7)) :action '(org-get-heading t t))

User Question:

“Find all tasks that are due next week.”

Converted org-ql Query:

(org-ql-select (org-agenda-files) '(and (todo) (deadline :from 0 :to 7)) :action '(org-get-heading t t))

User Question:

“Show me meetings I had in the last month.”

Converted org-ql Query:

(org-ql-select (org-agenda-files) '(and (tags "meeting") (ts :from -30 :to today)) :action '(org-get-heading t t))

User Question:

“Show me meetings I had last week.”

Converted org-ql Query:

(org-ql-select (org-agenda-files) '(and (tags "meeting") (ts :from -7 :to today)) :action '(org-get-heading t t))

User Question:

“List all tasks with priority 'A' that are scheduled for tomorrow.”

Converted org-ql Query:

(org-ql-select (org-agenda-files) '(and (priority "A") (scheduled :to 1)) :action '(org-get-heading t t))

User Question:

“Find all personal tasks that have a deadline next week.”

Converted org-ql Query:

(org-ql-select (org-agenda-files) '(and (category "personal") (todo) (deadline :from 7 :to 14)) :action '(org-get-heading t t))

User Question:

“Show tasks that are not done and have the tag 'admin'.”

Converted org-ql Query:

(org-ql-select (org-agenda-files) '(and (todo) (tags "admin")) :action '(org-get-heading t t))

User Question:

“Find all tasks with a priority of 'B' that were created this month.”

Converted org-ql Query:

(org-ql-select (org-agenda-files) '(and (todo) (priority "B") (ts :from -30 :to today)) :action '(org-get-heading t t))

User Question:

{question}

Converted org-ql Query:
"""


ORGQL_PROMPT = PromptTemplate(
    template=ORGQL_TEMPLATE, input_variables=["question"]
)


ORGQL_CHAIN = LLMChain(
    prompt=ORGQL_PROMPT,
    llm=ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-4o",
        temperature=0,
    ),
)


def gpt_answer_orgql(question: str) -> str:
    org_ql_query = ORGQL_CHAIN.run(question=question)
    print(org_ql_query)
    # DANGER!!!
    template = """(message (with-output-to-string (princ (mapconcat 'identity {ORGQL} "\\n"))))"""
    command = [
        'emacs',
        '-l',
        '$HOME/.emacs.d/init.el',
        '--batch',
        '--eval',
        template.replace('{ORGQL}', org_ql_query)
    ]
    process = subprocess.run(command, capture_output=True, text=True)
    result: str = process.stderr
    return result


AGENT_LLM = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name="gpt-4o",
    temperature=0,
)


SEARCH = SerpAPIWrapper(
    serpapi_api_key=SERP_API_KEY,
    search_engine="google",
)


TOOLS: List[Tool | BaseTool] = [
    Tool(
        name="OrgMode",
        func=gpt_answer_orgql,
        description="Useful for when you need to respond to a question about tasks, todos, meetings, and org-mode.",
    ),
    Tool(
        name="Search",
        func=SEARCH.run,
        description="Useful for when you need to answer questions about current events or the current state of the world. The input to this should be a single search term.",
    ),
    Tool(
        name="Notes",
        func=gpt_answer_notes,
        description="Useful for when you need to respond to a question about my notes or something I've written about before.",
    ),
]
TOOLS += BROWSER_TOOLS
MEMORY = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

AGENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Always include your sources at the end of your response."),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)
AGENT = create_openai_tools_agent(
    tools=TOOLS,
    llm=AGENT_LLM,
    prompt=AGENT_PROMPT,
)
AGENT_EXECUTOR = AgentExecutor(
    agent=AGENT, tools=TOOLS, memory=MEMORY
)


class ChatCmd(cmd.Cmd):
    prompt = "> "
    commands: List[str] = []

    def do_list(self, line: str) -> None:
        print(self.commands)

    def default(self, line: str) -> None:
        answer = AGENT_EXECUTOR.invoke({"input": line})
        print(answer['output'])
        # Write your code here by handling the input entered
        self.commands.append(line)

    def do_exit(self, line: str) -> bool:
        return True


if __name__ == "__main__":
    args = sys.argv
    ChatCmd().cmdloop()

import cmd
import sys
from typing import List

from langchain.globals import set_verbose
from langchain.agents import AgentExecutor, Tool, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
)
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferMemory
from langchain_community.tools import BaseTool
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import (
    create_sync_playwright_browser,
)
from langchain.retrievers.self_query.chroma import ChromaTranslator

from config import OPENAI_API_KEY, SERP_API_KEY
from index import search_index, task_index


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
        model_name="gpt-4",
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
        description="The timestamp the task or meeting was created formatted as an integer",
        type="int",
    ),
    AttributeInfo(
        name="deadline",
        description="The timestamp the task is due",
        type="int",
    ),
    AttributeInfo(
        name="scheduled",
        description="The timestamp the task is scheduled",
        type="int",
    ),
    AttributeInfo(
        name="tags",
        description="A list of tags as a comma separated string that categorize the item if available",
        type="string",
    ),
]


AGENT_LLM = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name="gpt-4",
    temperature=0,
)


TASKS_LLM = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name="gpt-4",
    temperature=0,
)


TASKS_SCHEMA = """\
<< Structured Request Schema >>
When responding use a markdown code snippet with a JSON object formatted in the following schema:

```json
{{{{
    "query": string \\ text string to compare to document contents
    "filter": string \\ logical condition statement for filtering documents
}}}}
```

The query string should contain only text that is expected to match the contents of documents. Any conditions in the filter should not be mentioned in the query as well.

A logical condition statement is composed of one or more comparison and logical operation statements.

A comparison statement takes the form: `comp(attr, val)`:
- `comp` ({allowed_comparators}): comparator
- `attr` (string):  name of attribute to apply the comparison to
- `val` (string): is the comparison value

A logical operation statement takes the form `op(statement1, statement2, ...)`:
- `op` ({allowed_operators}): logical operator
- `statement1`, `statement2`, ... (comparison statements or logical operation statements): one or more statements to apply the operation to

Make sure that you only use the comparators and logical operators listed above and no others.
Make sure that filters only refer to attributes that exist in the data source.
Make sure that filters only use the attributed names with its function names if there are functions applied on them.
Make sure that filters only use timestamp in seconds as an integer when handling date data typed values. If you need to convert them, translate the date into a unix epoch timstamp with UTC timezone and double check that it is the correct year. NEVER use the `eq` operator for timestamps, if the requested date is a single day, use the `gte` operator for the requested date AND a `lte` operator for the requested date plus one day. Be very careful with dates.
Make sure that filters take into account the descriptions of attributes and only make comparisons that are feasible given the type of data being stored.
Make sure that filters are only used as needed. If there are no filters that should be applied return "NO_FILTER" for the filter value.\
"""
TASKS_SCHEMA_PROMPT = PromptTemplate.from_template(TASKS_SCHEMA)

def gpt_answer_tasks(question: str) -> List[Document]:
    index = task_index()

    prompt = get_query_constructor_prompt(
        document_contents="My tasks",
        attribute_info=TASK_METADATA,
        schema_prompt=TASKS_SCHEMA_PROMPT,
    )
    output_parser = StructuredQueryOutputParser.from_components()
    query_constructor = prompt | TASKS_LLM | output_parser

    print(query_constructor.invoke({"query": question}))

    retriever = SelfQueryRetriever(
        query_constructor=query_constructor,
        structured_query_translator=ChromaTranslator(),
        vectorstore=index,
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
MEMORY = ConversationBufferMemory(memory_key="memory", return_messages=True)

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
AGENT_EXECUTOR = AgentExecutor(agent=AGENT, tools=TOOLS, memory=MEMORY)


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

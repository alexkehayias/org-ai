import cmd
import sys
import glob
import re
from typing import List

from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
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


def gpt_answer(question):
    index = search_index()
    # TODO: use the links in the document metadata to extract related
    # docs
    result = NOTES_CHAIN(
        {
            "input_documents": index.similarity_search(question, k=4),
            "question": question,
        },
        return_only_outputs=True,
    )

    return result["output_text"]


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
        func=gpt_answer,
        description="Useful for when you need to respond to a question about my notes or something I've written about before. The input to this should be a question or a phrase. If the input is a filename, only return content for the note that matches the filename.",
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


def extract_links_and_replace_text(text):
    regex = r'\[\[id:(.+?)\]\[(.+?)\]\]'
    matches = re.findall(regex, text)
    for match in matches:
        text = text.replace("[[id:{}][{}]]".format(match[0], match[1]), match[1])
    return matches, text


def extract_note(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    id = ''
    title = ''
    tags = []
    links = []
    body = ''
    read_body = False

    for line in lines:
        if line.startswith('#+TITLE:'):
            title = line.replace('#+TITLE: ', '').capitalize().strip()
            continue

        if line.startswith('#+FILETAGS:'):
            tags = line.replace('#+FILETAGS: ', '').split(' ')
            tags = [i.strip() for i in tags]
            continue

        if line.startswith('#+'):
            continue

        if line.startswith(':PROPERTIES:'):
            continue

        if line.startswith(':ID:'):
            id = line.replace(':ID:', '').strip()
            continue

        if line.startswith(':END:'):
            read_body = True
            continue

        if read_body:
            matches, text = extract_links_and_replace_text(line)
            links += matches
            body += text

    return id, title, tags, body.strip(), links


SKIP_NOTES_WITH_TAGS = [
    "journal",
    "project",
    "entity",
    "section",
]


def build_search_index_and_embeddings(path):
    """
    Builds a search index based on vectors of embeddings.
    """
    sources = []
    for filename in glob.glob(f"{path}/*.org"):
        id, title, tags, body, links = extract_note(filename)

        if not body:
            print(f"Skipping note because the body is empty: {filename}")
            continue

        # Skip anything we don't want indexed
        if set(tags).intersection(set(SKIP_NOTES_WITH_TAGS)):
            print(f"Skipping note because it contains skippable tags: {filename} {tags}")
            continue

        print(f"Indexing {filename}")

        doc = Document(
            page_content=body,
            metadata={
                "id": id,
                "links": links,
                "source": filename,
            },
        )

        sources.append(doc)

    index = FAISS.from_documents(
        documents=sources,
        embedding=OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
        ),
    )
    index.save_local(
        folder_path=f"{PROJECT_ROOT_DIR}/index",
        index_name="notes_search",
    )



class ChatCmd(cmd.Cmd):
    prompt = '> '
    commands: List[str] = []

    def do_list(self, line):
        print(self.commands)

    def default(self, line):
        print(line[::])
        answer = AGENT.run(input=line)
        print(answer)
        # Write your code here by handling the input entered
        self.commands.append(line)

    def do_exit(self, line):
        return True


if __name__ == '__main__':
    args = sys.argv
    if len(args) == 1:
        ChatCmd().cmdloop()
    if len(args) == 3:
        command = sys.argv[1]
        if command == "index":
            path = sys.argv[2]
            build_search_index_and_embeddings(path)
            print("Indexing complete!")
        else:
            print(f"Unknown command \"{command}\"")
    else:
        print("Too many arguments")

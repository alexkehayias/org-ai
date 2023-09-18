import os
import sys
import glob
import re
from enum import Enum

from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS, Chroma, utils
from langchain.document_loaders import UnstructuredOrgModeLoader

from config import PROJECT_ROOT_DIR, OPENAI_API_KEY


OPENAI_EMBEDDINGS = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
)


def search_index():
    return FAISS.load_local(
        folder_path=f"{PROJECT_ROOT_DIR}/index",
        index_name="notes_search",
        embeddings=OPENAI_EMBEDDINGS
    )


def task_index():
    return Chroma(
        persist_directory=f"{PROJECT_ROOT_DIR}/index",
        embedding_function=OPENAI_EMBEDDINGS
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
                "title": title,
                "tags": tags,
                "links": links,
                "source": filename,
            },
        )

        sources.append(doc)

    index = Chroma.from_documents(
        documents=sources,
        embedding=OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
        ),
    )
    index.persist()


def build_task_search_index_and_embeddings():
    """
    Builds a search index for org-agenda tasks based on vectors of
    embeddings.
    """

    # Get the list of agenda files by reading the default
    # customizations file for emacs
    emacs_customization_file = os.path.expanduser(f"~/.emacs.d/.customizations.el")

    agenda_files = []
    with open(emacs_customization_file) as f:
        found = False
        for line in f.readlines():
            if found:
                # Transform a lisp list to a python list
                file_paths = line.replace("'(", "").replace('))', '').strip().split(" ")
                for i in file_paths:
                    cleaned = i.replace("\"", "")
                    full_path = os.path.expanduser(cleaned)
                    agenda_files.append(full_path)
                break
            # Detect the configuration for org-agenda-files
            if line.startswith(" '(org-agenda-files"):
                # Set the sentinel value to indicate the next line is the
                # value we were looking for
                found = True
                continue

    sources = []
    for filename in agenda_files:
        print(f"Working on {filename}")
        loader = UnstructuredOrgModeLoader(file_path=filename, mode="elements")

        # TODO: work.org file isn't loading due to an xml error
        try:
            docs = loader.load()
        except Exception as e:
            print(f"Error: \n{e}")

        sources.extend(utils.filter_complex_metadata(docs))

    index = Chroma.from_documents(
        persist_directory=f"{PROJECT_ROOT_DIR}/index",
        documents=sources,
        embedding=OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
        ),
    )
    index.persist()


class Command(str, Enum):
    Index = "index"


class IndexSubCommand(str, Enum):
    Notes = "notes"
    Tasks = "tasks"


if __name__ == '__main__':
    args = sys.argv
    command = sys.argv[1]
    if command == IndexSubCommand.Notes:
        path = sys.argv[2]
        build_search_index_and_embeddings(path)
        print("Indexing complete!")
    elif command == IndexSubCommand.Tasks:
        build_task_search_index_and_embeddings()
        print("Indexing complete!")
    else:
        print(f"Unknown sub-command \"{command}\"")

import os
import sys
import glob
import re
import hashlib
from enum import Enum
from datetime import datetime
from typing import List, Tuple, Optional, Iterator

import chromadb
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS, Chroma, utils
import orgparse

from config import PROJECT_ROOT_DIR, OPENAI_API_KEY


OPENAI_EMBEDDINGS = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
)
TASK_DOC_COLLECTION_NAME = "tasks_v1"
TASK_INDEX_CHROMA_CLIENT_SETTINGS = chromadb.config.Settings(
    is_persistent=True,
    persist_directory=f"{PROJECT_ROOT_DIR}/index",
    anonymized_telemetry=False,
)


def hash_id(s: str) -> str:
    """
    Returns a hex encoded hash ID of the string.
    """

    return hashlib.sha1(s.encode('utf-8')).hexdigest()


def search_index() -> FAISS:
    return FAISS.load_local(
        folder_path=f"{PROJECT_ROOT_DIR}/index",
        index_name="notes_search",
        embeddings=OPENAI_EMBEDDINGS
    )


def task_index() -> Chroma:
    return Chroma(
        persist_directory=f"{PROJECT_ROOT_DIR}/index",
#        collection_name=TASK_DOC_COLLECTION_NAME,
        client_settings=TASK_INDEX_CHROMA_CLIENT_SETTINGS,
        embedding_function=OPENAI_EMBEDDINGS,
    )


def extract_links_and_replace_text(text) -> Tuple[List[str], str]:
    regex = r'\[\[id:(.+?)\]\[(.+?)\]\]'
    matches = re.findall(regex, text)
    for match in matches:
        text = text.replace("[[id:{}][{}]]".format(match[0], match[1]), match[1])
    return matches, text


def extract_note(file_path: str) -> Tuple[str, str, List[str], str, List[str]]:
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


def build_search_index_and_embeddings(path: str):
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
        persist_directory=f"{PROJECT_ROOT_DIR}/index",
#        collection_name=TASK_DOC_COLLECTION_NAME,
        documents=sources,
        embedding=OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
        ),
    )
    index.persist()


def org_agenda_files(emacs_customization_file: str) -> List[str]:
    """
    Get the list of agenda files by reading a customizations file for
    emacs. By default, this file is located in
    ~/.emacs.d/.customizations.el
    """
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

    return agenda_files


def org_element_to_doc(element, parent_metadata: Optional[dict]=None) -> Document:
    # TODO: Convert an element into a document
    # - If it's a TODO add metadata for a task
    # - If it's a meeting add metadata for that
    # - If it's an interview
    title = element.heading
    body = f"{title}\n\n{element.body}"

    # Optional metadata
    org_id = element.properties.get('ID') or hash_id(title)
    date_list = element.datelist
    created_date = datetime.strftime(date_list[0].start, '%Y-%m-%d') if date_list else None
    deadline = datetime.strftime(element.deadline.start, '%Y-%m-%d') if element.deadline else None
    scheduled = datetime.strftime(element.scheduled.start, '%Y-%m-%d') if element.scheduled else None

    # Clean up tags from orgparse which doesn't split by space
    # Note: also includes tags from the parent element
    tags = element.tags or []
    tags = [i for t in element.tags for i in t.split(' ')]

    # TODO: Handle recursion
    # if len(element.children) > 0:
    #     pass

    is_task = bool(element.todo)
    is_meeting = 'meeting' in tags

    metadata= {
        'id': org_id,
        'is_task': is_task,
        'is_meeting': is_meeting,
        'parent_id': parent_metadata['id'] if parent_metadata else None,
        # Chroma can't do arrays for metadata so change this to a string
        'tags': ', '.join(tags),
        'title': title,
        'created_date': created_date,
        'deadline': deadline,
        'scheduled': scheduled,
        'status': element.todo,
    }

    return Document(
        page_content=body,
        metadata=metadata
    )


def org_task_file_to_docs(file_path: str) -> Iterator[Document]:
    # TODO: recursively parse an org tasks file into a collection of
    # documents. Handles nested elements to better parse org trees.
    root = orgparse.load(file_path)

    # Parsing filetags doesn't seem to work correctly in orgparse so
    # we do it manually
    file_tags = root.get_file_property('filetags')
    tags = file_tags.split(' ') if file_tags else []

    title = root.get_file_property('title') or file_path
    org_id = root.properties.get('ID') or hash_id(title)
    created_date = root.get_file_property('date')

    # Parsing the body of a file root doesn't work correctly in
    # orgparse (it doesn't remove properties)
    lines = root.body.split('\n')
    lines = [i for i in lines if i and not i.startswith('#+')]
    body = '\n'.join(lines)
    body = f"{title}\n\n{body}"

    is_project = 'project' in tags

    # If this is a project, we know each item is related to the parent
    if is_project:
        # TODO handle turning project files into documents
        for el in root.children:
            yield org_element_to_doc(
                el,
                parent_metadata={
                    'id': org_id,
                    'title': title,
                    'created_date': created_date
                }
            )

        metadata = {
            'id': org_id,
            'title': title,
            'created_date': created_date,
            'tags': tags,
            'source': file_path,
            # TODO add metadata for child docs
        }

        # Create the overall document for the project
        yield Document(
            page_content=body,
            metadata=metadata
        )

    # Otherwise, we know each heading is independent
    for el in root.children:
        yield org_element_to_doc(el)


def build_task_search_index_and_embeddings():
    """
    Builds a search index for org-agenda tasks based on vectors of
    embeddings.
    """
    emacs_customization_file = os.path.expanduser("~/.emacs.d/.customizations.el")
    agenda_files = org_agenda_files(emacs_customization_file)

    documents = []
    for filename in agenda_files:
        print(f"Working on {filename}")
        for doc in org_task_file_to_docs(file_path=filename):
            documents.append(doc)

    # This is needed because all values in metadata must be strings
    documents = utils.filter_complex_metadata(documents)

    index = Chroma.from_documents(
        client_settings=TASK_INDEX_CHROMA_CLIENT_SETTINGS,
        documents=documents,
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

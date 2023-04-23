import glob

import nltk
from txtai.embeddings import Embeddings
from txtai.pipeline import Textractor


def download_corpus():
    nltk.download('punkt')


TEXTRACTOR = Textractor(paragraphs=True)


def extract_paragraphs(filepath: str):
    return TEXTRACTOR(filepath)


# Generate the index from org files
def notes_generator(path):
    for filename in glob.glob(f"{path}/*.org"):
        title = filename.split("/")[-1]
        if "journal" in title:
            continue

        for paragraph in extract_paragraphs(filename):
            yield title, paragraph


def build_index():
    # Create embeddings model, backed by sentence-transformers &
    # transformers Downloads the transformers on the first run if they
    # don't exist already
    embeddings = Embeddings({
        "path": "sentence-transformers/nli-mpnet-base-v2",
        "content": True,
        "objects": True
    })
    embeddings.index(
        [(uid, {"title": title, "text": text, "length": len(text)}, None)
         for uid, (title, text) in enumerate(notes_generator("/Users/alex/Org/notes"))]
     )
    embeddings.save("test_index_3")


def search_index(query: str):
    embeddings = Embeddings()
    embeddings.load("test_index_3")
    results = embeddings.search(query, 1)
    return results


while True:
    value = input("\nEnter search query:\n")
    embeddings = Embeddings()
    embeddings.load("test_index_3")
    for i in embeddings.search(f"select text, title, score from txtai where similar('{value}') and score >= 0.15 order by score desc"):
        print("-----------------------")
        print(f"{i['title']}\n{i['text']}\n")

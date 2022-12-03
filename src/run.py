import glob
from txtai.embeddings import Embeddings


# Generate the index from org files
def notes_generator(path):
    for filename in glob.glob(f"{path}/*.org"):
        with open(filename, "r") as f:
            title = filename.split("/")[-1]
            if title.startswith("journal"):
                continue

            # TODO split between properties and 'See also'
            text = f.read()
            yield title, text


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
    result = embeddings.search(query, 1)
    if result:
        return result[0]["title"]

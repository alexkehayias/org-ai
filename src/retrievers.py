# Write a langchain compatible Retriever that makes an API call to
# collect search results
from typing import Any, List
import requests

from langchain.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings


class NoteVectorStore(VectorStore):
    """
    Query the indexer service API for my notes
    https://github.com/alexkehayias/indexer
    """

    def __init__(self, api_url: str):
        self.api_url = api_url

    @staticmethod
    def _results_to_docs(results: List[Any]) -> List[Document]:
        return [
            Document(
                page_content=i.get('body', i.get('title')),
                metadata={**i, **{'source': i['file_name']}}
            )
            for i in results
        ]

    def _make_request(self, query: str, include_similarity: bool = False):
        response = requests.get(
            f"{self.api_url}/notes/search?query={query}&include_body=true{'&include_similarity=true' if include_similarity else ''}",
        )

        if response.status_code != 200:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

        return response.json()

    def search(self, query: str, top_k: int = 10) -> List[Document]:
        response_data = self._make_request(query, include_similarity=False)
        response_data = self._make_request(query)
        results = response_data.get('results', [])

        return self._results_to_docs(results[:top_k])

    def similarity_search(self, query: str, top_k: int = 10) -> List[Document]:
        response_data = self._make_request(query, include_similarity=True)
        response_data = self._make_request(query)
        results = response_data.get('results', [])

        return self._results_to_docs(results[:top_k])

    def with_config(self, **kwargs):
        return self

    @classmethod
    def from_texts(cls, texts: List[str], embeddings: Embeddings, api_url: str, api_key: str = None) -> 'NoteVectorStore':
        return cls(api_url=api_url)

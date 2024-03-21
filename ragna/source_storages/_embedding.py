import os

from ragna.core import Chunk

from ragna.core._components import Embedding, GenericEmbeddingModel
from typing import Union, List

class MiniLML6v2(GenericEmbeddingModel):
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def embed_chunks(self, chunks: list[Chunk]) -> list[Embedding]:
        return [Embedding(self.embed_text(chunk.text), chunk) for chunk in chunks]

    def embed_text(self, text: str) -> list[float]:
        return self.model.encode(text).tolist()


class TogetherEmbeddingModel(GenericEmbeddingModel):
    _MAX_SEQUENCE_LENGTH = 32768
    _MODEL_API_STRING: str

    def __init__(self):
        import os
        import requests

        api_key = os.getenv("TOGETHER_API_KEY")
        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        self.url = "https://api.together.xyz/api/v1/embeddings"
        self.model_api_string = self._MODEL_API_STRING

        self.session = requests.Session()

    def embed_chunks(self, chunks: list[Chunk]) -> list[Embedding]:
        texts = [chunk.text for chunk in chunks]
        embedding_floats = self.embed_text(texts)
        embeddings_list = zip(embedding_floats, chunks)
        embeddings = [Embedding(item[0], item[1]) for item in embeddings_list]
        return embeddings

    def embed_text(self, text: Union[List[str],str]) -> list[float]:

        embeddings = []

        if type(text) == str:
            text = [text]

        import tqdm
        for single_text in tqdm.tqdm(text):
            response = self.session.post(
                self.url,
                headers=self.headers,
                json={
                    "input": single_text,
                    "model": self.model_api_string
                }
            )
            if response.status_code != 200:
                raise ValueError(f"Request failed with status code {response.status_code}: {response.text}")

            embeddings += [item['embedding'] for item in response.json()['data']]
        return embeddings


class M2Bert80M32KRetrievalTogether(TogetherEmbeddingModel):
    _MAX_SEQUENCE_LENGTH = 32768
    _MODEL_API_STRING = 'togethercomputer/m2-bert-80M-32k-retrieval'


class M2Bert80M8KRetrievalTogether(TogetherEmbeddingModel):
    _MAX_SEQUENCE_LENGTH = 8192
    _MODEL_API_STRING = 'togethercomputer/m2-bert-80M-8k-retrieval'


class M2Bert80M2KRetrievalTogether(TogetherEmbeddingModel):
    _MAX_SEQUENCE_LENGTH = 2048
    _MODEL_API_STRING = 'togethercomputer/m2-bert-80M-2k-retrieval'

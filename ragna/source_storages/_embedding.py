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


def batch_text(text: list[str], threshold: int):
    batches = [[text[0]]]
    text.pop(0)

    for text_piece in text:
        if max([len(max(batches[-1], key=len)), len(text_piece)]) * (len(batches[-1]) + 1) < threshold:
            batches[-1].append(text_piece)
        else:
            batches.append([text_piece])
    return batches


class M2Bert80M32KRetrievalLocal(GenericEmbeddingModel):
    _MAX_SEQUENCE_LENGTH = 32768

    def __init__(self):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "togethercomputer/m2-bert-80M-32k-retrieval",
            trust_remote_code=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased",
            model_max_length=self._MAX_SEQUENCE_LENGTH
        )

    def embed_chunks(self, chunks: list[Chunk]) -> list[Embedding]:
        texts = [chunk.text for chunk in chunks]
        embedding_floats = self.embed_text(texts)
        embeddings_list = zip(embedding_floats, chunks)
        embeddings = [Embedding(item[0], item[1]) for item in embeddings_list]
        return embeddings

    def embed_text(self, text: Union[str, List[str]]) -> list[float]:
        if type(text) is not list:
            text = [text]

        embeddings = []

        import tqdm
        for single_text in tqdm.tqdm(text):
            input_ids = self.tokenizer(
                single_text,
                return_tensors="pt",
                padding="longest",
                return_token_type_ids=False,
                truncation=True,
                max_length=self._MAX_SEQUENCE_LENGTH
            )
            print(len(max(input_ids.data['input_ids'].tolist(), key=len)))
            outputs = self.model(**input_ids)
            embeddings += outputs['sentence_embedding'].tolist()
        return embeddings


class M2Bert80M32KRetrievalTogether(GenericEmbeddingModel):
    _MAX_SEQUENCE_LENGTH = 32768
    url = "https://api.together.xyz/api/v1/embeddings"
    model_api_string = 'togethercomputer/m2-bert-80M-32k-retrieval'
    def __init__(self):
        import os
        from transformers import AutoTokenizer
        api_key = os.getenv("TOGETHER_API_KEY")
        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        import requests
        self.session = requests.Session()

        self.tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased",
            model_max_length=self._MAX_SEQUENCE_LENGTH
        )

    def embed_chunks(self, chunks: list[Chunk]) -> list[Embedding]:
        texts = [chunk.text for chunk in chunks]
        embedding_floats = self.embed_text(texts)
        embeddings_list = zip(embedding_floats, chunks)
        embeddings = [Embedding(item[0], item[1]) for item in embeddings_list]
        return embeddings

    def get_tokenizer(self):
        return self.tokenizer

    def embed_text(self, text: Union[List[str],str]) -> list[float]:

        embeddings = []
        import tqdm
        for batch in tqdm.tqdm(batch_text(text, threshold=16000)):
            response = self.session.post(
                self.url,
                headers=self.headers,
                json={
                    "input": batch,
                    "model": self.model_api_string
                }
            )
            if response.status_code != 200:
                raise ValueError(f"Request failed with status code {response.status_code}: {response.text}")

            embeddings += [item['embedding'] for item in response.json()['data']]
        return embeddings

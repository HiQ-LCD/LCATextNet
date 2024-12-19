# -*- coding: utf-8 -*-
# @Time    : 2024/12/19 11:44
# @Author  : Biao
# @File    : embedding_model.py

import json
from typing import List
from .config import EMBEDDING_URL
import requests


class EmbeddingModel:

    def _load_model(self):
        pass

    def text_embedding(self, text, use_translate=False):
        pass


class RemoteEmbeddingModel(EmbeddingModel):

    def __init__(self, model_name="nomic-embed-text"):
        self.model_name = model_name
        self._load_model()

    def _load_model(self):
        pass

    def _text_embedding(self, text):
        ollama_input_data = {"model": self.model_name, "prompt": text}
        response = requests.post(data=json.dumps(ollama_input_data), url=EMBEDDING_URL)
        ollama_output_data = json.loads(response.content.decode())["embedding"]
        return ollama_output_data

    def text_embedding(self, text):
        return self._text_embedding(text)

    def document_embedding(self, documents: List[str]):
        collection = []
        for d in documents:
            collection.append(self._text_embedding(d))
        return collection

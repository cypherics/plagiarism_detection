from typing import Optional, List

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import HashingVectorizer


class Vectorizer:
    def run(self, **kwargs):
        raise NotImplementedError


class SE(Vectorizer):
    def __init__(self, model_id: Optional[str] = "all-MiniLM-L6-v2"):
        self._model = SentenceTransformer(model_id)

    def run(self, sentences: List, **kwargs):
        _embeddings = list()
        return self._model.encode(sentences)


class TFIDFHashing(Vectorizer):
    def __init__(self, n_features=20):
        self._model = HashingVectorizer(n_features=n_features)

    def run(self, sentences: List, **kwargs):
        return self._model.fit_transform(sentences).toarray()

from typing import Optional, Tuple, Dict

import hnswlib
import numpy as np
import pandas as pd
from nltk import word_tokenize
from sentence_transformers import SentenceTransformer, util
from torch import Tensor

from plagarism.constants import INPUT_COL
from plagarism.util import (
    case_conversion,
    apply_regex,
    remove_stop_words,
    lemmatize,
    sentences_from_para,
    remove_symbols_numbers_letters_consonants,
    generate_para_df,
)


class NN:
    def __init__(self, nn, **kwargs):
        self.__nn = nn

    def query(self, query_embeddings, **kwargs):
        raise NotImplementedError

    def init(self, embeddings, **kwargs):
        raise NotImplementedError


class ANN(NN):
    def __init__(self, search_space="cosine", nn: int = 10, **kwargs):
        super().__init__(nn, **kwargs)
        self._search_space = search_space
        self._nn = nn

        self._index = None

    def init(
        self,
        embeddings: np.ndarray,
        ef_construction: int = 400,
        m: int = 64,
        ef: int = 50,
    ):
        n, dim = embeddings.shape
        self._index = hnswlib.Index(space=self._search_space, dim=dim)
        self._index.init_index(max_elements=n, ef_construction=ef_construction, M=m)
        self._index.add_items(embeddings, list(range(n)))

        assert ef > self._nn, f"Given ef = {ef} and nn = {self._nn}, Expected ef > nn"
        self._index.set_ef(ef)

    def query(self, query_embeddings: np.ndarray, **kwargs) -> Tuple:
        return self._index.knn_query(query_embeddings, self._nn)


class SemanticSim(NN):
    def __init__(self, nn: int = 10, **kwargs):
        super().__init__(nn, **kwargs)
        self._nn = nn

    def init(self, embeddings, **kwargs):
        pass

    def query(self, query_embeddings, **kwargs):
        _sim = util.semantic_search(
            query_embeddings, kwargs["corpus_embeddings"], self._nn
        )
        _id = []
        _sim_v = []
        for i in _sim:
            _i = []
            _j = []
            for j in i:
                _i.append(j["corpus_id"])
                _j.append(j["score"])
            _id.append(_i)
            _sim_v.append(_j)
        return np.array(_id), np.array(_sim_v)


class Detector:
    def __init__(self):
        self.model = None
        self.sim = None

    @classmethod
    def init_with_input_path(cls, **kwargs):
        raise NotImplementedError

    def save(self, **kwargs):
        raise NotImplementedError

    def pre_process_data(self, data):
        raise NotImplementedError

    def generate_embeddings(self, **kwargs):
        raise NotImplementedError

    def load_embedding_model(self, model_id: Optional[str] = "all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_id)

    def load_sim(self, nn: int = 10, **kwargs):
        self.sim = ANN(nn=nn, **kwargs)

    def detect(self, **kwargs):
        raise NotImplementedError

    def generate_embedding_index(self, **kwargs):
        raise NotImplementedError


class ExtrinsicDetector(Detector):
    def __init__(self, source, suspicious):
        super().__init__()
        self._source = source
        self._suspicious = suspicious

        self._source_embeddings = []
        self._suspicious_embeddings = []

        self._source_sentences = {}
        self._suspicious_sentences = {}

    def pre_process_data(self, text: str):
        text = case_conversion(text)
        text = apply_regex(text)

        tokenized_text = word_tokenize(text)
        tokenized_text = remove_symbols_numbers_letters_consonants(tokenized_text)
        tokenized_text = remove_stop_words(tokenized_text)
        tokenized_text = lemmatize(tokenized_text)
        return tokenized_text

    def _input(self, df: pd.DataFrame) -> (np.ndarray, Dict):
        _ip_sent = []
        _embeddings = []
        for idx, row in df.iterrows():
            tokenized_sentences = []
            for sent in sentences_from_para(row[INPUT_COL]):
                tokenized_text = self.pre_process_data(row[INPUT_COL])
                tokenized_sentences.append(" ".join(tokenized_text))
                _ip_sent.append(sent)

            _embeddings.extend(self.model.encode(tokenized_sentences))
        _sentences = dict(zip(list(range(len(_ip_sent))), _ip_sent))

        return np.array(_embeddings), _sentences

    def generate_embeddings(self):
        self._source_embeddings, self._source_sentences = self._input(self._source)
        self._suspicious_embeddings, self._suspicious_sentences = self._input(
            self._suspicious
        )

    @classmethod
    def init_with_input_path(cls, source: str, suspicious: str):
        source = generate_para_df(source)
        suspicious = generate_para_df(suspicious)
        return cls(source, suspicious)

    def save(self, **kwargs):
        pass

    def detect(self, **kwargs):
        return self.sim.query(
            query_embeddings=self._suspicious_embeddings,
            courpus_embeddings=self._source_embeddings,
        )

    def generate_embedding_index(self, **kwargs):
        self.sim.init(embeddings=self._source_embeddings, **kwargs)

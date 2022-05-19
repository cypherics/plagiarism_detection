from typing import Optional

import pandas as pd
from sentence_transformers import SentenceTransformer

from plagarism.constants import INPUT_COL, PARA_COL
from plagarism.normalization import (
    case_conversion,
    apply_regex,
    remove_stop_words,
    lemmatize,
    sentences_from_para,
)


class Detector:
    def __init__(self):
        self.model = None

    @classmethod
    def init_with_input_path(cls, **kwargs):
        raise NotImplementedError

    def save(self, **kwargs):
        raise NotImplementedError

    def pre_process_data(self, data):
        raise NotImplementedError

    def _embedding_gen(self, data):
        self.pre_process_data(data)
        yield self.model.encode(list())

    def generate_embeddings(self, **kwargs):
        raise NotImplementedError

    def load_embedding_model(self, model_id: Optional[str] = None):
        self.model = SentenceTransformer(model_id)

    @staticmethod
    def _generate_para_df(filepath):
        para_content = list()
        with open(filepath, "r") as rf:
            _content = []
            for line in rf:
                if line == "\n":
                    para_content.append(" ".join(_content))
                    _content = []
                else:
                    _content.append(line.strip())
        return pd.DataFrame(para_content, columns=[PARA_COL])


class ExtrinsicDetector(Detector):
    def __init__(self, source, suspicious):
        super().__init__()
        self._source = source
        self._suspicious = suspicious

        self._source_embeddings = []
        self._suspicious_embeddings = []

    def pre_process_data(self, data):
        text = case_conversion(data)
        text = apply_regex(text)
        tokenized_text = remove_stop_words(text)
        tokenized_text = lemmatize(tokenized_text)
        return tokenized_text

    def generate_embeddings(self):
        for _, row in self._source.iterrows():
            for sent in sentences_from_para(row[INPUT_COL]):
                normalize_text = self.pre_process_data(sent)
                self._source_embeddings.append(self.model.encode(normalize_text))

    @classmethod
    def init_with_input_path(cls, source: str, suspicious: str):
        source = cls._generate_para_df(source)
        suspicious = cls._generate_para_df(suspicious)
        return cls(source, suspicious)

    def save(self, **kwargs):
        pass

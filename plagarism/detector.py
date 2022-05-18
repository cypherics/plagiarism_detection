import pandas as pd
from sentence_transformers import SentenceTransformer

from plagarism.constants import INPUT_COL
from plagarism.normalization import case_conversion, apply_regex, remove_stop_words, lemmatization


class Detector:
    def __init__(self, df: pd.DataFrame):
        self._df = df
        self._embeddings = None

    @classmethod
    def init_with_input_path(cls, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _text_normalization(text):
        text = case_conversion(text)
        text = apply_regex(text)
        tokenized_text = remove_stop_words(text)
        tokenized_text = lemmatization(tokenized_text)
        return tokenized_text

    def normalize_data(self, **kwargs):
        self._df[INPUT_COL] = self._df[INPUT_COL].apply(self._text_normalization)

    def save(self, **kwargs):
        raise NotImplementedError

    def _process_data_for_embeddings(self, data: pd.DataFrame):
        pass

    def generate_embeddings(self, model_id: str, data: pd.DataFrame, **kwargs):
        model = SentenceTransformer(model_id)
        self._process_data_for_embeddings(data)
        self._embeddings = model.encode(list())



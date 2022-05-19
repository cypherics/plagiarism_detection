from typing import Optional

from nltk import word_tokenize
from sentence_transformers import SentenceTransformer

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

    def load_embedding_model(self, model_id: Optional[str] = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_id)


class ExtrinsicDetector(Detector):
    def __init__(self, source, suspicious):
        super().__init__()
        self._source = source
        self._suspicious = suspicious

        self._source_embeddings = []
        self._suspicious_embeddings = []

    def pre_process_data(self, data: str):
        tokenized_sentences = []
        for sent in sentences_from_para(data):
            text = case_conversion(sent)
            text = apply_regex(text)

            tokenized_text = word_tokenize(text)
            tokenized_text = remove_symbols_numbers_letters_consonants(tokenized_text)
            tokenized_text = remove_stop_words(tokenized_text)
            tokenized_text = lemmatize(tokenized_text)
            tokenized_sentences.append(" ".join(tokenized_text))
        return tokenized_sentences

    def generate_embeddings(self):
        for _, row in self._source.iterrows():
            tokenized_sentences = self.pre_process_data(row[INPUT_COL])
            self._source_embeddings.extend(self.model.encode(tokenized_sentences))

    @classmethod
    def init_with_input_path(cls, source: str, suspicious: str):
        source = generate_para_df(source)
        suspicious = generate_para_df(suspicious)
        return cls(source, suspicious)

    def save(self, **kwargs):
        pass

import csv

import hnswlib
import numpy as np
import pandas as pd
import tensorflow_hub as hub

from typing import Optional, List
from nltk import word_tokenize
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel

from plagarism.constants import INPUT_COL
from plagarism.pipeline.base import PipelineComponent, Pipeline
from plagarism.util import (
    case_conversion,
    apply_regex,
    remove_symbols_numbers_letters_consonants,
    remove_stop_words,
    lemmatize,
    generate_para_df,
    sentences_from_para,
)


class DataNormalization(PipelineComponent):
    @staticmethod
    def _normalize(data: Optional[pd.DataFrame] = None):
        _ip_sent = []
        tokenized_sentences = []
        for idx, row in data.iterrows():
            for sent in sentences_from_para(row[INPUT_COL]):

                text = case_conversion(sent)
                text = apply_regex(text)

                tokenized_text = word_tokenize(text)
                tokenized_text = remove_symbols_numbers_letters_consonants(
                    tokenized_text
                )
                tokenized_text = remove_stop_words(tokenized_text)
                tokenized_text = lemmatize(tokenized_text)
                if len(tokenized_text) >= 5:
                    tokenized_sentences.append(" ".join(tokenized_text))
                    _ip_sent.append(sent)
        _sentences = dict(zip(list(range(len(_ip_sent))), _ip_sent))
        return _sentences, tokenized_sentences

    def execute(self, data: Optional[pd.DataFrame] = None, **kwargs):
        raise NotImplementedError


class SourceDataNormalization(DataNormalization):
    def execute(self, **kwargs):
        _sentences, tokenized_sentences = self._normalize(kwargs["source_df"])
        return {
            **{
                "source_sentences": _sentences,
                "source_tokenize_sentences": tokenized_sentences,
            },
            **kwargs,
        }


class SuspiciousDataNormalization(DataNormalization):
    def execute(self, data: Optional[pd.DataFrame] = None, **kwargs):
        _sentences, tokenized_sentences = self._normalize(kwargs["suspicious_df"])
        return {
            **{
                "suspicious_sentences": _sentences,
                "suspicious_tokenize_sentences": tokenized_sentences,
            },
            **kwargs,
        }


class ReadSourceData(PipelineComponent):
    def execute(self, source: Optional[str] = None, **kwargs) -> dict:
        return {**{"source_df": generate_para_df(source)}, **kwargs}


class ReadSuspiciousData(PipelineComponent):
    def execute(self, suspicious: Optional[str] = None, **kwargs) -> dict:
        return {**{"suspicious_df": generate_para_df(suspicious)}, **kwargs}


class CollectSourceWithSuspicious(PipelineComponent):
    def execute(self, **kwargs) -> dict:
        return {
            **{
                "sentences": {
                    "source_tokenize_sentences": kwargs["source_tokenize_sentences"],
                    "suspicious_tokenize_sentences": kwargs[
                        "suspicious_tokenize_sentences"
                    ],
                }
            },
            **kwargs,
        }


class USE(PipelineComponent):
    def __init__(
        self,
        model_id: Optional[
            str
        ] = "https://tfhub.dev/google/universal-sentence-encoder-large/5",
    ):
        # hub.KerasLayer(MD_PTH)
        # URL = "https://tfhub.dev/google/universal-sentence-encoder/4"
        # TRANSFORMER_MODEL = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
        self._model = hub.load(model_id)

    def execute(self, **kwargs) -> dict:
        sentences = kwargs["sentences"]

        _embedding_collection = {}
        for key, value in sentences.items():
            _embedding_collection[f"{key}_embeddings"] = self._model(value).numpy()
        return {**{"embeddings": _embedding_collection}, **kwargs}


class SE(PipelineComponent):
    def __init__(self, model_id: Optional[str] = "all-MiniLM-L6-v2"):
        self._model = SentenceTransformer(model_id)

    def execute(self, **kwargs) -> dict:
        sentences = kwargs["sentences"]

        _embedding_collection = {}
        for key, value in sentences.items():
            _embedding_collection[f"{key}_embeddings"] = self._model.encode(value)
        return {**{"embeddings": _embedding_collection}, **kwargs}


class BertCase(PipelineComponent):
    def __init__(self):
        self._tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        self._model = BertModel.from_pretrained("bert-base-cased")

    def execute(self, **kwargs) -> dict:
        sentences = kwargs["sentences"]

        _embedding_collection = {}
        for key, value in sentences.items():
            output_emd = []
            for i in value:
                encoded_input = self._tokenizer(i, return_tensors="pt")
                output_emd.append(
                    self._model(**encoded_input)["last_hidden_state"]
                    .detach()
                    .numpy()[0, -4:, :]
                    .mean(axis=0)
                )
            _embedding_collection[f"{key}_embeddings"] = np.array(output_emd)
        return {**{"embeddings": _embedding_collection}, **kwargs}


class NN(PipelineComponent):
    def execute(self, **kwargs) -> dict:
        embeddings_collection = kwargs["embeddings"]

        in_emd = embeddings_collection["source_tokenize_sentences_embeddings"]
        query_emd = embeddings_collection["suspicious_tokenize_sentences_embeddings"]

        ef_construction = 400
        m = 64
        ef = 50
        nn = 10
        n, dim = in_emd.shape
        index = hnswlib.Index(space="cosine", dim=dim)
        index.init_index(max_elements=n, ef_construction=ef_construction, M=m)
        index.add_items(in_emd, list(range(n)))

        index.set_ef(ef)
        nn, distances = index.knn_query(query_emd, nn)
        return {**{"detection": {"nn": nn, "score": 1 - distances}}, **kwargs}


class Output(PipelineComponent):
    def execute(self, **kwargs) -> dict:
        distance_threshold = 0.20

        nn = kwargs["detection"]["nn"]
        score = kwargs["detection"]["score"]

        source_sentences = kwargs["source_sentences"]
        suspicious_sentences = kwargs["suspicious_sentences"]

        header = ["suspicious", "plagarised from source", "score"]
        with open("output.csv", "w", encoding="UTF8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for ix, neighbours in enumerate(nn):
                for yx, neighbour in enumerate(neighbours):
                    if score[ix][yx] < distance_threshold:
                        continue
                    writer.writerow(
                        [suspicious_sentences[ix], source_sentences[yx], score[ix][yx]]
                    )
        return {**{"output": "output.csv"}, **kwargs}


class ExtrinsicPlagiarismPipeline(Pipeline):
    def components(self, component: List):
        for comp in component:
            self.pipe_line_components.append(comp.init())

    def execute(self, **data):
        for comp in self.pipe_line_components:
            data = comp.execute(**data)

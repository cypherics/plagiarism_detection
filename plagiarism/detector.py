import logging
import os
from typing import Optional, List

import hnswlib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from plagiarism.constants import ALL_DIR
from plagiarism.util import generate_para_df, normalize_data, get_sentences_from_df

logger = logging.getLogger()


class DocumentCollection:
    def __init__(self):
        self._df = None
        self.collect_source()

    @staticmethod
    def _sentences(files):
        logger.debug("SENTENCE GENERATION")
        df = pd.DataFrame()
        for i, file in enumerate(tqdm(files)):
            df1 = pd.DataFrame()

            sentences = get_sentences_from_df(generate_para_df(file))
            df1["filename"] = [str(os.path.basename(file))] * len(sentences)
            df1["sentences"] = sentences
            df = pd.concat([df, df1], ignore_index=True, sort=False)
        df["idx"] = range(0, len(df))
        return df

    def get_sentences(self):
        return self._df["sentences"].to_list()

    def collect_source(self):
        raise NotImplementedError


class SourceDocumentCollection(DocumentCollection):
    def __init__(self):
        super().__init__()

    def collect_source(self):
        files_collection = list()
        for sub_dir in ALL_DIR:
            for root, dirs, files in os.walk(sub_dir):
                for file in files:
                    if file.endswith(".txt"):
                        if "source-document" in file:
                            files_collection.append(os.path.join(root, file))

        self._df = self._sentences(files_collection)


class SuspiciousDocumentCollection(DocumentCollection):
    def __init__(self):
        super().__init__()

    def collect_source(self):
        files_collection = list()
        for sub_dir in ALL_DIR:
            for root, dirs, files in os.walk(sub_dir):
                for file in files:
                    if file.endswith(".txt"):
                        if "suspicious-document" in file:
                            files_collection.append(os.path.join(root, file))

        self._df = self._sentences(files_collection)


class Plagiarism:
    def __init__(self, model_id: Optional[str] = "all-MiniLM-L6-v2"):
        self._index = None
        self._model = SentenceTransformer(model_id)

    def sentence_embeddings(self, sentences: List):
        _embeddings = list()
        for sent in sentences:
            _embeddings.append(self._model.encode(sent))
        return np.array(_embeddings)

    def index_embedding(self, embeddings, pth, ef_construction=400, m=64, ef=50):
        n, dim = embeddings.shape
        self._index = hnswlib.Index(space="cosine", dim=dim)
        self._index.init_index(max_elements=n, ef_construction=ef_construction, M=m)
        self._index.add_items(embeddings, list(range(n)))
        self._index.save_index(pth)

        self._index.set_ef(ef)

    def load_index(self, pth, dim, ef):
        self._index = hnswlib.Index(space="cosine", dim=dim)
        self._index.load_index(pth)
        self._index.set_ef(ef)

    @staticmethod
    def normalize_sentences(sentences):
        tokenized_sentences = []
        for sent in sentences:
            tokenized_text = normalize_data(sent)
            if len(tokenized_text) >= 5:
                tokenized_sentences.append(" ".join(tokenized_text))
        return tokenized_sentences

    def query(self, **kwargs):
        raise NotImplementedError

    def generate_index(self, **kwargs):
        raise NotImplementedError


class Extrinsic(Plagiarism):
    def __init__(
        self,
        source_doc: SourceDocumentCollection,
        suspicious_doc: SuspiciousDocumentCollection,
        model_id: Optional[str] = "all-MiniLM-L6-v2",
    ):
        super().__init__(model_id)
        self.source_doc = source_doc
        self.suspicious_doc = suspicious_doc

    def generate_index(self, index_pth, ef_construction=400, m=64, ef=50):
        logger.debug("INDEX GENERATION")
        source_sentences = self.source_doc.get_sentences()
        embeddings = self.sentence_embeddings(
            self.normalize_sentences(source_sentences)
        )
        self.index_embedding(
            embeddings, index_pth, ef_construction=ef_construction, m=m, ef=ef
        )

    def query(self, nn=10):
        logger.debug("QUERY IN PROGRESS")
        suspicious_sentences = self.suspicious_doc.get_sentences()
        embeddings = self.sentence_embeddings(
            self.normalize_sentences(suspicious_sentences)
        )

        nn, distances = self._index.knn_query(embeddings, nn)

        return nn, 1 - distances

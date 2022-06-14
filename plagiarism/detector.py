import csv
import logging
import os
from typing import Optional, List, Any

import hnswlib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from tqdm import tqdm

from plagiarism.constants import ALL_DIR
from plagiarism.util import generate_para_df, normalize_data, get_sentences_from_df

logger = logging.getLogger()


class DocumentCollection:
    def __init__(self, pth):
        self._df = None
        self.pth = pth

        self.collect()

    @staticmethod
    def _sentences(files):
        logger.debug("SENTENCE GENERATION")
        df = pd.DataFrame()
        for i, file in enumerate(tqdm(files)):
            df1 = pd.DataFrame()

            sentences = get_sentences_from_df(generate_para_df(file))
            tokenized_sentences = dict()

            for j, sent in enumerate(sentences):
                tokenized_text = normalize_data(sent)
                if len(tokenized_text) >= 5:
                    tokenized_sentences[sent] = " ".join(tokenized_text)

            df1["filename"] = [str(os.path.basename(file))] * len(tokenized_sentences)
            df1["sentences"] = list(tokenized_sentences.keys())
            df1["normalised"] = list(tokenized_sentences.values())

            df = pd.concat([df, df1], ignore_index=True, sort=False)
        df["idx"] = range(0, len(df))
        return df

    def get_sentences(self):
        return self._df["sentences"].to_list()

    def get_normalised_sentences(self):
        return self._df["normalised"].to_list()

    def get_filename(self, idx):
        return self._df.loc[self._df["idx"] == idx]["filename"].values[0]

    def collect(self):
        raise NotImplementedError


class SourceDocumentCollection(DocumentCollection):
    def __init__(self, pth):
        super().__init__(pth)

    def collect(self):
        if os.path.exists(self.pth):
            logger.debug(f"READING FROM {self.pth}")
            self._df = pd.read_csv(self.pth)
        else:
            files_collection = list()
            for sub_dir in ALL_DIR:
                for root, dirs, files in os.walk(sub_dir):
                    for file in files:
                        if file.endswith(".txt"):
                            if "source-document" in file:
                                files_collection.append(os.path.join(root, file))

            self._df = self._sentences(files_collection)
            logger.info(f"Saving Generated sentences at {self.pth}")
            self._df.to_csv(self.pth)


class SuspiciousDocumentCollection(DocumentCollection):
    def __init__(self, pth):
        super().__init__(pth)

    def collect(self):
        if os.path.exists(self.pth):
            logger.debug(f"READING FROM {self.pth}")
            self._df = pd.read_csv(self.pth)
        else:
            files_collection = list()
            for sub_dir in ALL_DIR:
                for root, dirs, files in os.walk(sub_dir):
                    for file in files:
                        if file.endswith(".txt"):
                            if "suspicious-document" in file:
                                files_collection.append(os.path.join(root, file))

            self._df = self._sentences(files_collection)
            logger.info(f"Saving Generated sentences at {self.pth}")
            self._df.to_csv(self.pth)


class Plagiarism:
    def __init__(self):
        self._index = None

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

    def query(self, **kwargs):
        raise NotImplementedError

    def generate_index(self, **kwargs):
        raise NotImplementedError

    def save(self, **kwargs):
        raise NotImplementedError


class Approach:
    def run(self, **kwargs):
        raise NotImplementedError


class SE(Approach):
    def __init__(self, model_id: Optional[str] = "all-MiniLM-L6-v2"):
        self._model = SentenceTransformer(model_id)

    def run(self, sentences: List, **kwargs):
        _embeddings = list()
        for sent in sentences:
            _embeddings.append(self._model.encode(sent))
        return np.array(_embeddings)


class TFIDFHashing(Approach):
    def __init__(self, n_features=5):
        self._model = HashingVectorizer(n_features=n_features)

    def run(self, sentences: List, **kwargs):
        return self._model.fit_transform(sentences).toarray()


class Extrinsic(Plagiarism):
    def __init__(
        self,
        source_doc: SourceDocumentCollection,
        suspicious_doc: SuspiciousDocumentCollection,
        approach,
    ):
        super().__init__()
        self.source_doc = source_doc
        self.suspicious_doc = suspicious_doc
        self.approach = approach

    def generate_index(self, index_pth, ef_construction=400, m=64, ef=50):
        logger.debug("INDEX GENERATION")
        embeddings = self.approach.run(self.source_doc.get_normalised_sentences())
        self.index_embedding(
            embeddings, index_pth, ef_construction=ef_construction, m=m, ef=ef
        )

    def query(self, nn=10):
        logger.debug("QUERY IN PROGRESS")
        embeddings = self.approach.run(self.suspicious_doc.get_normalised_sentences())

        nn, distances = self._index.knn_query(embeddings, nn)

        return nn, 1 - distances

    def save(self, pth, nn, score, distance_threshold=0.20):
        header = [
            "suspicious_filename",
            "plagarised_filename",
            "suspicious",
            "plagarised",
            "score",
        ]

        suspicious_sentences = self.suspicious_doc.get_sentences()
        source_sentences = self.source_doc.get_sentences()

        with open(os.path.join(pth, "output.csv"), "w", encoding="UTF-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for ix, neighbours in enumerate(nn):
                for yx, neighbour in enumerate(neighbours):
                    if score[ix][yx] < distance_threshold:
                        continue

                    writer.writerow(
                        [
                            self.suspicious_doc.get_filename(ix),
                            self.source_doc.get_filename(yx),
                            suspicious_sentences[ix],
                            source_sentences[yx],
                            score[ix][yx],
                        ]
                    )

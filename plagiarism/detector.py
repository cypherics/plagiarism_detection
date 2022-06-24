import logging
import os
from dataclasses import dataclass
from typing import Optional, Any, List

import hnswlib
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

from plagiarism.doc import SourceDocumentCollection, SuspiciousDocumentCollection

logger = logging.getLogger()


@dataclass
class ExtrinsicOutput:
    nn: Any
    score: Any


@dataclass
class IntrinsicOutput:
    file_names: List
    sentences: List


class Plagiarism:
    def __init__(
        self,
        source_doc: Optional[SourceDocumentCollection] = None,
        suspicious_doc: Optional[SuspiciousDocumentCollection] = None,
        approach=None,
    ):

        self.source_doc = source_doc
        self.suspicious_doc = suspicious_doc
        self.approach = approach

        self.index = None

    def query(self, **kwargs):
        raise NotImplementedError

    def save(self, **kwargs):
        raise NotImplementedError


class Extrinsic(Plagiarism):
    def __init__(
        self,
        source_doc: Optional[SourceDocumentCollection] = None,
        suspicious_doc: Optional[SuspiciousDocumentCollection] = None,
        vector_model=None,
    ):
        super().__init__(source_doc, suspicious_doc, vector_model)
        self._header = [
            "suspicious_filename",
            "plagarised_filename",
            "suspicious",
            "plagarised",
            "score",
        ]

    def index_embedding(self, embeddings, pth, ef_construction=400, m=64, ef=50):
        n, dim = embeddings.shape
        self.index = hnswlib.Index(space="cosine", dim=dim)
        self.index.init_index(max_elements=n, ef_construction=ef_construction, M=m)
        self.index.add_items(embeddings, list(range(n)))
        logger.info(f"SAVING GENERATED INDEX AT {pth}")
        self.index.save_index(pth)

        self.index.set_ef(ef)

    def _load_saved_index(self, pth, dim, ef):
        self.index = hnswlib.Index(space="cosine", dim=dim)
        self.index.load_index(pth)
        self.index.set_ef(ef)

    def nn_index(
        self, index_pth, dim: int = None, ef_construction=400, m=64, ef=50, **kwargs
    ):
        if os.path.exists(index_pth):
            logger.info(f"LOADING INDEX FROM {index_pth}")
            self._load_saved_index(index_pth, dim, ef)
        else:
            logger.info("GENERATING INDEX")
            embeddings = self.approach.run(self.source_doc.get_normalised_sentences())
            self.index_embedding(
                embeddings, index_pth, ef_construction=ef_construction, m=m, ef=ef
            )
        return self

    def query(self, nn=5):
        logger.info("VECTORIZATION IN PROGRESS")
        embeddings = self.approach.run(self.suspicious_doc.get_normalised_sentences())

        logger.info("QUERYING DATA")
        nn, distances = self.index.knn_query(embeddings, nn)

        return ExtrinsicOutput(nn, 1 - distances)

    def save(self, pth, extrinsic_output: ExtrinsicOutput, distance_threshold=0.20):
        logger.info(f"SAVING IN PROGRESS AT {pth}")

        filtered_output_idx = np.where(extrinsic_output.score >= distance_threshold)

        suspicious_sentences_idx = filtered_output_idx[0]
        source_sentences_idx = extrinsic_output.nn[filtered_output_idx]

        suspicious_sentences_filtered = self.suspicious_doc.get_sentences()[
            suspicious_sentences_idx
        ]
        source_sentences_filtered = self.source_doc.get_sentences()[
            source_sentences_idx
        ]

        suspicious_file_filtered = self.suspicious_doc.get_file_names()[
            suspicious_sentences_idx
        ]
        source_file_filtered = self.source_doc.get_file_names()[source_sentences_idx]

        pd.DataFrame(
            np.column_stack(
                [
                    suspicious_file_filtered,
                    source_file_filtered,
                    suspicious_sentences_filtered,
                    source_sentences_filtered,
                    np.round(extrinsic_output.score[filtered_output_idx], 2),
                ]
            ),
            columns=self._header,
        ).to_csv(pth)


class Intrinsic(Plagiarism):
    def __init__(
        self,
        suspicious_doc: Optional[SuspiciousDocumentCollection] = None,
        vector_model=None,
    ):
        super().__init__(None, suspicious_doc, vector_model)
        self._header = [
            "suspicious_filename",
            "plagarised",
        ]
        self._lof = LocalOutlierFactor()

    def query(self, **kwargs):
        plagiarised_sent = []
        file_names = []
        logger.info("QUERYING DATA")
        for file_name, sentences in self.suspicious_doc.sentence_per_file_gen():
            embeddings = self.approach.run(sentences)
            plagiarised = list(
                sentences[np.where(self._lof.fit_predict(embeddings) == -1)]
            )
            file_names.extend([file_name] * len(plagiarised))
            plagiarised_sent.extend(plagiarised)
        return IntrinsicOutput(file_names, plagiarised_sent)

    def save(self, pth, intrinsic_output: IntrinsicOutput, **kwargs):
        pd.DataFrame(
            np.column_stack(
                [
                    intrinsic_output.file_names,
                    intrinsic_output.sentences,
                ]
            ),
            columns=self._header,
        ).to_csv(pth)

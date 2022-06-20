import csv
import logging
import os
from typing import Optional

import hnswlib
from plagiarism.doc import SourceDocumentCollection, SuspiciousDocumentCollection
from plagiarism.vectorizer import SE, TFIDFHashing

logger = logging.getLogger()


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

    def index_embedding(self, embeddings, pth, ef_construction=400, m=64, ef=50):
        n, dim = embeddings.shape
        self.index = hnswlib.Index(space="cosine", dim=dim)
        self.index.init_index(max_elements=n, ef_construction=ef_construction, M=m)
        self.index.add_items(embeddings, list(range(n)))
        self.index.save_index(pth)

        self.index.set_ef(ef)

    def _load_saved_index(self, pth, dim, ef):
        self.index = hnswlib.Index(space="cosine", dim=dim)
        self.index.load_index(pth)
        self.index.set_ef(ef)

    def query(self, **kwargs):
        raise NotImplementedError

    def nn_index(
        self, index_pth, dim: int = None, ef_construction=400, m=64, ef=50, **kwargs
    ):
        if os.path.exists(index_pth):
            logger.debug(f"LOADING INDEX FROM {index_pth}")
            self._load_saved_index(index_pth, dim, ef)
        else:
            logger.debug("GENERATING INDEX")
            embeddings = self.approach.run(self.source_doc.get_normalised_sentences())
            self.index_embedding(
                embeddings, index_pth, ef_construction=ef_construction, m=m, ef=ef
            )
        return self

    def save(self, **kwargs):
        raise NotImplementedError


class Extrinsic(Plagiarism):
    def __init__(
        self,
        source_doc: Optional[SourceDocumentCollection] = None,
        suspicious_doc: Optional[SuspiciousDocumentCollection] = None,
        vector_model: Optional[SE, TFIDFHashing] = None,
    ):
        super().__init__(source_doc, suspicious_doc, vector_model)

    def query(self, nn=10):
        logger.debug("QUERY IN PROGRESS")
        embeddings = self.approach.run(self.suspicious_doc.get_normalised_sentences())

        nn, distances = self.index.knn_query(embeddings, nn)

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

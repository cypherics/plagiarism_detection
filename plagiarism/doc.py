import logging
import os
from typing import List

import numpy as np
import pandas as pd
from nltk import TreebankWordDetokenizer
from tqdm import tqdm

from plagiarism.util import (
    get_sentences_from_df,
    generate_para_df,
    normalize_data,
    extrinsic_normalize_data,
    intrinsic_normalize_data,
)

logger = logging.getLogger()


class DocumentCollection:
    def __init__(self, pth, dir_iter: List):
        self._df = None
        self.pth = pth
        self.dir_iter = dir_iter

    def _sentences(self, files):
        logger.info("SENTENCE GENERATION")
        df = pd.DataFrame()
        for i, file in enumerate(tqdm(files)):
            df1 = pd.DataFrame()

            sentences = get_sentences_from_df(generate_para_df(file))
            tokenized_sentences = dict()

            for j, sent in enumerate(sentences):
                tokenized_text = self.normalize_data(sent)
                if len(tokenized_text) >= 5:
                    tokenized_sentences[sent] = TreebankWordDetokenizer().detokenize(
                        tokenized_text
                    )

            df1["filename"] = [str(os.path.basename(file))] * len(tokenized_sentences)
            df1["sentences"] = list(tokenized_sentences.keys())
            df1["normalised"] = list(tokenized_sentences.values())

            df = pd.concat([df, df1], ignore_index=True, sort=False)
        df["idx"] = range(0, len(df))
        return df

    def get_sentences(self) -> np.ndarray:
        return self._df["sentences"].to_numpy()

    def get_normalised_sentences(self) -> np.ndarray:
        return self._df["normalised"].to_numpy()

    def get_filename(self, idx):
        return self._df.loc[self._df["idx"] == idx]["filename"].values[0]

    def get_file_names(self) -> np.ndarray:
        return self._df["filename"].to_numpy()

    def extract_sentences(self):
        raise NotImplementedError

    def sentence_per_file_gen(self) -> np.ndarray:
        _file_name = list(pd.unique(self.get_file_names()))
        for file in tqdm(_file_name):
            yield file, self._df.loc[self._df["filename"] == file][
                "normalised"
            ].values, self._df.loc[self._df["filename"] == file]["sentences"].values

    def normalize_data(self, text, **kwargs):
        raise NotImplementedError


class ExSourceDocumentCollection(DocumentCollection):
    def __init__(self, pth, dir_iter: List):
        super().__init__(pth, dir_iter)

    def extract_sentences(self):
        if os.path.exists(self.pth):
            logger.info(f"READING FROM {self.pth}")
            self._df = pd.read_csv(self.pth)
        else:
            files_collection = list()
            for sub_dir in self.dir_iter:
                for root, dirs, files in os.walk(sub_dir):
                    for file in files:
                        if file.endswith(".txt"):
                            if "source-document" in file:
                                files_collection.append(os.path.join(root, file))

            self._df = self._sentences(files_collection)
            logger.info(f"Saving Generated sentences at {self.pth}")
            self._df.to_csv(self.pth)
        return self

    def normalize_data(self, text, **kwargs):
        return extrinsic_normalize_data(text)


class ExSuspiciousDocumentCollection(DocumentCollection):
    def __init__(self, pth, dir_iter: List):
        super().__init__(pth, dir_iter)

    def extract_sentences(self):
        if os.path.exists(self.pth):
            logger.info(f"READING FROM {self.pth}")
            self._df = pd.read_csv(self.pth)
        else:
            files_collection = list()
            for sub_dir in self.dir_iter:
                for root, dirs, files in os.walk(sub_dir):
                    for file in files:
                        if file.endswith(".txt"):
                            if "suspicious-document" in file:
                                files_collection.append(os.path.join(root, file))

            self._df = self._sentences(files_collection)
            logger.info(f"Saving Generated sentences at {self.pth}")
            self._df.to_csv(self.pth)
        return self

    def normalize_data(self, text, **kwargs):
        return extrinsic_normalize_data(text)


class InSuspiciousDocumentCollection(ExSuspiciousDocumentCollection):
    def __init__(self, pth, dir_iter: List):
        super().__init__(pth, dir_iter)

    def normalize_data(self, text, **kwargs):
        return intrinsic_normalize_data(text)

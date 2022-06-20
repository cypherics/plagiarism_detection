import logging
import os
from typing import List

import pandas as pd
from tqdm import tqdm

from plagiarism.util import get_sentences_from_df, generate_para_df, normalize_data

logger = logging.getLogger()


class DocumentCollection:
    def __init__(self, pth, dir_iter: List):
        self._df = None
        self.pth = pth
        self.dir_iter = dir_iter

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

    def extract_sentences(self):
        raise NotImplementedError


class SourceDocumentCollection(DocumentCollection):
    def __init__(self, pth, dir_iter: List):
        super().__init__(pth, dir_iter)

    def extract_sentences(self):
        if os.path.exists(self.pth):
            logger.debug(f"READING FROM {self.pth}")
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


class SuspiciousDocumentCollection(DocumentCollection):
    def __init__(self, pth, dir_iter: List):
        super().__init__(pth, dir_iter)

    def extract_sentences(self):
        if os.path.exists(self.pth):
            logger.debug(f"READING FROM {self.pth}")
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

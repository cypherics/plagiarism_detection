from typing import Optional, List

import numpy as np
from authorstyle import get_feature_vector, all_feature_functions, Text
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import HashingVectorizer


class Vectorizer:
    def run(self, **kwargs):
        raise NotImplementedError


class SE(Vectorizer):
    def __init__(self, model_id: Optional[str] = "all-MiniLM-L6-v2"):
        self._model = SentenceTransformer(model_id)

    def run(self, sentences: List, **kwargs):
        _embeddings = list()
        return self._model.encode(sentences)


class TFIDFHashing(Vectorizer):
    def __init__(self, n_features=20):
        self._model = HashingVectorizer(n_features=n_features)

    def run(self, sentences: List, **kwargs):
        return self._model.fit_transform(sentences).toarray()


class StyleEmbedding(Vectorizer):
    def __init__(self, features_to_use: List, **kwargs):
        self._features_to_use = []
        for feat in all_feature_functions():
            if feat.__name__ in features_to_use:
                self._features_to_use.append(feat)

    def run(self, sentences: List, **kwargs):
        tex = Text(" ".join(sentences))
        tex.set_sliding_window(window_size=5, step_size=1, unit=sentences)
        feat_vec = np.array(
            get_feature_vector(tex, self._features_to_use, fragments=True)
        )
        # normalised_feat_vec = self._norm_func.fit_transform(
        #     feat_vec.astype(np.float32).T
        # ).T
        return feat_vec / np.linalg.norm(feat_vec, axis=-1).reshape(-1, 1)

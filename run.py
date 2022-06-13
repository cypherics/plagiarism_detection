from plagarism.pipeline.components.ext_components import (
    SuspiciousDataNormalization,
    ReadTfidfEmbeddings,
    ReadVocab,
    TFIDFForSuspicious,
    SE,
    NN,
    Output, ReadSuspiciousData, IndexSourceDataWithANN,
)
from plagarism.pipeline.ext_pipeline import ExtrinsicPlagiarismPipeLineWithIndexedTFIDF, IndexingSourceDataPipeline

# CREATE INDEX FOR GENERTAED EMBEDDING
args = {
    "tfidf_path": r"/home/emin/Fax/NLP/code/plagiarism_detection/plagiarism-detector/calculation_data/tfid_matrix.npz",
    "index_path": r"home/emin/Fax/NLP/code/plagiarism_detection/plagiarism-detector/calculation_data/",
}

index_pipeline = IndexingSourceDataPipeline()
index_pipeline.components([ReadTfidfEmbeddings, IndexSourceDataWithANN])
index_pipeline.execute(**args)

# class SemanticSim(NN):
#     def __init__(self, nn: int = 10, **kwargs):
#         super().__init__(nn, **kwargs)
#         self._nn = nn
#
#     def init(self, embeddings, **kwargs):
#         pass
#
#     def query(self, query_embeddings, **kwargs):
#         _sim = util.semantic_search(
#             query_embeddings, kwargs["corpus_embeddings"], self._nn
#         )
#         _id = []
#         _sim_v = []
#         for i in _sim:
#             _i = []
#             _j = []
#             for j in i:
#                 _i.append(j["corpus_id"])
#                 _j.append(j["score"])
#             _id.append(_i)
#             _sim_v.append(_j)
#         return np.array(_id), np.array(_sim_v)

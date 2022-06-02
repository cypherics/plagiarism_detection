from plagarism.pipeline.extrinsic import (
    ExtrinsicPlagiarismPipeline,
    ReadSourceData,
    ReadSuspiciousData,
    SourceDataNormalization,
    SuspiciousDataNormalization,
    CollectSourceWithSuspicious,
    USE,
    SE,
    NN,
    Output,
    BertCase,
)

arg = {
    "source": r"C:\Users\Fuzail.Palnak\Downloads\pan-plagiarism-corpus-2009\external-analysis-corpus\source-documents\part2\source-document02007.txt",
    "suspicious": r"C:\Users\Fuzail.Palnak\Downloads\pan-plagiarism-corpus-2009.part2\pan-plagiarism-corpus-2009\external-analysis-corpus\suspicious-documents\part2\suspicious-document02007.txt",
}
e = ExtrinsicPlagiarismPipeline()
e.components(
    [
        ReadSourceData,
        ReadSuspiciousData,
        SourceDataNormalization,
        SuspiciousDataNormalization,
        CollectSourceWithSuspicious,
        SE,
        NN,
        Output,
    ]
)
e.execute(**arg)

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

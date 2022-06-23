from plagiarism.detector import (
    Extrinsic,
    SourceDocumentCollection,
    SuspiciousDocumentCollection,
)
from plagiarism.vectorizer import SE, TFIDFHashing

source_doc = SourceDocumentCollection(
    pth="dataset/source_sent_all_three_subset.csv",
    dir_iter=[
        "dataset/subset/subset_1/sou_1",
        "dataset/subset/subset_2/sou_2",
        "dataset/subset/subset_3/sou_3",
    ],
).extract_sentences()
suspicious_doc = SuspiciousDocumentCollection(
    pth="dataset/suspicious_sent.csv", dir_iter=["dataset/subset/subset_1/sus_1"]
).extract_sentences()

e = Extrinsic(source_doc, suspicious_doc, vector_model=SE())
e.nn_index("dataset/output/se_index_subset_1_2_3.index")
nn, score = e.query()
e.save(
    "dataset/output/set1/SE/se_output_subset_1_with_all_three_source.csv",
    nn,
    score,
    distance_threshold=0.90,
)

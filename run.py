from plagiarism.detector import (
    Extrinsic,
    SourceDocumentCollection,
    SuspiciousDocumentCollection,
    SE,
)

ALL_DIR = [
    r"C:\Users\Fuzail.Palnak\PycharmProjects\crefts\plagiarism_detection\dataset\pan-plagiarism-corpus-2009",
    r"C:\Users\Fuzail.Palnak\PycharmProjects\crefts\plagiarism_detection\dataset\pan-plagiarism-corpus-2009.part2",
    r"C:\Users\Fuzail.Palnak\PycharmProjects\crefts\plagiarism_detection\dataset\pan-plagiarism-corpus-2009.part3",
]

source_doc = SourceDocumentCollection(
    pth="dataset/source_sent.csv", dir_iter=ALL_DIR
).extract_sentences()
suspicious_doc = SuspiciousDocumentCollection(
    pth="dataset/suspicious_sent.csv", dir_iter=ALL_DIR
)
vector_model = SE()

e = Extrinsic(source_doc, suspicious_doc, vector_model)
e.nn_index("dataset/source.index")
# nn, score = e.query()
# e.save("dataset", nn, score, distance_threshold=0.90)

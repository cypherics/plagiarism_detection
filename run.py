from plagiarism.detector import (
    Extrinsic,
    DocumentCollection,
    SourceDocumentCollection,
    SuspiciousDocumentCollection,
    TFIDFHashing,
)

source_doc = SourceDocumentCollection(pth="dataset/source_sent.csv")
suspicious_doc = SuspiciousDocumentCollection(pth="dataset/suspicious_sent.csv")
approach = TFIDFHashing()

e = Extrinsic(source_doc, suspicious_doc, approach)
e.generate_index("dataset/source.index")
nn, score = e.query()
e.save("dataset", nn, score)

from plagiarism.detector import Extrinsic, DocumentCollection, SourceDocumentCollection, SuspiciousDocumentCollection

source_doc = SourceDocumentCollection()
suspicious_doc = SuspiciousDocumentCollection()

e = Extrinsic(source_doc, suspicious_doc)
e.generate_index("dataset/source.index")
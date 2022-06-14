from plagiarism.detector import Extrinsic, DocumentCollection, SourceDocumentCollection, SuspiciousDocumentCollection, \
    TFIDFHashing

source_doc = SourceDocumentCollection()
suspicious_doc = SuspiciousDocumentCollection()
approach = TFIDFHashing()

e = Extrinsic(source_doc, suspicious_doc, approach)
e.generate_index("/home/emin/Fax/NLP/code/plagiarism_detection/plagiarism-detector/data/source.index")
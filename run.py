from plagiarism.detector import Extrinsic, DocumentCollection

document = DocumentCollection()
e = Extrinsic(document)
e.generate_index("dataset/source.index")
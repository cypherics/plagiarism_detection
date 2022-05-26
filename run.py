from plagarism.detector import ExtrinsicPlagiarismPipeline
from plagarism.pipeline import DataNormalization, USE, ReadSourceData

e = ExtrinsicPlagiarismPipeline()
e.components([ReadSourceData, DataNormalization, USE])
e.execute()

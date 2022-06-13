from typing import List

from plagarism.pipeline.base import Pipeline
from plagarism.pipeline.components.ext_components import IndexSourceDataWithANN


class ExtrinsicPlagiarismPipeline(Pipeline):
    def components(self, component: List):
        for comp in component:
            self.pipe_line_components.append(comp.init())

    def execute(self, **data):
        for comp in self.pipe_line_components:
            data = comp.execute(**data)


class IndexingSourceDataPipeline(Pipeline):
    def components(self, component: List):
        for comp in component:
            self.pipe_line_components.append(comp.init())

    def execute(self, **data):
        for comp in self.pipe_line_components:
            if not isinstance(data, IndexSourceDataWithANN):
                data = comp.execute(**data)
        if isinstance(self.pipe_line_components[-1], IndexSourceDataWithANN):
            data = self.pipe_line_components[-1].execute(**data)
        else:
            data = IndexSourceDataWithANN.init().execute(**data)
        return data


class ExtrinsicPlagiarismPipeLineWithIndexedTFIDF(Pipeline):
    def components(self, component: List):
        for comp in component:
            self.pipe_line_components.append(comp.init())

    def execute(self, **data):
        for comp in self.pipe_line_components:
            data = comp.execute(**data)
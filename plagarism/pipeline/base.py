from typing import List


class Pipeline:
    def __init__(self):
        self.pipe_line_components = []

    def components(self, component: List):
        raise NotImplementedError

    def execute(self, **kwargs):
        raise NotImplementedError


class PipelineComponent:
    @classmethod
    def init(cls):
        return cls()

    def execute(self, **kwargs) -> dict:
        raise NotImplementedError

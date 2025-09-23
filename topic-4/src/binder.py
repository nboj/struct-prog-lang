from src.parser import Program
from src.source_map import SourceMap


class Binder:
    program: Program
    sm: SourceMap

    def __init__(self, program: Program, source_map: SourceMap):
        self.program = program
        self.sm = source_map


from src.source_map import SourceMap
from src.parser import Program


class Lowering:
    sm: SourceMap
    program: Program

    def __init__(self, program: Program, source_map: SourceMap):
        self.sm = source_map
        self.program = program

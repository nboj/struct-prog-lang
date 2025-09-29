from .source_map import SourceMap
from .parser import Program
from dataclasses import dataclass


@dataclass(frozen=True)
class Instr:
    pass


@dataclass(frozen=True)
class Block:
    instrs: list[Instr]


class Lowering:
    sm: SourceMap
    program: Program

    def __init__(self, program: Program, source_map: SourceMap):
        self.sm = source_map
        self.program = program

    def lower(self):
        pass

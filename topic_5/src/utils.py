from contextlib import contextmanager
import io
import sys
from .vm import VM
from .types import CodeObject, Instr
from .lowering import Lowering
from .binder import Binder
from .parser import Parser
from .tokenizer import Tokenizer


def compile_trivial(text: str):
    tokenizer = Tokenizer(text)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens, tokenizer.sm)
    parsed = parser.parse()
    binder = Binder(parsed, tokenizer.sm)
    bound = binder.bind()
    lowering = Lowering(bound, tokenizer.sm)
    return lowering.lower()

def pretty_ir(ir: list[Instr]):
    for idx, instr in enumerate(ir):
        print (f"| {idx+1}: {instr}")

@contextmanager
def capture_stdout():
    old = sys.stdout
    buf = io.StringIO()
    sys.stdout = buf
    try:
        yield buf
    finally:
        sys.stdout = old

def run_vm(codeobj: CodeObject):
    vm = VM(codeobj)
    with capture_stdout() as buf:
        vm.run()
    return buf.getvalue(), vm


def parse_optional(s: str):
    s = s.strip()
    sl = s.lower()
    if sl == "true":  return True
    if sl == "false": return False
    try:
        if s.count(".") == 0 and "e" not in sl and "E" not in s:
            return int(s.replace("_", ""))
    except ValueError:
        pass
    try:
        return float(s.replace("_", ""))
    except ValueError:
        return s

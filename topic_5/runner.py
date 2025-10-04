import sys
from pathlib import Path
import time
from src.tokenizer import Tokenizer, Token
from src.parser import Parser
from src.vm import VM
from src.lowering import Lowering
from src.binder import Binder

if __name__ == "__main__":
    args = sys.argv
    if len(args) > 1:
        text = Path(args[1]).read_text()
        if text:
            tokenizer = Tokenizer(text)
            tokens: list[Token] = tokenizer.tokenize()

            parser = Parser(tokens, tokenizer.sm)
            parsed = parser.parse()

            binder = Binder(parsed, tokenizer.sm)
            bound = binder.bind()

            lowering = Lowering(bound, tokenizer.sm)
            lowered = lowering.lower()

            vm = VM(lowered)

            start = time.perf_counter_ns()
            vm.run()
            end = time.perf_counter_ns()
            print(f"=== {(end-start)/1e+9} ===")
        else:
            print("No text.")

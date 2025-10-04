import sys
from pathlib import Path
import time
from src.lowering import Lowering
from src.tokenizer import Tokenizer, Token
from src.parser import Parser
from src.debug_ast import render_ast
from src.vm import VM, debug_stack
from src.binder import Binder
from src.utils import run_vm

if __name__ == "__main__":
    args = sys.argv
    if len(args) > 1:
        text = Path(args[1]).read_text()
        if text:
            tokenizer = Tokenizer(text)
            tokens: list[Token] = tokenizer.tokenize()

            tokenizer.debug_print()
            print(tokenizer.tokens_pretty_gutter())

            parser = Parser(tokens, tokenizer.sm)
            parsed = parser.parse()
            print(f"parsed: \n{render_ast(root=parsed,
                  source_map=tokenizer.sm, show_spans=False)}")
            print(parsed)

            binder = Binder(program=parsed, source_map=tokenizer.sm)
            bound = binder.bind()
            print(f"parsed: \n{render_ast(root=bound,
                  source_map=tokenizer.sm, show_spans=False)}")
            lowering = Lowering(bound, tokenizer.sm)
            code = lowering.lower()
            print()
            for idx, instr in enumerate(code.code):
                print(f"| {idx}: ", instr)

            pre_consts = code.consts
            pre_nglobals = code.nglobals

            vm = VM(code)
            print("=== OUT ===")
            start = time.perf_counter_ns()
            vm.run()
            end = time.perf_counter_ns()
            print(f"=== {(end-start)/1e+9} ===")

            print()
            print("=== PRE EVAL ===")
            print(f"consts: {pre_consts}")
            print(f"globals: [nil] * {pre_nglobals} (number of globals)")
            print(f"stack: [nil] * 4096")
            print("sp: 0 (stack pointer)")
            print()
            print("=== POST EVAL ===")
            print(f"globals: {vm.globals}")
            print(f"stack: {debug_stack(vm.stack)}")
            print(f"sp={vm.sp}")
            print()

        else:
            print("No text.")
        print("Exiting.")

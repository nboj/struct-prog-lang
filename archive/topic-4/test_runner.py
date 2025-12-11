import sys
from pathlib import Path
from src.tokenizer import Tokenizer, Token
from src.parser import Parser
from src.debug_ast import render_ast
from src.vm import VM
from src.binder import Binder

if __name__ == "__main__":
    args = sys.argv
    if len(args) > 1:
        text = Path(args[1]).read_text()
        if text:
            tokenizer = Tokenizer(text)
            tokens: [Token] = tokenizer.tokenize()

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

            vm = VM(parsed, tokenizer.sm)
            vm.run()

        else:
            print("No text.")
        print("Exiting.")

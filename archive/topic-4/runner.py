import sys
from pathlib import Path
from src.tokenizer import Tokenizer, Token
from src.parser import Parser
from src.debug_ast import render_ast
from src.vm import VM
from src.lowering import Lowering

if __name__ == "__main__":
    args = sys.argv
    if len(args) > 1:
        text = Path(args[1]).read_text()
        if text:
            tokenizer = Tokenizer(text)
            tokens: [Token] = tokenizer.tokenize()

            parser = Parser(tokens, tokenizer.sm)
            parsed = parser.parse()
            lowering = Lowering(parsed, tokenizer.sm)
            lowered = lowering.lower()
            print(lowered)
            vm = VM(parsed, tokenizer.sm)
            vm.run()

        else:
            print("No text.")

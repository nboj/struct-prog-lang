import sys
from pathlib import Path
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
            vm.run()

        else:
            print("No text.")

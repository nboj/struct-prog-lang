from src.tokenizer import Tokenizer, Token
from src.parser import Parser
from pathlib import Path
from src.debug_ast import render_ast

if __name__ == "__main__":
    text = Path("./test.txt").read_text()
    if text:
        tokenizer = Tokenizer(text)
        tokens: [Token] = tokenizer.tokenize()

        tokenizer.debug_print()
        print(tokenizer.tokens_pretty_gutter())  # human-friendly with carets

        parser = Parser(tokens)
        parsed = parser.parse()
        print(f"parsed: \n{render_ast(root=parsed, source_map=tokenizer.sm, show_spans=False)}")

    else:
        print("No text.")
    print("Exiting.")

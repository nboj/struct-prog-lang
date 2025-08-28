from src.tokenizer import Tokenizer, Token
from src.parser import Parser
from pathlib import Path

if __name__ == "__main__":
    text = Path("./test.txt").read_text()
    if text:
        tokenizer = Tokenizer(text)
        tokens: [Token] = tokenizer.tokenize()

        tokenizer.debug_print()
        print(tokenizer.tokens_pretty_gutter())  # human-friendly with carets

        parser = Parser(tokens)
        parsed = parser.parse()
    else:
        print("No text.")
    print("Exiting.")

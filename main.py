from src.tokenizer import Tokenizer, Token
from pathlib import Path

if __name__ == "__main__":
    text = Path("./test.txt").read_text()
    if text:
        tokenizer = Tokenizer(text)
        tokens: [Token] = tokenizer.tokenize()

        tokenizer.debug_print()


        #print(tokenizer.tokens_debug())             # snapshot-friendly
        print(tokenizer.tokens_pretty_gutter())  # human-friendly with carets
    else:
        print("No text.")
    print("Exiting.")

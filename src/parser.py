from src.tokenizer import Token, TokenType


class Parser:
    tokens: list[Token]
    index: int

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.index = 0
        pass

    def bump(self):
        tmp = self.tokens[self.index]
        self.index += 1
        return tmp

    def peek(self):
        return self.tokens[self.index]

    def parse_atom(self):
        """

        """
        token = self.peek()
        match token.kind:
            case TokenType.Number:
                self.bump()
            case _:
                raise AssertionError(f"Invalid token in atom {token.kind}")

    def parse_factor(self):
        """
        factor = Atom | Expr
        """
        token = self.peek()
        match token.kind:
            case TokenType.LParen:
                self.parse_expr()
            case _:
                self.parse_atom()

    def parse_term(self):
        """
        term = factor Optional{ /|* factor}
        """
        self.parse_factor()
        token = self.peek()
        while True:
            match token.kind:
                case TokenType.Star | TokenType.Divide:
                    self.bump()
                    self.parse_factor()
                case _:
                    break
            token = self.peek()


    def parse_expr(self):
        """
        expr = term Optional{ +|- term }
        """
        self.bump()
        self.parse_term()
        token = self.peek()
        while True:
            match token.kind:
                case TokenType.Plus | TokenType.Minus:
                    self.bump()
                case TokenType.RParen:
                    break
                case _:
                    self.parse_term()
            token = self.peek()
        assert self.bump().kind == TokenType.RParen

    def parse_if(self):
        self.parse_expr()
        assert self.peek().kind == TokenType.Nl

    def parse_print(self):
        """
        stmt = <print>(expr)
        """
        self.bump()
        self.parse_expr()
        assert self.bump().kind == TokenType.Nl


    def parse(self):
        while self.index < len(self.tokens)-1:
            token = self.peek()
            match token.kind:
                case TokenType.If:
                    pass
                case TokenType.Let:
                    pass
                case TokenType.Print:
                    """
                    stmt = <print> expr
                    """
                    self.parse_print()
                case TokenType.Nl:
                    self.bump()
                case TokenType.Eof:
                    break
                case _:
                    raise AssertionError(f"Reached invalid token {token.kind}")

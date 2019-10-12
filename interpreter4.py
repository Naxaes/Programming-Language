"""
    ---- Grammar ----

    program      : (statement)*
    statement    : expression END_STATEMENT
                 | block                          Should a block require an end statement?
                 | IF condition THEN statement [ELSE IF condition block]* [ELSE block]    Block or statement???
                 | WHILE condition THEN statement
    block        : INDENTION (statement)* DEDENTION
    declaration  : variable DECLARE DATA_TYPE [ASSIGN expression]
                 | variable DECLARE_ASSIGN expression
    assignment   : variable ASSIGN expression
    expression   : assignment
                 | declaration
                 | function
                 | term ((ADD | SUB) term)*
    term         : factor ((MUL | INT_DIV | REAL_DIV) factor)*
    factor       : INTEGER_CONST
                   | REAL_CONST
                   | variable
                   | NEG factor
                   | LPAREN expr RPAREN
    variable     :  ID

"""
INDENTION, DEDENTION, EMPTY = 'INDENTION', 'DEDENTION', 'EMPTY'

# Data types
DATA_TYPE = 'DATA_TYPE'
INTEGER, REAL, STRING, FUNCTION = 'INTEGER', 'REAL', 'STRING', 'FUNCTION'

# Constants
CONSTANT = 'CONSTANT'
INTEGER_CONST, REAL_CONST, STRING_CONST = 'INTEGER_CONST', 'REAL_CONST', 'STRING_CONST'

# Binary operations
BINARY_OPERATION = 'BINARY_OPERATION'
ADD, SUB, MUL, INT_DIV, REAL_DIV, POW = 'ADD', 'SUB', 'MUL', 'INT_DIV', 'REAL_DIV', 'POW'

# Unary operations
UNARY_OPERATION = 'UNARY_OPERATION'
NEGATE, SQRT, INCREMENT, DECREMENT = 'SUB', 'SQRT', 'INC', 'DEC'

# Built-in functions
PRINT, SUM = 'PRINT', 'SUM'

# Parenthesis
PARENTHESIS = 'PARENTHESIS'
LPARENS, RPARENS, LBRACKET, RBRACKET, LSQUARE, RSQUARE = 'LPARENS', 'RPARENS', 'LBRACKET', 'RBRACKET', 'LSQUARE', 'RSQUARE'

# Operators.
OPERATORS = 'OPERATORS'
GREATER_THAN, GREATER_EQUAL_THAN, EQUAL, LESS_EQUAL_THAN, LESS_THAN, NOT_EQUAL = (
    'GREATER_THAN', 'GREATER_EQUAL_THAN', 'EQUAL', 'LESS_EQUAL_THAN', 'LESS_THAN', 'NOT_EQUAL'
)
# Branch.
BRANCH = 'BRANCH'
IF, THEN, ELSE, WHILE, OR, AND = 'IF', 'THEN', 'ELSE', 'WHILE', 'OR', 'AND'

# Other
COMMA, END_STATEMENT, EOF, DECLARE_ASSIGN, ASSIGN, CALL, DECLARE, INLINE = (
    'COMMA', 'END_STATEMENT', 'EOF', 'DECLARE_ASSIGN', 'ASSIGN', 'CALL', 'DECLARE', 'INLINE'
)

ID = 'ID'

IS, THAN, TO, NOT = 'IS', 'THAN', 'TO', 'NOT'

UNKNOWN = 'UNKNOWN'

START_BLOCK = 'START_BLOCK'


class Token:
    def __init__(self, type, value):
        self.type = type
        self.value = value

    def __repr__(self):
        return 'Token(type={}, value={})'.format(self.type, self.value)


class EndOfFile(Exception): pass

class InputStream:

    def __init__(self, source):
        self.source = source

        self.index = 0
        self.last_index = len(source)

        self.row = 1
        self.column = 1

    def peek(self):
        if self.index >= self.last_index:
            raise EndOfFile()
        return self.source[self.index]

    def next(self):
        character = self.peek()

        if character == '\n':
            self.row += 1
            self.column = 0
        else:
            self.column += 1

        self.index += 1
        return character


KEYWORDS = {
    'int'   : Token(INTEGER, INTEGER),
    'real'  : Token(REAL, REAL),
    'string': Token(STRING, STRING),
    'inc'   : Token(INCREMENT, INCREMENT),
    'dec'   : Token(DECREMENT, DECREMENT),
    'sqrt'  : Token(SQRT, SQRT),
    'pow'   : Token(POW, POW),
    'print' : Token(FUNCTION, PRINT),
    'sum'   : Token(FUNCTION, SUM),
    'call'  : Token(CALL, CALL),
    'if'    : Token(IF, IF),
    'then'  : Token(THEN, THEN),
    'else'  : Token(ELSE, ELSE),
    'while' : Token(WHILE, WHILE),
    ';'     : Token(END_STATEMENT, END_STATEMENT),
    # '\t'    : Token(INLINE, INLINE),
    # '    '  : Token(INLINE, INLINE),

    '/' : Token(REAL_DIV, REAL_DIV),
    ',' : Token(COMMA, COMMA),
    '(' : Token(LPARENS, LPARENS),
    ')' : Token(RPARENS, RPARENS),
    '{' : Token(LBRACKET, LBRACKET),
    '}' : Token(RBRACKET, RBRACKET),

    '*': Token(MUL, MUL),
    '+': Token(ADD, ADD),
    '-': Token(SUB, SUB),
    ':': Token(DECLARE, DECLARE),
    '=': Token(ASSIGN, ASSIGN),
    '>': Token(GREATER_THAN, GREATER_THAN),
    '<': Token(LESS_THAN, LESS_THAN),
    '!': None,

    '**': Token(POW, POW),
    '++': Token(INCREMENT, INCREMENT),
    '--': Token(DECREMENT, DECREMENT),
    ':=': Token(DECLARE_ASSIGN, DECLARE_ASSIGN),
    '==': Token(EQUAL, EQUAL),
    '>=': Token(GREATER_EQUAL_THAN, GREATER_EQUAL_THAN),
    '<=': Token(LESS_EQUAL_THAN, LESS_EQUAL_THAN),
    '!=': Token(NOT_EQUAL, NOT_EQUAL),

    '≥': Token(GREATER_EQUAL_THAN, GREATER_EQUAL_THAN),
    '≤': Token(LESS_EQUAL_THAN, LESS_EQUAL_THAN),
    '√': Token(SQRT, SQRT),
    '^': Token(POW, POW),
    '∑': Token(FUNCTION, SUM),
}


class Tokenizer:

    def __init__(self, stream):
        self.stream = stream
        self.character = self.stream.next()
        self.current_indention = 0

    def read_indention(self):
        indention = 0
        while self.character == '\n':
            self.character = self.stream.next()
        while self.character == ' ':
            indention += 1
            self.character = self.stream.next()
        level = indention // 4

        type = None
        if level > self.current_indention + 1:
            raise IndentationError()
        elif level >= self.current_indention:
            type = INDENTION
        elif level < self.current_indention:
            type = DEDENTION

        self.current_indention = level

        return None if level == 0 else Token(type, level)

    def read_identifier(self):
        result = ''
        while self.character.isalpha():
            result += self.character
            self.character = self.stream.next()
        return KEYWORDS.get(result, Token(ID, result))

    def read_number(self):
        result = ''
        while self.character.isdigit():
            result += self.character
            self.character = self.stream.next()
        return Token(INTEGER_CONST, result)

    def read_string(self):
        result = ''
        if self.character == '"':
            result += self.character
            self.character = self.stream.next()
        while self.character != '"':
            result += self.character
            self.character = self.stream.next()
        result += self.character
        self.character = self.stream.next()
        return Token(STRING_CONST, result)

    def read_operator(self):
        first = self.character
        self.character = self.stream.next()
        second = self.character

        token = KEYWORDS.get(first + second)
        if token:
            self.character = self.stream.next()
        else:
            token = KEYWORDS.get(first)

        return token

    def read_whitespace(self):
        while self.character == ' ':
            self.character = self.stream.next()
        return None

    def read_comment(self):
        while self.character != '\n':
            self.character = self.stream.next()
        return None

    def get_token(self):
        if self.character == '\n':
            return self.read_indention()
        elif self.character == ' ':
            return self.read_whitespace()
        elif self.character.isalpha():
            return self.read_identifier()
        elif self.character.isdigit():
            return self.read_number()
        elif self.character == '"':
            return self.read_string()
        elif self.character == '#':
            return self.read_comment()
        else:
            return self.read_operator()

    def get_tokens(self):
        tokens = []
        token = self.get_token()
        while True:
            if token:
                tokens.append(token)
            try:
                token = self.get_token()
            except EndOfFile:
                return tokens




source = """
while a := 10 then            
    test ++ 
        a
            a
                a
a -= 4
;
"""
tokenizer = Tokenizer(InputStream(source))
print(*tokenizer.get_tokens(), sep='\n')





class Node:

    def __init__(self, operation, **kwargs):
        self.operation = operation
        self.kwargs = kwargs

class Parser:
    """
    ---- Grammar ----

    program      : (statement)*
    block        : INDENTION (statement)* DEDENTION
    statement    : expression END_STATEMENT
                    | IF condition THEN block [ELSE IF condition THEN block] [ELSE block]
                    | WHILE condition THEN block
    assignment   : variable DECLARE_ASSIGN expression
                    | variable DECLARE DATA_TYPE ASSIGN expression
                    | variable ASSIGN expression
    expression   : term ((ADD | SUB) term)*
                    | assignment
    term         : factor ((MUL | INT_DIV | REAL_DIV) factor)*
    factor       : INTEGER_CONST
                   | REAL_CONST
                   | variable
                   | NEG factor
                   | LPAREN expr RPAREN
    variable     :  ID
    """
    def __init__(self, tokens):
        self.index = 0
        self.end = len(tokens)
        self.tokens = tokens
        self.current_token = tokens[0]


    def parse(self):
        return self.program()

    def program(self):
        statements = []
        while self.index < self.end:
            statements.append(
                Node('statement', statement=self.statement())
            )
        return Node('program', statements=self.program())

    def statement(self):
        token = self.tokens[self.index]

        if token.type == IF:
            pass
        elif token.type == WHILE:
            pass
        elif token.type == ID:
            pass

    def block(self):
        return























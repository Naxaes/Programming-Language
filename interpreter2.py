from collections import OrderedDict

from inputstream import InputStream


# from collections import defaultdict
# from collections import namedtuple


PRINT_JUST_VALUE = True


# ---- Token types ----

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
RETURN = 'RETURN'

class Token:
    def __init__(self, type_, value):
        self.type  = type_
        self.value = value
    def __repr__(self):
        return 'Token(type={}, value={})'.format(self.type, self.value)


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
    'return': Token(RETURN, RETURN),

    '/' : Token(REAL_DIV, REAL_DIV),
    ',' : Token(COMMA, COMMA),
    '(' : Token(LPARENS, LPARENS),
    ')' : Token(RPARENS, RPARENS),
    '{' : Token(LBRACKET, LBRACKET),
    '}' : Token(RBRACKET, RBRACKET),

     # FIRST                                           SECOND
    '*': {'default': Token(MUL, MUL),                  '*': Token(POW, POW)},
    '+': {'default': Token(ADD, ADD),                  '+': Token(INCREMENT, INCREMENT)},
    '-': {'default': Token(SUB, SUB),                  '-': Token(DECREMENT, DECREMENT)},
    ':': {'default': Token(DECLARE, DECLARE),          '=': Token(DECLARE_ASSIGN, DECLARE_ASSIGN)},
    '=': {'default': Token(ASSIGN, ASSIGN),            '=': Token(EQUAL, EQUAL)},
    '>': {'default': Token(GREATER_THAN, GREATER_THAN),'=': Token(GREATER_EQUAL_THAN, GREATER_EQUAL_THAN)},
    '<': {'default': Token(LESS_THAN, LESS_THAN),      '=': Token(LESS_EQUAL_THAN, LESS_EQUAL_THAN)},
    '!': {'default': None,                             '=': Token(NOT_EQUAL, NOT_EQUAL)},

    '≥': Token(GREATER_EQUAL_THAN, GREATER_EQUAL_THAN),
    '≤': Token(LESS_EQUAL_THAN, LESS_EQUAL_THAN),
    '√': Token(SQRT, SQRT),
    '^': Token(POW, POW),
    '∑': Token(FUNCTION, SUM),


    'less'   : Token(LESS_THAN, LESS_THAN),
    'greater': Token(GREATER_THAN, GREATER_THAN),
    'not'    : Token(NOT, NOT),
    'equal'  : Token(EQUAL, EQUAL),

    'is'     : Token(IS, IS),
    'than'   : Token(THAN, THAN),
    'or'     : Token(OR, OR),
    'and'    : Token(AND, AND),
    'to'     : Token(TO, TO),


}

operator_logic = {
    (LESS_THAN, EQUAL)        : LESS_EQUAL_THAN,
    (GREATER_THAN, EQUAL)     : GREATER_EQUAL_THAN,
    (EQUAL, LESS_THAN)        : LESS_EQUAL_THAN,
    (EQUAL, GREATER_THAN)     : GREATER_EQUAL_THAN,
    (LESS_THAN, GREATER_THAN) : NOT_EQUAL,
    (GREATER_THAN, LESS_THAN) : NOT_EQUAL,

    (NOT, EQUAL)              : NOT_EQUAL,
    (NOT, NOT_EQUAL)          : EQUAL,
    (NOT, GREATER_THAN)       : LESS_EQUAL_THAN,
    (NOT, LESS_THAN)          : GREATER_EQUAL_THAN,
    (NOT, GREATER_EQUAL_THAN) : LESS_THAN,
    (NOT, LESS_EQUAL_THAN)    : GREATER_THAN,
}


class Tokenizer:

    def __init__(self, stream):
        self.stream = stream
        self.current_character = self.stream.next()

    def advance(self, times=1):
        current_character = ''
        character = self.current_character
        for i in range(times):
            current_character = self.stream.next()
        self.current_character = current_character
        return character

    def skip_comment(self):
        while self.current_character != '\n' and self.current_character != '':
            self.advance()
        self.advance()

    def skip_whitespace(self):
        while self.current_character.isspace():
            self.current_character = self.stream.next()

    def read_identifier(self):
        result = self.advance()
        while self.current_character.isalnum() or self.current_character == '_':
            result += self.advance()
        return KEYWORDS.get(result.lower(), Token(ID, result))  # Case-insensitive for keywords

    def read_number(self):
        result = self.advance()
        while self.current_character.isdigit():
            result += self.advance()
        if self.current_character == '.':
            result += self.advance()
            while self.current_character.isdigit():
                result += self.advance()
            return Token(REAL_CONST, result)
        else:
            return Token(INTEGER_CONST, result)

    def read_string(self):
        self.advance()  # Advance pass the first quotation mark.
        result = ''
        while self.current_character != '"':
            character = self.advance()
            if character == '\\':
                next_character = self.advance()
                if next_character == 'n':
                    result += '\n'
                continue
            result += character
        self.advance()  # Advance pass the last quotation mark.
        return Token(STRING_CONST, result)

    def skip_multiple_end_statements(self):
        if self.current_character== '\n':
            while self.stream.peek() == '\n':
                self.advance()
            if self.current_character == '/' and self.stream.peek() == '/':
                self.skip_comment()
            self.skip_multiple_end_statements()
 

    def get_next_token(self):

        while self.current_character.isspace() or self.current_character == '/' and self.stream.peek() == '/':
            if self.current_character.isspace():
                self.skip_whitespace()
            if self.current_character == '/' and self.stream.peek() == '/':
                self.skip_comment()

        character = self.current_character

        if character.isalpha() or character == '_':
            return self.read_identifier()
        elif character.isdigit():
            return self.read_number()
        elif character == '"':
            return self.read_string()

        token = KEYWORDS.get(character)
        if isinstance(token, Token):
            self.advance()  # Skip current character.
            return token
        elif isinstance(token, dict):
            self.advance()  # Skip current character.
            character = self.current_character
            if character in token:
                self.advance()
                token = token[character]
            else:
                token = token['default']
            return token
        elif character != '':
            return Token(UNKNOWN, character)
        else:
            return Token(EOF, '')


# ------------------------------ NODES ------------------------------
# AtomNode = namedtuple('AtomNode', 'name, type, str')
class Atom:
    """
    Internal representation of an atom.
    """
    TYPE = {INTEGER: int, REAL: float, STRING: str, FUNCTION: callable}
    def __init__(self, type_: (str, None), value: (str, callable)):
        assert type_ in Atom.TYPE or type_ is None, '{} not valid type'.format(type_)
        self.type  = type_
        self.value = value

    def type_check(self, other):
        if self.type == other.type or other.type is None:
            return 1
        elif self.type == INTEGER and other.type == REAL:
            # Only occurs for expressions. Cannot change type of variable.
            self.type = REAL
            return 1
        elif self.type == REAL and other.type == INTEGER:
            other.type = REAL
            return 1
        raise TypeError('')

    def __add__(self, other):
        if self.type_check(other):
            return Atom(self.type, str(Atom.TYPE[self.type](self.value) + Atom.TYPE[self.type](other.value)))

    def __sub__(self, other):
        if self.type_check(other):
            return Atom(self.type, str(Atom.TYPE[self.type](self.value) - Atom.TYPE[self.type](other.value)))

    def __mul__(self, other):
        if self.type_check(other):
            return Atom(self.type, str(Atom.TYPE[self.type](self.value) * Atom.TYPE[self.type](other.value)))

    def __floordiv__(self, other):
        if self.type_check(other):
            return Atom(self.type, str(Atom.TYPE[self.type](self.value) // Atom.TYPE[self.type](other.value)))

    def __truediv__(self, other):
        if self.type_check(other):
            return Atom(self.type, str(Atom.TYPE[self.type](self.value) / Atom.TYPE[self.type](other.value)))

    def __pow__(self, other):
        if self.type_check(other):
            return Atom(self.type, str(Atom.TYPE[self.type](self.value) ** Atom.TYPE[self.type](other.value)))


    def __eq__(self, other):
        return Atom.TYPE[self.type](self.value) == Atom.TYPE[self.type](other.value)

    def __ne__(self, other):
        return Atom.TYPE[self.type](self.value) != Atom.TYPE[self.type](other.value)

    def __ge__(self, other):
        return Atom.TYPE[self.type](self.value) >= Atom.TYPE[self.type](other.value)

    def __le__(self, other):
        return Atom.TYPE[self.type](self.value) <= Atom.TYPE[self.type](other.value)

    def __gt__(self, other):
        return Atom.TYPE[self.type](self.value) > Atom.TYPE[self.type](other.value)

    def __lt__(self, other):
        return Atom.TYPE[self.type](self.value) < Atom.TYPE[self.type](other.value)

    def __repr__(self):
        if PRINT_JUST_VALUE:
            return str(self.value)
        return 'Atom({!r} {!r})'.format(self.type, self.value)


class ConstantNode:
    """
    Container of a value.
    """
    TYPES = INTEGER_CONST, REAL_CONST, STRING_CONST
    def __init__(self, inferred_type, value: str):
        assert inferred_type in ConstantNode.TYPES
        assert isinstance(value, str)
        self.inferred_type = inferred_type
        self.value = value

class NameNode:
    """
    Container of a name.
    """
    def __init__(self, name: str):
        assert isinstance(name, str)
        self.name = name


class ListNode:

    def __init__(self, size, data_type, *data):
        self.size = size
        self.data = data
        self.data_type = data_type

class AssignNode:
    TYPES = INTEGER, REAL, STRING, FUNCTION
    def __init__(self, name, type_, expression):
        assert isinstance(name, NameNode)
        assert type_ in AssignNode.TYPES or type_ is None, '{} is not a valid type!'.format(type_)
        self.name       = name
        self.type       = type_
        self.expression = expression  # Needs to be evaluated

class ReAssign:
    def __init__(self, name, expression):
        assert isinstance(name, NameNode)
        self.name = name
        self.expression = expression

class BinaryOperationNode:
    OPERATIONS = ADD, SUB, MUL, INT_DIV, REAL_DIV, POW
    def __init__(self, left_expression, operation, right_expression):
        assert operation in BinaryOperationNode.OPERATIONS
        self.left_expression  = left_expression
        self.operation        = operation
        self.right_expression = right_expression

class UnaryOperationNode:
    OPERATIONS = NEGATE, SQRT, INCREMENT, DECREMENT
    def __init__(self, operation, expression):
        assert operation in UnaryOperationNode.OPERATIONS
        self.operation  = operation
        self.expression = expression

class BlockNode:
    def __init__(self, statements):
        self.statements = statements
        self.namespace  = OrderedDict()

class BranchNode:
    def __init__(self, left, condition, right=None):
        assert isinstance(condition, ConditionNode)
        self.left      = left
        self.condition = condition
        self.right     = right if right is not None else NoOperationNode()

class WhileLoopNode:
    def __init__(self, condition, block):
        assert isinstance(condition, ConditionNode)
        self.condition = condition
        self.block     = block

class ConditionNode:
    OPERATIONS = GREATER_THAN, GREATER_EQUAL_THAN, EQUAL, LESS_EQUAL_THAN, LESS_THAN, NOT_EQUAL, AND, OR
    def __init__(self, left, operator, right):
        assert operator in ConditionNode.OPERATIONS, '{} not a valid operator!'.format(operator)
        self.left      = left
        self.operator  = operator
        self.right     = right

class CallNode:
    def __init__(self, name, arguments=None):
        assert isinstance(name, str)
        self.name = name
        self.arguments = arguments

class BuiltInFunction:
    def __init__(self, name, arguments):
        assert isinstance(name, str)
        self.name = name
        self.arguments = list(arguments)

class NoOperationNode:
    pass






# ------------------------------ PARSER ------------------------------

class ParserError(Exception):
    def __init__(self, parser, error_message=''):
        message = "Error around row {}, column {}:"" \
        ""\n    Illogical token sequence: {}, {}".format(
            parser.token_stream.stream.row, parser.token_stream.stream.column,
            parser.previous_token, parser.current_token
        )
        super().__init__(error_message + '\n' + message)


class Parser:
    """
    ---- Grammar ----
    
    program      : (statement)*
    block        : LBRACKET (statement)* RBRACKET
                   | statement statement
    statement    : assignment END_STATEMENT
    assignment   : variable DECLARE_ASSIGN expression
    expression   : term ((ADD | SUB) term)*
    term         : factor ((MUL | INT_DIV | REAL_DIV) factor)*
    factor       : INTEGER_CONST
                   | REAL_CONST
                   | variable
                   | NEG factor
                   | LPAREN expr RPAREN
    variable     :  ID
    """

    def __init__(self, token_stream):
        self.token_stream = token_stream
        self.previous_token = None
        self.current_token = self.token_stream.get_next_token()

    def parse(self):
        return self.program()

    def program(self):
        """         
        program   :  (statement)* EOF
        """
        statements = []
        while self.current_token.type != EOF:
            statements.append(self.statement())
        return BlockNode(statements)  # ProgramNode?

    def statement(self):
        """
        statement    : assignment END_STATEMENT       Is also for declaration.
                     | function END_STATEMENT
                     | block                          Should a block require an end statement?
                     | call END_STATEMENT
                     | IF condition THEN statement (ELSE IF condition block)*               Block or statement???
                     | IF condition THEN statement (ELSE IF condition block)* ELSE block
                     | WHILE condition THEN statement
                     | END_STATEMENT

        return  :  AssignNode, FunctionNode, CallNode, BlockNode or BranchNode.
        """

        if self.current_token.type == ID:    # Will fail when using user-defined functions.
            node = self.assignment()
            self.consume(END_STATEMENT)
            return node
        elif self.current_token.type == FUNCTION:
            node = self.function()
            self.consume(END_STATEMENT)
            return node
        elif self.current_token.type == LBRACKET:
            node = self.block()
            return node
        elif self.current_token.type == CALL:
            node = self.call()
            self.consume(END_STATEMENT)
            return node
        elif self.current_token.type == IF:
            self.consume(IF)
            condition = self.condition()
            self.consume(THEN)
            node = BranchNode(self.statement(), condition)
            while self.current_token.type == ELSE:
                self.consume(ELSE)
                if self.current_token.type == IF:
                    node.right = self.statement()  # Since token type is 'IF', it'll come back here.
                else:
                    node.right = self.statement()
            return node
        elif self.current_token.type == WHILE:
            self.consume(WHILE)
            condition = self.condition()
            self.consume(THEN)
            return WhileLoopNode(condition, self.block())
        elif self.current_token.type == RETURN:
            self.consume(RETURN)
            return self.expression()

        raise ParserError(self)

    def condition(self, previous_condition=None):
        """
        condition  :  expression OPERATOR expression
                   |  expression [IS] OPERATOR [OR OPERATOR] [THAN|TO] expression
                   |  expression [IS] NOT [OPERATOR [OR OPERATOR]] [THAN|TO] expression
                   |  condition ((AND|OR) condition)* 
        """
        if self.current_token.type in (EQUAL, NOT_EQUAL, GREATER_THAN, GREATER_EQUAL_THAN, LESS_THAN, LESS_EQUAL_THAN, NOT, IS) \
                and previous_condition is not None:
            node = previous_condition
            if isinstance(node.left, NameNode):
                left = node.left
            else:
                raise Exception('Only left variable can be chained. Switch {} with {}.'.format(node.left.value, node.right.name))
        else:
            left = self.expression()

        if self.current_token.type == IS:
            self.consume(IS)

        operator = self.current_token.type
        self.consume(operator)

        if operator == NOT:  # Already consumed!
            if self.current_token.type in (EQUAL, NOT_EQUAL, GREATER_THAN, GREATER_EQUAL_THAN, LESS_THAN, LESS_EQUAL_THAN):
                operator2 = self.current_token.type
                self.consume(operator2)
                if self.current_token.type == OR:
                    self.consume(OR)
                    operator3 = self.current_token.type
                    self.consume(operator3)
                    operator2 = operator_logic[operator2, operator3]
                operator = operator_logic[operator, operator2]
            else:
                operator = NOT_EQUAL
        elif self.current_token.type == OR:
            self.consume(OR)
            operator2 = self.current_token.type
            self.consume(operator2)
            operator = operator_logic[operator, operator2]

        if self.current_token.type in (THAN, TO):
            self.consume(self.current_token.type)

        right = self.expression()

        node = ConditionNode(left, operator, right)

        if self.current_token.type in (AND, OR):
            logic_operator = self.current_token.type
            self.consume(logic_operator)
            return ConditionNode(node, logic_operator, self.condition(node))

        return node

    def call(self):
        """
        call  :  CALL variable [LPARENS expression (COMMA expression)* RPARENS]
        """
        self.consume(CALL)
        name = self.current_token.value
        self.consume(ID)

        arguments = None

        if self.current_token.type == LPARENS:
            self.consume(LPARENS)
            arguments = [self.assignment()]
            while self.current_token.type == COMMA:
                self.consume(COMMA)
                arguments.append(self.assignment())
            self.consume(RPARENS)

        return CallNode(name, arguments)

    def block(self):
        """
        block   :  LBRACKET (statement)* RBRACKET

        return  :  BlockNode.
        """
        self.consume(LBRACKET)
        statements = []
        while self.current_token.type != RBRACKET:
            statements.append(self.statement())
        self.consume(RBRACKET)
        return BlockNode(statements)

    def function(self):  # Procedure?
        """
        function  :  FUNCTION [arguments]*
        """
        token = self.current_token
        self.consume(FUNCTION)
        node = BuiltInFunction(token.value, self.arguments())
        return node

    def assignment(self):
        """
        assignment  : variable DECLARE_ASSIGN expression
                    | variable DECLARE_ASSIGN block                     Procedure.
                    | variable DECLARE DATA_TYPE ASSIGN expression
                    | variable DECLARE DATA_TYPE                        Just a declaration
                    | variable ASSIGN expression                        Must be declared first.

        returns  :  AssignNode or DeclarationNode
        """
        variable = NameNode(self.current_token.value)
        self.consume(ID)


        if self.current_token.type == DECLARE_ASSIGN:
            self.consume(DECLARE_ASSIGN)

            if self.current_token.type == LBRACKET:
                return AssignNode(variable, FUNCTION, self.block())
            else:
                return AssignNode(variable, None, self.expression())

        elif self.current_token.type == DECLARE:
            self.consume(DECLARE)

            if self.current_token.type == LPARENS:
                self.consume(LPARENS)
                arguments = self.parameters()
                self.consume(RPARENS)
                self.consume(ASSIGN)
                body = self.block()
                for arg_assignments in arguments:
                    body.statements.insert(0, arg_assignments)
                return AssignNode(variable, FUNCTION, body)

            data_type = self.current_token.value
            self.consume(self.current_token.type)


            if self.current_token.type == ASSIGN:
                self.consume(ASSIGN)
                return AssignNode(variable, data_type, self.expression())
            else:
                return AssignNode(variable, data_type, NoOperationNode())

        elif self.current_token.type == ASSIGN:
            self.consume(ASSIGN)
            return ReAssign(variable, self.expression())

        else:
            raise ParserError(self)

    def parameters(self):
        arguments = [self.assignment()]

        while self.current_token.type == COMMA:
            self.consume(COMMA)
            arguments.append(self.assignment())

        return arguments

    def arguments(self):

        if self.current_token.type == LPARENS:
            self.consume(LPARENS)

            arguments = []

            while self.current_token.type == COMMA:
                self.consume(COMMA)
                arguments.append(self.expression())

            self.consume(RPARENS)

        else:
            arguments = [self.expression()]

            while self.current_token.type == COMMA:
                self.consume(COMMA)
                arguments.append(self.expression())

        return arguments


    def expression(self):
        """
        expression  : term ((ADD | SUB) term)* (INC|DEC)*

        returns  :  DataNode, VariableNode, UnaryOperationNode or BinaryOperationNode.
        """
        node = self.term()

        while self.current_token.type in (ADD, SUB):
            token = self.current_token
            self.consume(token.type)
            node = BinaryOperationNode(node, token.type, self.term())

        while self.current_token.type in (INCREMENT, DECREMENT):
            token = self.current_token
            self.consume(token.type)
            node = UnaryOperationNode(token.type, node)

        return node

    def term(self):
        """
        term  : factor ((MUL | INT_DIV | REAL_DIV) factor)* (INC|DEC)*

        returns  :  DataNode, VariableNode, UnaryOperationNode or BinaryOperationNode.
        """
        node = self.power()

        while self.current_token.type in (MUL, REAL_DIV, INT_DIV):
            token = self.current_token
            self.consume(token.type)
            node = BinaryOperationNode(node, token.type, self.power())

        while self.current_token.type in (INCREMENT, DECREMENT):
            token = self.current_token
            self.consume(token.type)
            node = UnaryOperationNode(token.type, node)

        return node

    def power(self):
        node = self.factor()

        while self.current_token.type == POW:
            token = self.current_token
            self.consume(token.type)
            node = BinaryOperationNode(node, token.type, self.factor())

        while self.current_token.type == SQRT:
            token = self.current_token
            self.consume(token.type)
            node = UnaryOperationNode(token.type, node)

        return node


    def factor(self):
        """
        factor : INTEGER_CONST
               | REAL_CONST
               | variable
               | (NEGATE|SQRT|INC|DEC|PRINT) factor
               | LPAREN expr RPAREN
               | FUNCTION [arguments | (LPARENS arguments RPARENS)]

        returns  :  DataNode, VariableNode, UnaryOperationNode or BinaryOperationNode.
        """
        token = self.current_token

        if token.type == INTEGER_CONST:
            self.consume(INTEGER_CONST)
            return ConstantNode(token.type, token.value)
        elif token.type == REAL_CONST:
            self.consume(REAL_CONST)
            return ConstantNode(token.type, token.value)
        elif token.type == STRING_CONST:
            self.consume(STRING_CONST)
            return ConstantNode(token.type, token.value)
        elif token.type == ID:
            self.consume(ID)
            return NameNode(token.value)
        elif token.type in (NEGATE, SQRT, INCREMENT, DECREMENT, PRINT):
            self.consume(token.type)
            return UnaryOperationNode(token.type, self.factor())
        elif token.type == LPARENS:
            self.consume(LPARENS)
            node = self.expression()  # DataNode, VariableNode, UnaryOperationNode, BinaryOperationNode
            self.consume(RPARENS)
            return node
        elif token.type == FUNCTION:
            name = token.value
            self.consume(FUNCTION)
            if self.current_token.type == LPARENS:
                self.consume(LPARENS)
            node = BuiltInFunction(name, self.arguments())
            if self.current_token.type == RPARENS:
                self.consume(RPARENS)
            return node
        else:
            raise ParserError(self)

    def consume(self, token_type):
        if self.current_token.type == token_type:
            self.previous_token = self.current_token
            self.current_token = self.token_stream.get_next_token()
        else:
            raise ParserError(self, 'Got {}, expected {}'.format(self.current_token.type, token_type))







# ------------------------------ INTERPRETER ------------------------------




class Interpreter:

    def __init__(self, parser):
        self.parser = parser
        self.current_namespace = None


    def visit_error(self, node):
        error_message = 'Could not find method visit_{}.'.format(node.__class__.__name__)
        raise ParserError(self.parser, error_message)

    def error(self, node):
        error_message = 'Error in node {}.'.format(node.__class__.__name__.replace('Node'))
        raise ParserError(self.parser, error_message)

    def visit(self, node):
        method_name = 'visit_' + node.__class__.__name__
        method = getattr(self, method_name, self.visit_error)
        return method(node)


    def interpret(self):
        return self.visit(self.parser.parse())


    def visit_BlockNode(self, node):

        current_namespace = self.current_namespace

        new_namespace = node.namespace
        new_namespace['__global_namespace__'] = current_namespace

        self.current_namespace = new_namespace  # Use new namespace as current namespace.

        for statement in node.statements:
            self.visit(statement)

        new_namespace.clear()

        self.current_namespace = current_namespace  # Restore old namespace.


    def visit_BuiltInFunction(self, node):
        if node.name == PRINT:
            return print(*(self.visit(argument) for argument in node.arguments))
        elif node.name == SUM:
            result = Atom(REAL, '0')
            for argument in node.arguments:
                result = result + self.visit(argument)
            return result
        self.error(node)


    def visit_ListNode(self, node):
        pass

    def visit_UnaryOperationNode(self, node):
        operation = node.operation

        if operation == NEGATE:
            return -self.visit(node.expression)
        if operation == SQRT:
            return self.visit(node.expression) ** Atom(REAL, '0.5')
        if operation == INCREMENT:
            return self.visit(node.expression) + Atom(None, '1')
        if operation == DECREMENT:
            return self.visit(node.expression) - Atom(None, '1')

        self.error(node)

    def visit_BinaryOperationNode(self, node):
        operation = node.operation

        if operation == ADD:
            return self.visit(node.left_expression) + self.visit(node.right_expression)
        if operation == SUB:
            return self.visit(node.left_expression) - self.visit(node.right_expression)
        if operation == MUL:
            return self.visit(node.left_expression) * self.visit(node.right_expression)
        if operation == INT_DIV:
            return self.visit(node.left_expression) // self.visit(node.right_expression)
        if operation == REAL_DIV:
            return self.visit(node.left_expression) / self.visit(node.right_expression)
        if operation == POW:
            return self.visit(node.left_expression) ** self.visit(node.right_expression)

        self.error(node)

    def visit_NameNode(self, node):
        namespace = self.current_namespace

        while namespace is not None:
            value = namespace.get(node.name)
            if value is not None:
                return value
            else:
                namespace = namespace.get('__global_namespace__')

        raise NameError('{} is undefined'.format(node.name))


    def visit_ConstantNode(self, node):
        data_type = node.inferred_type

        if data_type == INTEGER_CONST:
            return Atom(INTEGER, node.value)
        if data_type == REAL_CONST:
            return Atom(REAL, node.value)
        if data_type == STRING_CONST:
            return Atom(STRING, node.value)

        self.error(node)

    def visit_AssignNode(self, node):

        name = node.name.name  # AssignNode.NameNode.name
        data_type = node.type

        if name in self.current_namespace:
            print(self.current_namespace)
            raise NameError('{} already defined!'.format(name))

        if data_type == FUNCTION:
            self.current_namespace[name] = Atom(FUNCTION, node.expression)
            return

        atom = self.visit(node.expression)  # NO TYPE-CHECKING AT DECLARATION, ONLY REASSIGNMENT.
        if data_type is None:
            self.current_namespace[name] = Atom(atom.type, atom.value)
        elif atom is None:
            self.current_namespace[name] = Atom(data_type, None)
        else:
            self.current_namespace[name] = Atom(data_type, atom.value)

    def visit_ReAssign(self, node):
        name = node.name.name  # ReAssignNode.NameNode.name

        atom = self.visit(node.name)

        new_atom = self.visit(node.expression)
        if atom.type != new_atom.type:
            raise TypeError('{} cannot be assigned to {}'.format(new_atom.type, atom.type))
        else:
            namespace = self.current_namespace

            while namespace is not None:
                value = namespace.get(name)
                if value is not None:
                    namespace[name].value = new_atom.value
                    return
                else:
                    namespace = namespace.get('__global_namespace__')

            raise NameError('{} is undefined'.format(name))

    def visit_CallNode(self, node):

        namespace = self.current_namespace

        while namespace is not None:
            atom = namespace.get(node.name)
            if atom is not None:
                block = atom.value
                if node.arguments is not None:
                    offset = len(node.arguments)
                    for i, argument in enumerate(node.arguments):
                        block.statements.insert(offset + i, argument)
                return self.visit(block)
            else:
                namespace = namespace.get('__global_namespace__')

        raise NameError('{} is undefined'.format(node.name))

    def visit_ConditionNode(self, node):
        left  = self.visit(node.left)
        right = self.visit(node.right)
        operator = node.operator

        if   operator == EQUAL:
            return left == right
        elif operator == LESS_THAN:
            return left < right
        elif operator == LESS_EQUAL_THAN:
            return left <= right
        elif operator == GREATER_THAN:
            return left > right
        elif operator == GREATER_EQUAL_THAN:
            return left >= right
        elif operator == NOT_EQUAL:
            return left != right
        elif operator == OR:
            return left or right
        elif operator == AND:
            return left and right
        else:
            self.error(node)

    def visit_BranchNode(self, node):
        if self.visit(node.condition):
            self.visit(node.left)
        else:
            self.visit(node.right)

    def visit_NoOperationNode(self, node):
        pass

    def visit_WhileLoopNode(self, node):
        while self.visit(node.condition):
            self.visit(node.block)











import sys
import datetime

source_path = ''
if len(sys.argv) >= 2:  # Running from the command line with path argument
    source_path = sys.argv[1]
    source_code = open(source_path).read()

    start_time = datetime.datetime.now()

    print('Running: {} at {}\n'.format(source_path, start_time))

    interpreter = Interpreter(Parser(Tokenizer(InputStream(source_code))))
    interpreter.interpret()

    print('\nFinished in {} s\n'.format(datetime.datetime.now() - start_time))
    exit()



interpreter = Interpreter(Parser(Tokenizer(InputStream(open('scripts/test2.ted').read()))))
interpreter.interpret()
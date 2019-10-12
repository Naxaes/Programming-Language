# A token is the smallest component of the program. For example a keyword, operation, number, etc.
# Grammar is divided in terminals and non-terminals, where non-terminals is made up of both terminals and non-terminals.
# A terminal is a token.

from collections import OrderedDict
from collections import defaultdict

from inputstream import InputStream

# ---- Token types ----

# Data types
DATA_TYPE = 'DATA_TYPE'
INTEGER, REAL, STRING = 'INTEGER', 'REAL', 'STRING'

# Constants
INTEGER_CONST, REAL_CONST, STRING_CONST = 'INTEGER_CONST', 'REAL_CONST', 'STRING_CONST'

# Binary operations
ADD, SUB, MUL, INT_DIV, REAL_DIV, POW = 'ADD', 'SUB', 'MUL', 'INT_DIV', 'REAL_DIV', 'POW'

# Unary operations
NEGATE, SQRT, INCREMENT, DECREMENT = 'SUB', 'SQRT', 'INC', 'DEC'

# Built-in functions
FUNCTION, PRINT = 'FUNCTION', 'PRINT'

# Parenthesis
LPARENS, RPARENS, LBRACKET, RBRACKET = 'LPARENS', 'RPARENS', 'LBRACKET', 'RBRACKET'

# Operators.
GREATER_THEN, GREATER_EQUAL_THEN, EQUAL, LESS_EQUAL_THEN, LESS_THEN, NOT_EQUAL = (
    'GREATER_THEN', 'GREATER_EQUAL_THEN', 'EQUAL', 'LESS_EQUAL_THEN', 'LESS_THEN', 'NOT_EQUAL'
)
# Branch.
IF, THEN, ELSE, WHILE = 'IF', 'THEN', 'ELSE', 'WHILE'

# Other
ID, COMMA, END_STATEMENT, EOF, DECLARE_ASSIGN, ASSIGN, CALL, DECLARE = (
    'ID', 'COMMA', 'END_STATEMENT', 'EOF', 'DECLARE_ASSIGN', 'ASSIGN', 'CALL', 'DECLARE'
)


class Token:
    def __init__(self, type, value):
        self.type  = type
        self.value = value
    def __repr__(self):
        return 'Token(type={}, value={})'.format(self.type, self.value)


KEYWORDS = {
    'int'   : Token(DATA_TYPE, INTEGER),
    'real'  : Token(DATA_TYPE, REAL),
    'string': Token(DATA_TYPE, STRING),
    'pow'   : Token(POW, POW),
    'sqrt'  : Token(SQRT, SQRT),
    'inc'   : Token(INCREMENT, INCREMENT),
    'dec'   : Token(DECREMENT, DECREMENT),
    'print' : Token(FUNCTION, PRINT),
    'call'  : Token(CALL, CALL),
    'if'    : Token(IF, IF),
    'then'  : Token(THEN, THEN),
    'else'  : Token(ELSE, ELSE),
    'while' : Token(WHILE, WHILE),

    '*': Token(MUL, '*'),
    '/': Token(REAL_DIV, '/'),
    ',': Token(COMMA, ','),
    ';': Token(END_STATEMENT, ';'),
    '(': Token(LPARENS, '('),
    ')': Token(RPARENS, ')'),
    '{': Token(LBRACKET, '{'),
    '}': Token(RBRACKET, '}'),

  # FIRST                                               SECOND
    '=': defaultdict(lambda: Token(ASSIGN, '='),       **{'=': Token(EQUAL, '==')}),
    '+': defaultdict(lambda: Token(ADD, '+'),          **{'+': Token(INCREMENT, '++')}),
    '-': defaultdict(lambda: Token(SUB, '-'),          **{'-': Token(DECREMENT, '--')}),
    ':': defaultdict(lambda: Token(DECLARE, ':'),      **{'=': Token(DECLARE_ASSIGN, ':=')}),
    '>': defaultdict(lambda: Token(GREATER_THEN, '>'), **{'=': Token(GREATER_EQUAL_THEN, '>=')}),
    '<': defaultdict(lambda: Token(LESS_THEN, '<'),    **{'=': Token(LESS_EQUAL_THEN, '<=')}),
    '!': defaultdict(lambda: None,                     **{'=': Token(NOT_EQUAL, '!=')})
}


# ---- Nodes ----

# Every rule in the grammar is a node (No?).
# Every operation is a node.
# Every data type is a node.
# Several rules/operations/data types can be put into a single note. Example: Number, BinaryOperation, etc.

# A node binds two children.

class UnaryOperationNode:
    """
    Operation between one values. This is a combination of:   
        NegateNode, SqrtNode, IncNode and DecNode.
    
    Unary nodes construct expressions of type UnaryOperation Expression.
    """
    OPERATIONS = NEGATE, SQRT, INCREMENT, DECREMENT
    def __init__(self, operation, child):
        assert isinstance(operation, Token) and operation.type in UnaryOperationNode.OPERATIONS
        assert child.is_expression
        self.operation = operation.type
        self.child     = child

        self.is_expression = True

class BinaryOperationNode:
    """
    Operation between two values. This is a combination of:   
        AddNode, SubNode, MulNode, IntDivNode, RealDivNode and PowNode.
    
    Binary nodes construct expressions of type Expression BinaryOperation Expression.
    """
    OPERATIONS = ADD, SUB, MUL, INT_DIV, REAL_DIV, POW
    def __init__(self, left, operation, right):
        assert left.is_expression
        assert isinstance(operation, Token) and operation.type in BinaryOperationNode.OPERATIONS
        assert right.is_expression
        self.left      = left
        self.operation = operation.type
        self.right     = right

        self.is_expression = True

class VariableNode:
    """
    Just a single identifier.
    
    A variable is an expression and it's thus syntactically correct to have uninitialized variables when
    an expression is expected, but it's not logically correct and will throw an error.
    """
    def __init__(self, name):
        assert isinstance(name, Token) and name.type == ID
        self.name  = name.value

        self.is_expression = True

class DataNode:
    """
    A data type object.
    
    The value 
    """
    DATA_TYPES = INTEGER_CONST, REAL_CONST, STRING_CONST
    def __init__(self, value, inferred_type):
        assert isinstance(value, (int, float, str))
        assert inferred_type in DataNode.DATA_TYPES
        self.value = value
        self.inferred_type = inferred_type

        self.is_expression = True

class AssignNode:
    """
    Link between a variable and a value/expression.
    
    An assignment will not evaluate to a value and is thus not an expression.
    """
    DATA_TYPES = INTEGER, REAL, STRING, FUNCTION
    def __init__(self, left, right, type=None):
        assert isinstance(left, VariableNode)
        # assert right.is_expression
        assert type is None or type.value in AssignNode.DATA_TYPES, "{} is not a valid type!".format(type.value)
        self.left  = left
        self.right = right
        self.type = type.value if type is not None else None

        self.is_expression = False

class ReAssign:
    DATA_TYPES = INTEGER, REAL, STRING, FUNCTION
    def __init__(self, left, right, type=None):
        assert isinstance(left, VariableNode)
        # assert right.is_expression
        assert type is None or type.value in AssignNode.DATA_TYPES, "{} is not a valid type!".format(type.value)
        self.left  = left
        self.right = right
        self.type = type.value if type is not None else None

        self.is_expression = False

class DeclarationNode:
    DATA_TYPES = INTEGER, REAL, STRING, FUNCTION
    def __init__(self, name, data_type):
        assert isinstance(name, VariableNode)
        assert data_type is None or isinstance(data_type, Token) and data_type.type in DeclarationNode.DATA_TYPES
        self.name = name
        self.type = data_type.type if data_type is not None else None

class BlockNode:
    def __init__(self, children):
        self.children = children
        self.namespace = {}

        self.is_expression = False

class ProgramNode:
    def __init__(self, children):
        self.children = children

        self.is_expression = False

class BuiltInFunction:
    FUNCTIONS = PRINT,
    def __init__(self, name, *args):
        assert isinstance(name, Token) and name.type == FUNCTION and name.value in BuiltInFunction.FUNCTIONS
        assert all(argument.is_expression for argument in args)
        self.name = name.value
        self.args = list(args)

        self.is_expression = False


class CallNode:
    def __init__(self, name):
        assert isinstance(name, Token) and name.type == ID
        self.name  = name.value


class ConditionNode:
    OPERATORS = GREATER_THEN, GREATER_EQUAL_THEN, EQUAL, LESS_EQUAL_THEN, LESS_THEN, NOT_EQUAL
    def __init__(self, left, operator, right):
        assert left.is_expression and right.is_expression
        assert isinstance(operator, Token)
        assert operator.type in ConditionNode.OPERATORS
        self.left     = left
        self.operator = operator.type
        self.right    = right


class BranchNode:
    def __init__(self, left, condition, right=None):
        assert isinstance(condition, ConditionNode)
        self.condition = condition
        self.left      = left
        self.right     = right if right is not None else NoOperationNode()


class NoOperationNode:
    pass


class WhileLoopNode:
    def __init__(self, condition, block):
        assert isinstance(condition, ConditionNode)
        assert isinstance(block, BlockNode)
        self.condition = condition
        self.block     = block


# ---- Grammar ----
#
# program      : compound
# compound     : statement
#                | statement statement
# statement    : (assignment | expression) SEMI
# assignment   : variable DECLARE_ASSIGN expression
# expression   : term ((ADD | SUB) term)*
# term         : factor ((MUL | INT_DIV | REAL_DIV) factor)*
# factor       : INTEGER_CONST
#                | REAL_CONST
#                | variable
#                | NEG factor
#                | LPAREN expr RPAREN
# variable     :  ID



# Some rules evaluates into a number. Example: expression, term, factor.

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
        elif isinstance(token, defaultdict):
            self.advance()  # Skip current character.
            following_character = self.advance()
            token = token[following_character]
            return token
        else:
            return Token(EOF, '')


class ParserError(Exception):
    def __init__(self, parser):
        message = "Error around row {}, column {}:"" \
        ""\n    Illogical token sequence: {}, {}".format(
            parser.token_stream.stream.row, parser.token_stream.stream.column,
            parser.current_token, parser.token_stream.get_next_token()
        )
        super().__init__(message)


class Parser:
    
    # ---- Grammar ----
    #
    # program      : (statement)*
    # block        : LBRACKET (statement)* RBRACKET
    #                | statement statement
    # statement    : assignment END_STATEMENT
    # assignment   : variable DECLARE_ASSIGN expression
    # expression   : term ((ADD | SUB) term)*
    # term         : factor ((MUL | INT_DIV | REAL_DIV) factor)*
    # factor       : INTEGER_CONST
    #                | REAL_CONST
    #                | variable
    #                | NEG factor
    #                | LPAREN expr RPAREN
    # variable     :  ID
    
    def __init__(self, token_stream):
        self.token_stream = token_stream
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
                     | IF condition THEN block (ELSE IF condition block)*               Block or statement???
                     | IF condition THEN block (ELSE IF condition block)* ELSE block
                     | WHILE condition THEN block
        
        return  :  AssignNode, FunctionNode, CallNode, BlockNode or BranchNode.
        """
        if self.current_token.type == ID:
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
            node = BranchNode(self.block(), condition)
            while self.current_token.type == ELSE:
                self.consume(ELSE)
                if self.current_token.type == IF:
                    node.right = self.statement()  # Since token type is 'IF', it'll come back here.
                else:
                    node.right = self.block()
            return node
        elif self.current_token.type == WHILE:
            self.consume(WHILE)
            condition = self.condition()
            self.consume(THEN)
            return WhileLoopNode(condition, self.block())

        raise ParserError(self)

    def condition(self):
        """
        condition  :  expression OPERATOR expression
        """
        left = self.expression()
        operator = self.current_token
        self.consume(self.current_token.type)
        right = self.expression()
        return ConditionNode(left, operator, right)

    def call(self):
        """
        call  :  CALL variable 
        """
        self.consume(CALL)
        name = self.current_token
        self.consume(ID)
        return CallNode(name)


    def block(self):
        """
        block   :  LBRACKET (statement)* RBRACKET

        return  :  BlockNode.
        """
        self.consume(LBRACKET)
        statement = []
        while self.current_token.type != RBRACKET:
            statement.append(self.statement())
        self.consume(RBRACKET)
        return BlockNode(statement)

    def function(self):  # Procedure?
        """
        function  :  PRINT expression (COMMA expression)*
        """
        token = self.current_token
        self.consume(FUNCTION)
        node = BuiltInFunction(token, self.expression())
        while self.current_token.type == COMMA:
            self.consume(COMMA)
            node.args.append(self.expression())
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
        token = self.current_token
        self.consume(ID)

        variable = VariableNode(token)
        data_type = None

        if self.current_token.type == DECLARE_ASSIGN:
            self.consume(DECLARE_ASSIGN)
            if self.current_token.type == LBRACKET:
                data_type = Token(DATA_TYPE, FUNCTION)
                return AssignNode(variable, self.block(), data_type)
            else:
                return AssignNode(variable, self.expression(), data_type)
        elif self.current_token.type == DECLARE:
            self.consume(DECLARE)
            data_type = self.current_token
            self.consume(data_type.type)
            if self.current_token.type == ASSIGN:
                self.consume(ASSIGN)
                return AssignNode(variable, self.expression(), data_type)
            else:
                return AssignNode(variable, None, data_type)
        elif self.current_token.type == ASSIGN:
            self.consume(ASSIGN)
            return ReAssign(variable, self.expression(), data_type)
        else:
            raise ParserError(self)

    def expression(self):
        """
        expression  : term ((ADD | SUB) term)* (INC|DEC)*
        
        returns  :  DataNode, VariableNode, UnaryOperationNode or BinaryOperationNode.
        """
        node = self.term()

        while self.current_token.type in (ADD, SUB):
            token = self.current_token
            self.consume(token.type)
            node = BinaryOperationNode(node, token, self.term())

        while self.current_token.type in (INCREMENT, DECREMENT):
            token = self.current_token
            self.consume(token.type)
            node = UnaryOperationNode(token, node)

        return node

    def term(self):
        """
        term  : factor ((MUL | INT_DIV | REAL_DIV) factor)* (INC|DEC)*
        
        returns  :  DataNode, VariableNode, UnaryOperationNode or BinaryOperationNode.
        """
        node = self.factor()

        while self.current_token.type in (MUL, REAL_DIV, INT_DIV, POW, SQRT):
            token = self.current_token
            self.consume(token.type)
            node = BinaryOperationNode(node, token, self.factor())

        while self.current_token.type in (INCREMENT, DECREMENT):
            token = self.current_token
            self.consume(token.type)
            node = UnaryOperationNode(token, node)

        return node


    def factor(self):
        """
        factor : INTEGER_CONST
               | REAL_CONST
               | variable
               | (NEGATE|SQRT|INC|DEC|PRINT) factor
               | LPAREN expr RPAREN
        
        returns  :  DataNode, VariableNode, UnaryOperationNode or BinaryOperationNode.
        """
        token = self.current_token

        if token.type == INTEGER_CONST:
            self.consume(INTEGER_CONST)
            return DataNode(token.value, token.type)
        elif token.type == REAL_CONST:
            self.consume(REAL_CONST)
            return DataNode(token.value, token.type)
        elif token.type == STRING_CONST:
            self.consume(STRING_CONST)
            return DataNode(token.value, token.type)
        elif token.type == ID:
            self.consume(ID)
            return VariableNode(token)
        elif token.type in (NEGATE, SQRT, INCREMENT, DECREMENT, PRINT):
            self.consume(token.type)
            return UnaryOperationNode(token, self.factor())
        elif token.type == LPARENS:
            self.consume(LPARENS)
            node = self.expression()  # DataNode, VariableNode, UnaryOperationNode, BinaryOperationNode
            self.consume(RPARENS)
            return node
        else:
            raise ParserError(self)


    def consume(self, token_type):
        if self.current_token.type == token_type:
            self.current_token = self.token_stream.get_next_token()
        else:
            error_message = 'Got {}, expected {}'.format(self.current_token.type, token_type)
            raise ParserError(self)



class Interpreter:

    def __init__(self, parser):
        self.parser = parser
        self.current_namespace = OrderedDict()

    def error(self, node):
        error_message = 'Could not find method visit_{}.'.format(node.__class__.__name__)
        raise SyntaxError(error_message)

    def visit(self, node):
        method_name = 'visit_' + node.__class__.__name__
        method = getattr(self, method_name, self.error)
        return method(node)

    def interpret(self):
        return self.visit(self.parser.parse())

    def visit_BuiltInFunction(self, node):
        if node.name == PRINT:
            return print(*[self.visit(argument) for argument in node.args])

        self.error(node)

    def visit_UnaryOperationNode(self, node):
        operation = node.operation

        if operation == NEGATE:
            return -self.visit(node.child)
        if operation == SQRT:
            return self.visit(node.child) ** (1/2)
        if operation == INCREMENT:
            return self.visit(node.child) + 1
        if operation == DECREMENT:
            return self.visit(node.child) - 1

        self.error(node)

    def visit_BinaryOperationNode(self, node):
        operation = node.operation

        if operation == ADD:
            return self.visit(node.left) + self.visit(node.right)
        if operation == SUB:
            return self.visit(node.left) - self.visit(node.right)
        if operation == MUL:
            return self.visit(node.left) * self.visit(node.right)
        if operation == INT_DIV:
            return self.visit(node.left) // self.visit(node.right)
        if operation == REAL_DIV:
            return self.visit(node.left) / self.visit(node.right)
        if operation == POW:
            return self.visit(node.left) ** self.visit(node.right)

        return self.error(node)

    def visit_VariableNode(self, node):
        namespace = self.current_namespace

        while namespace is not None:
            value = namespace.get(node.name)
            if value is not None:
                return value
            else:
                namespace = namespace.get('__global_namespace__')

        raise NameError('{} is undefined'.format(node.name))

    def visit_DataNode(self, node):
        data_type = node.inferred_type

        if data_type == INTEGER_CONST:
            return int(node.value)
        if data_type == REAL_CONST:
            return float(node.value)
        if data_type == STRING_CONST:
            return str(node.value)

        self.error(node)

    def visit_AssignNode(self, node):

        name = node.left.name
        data_type = node.type

        if data_type == FUNCTION:
            self.current_namespace[name] = node.right
        elif data_type is None:
            self.current_namespace[name] = self.visit(node.right)
        elif data_type == INTEGER:
            self.current_namespace[name] = int(self.visit(node.right))
        elif data_type == REAL:
            self.current_namespace[name] = float(self.visit(node.right))
        elif data_type == STRING:
            self.current_namespace[name] = str(self.visit(node.right))
        else:
            self.error(node)

    def visit_ReAssign(self, node):
        name = node.left.name
        data_type = node.type  # TODO(ted): Check data types match!

        namespace = self.current_namespace

        while namespace is not None:
            value = namespace.get(name)
            if value is not None:
                if data_type == FUNCTION:
                    namespace[name] = node.right
                    return
                elif data_type is None:
                    namespace[name] = self.visit(node.right)
                    return
                elif data_type == INTEGER:
                    namespace[name] = int(self.visit(node.right))
                    return
                elif data_type == REAL:
                    namespace[name] = float(self.visit(node.right))
                    return
                elif data_type == STRING:
                    namespace[name] = str(self.visit(node.right))
                    return
                else:
                    self.error(node)
            else:
                namespace = namespace.get('__global_namespace__')

        raise NameError('{} is undefined'.format(node.name))


    def visit_BlockNode(self, node):
        previous_namespace = self.current_namespace
        if self.current_namespace is not node.namespace:
            self.current_namespace = node.namespace
            self.current_namespace['__global_namespace__'] = previous_namespace
        for child in node.children:
            self.visit(child)
        self.current_namespace = previous_namespace

    def visit_CallNode(self, node):

        namespace = self.current_namespace

        while namespace is not None:
            block = namespace.get(node.name)
            if block is not None:
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
        elif operator == LESS_THEN:
            return left < right
        elif operator == LESS_EQUAL_THEN:
            return left <= right
        elif operator == GREATER_THEN:
            return left > right
        elif operator == GREATER_EQUAL_THEN:
            return left >= right
        elif operator == NOT_EQUAL:
            return left != right
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

    print('\nRunning: {} at {}\n'.format(source_path, start_time))

    interpreter = Interpreter(Parser(Tokenizer(InputStream(source_code))))
    interpreter.interpret()

    print('\nFinished in {}.\n'.format(datetime.datetime.now() - start_time))
    exit()

#
# with open('beginner.ted') as text:
#     tokenizer = Tokenizer(InputStream(text.read()))
#     token = tokenizer.get_next_token()
#     while token.type != EOF:
#         print(token, end=', ')
#         token = tokenizer.get_next_token()
#     print()
#
#
#
#
#
"""


                 ROOT
                 /  \ 
          Interior  Interior
            Node      Node
            /  \        \  
         Leaf  Leaf     Leaf


The leaf must be a built-in data type, such as integer, real or string, or an identifier.


We can divide our code into 2 categories: 

    1. things that evaluates to a value.
        A value is a integer, real or string.
        Binary operations returns a value.
        Expressions returns a value.
        Functions returns a value.
    
    
    2. things that do stuff.

"""

source_code = open('beginner.ted').read()
interpreter = Interpreter(Parser(Tokenizer(InputStream(source_code))))
interpreter.interpret()

# print('\nGLOBALS:\n', *['{} = {}'.format(a, b) for a, b in interpreter.globals.items()], sep='\n')


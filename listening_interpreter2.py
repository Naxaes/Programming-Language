from collections import OrderedDict
from collections import defaultdict
from difflib import SequenceMatcher

from inputstream import InputStream

# from collections import namedtuple


# ---- Token types ----
ALL_TOKENS = \
    START, INTEGER, CREATE, ASSIGN, VARIABLE, AND, IF, PRINT, UNKNOWN, EOF, STOP, ID, PLUS, MINUS, \
    TIMES, OVER, THE, TO, VALUE, EQUAL, LESS, LESS_EQUAL, GREATER, GREATER_EQUAL, NOT_EQUAL, OR, \
    NOT, IS, THAN, THAT, REAL, IT, THEN = (
        'START','INTEGER', 'CREATE', 'ASSIGN', 'VARIABLE', 'AND', 'IF', 'PRINT', 'UNKNOWN', 'EOF',
        'STOP', 'ID', 'PLUS', 'MINUS', 'TIMES', 'OVER', 'THE', 'TO', 'VALUE', 'EQUAL', 'LESS', 'LESS_EQUAL',
        'GREATER', 'GREATER_EQUAL', 'NOT_EQUAL', 'OR', 'NOT', 'IS', 'THAN', 'THAT', 'REAL', 'IT', 'THEN'
)

NUMBERS = 'ZERO' , 'ONE' , 'TWO' , 'THREE' , 'FOUR' , 'FIVE' , 'SIX' , 'SEVEN' , 'EIGHT' , 'NINE'



class Token:

    def __init__(self, type_, value):
        self.type = type_
        self.value = value

    def __repr__(self):
        return 'Token({}, {})'.format(self.type, self.value)


KEYWORDS = {
    'zero'    : Token(INTEGER, '0'),
    'one'     : Token(INTEGER, '1'),
    'two'     : Token(INTEGER, '2'),
    'three'   : Token(INTEGER, '3'),
    'four'    : Token(INTEGER, '4'),
    'five'    : Token(INTEGER, '5'),
    'six'     : Token(INTEGER, '6'),
    'seven'   : Token(INTEGER, '7'),
    'eight'   : Token(INTEGER, '8'),
    'nine'    : Token(INTEGER, '9'),

    'create'  : Token(CREATE, CREATE),
    'set'     : Token(ASSIGN, ASSIGN),      # SET or ASSIGN!? 'assign' is often heard as 'the sign' which break the program.
    'variable': Token(VARIABLE, VARIABLE),  # Change 'variable' to 'name'?

    'stop'    : Token(STOP, STOP),
    'and'     : Token(AND, AND),  # Could be used as 'stop' in certain situations?
    'or'      : Token(OR, OR),
    'if'      : Token(IF, IF),
    'equal'   : Token(EQUAL, EQUAL),
    'less'    : Token(LESS, LESS),
    'greater' : Token(GREATER, GREATER),
    'not'     : Token(NOT, NOT),
    'to'      : Token(TO, TO),
    'is'      : Token(IS, IS),
    'than'    : Token(THAN, THAN),
    'that'    : Token(THAT, THAT),
    'the'     : Token(THE, THE),
    'it'      : Token(IT, IT),
    'then'    : Token(THEN, THEN),

    'value'   : Token(VALUE, VALUE),

    'print'   : Token(PRINT, PRINT),

    'plus'    : Token(PLUS, PLUS),
    'minus'   : Token(MINUS, MINUS),
    'times'   : Token(TIMES, TIMES),
    'over'    : Token(OVER, OVER),

    '+'       : Token(PLUS, PLUS),
    '-'       : Token(MINUS, MINUS),
    '/'       : Token(OVER, OVER),
    'x'       : Token(TIMES, TIMES),  # Google speech api converts audio 'times' to 'x',
    '*'       : Token(TIMES, TIMES),  # and sometimes to '*'.

}


to_text_map = {
    '0': 'zero' ,
    '1': 'one'  ,
    '2': 'two'  ,
    '3': 'three',
    '4': 'four' ,
    '5': 'five' ,
    '6': 'six'  ,
    '7': 'seven',
    '8': 'eight',
    '9': 'nine' ,

    '+': 'plus' ,
    '-': 'minus',
    '/': 'over' ,
    'x': 'times',
    '*': 'times',
}


operator_logic = {
    (LESS, EQUAL)       : LESS_EQUAL,
    (EQUAL, LESS)       : LESS_EQUAL,
    (EQUAL, GREATER)    : GREATER_EQUAL,
    (GREATER, EQUAL)    : GREATER_EQUAL,
    (LESS, GREATER)     : NOT_EQUAL,
    (GREATER, LESS)     : NOT_EQUAL,

    (NOT, EQUAL)        : NOT_EQUAL,
    (NOT, NOT_EQUAL)    : EQUAL,
    (NOT, GREATER)      : LESS_EQUAL,
    (NOT, LESS)         : GREATER_EQUAL,
    (NOT, GREATER_EQUAL): LESS,
    (NOT, LESS_EQUAL)   : GREATER,
}


def text_to_numeric(*sequence):
    """
    one_use         :  zero    | ten    | eleven | twelve | thirteen | fifteen

    below_ten       :  one     | two    | three  | four   | five     | six      | seven  | eight  | nine

    below_twenty    : below_ten [teen]
                    | one_use

    below_hundred   : twenty  | thirty | forty  | fifty  | sixty    | seventy  | eighty | ninety
                    | [twenty | thirty | forty  | fifty  | sixty    | seventy  | eighty | ninety] below_ten
                    | below_ten

    below_thousand  : below_ten      [hundred  [[and] below_hundred]]
    below_million   : below_thousand [thousand [[and] below_hundred]]
    below_billion   : below_million  [million  [[and] below_thousand]]


    one hundred twenty two                     122
    six thousand five hundred ninty nine     6,599
    eleven thousand                         11,000
    three thousand sixty seven               3,067
    one hundred thousand one               100,001
    two thousand seventeen                   2,017
    six hundred eight                          608
    nineteen thousand fifteen               19,015
    nine hundred sixty five million two thousand two hundred seventy two  965,110


    Iterate backwards and add the value to the total. If hundred, thousand, million or billion is encountered,
    multiply it by the following number (since we iterate backwards it'll actually be the preceding number).
    """
    single = {
        'zero'     : 0,
        'one'      : 1,
        'two'      : 2,
        'three'    : 3,
        'four'     : 4,
        'five'     : 5,
        'six'      : 6,
        'seven'    : 7,
        'eight'    : 8,
        'nine'     : 9,
        'ten'      : 10,
        'eleven'   : 11,
        'twelve'   : 12,
        'thirteen' : 13,
        'fourteen' : 14,
        'fifteen'  : 15,
        'sixteen'  : 16,
        'seventeen': 17,
        'eighteen' : 18,
        'nineteen' : 19,
        'twenty'   : 20,
        'thirty'   : 30,
        'forty'    : 40,
        'fifty'    : 50,
        'sixty'    : 60,
        'seventy'  : 70,
        'eighty'   : 80,
        'ninety'   : 90,
    }
    multipliers = {
        'hundred' : 100,
        'thousand': 1000,
        'million' : 1000000,
    }

    total = 0
    index = len(sequence) - 1
    previous = 0

    while index >= 0:

        number = sequence[index]

        if number in single:
            value = single[number]

        elif number in multipliers:
            value = multipliers[number]

            index -= 1
            number = sequence[index]
            while number in multipliers:
                index -= 1
                other_value = multipliers[number]

                if value < other_value:    # Example 'one thousand hundred'
                    raise Exception('value={}, other_value={}'.format(value, other_value))

                value *= other_value
                number = sequence[index]

            value = (single[number] * value)

        else:
            raise Exception('BUHU!')

        if value < previous:
            raise Exception('value={}, previous={}'.format(value, previous))

        previous = value
        total += value
        index -= 1

    return total


class Tokenizer:

    def __init__(self, stream):
        self.stream = stream
        self.current_character = self.stream.next()
        self.tokens = []

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
        text = KEYWORDS.get(result.lower())  # Case-insensitive for keywords

        if text is None:
            return Token(UNKNOWN, result)
        else:
            return text

    def read_number(self):
        result = self.advance()
        while self.current_character.isdigit():
            result += self.advance()

        return Token(INTEGER, result)


    def skip_multiple_end_statements(self):
        if self.current_character == '\n':
            while self.stream.peek() == '\n':
                self.advance()
            if self.current_character == '/' and self.stream.peek() == '/':
                self.skip_comment()
            self.skip_multiple_end_statements()

    def get_tokens(self):
        while True:

            while self.current_character.isspace() or self.current_character == '/' and self.stream.peek() == '/':
                if self.current_character.isspace():
                    self.skip_whitespace()
                if self.current_character == '/' and self.stream.peek() == '/':
                    self.skip_comment()

            character = self.current_character

            if character.isalpha() or character == '_':
                self.tokens.append(self.read_identifier())
                continue
            elif character.isdigit():
                self.tokens.append(self.read_number())
                continue

            token = KEYWORDS.get(character)
            if isinstance(token, Token):
                self.advance()  # Skip current character.
                self.tokens.append(token)
                continue
            elif isinstance(token, defaultdict):
                self.advance()  # Skip current character.
                following_character = self.advance()
                token = token[following_character]
                self.tokens.append(token)
                continue
            else:
                # self.tokens.append(Token(EOF, EOF))
                return self.tokens


# ------------------------------ NODES ------------------------------
# AtomNode = namedtuple('AtomNode', 'name, type, str')
class Atom:
    """
    Internal representation of an atom.
    """
    TYPE = {INTEGER: int}

    def __init__(self, type_: (str, None), value: (str, callable)):
        assert type_ in Atom.TYPE or type_ is None, '{} not valid type'.format(type_)
        self.type = type_
        self.value = value

    def type_check(self, other):
        if self.type == other.type or other.type is None:
            return 1

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
        return self.value
        # return 'Atom({!r} {!r})'.format(self.type, self.value)

class ItNode: pass

class ConstantNode:
    """
    Container of a value.
    """
    TYPES = INTEGER

    def __init__(self, inferred_type, value: str):
        assert inferred_type in ConstantNode.TYPES
        assert isinstance(value, str)
        self.inferred_type = inferred_type
        self.value = value

class NameNode:
    """
    Container of a name.
    """

    def __init__(self, value: str):
        assert isinstance(value, str)
        self.value = value

class AssignNode:
    TYPES = INTEGER
    def __init__(self, name, type_, expression):
        assert isinstance(name, NameNode) or isinstance(name, ItNode) or name is None
        assert type_ in AssignNode.TYPES or type_ is None, '{} is not a valid type!'.format(type_)
        self.name = name
        self.type = type_
        self.expression = expression  # Needs to be evaluated

class CreateNode:
    TYPES = INTEGER
    def __init__(self, name, type_):
        assert isinstance(name, NameNode) or name is None, '{} is not a valid name!'.format(name)
        assert type_ in AssignNode.TYPES or type_ is None, '{} is not a valid type!'.format(type_)
        self.name = name
        self.type = type_

class CreateAssignNode:
    def __init__(self, create_node, assign_node):
        assert isinstance(create_node, CreateNode)
        assert isinstance(assign_node, AssignNode)
        self.create_node = create_node
        self.assign_node = assign_node

class BinaryOperationNode:
    OPERATIONS = PLUS, MINUS, TIMES, OVER

    def __init__(self, left_expression, operation, right_expression):
        assert operation in BinaryOperationNode.OPERATIONS
        self.left_expression = left_expression
        self.operation = operation
        self.right_expression = right_expression

class UnaryOperationNode:
    OPERATIONS = None

    def __init__(self, operation, expression):
        assert operation in UnaryOperationNode.OPERATIONS
        self.operation = operation
        self.expression = expression

class BlockNode:
    def __init__(self, statements):
        self.statements = statements
        self.namespace = OrderedDict()

class BranchNode:
    def __init__(self, left, condition, right=None):
        assert isinstance(condition, ConditionNode)
        self.left = left
        self.condition = condition
        self.right = right if right is not None else NoOperationNode()

class WhileLoopNode:
    def __init__(self, condition, block):
        assert isinstance(condition, ConditionNode)
        self.condition = condition
        self.block = block

class ConditionNode:
    OPERATIONS = GREATER, GREATER_EQUAL, EQUAL, LESS_EQUAL, LESS, NOT_EQUAL, AND, OR

    def __init__(self, left_expression, operator, right_expression):
        assert operator in ConditionNode.OPERATIONS, '{} not a valid operator!'.format(operator)
        self.left_expression = left_expression
        self.operator = operator
        self.right_expression = right_expression

class CallNode:
    def __init__(self, name, arguments=None):
        assert isinstance(name, str)
        self.name = name
        self.arguments = arguments

class PrintNode:
    def __init__(self, argument):
        self.arguments = argument

class NoOperationNode:
    pass


# ------------------------------ PARSER ------------------------------


class Parser:
    """
    ---- Grammar ----

    program    : (statement)*
    
    statement  :  create STOP
               |  assign STOP
               |  print  STOP
               |  IF condition THEN statement
               |  STOP
    
    
    create     : CREATE variable
               | CREATE variable AND ASSIGN expression
    assign     : ASSIGN [TO] variable expression
    print      : PRINT expression

    
    condition  : expression OPERATOR expression
    operator   : LESS | GREATER | EQUAL
               | [IS] (LESS | GREATER | EQUAL) [THAN | TO]
               | [IS] (LESS | GREATER | EQUAL) OR (LESS | GREATER | EQUAL) [THAN | TO]
               | [IS] NOT operator

               
    expression : term   ((PLUS  | MINUS) term)*     
    term       : factor ((TIMES | OVER)  factor)*
    factor     : constant
               | variable
               | it
    
    type       : VALUE | INTEGER | REAL
    constant   : [THE] [type] NUMBER
    variable   : [THE] VARIABLE  ID
    it         : constant | variable
    
    """

    def __init__(self, tokens):
        self.tokens = tokens
        self.token_index = 0
        self.previous_token = Token(START, START)
        self.current_token = self.tokens[self.token_index]

    def parse(self):
        """         
        program   :  (statement)* EOF
        """

        statements = []
        while self.current_token.type != EOF:
            statements.append(self.statement())

        return BlockNode(statements)  # ProgramNode?

    def statement(self):
        """
        statement  :  create (STOP | AND)
                   |  assign (STOP | AND)
                   |  print  (STOP | AND)
                   |  IF condition THEN statement (STOP | AND)
                   |  WHILE condition THEN statement (STOP | AND)
                   |  (STOP | AND)
        """
        self.check(CREATE, ASSIGN, PRINT, IF, STOP, AND)

        node = NoOperationNode()

        if self.current_token.type == CREATE:
            node = self.create()
        elif self.current_token.type == ASSIGN:
            node = self.assign()
        elif self.current_token.type == PRINT:
            node = self.print()
        elif self.current_token.type == IF:
            self.check(IF)
            self.consume(IF)

            condition = self.condition()

            self.check(THEN)
            self.consume(THEN)

            return BranchNode(self.statement(), condition)

        self.check(STOP, AND)
        self.consume(STOP, AND)

        return node

    def create(self):
        """
        create  :  CREATE variable
        """
        self.check(CREATE)
        self.consume(CREATE)

        return CreateNode(self.variable(), INTEGER)


    def assign(self):
        """
        assign  : ASSIGN variable [TO] expression
        """
        self.check(ASSIGN)
        self.consume(ASSIGN)

        name = self.variable()

        if self.check_optional(TO, otherwise=(THE, VARIABLE, IT)):
            self.consume(TO)

        node = AssignNode(name, INTEGER, self.expression())

        return node


    def print(self):
        """
        print  :  PRINT expression
        """
        self.check(PRINT)
        self.consume(PRINT)

        node = PrintNode(self.expression())

        return node


    def condition(self):
        """
        condition  :  expression OPERATOR expression
        """
        left  = self.expression()
        operator = self.operator()
        right = self.expression()

        node = ConditionNode(left, operator, right)

        return node


    def operator(self):
        """
        operator : LESS | GREATER | EQUAL
                 | [IS] (LESS | GREATER | EQUAL) [THAN | TO]
                 | [IS] (LESS | GREATER | EQUAL) OR (LESS | GREATER | EQUAL) [THAN | TO]
                 | [IS] NOT operator
        """
        if self.check_optional(IS, otherwise=(EQUAL, LESS, GREATER, NOT)):
            self.consume(IS)

        if self.check_optional(NOT, otherwise=(EQUAL, LESS, GREATER)):
            self.consume(NOT)
            return operator_logic[NOT, self.operator()]

        self.check(EQUAL, LESS, GREATER)
        operator = self.current_token.type
        self.consume(EQUAL, LESS, GREATER)

        if self.check_optional(TO, THAN, OR, otherwise=(INTEGER, VARIABLE, IT)):
            if self.current_token.type != OR:
                self.consume(TO, THAN)
                return operator
            else:
                self.consume(OR)

                self.check(EQUAL, LESS, GREATER)
                operator2 = self.current_token.type
                self.consume(EQUAL, LESS, GREATER)

                if self.check_optional(TO, THAN, otherwise=(INTEGER, VARIABLE, IT)):
                    self.consume(TO, THAN)

                return operator_logic[operator, operator2]

        return operator


    def expression(self):
        """
        expression  :  term ((PLUS | MINUS) term)* 
        """
        node = self.term()

        while self.check_optional(PLUS, MINUS, otherwise=(IS, IF, STOP, AND, THEN)):

            if self.current_token.type == PLUS:
                self.consume(PLUS)
                node = BinaryOperationNode(node, PLUS, self.term())

            elif self.current_token.type == MINUS:
                self.consume(MINUS)
                node = BinaryOperationNode(node, MINUS, self.term())

        return node


    def term(self):
        """
        term  : factor ((TIMES | OVER) factor)*
        """
        node = self.factor()

        while self.check_optional(TIMES, OVER, otherwise=(IS, IF, STOP, AND, PLUS, MINUS, THEN)):

            if self.current_token.type == TIMES:
                self.consume(TIMES)
                node = BinaryOperationNode(node, TIMES, self.factor())

            elif self.current_token.type == OVER:
                self.consume(OVER)
                node = BinaryOperationNode(node, OVER, self.factor())

        return node


    def factor(self):
        """
        factor : [[THE] VALUE]   NUMBER
               | [THE] VARIABLE  ID
        """
        if self.check_optional(THE, VALUE, otherwise=(INTEGER, VARIABLE, IT)):
            if self.current_token.type == THE:
                self.consume(THE)

                if self.check_optional(VALUE, otherwise=(INTEGER, VARIABLE, IT)):
                    self.consume(VALUE)
            else:
                self.consume(VALUE)


        self.check(INTEGER, VARIABLE, IT)

        if self.current_token.type == INTEGER:
            value = self.current_token.value
            self.consume(INTEGER)

            return ConstantNode(INTEGER, value)

        elif self.current_token.type == VARIABLE:
            self.consume(VARIABLE)

            self.check(ID)
            name = self.current_token.value
            self.consume(ID)

            return NameNode(name)

        elif self.current_token.type == IT:
            self.consume(IT)
            return ItNode()


    def constant(self):
        """
        constant: [THE] (VALUE | INTEGER | REAL) NUMBER 
        """
        if self.check_optional(THE, otherwise=(VALUE, INTEGER, REAL)):
            self.consume(THE)

        self.check(VALUE, INTEGER, REAL)
        value = self.current_token.value
        self.consume(VALUE, INTEGER, REAL)

        return ConstantNode(INTEGER, value)  # Only integer types for now.


    def variable(self):
        """
        variable : [THE] VARIABLE ID 
                 | IT
        """

        if self.check_optional(THE, IT, otherwise=(VARIABLE, )):
            if self.current_token.type == IT:
                self.consume(IT)
                return ItNode()
            else:
                self.consume(THE)

        self.check(VARIABLE)
        self.consume(VARIABLE)

        self.check(ID)
        value = self.current_token.value
        self.consume(ID)

        return NameNode(value)  # Only integer types for now.


    def check_optional(self, *token_types, otherwise=()):
        """
        Check whether the current token is any of 'token_types'.
        
        The optional argument 'otherwise' is the token that will be followed if the current token is *not* any of
        the 'token_types'.
        
        Example:
            # token sequence: PRINT [THE] VARIABLE ID STOP
            
            self.check(PRINT)
            self.consume(PRINT)
            
            if self.check_optional(THE, otherwise=(VARIABLE)):
                self.consume(THE)
            
            self.check(VARIABLE)
            self.consume(VARIABLE)
        
        """
        if otherwise == ALL_TOKENS:
            possible_tokens = ALL_TOKENS
        else:
            assert isinstance(otherwise, tuple)
            possible_tokens = token_types + otherwise

        if self.current_token.type == UNKNOWN or self.current_token.type not in possible_tokens:
            self.evaluate_unknown(*possible_tokens)

        if self.current_token.type in token_types:
            return True
        elif self.current_token.type in otherwise:
            return False
        else:
            raise Exception('Illogical token sequence: {} and {}'.format(self.previous_token, self.current_token))


    def check(self, *token_types):
        if self.current_token.type == UNKNOWN or self.current_token.type not in token_types:
            self.evaluate_unknown(*token_types)

        if self.current_token.type not in token_types:
            raise Exception('Illogical token sequence: {} and {}'.format(self.previous_token, self.current_token))


    def evaluate_unknown(self, *expected_tokens):
        if self.previous_token.type == VARIABLE:
            self.current_token.type = ID
        else:

            value = self.current_token.value
            if value in to_text_map:
                value = to_text_map[value.lower()].upper()
            else:
                value = value.upper()

            if INTEGER in expected_tokens:
                expected_tokens += NUMBERS

            best_match, best_match_value = None, 0.34

            for token_type in expected_tokens:
                match_value = SequenceMatcher(None, token_type, value).ratio()

                if match_value > best_match_value:
                    best_match = token_type
                    best_match_value = match_value

            if best_match is not None:
                if best_match in NUMBERS:
                    token = KEYWORDS[best_match.lower()]
                    print('Changed Token(UNKNOWN, {value}) to {token}.'.format(value=self.current_token.value, token=token))
                    self.current_token = token
                else:
                    print('Changed Token(UNKNOWN, {value}) to Token({type}, {type}).'.format(value=self.current_token.value, type=best_match))
                    self.current_token = Token(best_match, best_match)
            else:
                raise Exception('Could not identify {}.'.format(self.current_token))


    def consume(self, *token_types):
        if self.current_token.type in token_types:
            self.token_index += 1
            if self.token_index >= len(self.tokens):
                self.tokens.append(Token(EOF, EOF))
            self.previous_token = self.current_token
            self.current_token = self.tokens[self.token_index]
        else:
            raise Exception('Expected {}, got {}'.format(token_types, self.current_token.type))

# ------------------------------ INTERPRETER ------------------------------




class Interpreter:

    def __init__(self, abstract_syntax_tree):
        self.abstract_syntax_tree = abstract_syntax_tree
        self.current_namespace = None
        self.last_atom = None

    def interpret(self):
        self.visit(self.abstract_syntax_tree)

    def visit(self, node):
        name = 'visit_' + node.__class__.__name__
        method = getattr(self, name)
        return method(node)

    def visit_ConstantNode(self, node):
        return Atom(INTEGER, node.value)

    def visit_ItNode(self, node):
        return self.last_atom

    def visit_NameNode(self, node):
        if node.value in self.current_namespace:
            atom = self.current_namespace[node.value]
            if atom is None:
                raise Exception('{} is not initialized!'.format(node.value))
            else:
                self.last_atom = atom
                return atom
        else:
            best_match, best_match_value = None, 0.35

            for name in self.current_namespace:
                match_value = SequenceMatcher(None, node.value, name).ratio()

                if match_value > best_match_value:
                    best_match = name
                    best_match_value = match_value

            if best_match is None:
                raise NameError('{} is not defined!'.format(node.name.value))
            else:
                print('Changed variable name {} to {}'.format(node.value, best_match))
                node.value = best_match

            atom = self.current_namespace[node.value]
            self.last_atom = atom
            return atom


    def visit_BinaryOperationNode(self, node):
        operation = node.operation

        if operation == PLUS:
            return self.visit(node.left_expression) + self.visit(node.right_expression)
        if operation == MINUS:
            return self.visit(node.left_expression) - self.visit(node.right_expression)
        if operation == TIMES:
            return self.visit(node.left_expression) * self.visit(node.right_expression)
        if operation == OVER:
            return self.visit(node.left_expression) // self.visit(node.right_expression)

    def visit_CreateAssignNode(self, node):
        self.visit(node.create_node)
        self.visit(node.assign_node)

    def visit_AssignNode(self, node):

        if isinstance(node.name, ItNode):
            self.last_atom.value = self.visit(node.expression).value
            return

        if node.name.value not in self.current_namespace:
            best_match, best_match_value = None, 0.35

            for name in self.current_namespace:
                match_value = SequenceMatcher(None, node.name.value, name).ratio()

                if match_value > best_match_value:
                    best_match = name
                    best_match_value = match_value

            if best_match is None:
                raise NameError('{} is not defined!'.format(node.name.value))
            else:
                print('Changed variable name {} to {}'.format(node.name.value, best_match))
                node.name.value = best_match

        self.current_namespace[node.name.value] = self.visit(node.expression)
        self.last_atom = self.current_namespace[node.name.value]

    def visit_CreateNode(self, node):
        if node.name.value in self.current_namespace:
            raise NameError('{} is already defined.'.format(node.name.value))
        else:
            atom = Atom(node.type, None)
            self.last_atom = atom
            self.current_namespace[node.name.value] = atom

    def visit_PrintNode(self, node):
        print(self.visit(node.arguments))

    def visit_BlockNode(self, node):
        previous_namespace = self.current_namespace

        new_namespace = node.namespace                      # Should be empty
        new_namespace['__globals__'] = previous_namespace

        self.current_namespace = new_namespace

        for statement in node.statements:
            self.visit(statement)

        node.namespace.clear()
        self.current_namespace = previous_namespace

    def visit_ConditionNode(self, node):
        left = self.visit(node.left_expression)
        right = self.visit(node.right_expression)
        operator = node.operator

        if   operator == EQUAL:
            return left == right
        elif operator == LESS:
            return left < right
        elif operator == LESS_EQUAL:
            return left <= right
        elif operator == GREATER:
            return left > right
        elif operator == GREATER_EQUAL:
            return left >= right
        elif operator == NOT_EQUAL:
            return left != right
        elif operator == OR:
            return left or right
        elif operator == AND:
            return left and right

    def visit_BranchNode(self, node):
        if self.visit(node.condition):
            self.visit(node.left)
        else:
            self.visit(node.right)

    def visit_NoOperationNode(self, node):
        pass





# source_code = """
# create variable count stop
# assign to variable count five stop
# assign to variable count variable count plus five stop
# print variable count stop
# print five times six minus four over three stop
# """

source_code = """
create variable count and set it to four and

create variable test stop
set it to nine stop

if it is greater than 3 then print it stop

print variable count plus variable test stop

if variable count plus variable test is greater or equal to 13 then print 5 times 6 stop
"""


tokenizer = Tokenizer(InputStream(source_code))
token_list = tokenizer.get_tokens()

print(*token_list, sep='\n', end='\n')

parser = Parser(token_list)
abstract_syntax_tree_ = parser.parse()
interpreter = Interpreter(abstract_syntax_tree_)

print('\nProgram output:')
interpreter.interpret()

exit()











import speech_recognition as sr

recognizer = sr.Recognizer()
with sr.Microphone() as source:
    token_list = []
    code = None
    full_code = []
    print("Speak your code. Use a normal speaking voice and try not to exaggerate pronunciation.")
    while True:
        print('Input:')
        audio = recognizer.listen(source)
        try:
            code = recognizer.recognize_google(audio)
            if code.lower() == 'terminate':
                token_list.append([Token(EOF, EOF)])
                print('\tTerminating...', end='\n\n')
                break
            elif code.lower() == 'undo':
                print('\tUndoing:', full_code.pop())
                token_list.pop()
                continue

            code += ' stop'
            full_code.append(code + '\n')

            print('\t', code)

            tokenizer = Tokenizer(InputStream(code))
            temp = tokenizer.get_tokens()

            print('Tokens:\n\t', end='')
            print(*temp, '\n', sep=', ')
            token_list.append(temp)

        except sr.UnknownValueError:
            pass

    print('\nFULL CODE:')
    print(''.join(full_code))

    token_list = [token for sub_list in token_list for token in sub_list]
    abs = Parser(token_list).parse()
    interpreter = Interpreter(abstract_syntax_tree=abs)

    print('\nPROGRAM OUTPUT:\n')
    interpreter.interpret()
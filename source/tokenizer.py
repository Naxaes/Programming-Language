# identifiers  - names the programmer chooses;
# keywords     - names already in the programming language;
# separators   - punctuation characters and paired-delimiters;
# operators    - symbols that operate on arguments and produce results;
# literals     - numeric, logical, textual, reference literals;
# comments     - line, block.


IDENTIFIER = 'IDENTIFIER'
KEYWORD    = 'KEYWORD'
SEPARATOR  = 'SEPARATOR'
OPERATOR   = 'OPERATOR'
LITERAL    = 'LITERAL'
COMMENT    = 'COMMENT'



class Tokenizer:

    def __init__(self, stream):
        self.stream = stream


    def next_token(self):
        character = self.stream.next_character()

        while character.isspace():
            character = self.stream.next_character()

        if character.isdigit():
            number = character
            character = self.stream.next_character()
            while character.isdigit():
                number += character
                character = self.stream.next_character()


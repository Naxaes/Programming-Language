class InputStream:

    @classmethod
    def from_path(cls, path):
        return cls(open(path).read())

    def __init__(self, source):
        self.source = source

        self.index = 0
        self.last_index = len(source)

        self.row = 1
        self.column = 1

    def peek_next_character(self):
        if self.index >= self.last_index:
            return ''
        return self.source[self.index]

    def next_character(self):
        character = self.peek_next_character()

        if character == '\n':
            self.row += 1
            self.column = 0
        else:
            self.column += 1

        self.index += 1
        return character
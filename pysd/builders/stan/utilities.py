class IndentedString:
    def __init__(self, indent_level=0):
        self.indent_level = indent_level
        self.string = ""

    def __iadd__(self, other: str):
        prefix = " " * 4 * self.indent_level
        if other != "\n":
            self.string += prefix
        self.string += other
        return self

    def add_raw(self, string, ignore_indent=False):
        if ignore_indent:
            self.string += string
        else:
            self.__iadd__(string)

    def __str__(self):
        return self.string


def name_to_identifier(name: str):
    return name.lower().replace(" ", "_")
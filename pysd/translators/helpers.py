def clean_identifier(string):
    """
        at the moment, we may have trailing spaces on an identifier that need to be dealt with
        in the future, it would be better to improve the parser so that it handles that whitespace properly
    """
    string = string.lower()
    string = string.strip()
    string = string.replace(' ', '_')
    return string



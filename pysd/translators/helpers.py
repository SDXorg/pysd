import collections

def clean_identifier(string):
    """
        at the moment, we may have trailing spaces on an identifier that need to be dealt with
        in the future, it would be better to improve the parser so that it handles that whitespace properly
    """
    string = string.strip()
    string = string.replace(' ', '_')
    return string



def update(d, u):
    """
        Facilitates nested dictionary updating.
        This is stolen from:
        http://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
        """
    for k, v in u.iteritems():
        if isinstance(v, collections.Mapping):
            r = update(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d



def getChildren(node, name):
    """
        Returns the children of node 'node' that are named 'name'
        in a list.
        """
    return [child for child in node.children if child.expr_name == name]


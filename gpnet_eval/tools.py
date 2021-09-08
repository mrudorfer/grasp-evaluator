# tools.py

def flatten_nested_list(nested_list):
    """
    flattens a list of lists. note that all elements of the primary list need to be lists as well.

    :param nested_list: a nested list.
    :return: a flattened list.
    """
    return [item for sublist in nested_list for item in sublist]


def log_line(item_list, delimiter='\t'):
    """
    prepares items as a line for csv files, delimited by delimiter.
    :param item_list: list of objects that must have a str(object) function
    :param delimiter: delimiter between string representations
    :return: a concatenated string
    """
    line = delimiter.join([str(obj) for obj in item_list])
    line += '\n'
    return line

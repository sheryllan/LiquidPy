import pandas as pd
from itertools import groupby
from tabulate import tabulate
from commonlib.commonfuncs import *


def to_tbstring(data, dtypes, cols=None, tablefmt='simple'):
    if nontypes_iterable(data, dtypes):
        for d in data:
            yield to_tbstring(d, dtypes, cols, tablefmt)
    elif isinstance(data, dtypes):
        try:
            table = select_mapping(data, cols)
        except AttributeError as e:
            raise ValueError('Invalid type of row: must be either pandas Series or dict').with_traceback(
                e.__traceback__)

        if isinstance(table, pd.Series):
            return tabulate([table.tolist()], tablefmt)
        elif isinstance(table, dict):
            return tabulate([table.values()], tablefmt)
        else:
            return tabulate([list(table)], tablefmt)
    else:
        raise TypeError('Inconsistent function input: data parameter must contain data of dtype')


def tabulate_rows(data, outcols=None, grouping=None):
    first, data = peek_iter(data)
    if first is None:
        return ''

    if grouping is not None:
        for key, subitems in groupby(data, key=grouping):
            yield tabulate([key], 'plain')
            yield from to_tbstring(subitems, type(first), outcols)
    else:
        yield from to_tbstring(data, type(first), outcols)

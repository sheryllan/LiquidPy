import pandas as pd
from itertools import groupby
from tabulate import tabulate

from commonlib import commonfuncs


def tabulate_rows(data, outcols=None, grouping=None):

    def to_tbstring(rows, tablefmt='simple'):
        table = list(map(lambda x: x.tolist() if isinstance(x, pd.Series) else list(x.values()), rows))
        return tabulate(table, tablefmt)

    if grouping is not None:
        for group, subgroups in groupby(data, key=grouping):
            yield to_tbstring(group, 'plain')
            yield to_tbstring(subgroups.loc[outcols])
    else:
        for row in data:
            yield to_tbstring(row.loc[outcols])

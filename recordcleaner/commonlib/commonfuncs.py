import types
from collections import Iterable
from itertools import tee
import warnings
from sortedcontainers import SortedDict
from argparse import ArgumentTypeError
import pandas as pd
import traceback
from datetime import date
from dateutil.relativedelta import relativedelta


def last_n_year(n=1):
    return (date.today() + relativedelta(years=-n)).year


def last_n_month(n=1):
    return (date.today() + relativedelta(months=-n)).month


def fmt_date(year, month=None, day=1, fmt='%Y%m'):
    if month is None:
        return str(year)
    return date(int(year), int(month), int(day)).strftime(fmt)


def nontypes_iterable(arg, excl_types=(str,)):
    return isinstance(arg, Iterable) and not isinstance(arg, excl_types)


def flatten_iter(items, level=None, excl_types=(str,)):
    if nontypes_iterable(items, excl_types):
        level = None if level is None else level + 1
        for sublist in items:
            yield from flatten_iter(sublist, level, excl_types)
    else:
        level_item = items if level is None else (level - 1, items)
        yield level_item


def map_recursive(f, items):
    if not nontypes_iterable(items):
        return f(items)

    subitems = (map_recursive(f, item) for item in items)
    if isinstance(items, types.GeneratorType):
        return subitems
    return type(items)(subitems)


def df_groupby(df, cols):
    cols = to_iter(cols)
    if any(c not in df for c in cols):
        return None

    def groupby_rcs(gdf, rcols):
        if not rcols:
            return gdf
        else:
            group_dict = dict()
            for key, new_df in gdf.groupby(rcols[0]):
                group_dict[key] = groupby_rcs(new_df, rcols[1:])
            return group_dict

    return groupby_rcs(df, cols)


def group_every_n(items, n, gtype=list):
    return (gtype(items[i: i + n]) for i in range(0, len(items), n))


def to_iter(x, excl_types=(str,), ittype=list):
    return [x] if not nontypes_iterable(x, excl_types) else ittype(x)


def find_first_n(arry, condition, n=1):
    result = list()
    for a in arry:
        if n <= 0:
            break
        if condition(a):
            result.append(a)
            n -= 1
    return result if len(result) != 1 else result[0]


def swap(a, b):
    tmp = a
    a = b
    b = tmp
    return a, b


def mapping_updated(data, values, insert=True, condition=lambda k, v: True):
    val_dict = dict(values)
    for k in val_dict:
        if (insert or k in data) and condition(k, val_dict[k]):
            data[k] = val_dict[k]
    return data


def select_mapping(data, keys, keepnone=True, rtype=None):
    if keys is None:
        return data

    rtype = type(data) if rtype is None else rtype
    result = rtype()
    for k in keys:
        if keepnone:
            result[k] = data.get(k, None)
        elif k in data:
            result[k] = data[k]
    return result


def rename_mapping(data, mapping):
    if mapping is None:
        return data
    if isinstance(data, pd.DataFrame):
        return data.rename(columns=mapping)
    elif isinstance(data, pd.Series):
        return data.rename(index=mapping)
    else:
        return type(data)({mapping.get(k, k): data[k] for k in data})


def rreplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)


def last_indexof(items, matchfunc):
    for i in range(len(items) - 1, -1, -1):
        if matchfunc(items[i]):
            return i
    return None


def get_indexes(arry, condition):
    return [i for i, a in enumerate(arry) if condition(a)]


def index_culmulative(items, func):
    idx, _ = func(enumerate(items), key=lambda x: x[1])
    return idx


def verify_non_decreasing(array):
    for i in range(1, len(array)):
        if array[i - 1] > array[i]:
            return False
    return True


def peek_iter(items, n=1):
    if not nontypes_iterable(items):
        raise TypeError('The input must be iterable')
    if n > 20:
        warnings.warn('Costly using this method if n > 20: suggest to use list instead')

    _, iteritems = tee(items)
    return next(iteritems, None) if n == 1 else list(filter(None, (next(iteritems, None) for _ in range(0, n))))


def slicing_gen(items, stopfunc):
    while items:
        if stopfunc(items[0]):
            break
        else:
            yield items[0]
            del items[0]


def unique_gen(items):
    seen = set()
    for item in items:
        if item in seen:
            continue
        yield item
        seen.add(item)


def format_ex_str(ex):
    return ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__)) \
                    if isinstance(ex, Exception) else ''


def hierarch_groupby(orig_dict, key_funcs, sort=False):

    def groupby_rcsv(entry, key_funcs, output_dict):
        if not key_funcs:
            return entry
        new_key = key_funcs[0](entry)
        new_outdict = output_dict[new_key] if new_key in output_dict else dict()
        output_dict.update({new_key: groupby_rcsv(entry, key_funcs[1:], new_outdict)})
        return output_dict

    output_dict = SortedDict() if sort else dict()
    for k, v in orig_dict.items():
        groupby_rcsv(v, key_funcs, output_dict)
    return output_dict


def argconv(keeporig=True, **convs):
    def parse_argument(arg):
        if arg in convs:
            return convs[arg]
        elif not keeporig:
            msg = "invalid choice: {!r} (choose from {})"
            choices = ", ".join(sorted(repr(choice) for choice in convs.keys()))
            raise ArgumentTypeError(msg.format(arg, choices))
        else:
            return arg
    return parse_argument


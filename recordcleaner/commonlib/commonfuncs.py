from itertools import chain
from collections import Iterable
from sortedcontainers import SortedDict
import types


def nontypes_iterable(arg, excl_types=(str,)):
    return isinstance(arg, Iterable) and not isinstance(arg, excl_types)


def flatten_iter(items, level=None, types=(str,)):
    if nontypes_iterable(items, types):
        level = None if level is None else level + 1
        for sublist in items:
            yield from flatten_iter(sublist, level, types)
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


def group_every_n(items, n, gtype=list):
    return (gtype(items[i: i + n]) for i in range(0, len(items), n))


def to_list(x, excl_types=(str,)):
    return [x] if not nontypes_iterable(x, excl_types) else list(x)


def find_first_n(arry, condition, n=1):
    result = list()
    for a in arry:
        if n == 0:
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


def mapping_updated(dct, values):
    dct.update(values)
    return dct


def select_mapping(dct, keys):
    if keys is None:
        return dct
    return {k: dct.get(k, None) for k in keys}


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


def peek_iter(items):
    if not nontypes_iterable(items):
        return items
    gen = iter(items)
    peek = next(gen, None)
    items = chain([peek], gen) if peek is not None else iter('')
    return peek, items


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


# def validate_dfcols(dfs_dict, cols_dict):
#     for key, df in dfs_dict.items():
#         if not all(c in df.columns for c in cols_dict[key]):
#             raise ValueError('Input {} dataframe is missing necessary columns'.format(key))

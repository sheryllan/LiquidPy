import types
from collections import Iterable
from itertools import tee
import warnings
from sortedcontainers import SortedDict


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


def to_iter(x, excl_types=(str,), ittype=list):
    return [x] if not nontypes_iterable(x, excl_types) else ittype(x)


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


def mapping_updated(data, values, insert=True):
    val_dict = dict(values)
    for k in val_dict:
        if insert or k in data:
            data[k] = val_dict[k]
    return data


def select_mapping(data, keys, keepnone=True):
    if keys is None:
        return data

    result = type(data)()
    for k in keys:
        if keepnone:
            result[k] = data.get(k, None)
        elif k in data:
            result[k] = data[k]
    return result

def rename_mapping(data, mapping):
    if mapping is None:
        return data
    return type(data)({mapping.get(k, k): data[k] for k in data.keys()})


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

    # if hasattr(items, '__getitem__'):
    #     return (None, items) if not items else (list(items)[0], items)
    # if hasattr(items, '__next__'):
    #     peek = next(items, None)
    #     gen = chain([peek], items) if peek is not None else iter('')
    #     return peek, gen


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

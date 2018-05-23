from collections import Iterable
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


def to_list(x):
    return [x] if not nontypes_iterable(x) else list(x)


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


def to_dict(items, tkey, tval):
    return {tkey(x): tval(x) for x in items}


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

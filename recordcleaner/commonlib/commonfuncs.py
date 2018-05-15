from collections import Iterable


def nontypes_iterable(arg, excl_types=(str,)):
    return isinstance(arg, Iterable) and not isinstance(arg, excl_types)


def flatten_iter(items, incl_level=False, types=(str,)):
    def flattern_iter_rcrs(items, flat_list, level):
        if not items:
            return flat_list

        if nontypes_iterable(items, types):
            level = None if level is None else level + 1
            for sublist in items:
                flat_list = flattern_iter_rcrs(sublist, flat_list, level)
        else:
            level_item = items if level is None else (level, items)
            flat_list.append(level_item)
        return flat_list

    level = -1 if incl_level else None
    return flattern_iter_rcrs(items, list(), level)


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


def last_indexof(items, target):
    j = None
    for i in range(len(items) - 1, -1, -1):
        if items[i] == target:
            j = i
            break
    return j


def get_indexes(arry, condition):
    return [i for i, a in enumerate(arry) if condition(a)]



from collections import namedtuple
from itertools import groupby
import numpy as np

from commonlib.datastruct import namedtuple_with_defaults
from commonlib.iohelper import XlsxWriter
from configfilesparser import *
from productmatcher import *

RECORDED = 'Recorded'
GROUP = 'Group'


class ProductKey(namedtuple_with_defaults(namedtuple('ProductKey', [CO_PRODCODE, CO_TYPE]))):
    FD_PRODCODE = CO_PRODCODE
    FD_TYPE = CO_TYPE

    def __eq__(self, other):
        this_tuple = tuple(map(str, self))
        other_tuple = tuple(map(str, other))
        eq_code = this_tuple[0].lower() == this_tuple[0].lower()
        eq_type = MatchHelper.match_in_string(this_tuple[1], other_tuple[1], one=False, stemming=True)
        return eq_code and eq_type

    def __ne__(self, other):
        this_tuple = tuple(map(str, self))
        other_tuple = tuple(map(str, other))
        eq_code = this_tuple[0].lower() == this_tuple[0].lower()
        eq_type = MatchHelper.match_in_string(this_tuple[1], other_tuple[1], one=False, stemming=True)
        return not (eq_code or eq_type)

    def hashing_type(self):
        this_tuple = tuple(map(lambda x: str(x) if x else x, self))
        t_code = this_tuple[0].lower() if this_tuple[0] else this_tuple[0]
        t_type = MatchHelper.to_singular_noun(this_tuple[1].lower())
        return tuple([t_code, t_type])

    def __hash__(self):
        return hash(self.hashing_type())


def sum_unique(data, aggr_col):
    if nontypes_iterable(data):
        return sum(set(map(lambda x: x[aggr_col], data)))


def print_duplicate(group, duplicate):
    print()
    print('In group: {}'.format(group))
    print('Duplicate: {}'.format(str(duplicate)))


def dfgroupby_aggr(df, group_key, aggr_col, aggr_func, inplace=True):
    if not inplace:
        df = df.copy()
    if group_key in df:
        for group, subdf in df.groupby(group_key):
            aggr_val = aggr_func(subdf, aggr_col)
            df.loc[subdf.index, aggr_col] = aggr_val
    return df


def get_config_dict(exch, keycols=(ProductKey.FD_PRODCODE, ProductKey.FD_TYPE), keygen=ProductKey, valcols=None):
    src = get_src_file(exch)
    config_data = parse_config(exch, src, attrs=valcols)
    return {keygen(**select_mapping(d, keycols)): d for d in config_data}


def df_lower_limit(df, col, lower_limit):
    return df[df[col] >= lower_limit]


def mark_recorded(data, col_pcode, col_type, config_dict, inplace=True):
    df = data.copy() if not inplace else data
    df[RECORDED] = df[col_pcode, col_type].apply(lambda x: ProductKey(*x) in config_dict, axis=1)
    return df


# def filter_mark_rows(data_rows, filterfunc, keyfunc, config_dict):
#     for row in data_rows:
#         row = pd.Series(row)
#         if filterfunc and not filterfunc(row):
#             continue
#         recorded = keyfunc(row) in config_dict
#         yield pd.concat([row, pd.Series({RECORDED: recorded})])


def count_unique(data, col=RECORDED):
    arr = np.array(data[col] if isinstance(data, pd.DataFrame) else [d[col] for d in data])
    uniques, counts = np.unique(arr, return_counts=True)
    return {key: val for key, val in zip(uniques, counts)}


# region unused methods

def groupby_aggr(data, groupfunc, aggr_col, aggr_func, groupkey=None):
    for key, group in groupby(data, key=groupfunc):
        rows = list(group)
        aggr_val = aggr_func(rows, aggr_col)
        for row in rows:
            row = pd.Series(row)
            if groupkey is not None:
                mapping_updated(row, {groupkey: key})
            yield mapping_updated(row, {aggr_col: aggr_val})


# parameter file2sheet is a tuple
def xl_consolidate(file2sheet, dest):
    wrt = XlsxWriter.create_xlwriter(dest, False)
    for fl, sht in file2sheet:
        xl = pd.ExcelFile(fl)
        dt = xl.parse(sht)
        XlsxWriter.to_xlsheet(dt, wrt, sht)
    wrt.save()


def divide_dict_by(orig_dict, key_cols, left_sort=False, right_sort=False):
    left_dict = SortedDict() if left_sort else dict()
    right_dict = SortedDict() if right_sort else dict()
    for k, v in orig_dict.items():
        right_key = tuple(v[col] for col in key_cols)
        left_dict.update({k: right_key})
        right_dict.update(({right_key: v}))
    return left_dict, right_dict


# endregion


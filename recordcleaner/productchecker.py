from collections import namedtuple
from itertools import groupby
import numpy as np

from commonlib.datastruct import namedtuple_with_defaults
from commonlib.iohelper import XlsxWriter
from configfilesparser import *
from productmatcher import *
from datascraper import *

RECORDED = 'Recorded'
GROUP = 'Group'
PRODUCTKEY = 'ProductKey'


class ProductKey(namedtuple_with_defaults(namedtuple('ProductKey', [CO_PRODCODE, CO_TYPE]))):
    FD_PRODCODE = CO_PRODCODE
    FD_TYPE = CO_TYPE

    def __init__(self, *args, **kwargs):
        if any(p is None for p in self):
            raise ValueError('Error field value {}: fields of ProductKey must not be None'.format(self))

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


def sum_mapping(data, aggr_col):
    if isinstance(data, pd.DataFrame):
        return data[aggr_col].sum()
    elif nontypes_iterable(data, excl_types=[pd.DataFrame, pd.Series, dict]):
        return sum(map(lambda x: x[aggr_col], data))


def dfgroupby_aggr(df, group_key, aggr_col, aggr_func, inplace=True):
    if not inplace:
        df = df.copy()

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


def mark_recorded(data, config_dict, inplace=True):
    df = data.copy() if not inplace else data
    df[RECORDED] = pd.Series(data.index.get_level_values(PRODUCTKEY).map(lambda x: ProductKey(*x) in config_dict), index=data.index)
    return df


def get_prod_keys(data, keyfunc=lambda x: x):
    return data.apply(lambda x: keyfunc(x), axis=1)


def count_unique(data, col=RECORDED):
    if isinstance(data, pd.DataFrame):
        arr = np.array(data[col]) if col is not None else data.values
    else:
        arr = np.array([d[col] for d in data]) if col is not None else np.array(list(data))
    uniques, counts = np.unique(arr, return_counts=True)
    return {key: val for key, val in zip(uniques, counts)}


def set_check_index(data, prod_keys, group_keys=None, duplicates=False, drop=True):
    if group_keys is None:
        names = [PRODUCTKEY]
        indices = [prod_keys]
    else:
        names = [GROUP, PRODUCTKEY]
        indices = [group_keys, prod_keys]

    mindex = pd.MultiIndex.from_arrays(indices, names=names)
    data_set = data.set_index(mindex, drop=drop)
    if not duplicates:
        data_set = data_set[~data_set.index.duplicated(keep='first')]

    return data_set


def validate_precheck(data):
    if not isinstance(data, dict):
        raise TypeError('The checked results must be a dict with keys of exchange')
    if any(not isinstance(data[e], pd.DataFrame) for e in data):
        raise TypeError('The checked data in the results must be a Dataframe')


def postcheck(data, cols_mapping, outcols, logger=None):
    for exch in data:
        if cols_mapping:
            logger.debug('Renaming {} data columns: {}'.format(exch, list(cols_mapping.keys())))
        df = data[exch]
        df = rename_filter(df, cols_mapping, outcols)
        if logger is not None:
            logger.debug('Output {} data columns: {}'.format(exch, list(df.columns)))
        data[exch] = df



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


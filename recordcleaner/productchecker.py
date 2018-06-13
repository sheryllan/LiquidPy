from collections import namedtuple
from sortedcontainers import SortedDict

from commonlib.datastruct import namedtuple_with_defaults
from commonlib.iohelper import XlsxWriter
from configfilesparser import *
from productmatcher import *

RECORDED = 'Recorded'


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


def sum_unique(subdf, aggr_col):
    return sum(subdf[aggr_col].unique())


# parameter file2sheet is a tuple
def xl_consolidate(file2sheet, dest):
    wrt = XlsxWriter.create_xlwriter(dest, False)
    for fl, sht in file2sheet:
        xl = pd.ExcelFile(fl)
        dt = xl.parse(sht)
        XlsxWriter.to_xlsheet(dt, wrt, sht)
    wrt.save()


def print_duplicate(group, duplicate):
    print()
    print('In group: {}'.format(group))
    print('Duplicate: {}'.format(str(duplicate)))


def dfgroupby_aggr(df, group_key, aggr_col, aggr_func):
    groups = df_groupby(df, group_key)
    for group, subdf in groups.items():
        aggr_val = aggr_func(subdf, aggr_col)
        for _, row in subdf.iterrows():
            row[aggr_col] = aggr_val
            yield row


def df_todict(df, keyfunc):
    return {keyfunc(row): row for _, row in df.iterrows()}


def divide_dict_by(orig_dict, key_cols, left_sort=False, right_sort=False):
    left_dict = SortedDict() if left_sort else dict()
    right_dict = SortedDict() if right_sort else dict()
    for k, v in orig_dict.items():
        right_key = tuple(v[col] for col in key_cols)
        left_dict.update({k: right_key})
        right_dict.update(({right_key: v}))
    return left_dict, right_dict


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


def get_config_dict(exch, keycols=(ProductKey.FD_PRODCODE, ProductKey.FD_TYPE), keygen=ProductKey, valcols=None):
    config_data = parse_config(exch, attrs=valcols)
    return {keygen(**select_dict(d, keycols)): d for d in config_data}


def filter_mark_prods(data_rows, filterfunc, keyfunc, config_dict):
    for row in data_rows:
        if filterfunc and not filterfunc(row):
            continue
        recorded = keyfunc(row) in config_dict
        yield {**row, RECORDED: recorded}





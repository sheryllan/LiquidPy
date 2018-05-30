import pandas as pd
from collections import namedtuple
from sortedcontainers import SortedDict

from configparser import *
from productmatcher import *
from commonlib.iohelper import XlsxWriter
from commonlib.datastruct import namedtuple_with_defaults


# reports_path = '/home/slan/Documents/exch_report/'
# configs_path = '/home/slan/Documents/config_files/'
# # checked_path = '/home/slan/Documents/checked_report/'
# checked_path = os.getcwd()
#
# EXCHANGES = ['asx', 'bloomberg', 'cme', 'cbot', 'nymex_comex', 'eurex', 'hkfe', 'ice', 'ose', 'sgx']
# REPORT_FMTNAME = 'Web_ADV_Report_{}.xlsx'
#
# REPORT_FILES = {e: REPORT_FMTNAME.format(e.upper()) for e in EXCHANGES}

# region output keys
PRODCODE = 'prod_code'
TYPE = 'type'
RECORDED = 'recorded'
# endregion


class ProductKey(namedtuple_with_defaults(namedtuple('ProductKey', [PRODCODE, TYPE]))):
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

    def __hash__(self):
        this_tuple = tuple(map(lambda x: str(x) if x else x, self))
        t_code = this_tuple[0].lower() if this_tuple[0] else this_tuple[0]
        t_type = MatchHelper.to_singular_noun(this_tuple[1].lower())
        return hash(tuple([t_code, t_type]))


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
    print('duplicate: {}'.format(str(duplicate)))


def dfgroupby_aggr(df, group_key, aggr_col, aggr_func):
    groups = df_groupby(df, group_key)
    for group, subdf in groups.items():
        aggr_val = aggr_func(subdf, aggr_col)
        for _, row in subdf.iterrows():
            row[aggr_col] = aggr_val
            yield row
            # dict_key = dict_keyfunc(row)
            # if dict_key is not None:
            #     if dict_key in output_dict:
            #         print_duplicate(group, dict_key)
            #     else:
            #         row.update(pd.Series([aggr_val], index=[aggr_col]))
            #         output_dict.update({dict_key: row})
    # return output_dict


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


def get_config_dict(exch, keycols=(CO_PRODCODE, CO_TYPE), valcols=None):
    config_data = parse_config(exch, columns=valcols)
    return {tuple(d[col] for col in keycols): d for d in config_data}


def filter_mark_prods(data_rows, filterfunc, keyfunc, config_dict):
    for row in data_rows:
        if filterfunc and not filterfunc(row):
            continue
        recorded = keyfunc(row) in config_dict
        yield {**row, RECORDED: recorded}



class OSEChecker(object):
    def __init__(self):
        self.cfg_properties = ATTR_NAMES['ose']


    # def run_pd_check(self, df):





# xl_consolidate(test_input, test_output)
# xl = pd.ExcelFile(test_input[0][0])
# summary = xl.parse(test_input[0][1])
# products = xl.parse(test_input[1][1])['commodity_name']
# exp = lambda x: x in products.tolist()
# results = summary[list(filter(summary, 'Globex',  exp))]
# print((summary[list(filter(summary, 'Globex',  exp))].head()))

# cmechecker = CMEGChecker()
# cmechecker.run_pd_check(dict())





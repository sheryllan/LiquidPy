import pandas as pd
from collections import namedtuple
from sortedcontainers import SortedDict

import configparser as cp
from productmatcher import *
from commonlib.iohelper import XlsxWriter
from commonlib.commonfuncs import *
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
GROUP = 'group'
PRODUCT = 'product'
PRODCODE = 'prod_code'
TYPE = 'type'
RECORDED = 'recorded'
# endregion


ProductKey = namedtuple_with_defaults(namedtuple('ProductKey', [PRODCODE, TYPE]))


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
    print('duplicate (pd_code, cleared_as): {}'.format(duplicate))


def aggregate_todict(df, group_key, aggr_col, aggr_func, dict_keyfunc):
    groups = df_groupby(df, group_key)
    output_dict = dict()
    for group, subdf in groups.items():
        aggr_val = aggr_func(subdf, aggr_col)
        for _, row in subdf.iterrows():
            dict_key = dict_keyfunc(row)
            if dict_key is not None:
                if dict_key in output_dict:
                    print_duplicate(group, dict_key)
                else:
                    row.update(pd.Series([aggr_val], index=[aggr_col]))
                    output_dict.update({dict_key: row})
    return groups, output_dict


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


def get_cf_type(rp_type):
    cf_type = find_first_n(cp.INSTRUMENT_TYPES.keys(),
                           lambda x: MatchHelper.match_in_string(x, rp_type, one=False, stemming=True))
    return cf_type if cf_type else None


def get_config_keys(exch, cols, name='configkey'):
    config_data = cp.parse_config(exch)[cols]
    return set(key for key in config_data.itertuples(False, name))






class OSEChecker(object):
    def __init__(self):
        self.cfg_properties = cp.PROPERTIES['ose']


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





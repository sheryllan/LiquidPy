import pandas as pd
import numpy as np
import configparser as cp
import os
import re
import inflect
import itertools
import math
from sortedcontainers import SortedDict
from productmatcher import *

from configparser import XlsxWriter


# Parse the config files
# cp.parse_save()


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


def sum_unique(subdf, aggr_col):
    return sum(subdf[aggr_col].unique())


# parameter file2sheet is a tuple
def xl_consolidate(file2sheet, dest):
    wrt = cp.XlsxWriter.create_xlwriter(dest, False)
    for fl, sht in file2sheet:
        xl = pd.ExcelFile(fl)
        dt = xl.parse(sht)
        cp.XlsxWriter.to_xlsheet(dt, wrt, sht)
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


class CMEGChecker(object):
    EXCHANGES = ['cme', 'cbot', 'nymex_comex']
    REPORT_FMTNAME = 'Web_ADV_Report_{}.xlsx'
    PRODSLAT_FILE = 'Product_Slate.xls'

    def __init__(self, checked_path=None):
        self.report_files = [self.REPORT_FMTNAME.format(e.upper()) for e in self.EXCHANGES]
        self.checked_path = checked_path if checked_path is not None else os.getcwd()
        self.cmeg_prds_file = os.path.join(self.checked_path, self.PRODSLAT_FILE)
        self.cmeg_adv_files = [os.path.join(self.checked_path, f) for f in self.report_files]
        self.matcher = CMEGMatcher(self.cmeg_adv_files, self.cmeg_prds_file, '2017', self.checked_path)

    def get_prod_code(self, row):
        if not pd.isnull(row[self.matcher.F_GLOBEX]):
            return row[self.matcher.F_GLOBEX]
        elif not pd.isnull(row[self.matcher.F_CLEARING]):
            return row[self.matcher.F_CLEARING]
        else:
            print('no code: {}'.format(row[self.matcher.F_PRODUCT_NAME]))
            return None

    def get_prod_key(self, row):
        if pd.isnull(row[self.matcher.F_PRODUCT_NAME]):
            return None
        pd_code = self.get_prod_code(row)
        if pd_code is not None:
            return pd_code, row[self.matcher.CLEARED_AS]
        return None

    def run_pd_check(self, vol_threshold=1000, outpath=None):
        dfs_dict = self.matcher.run_pd_mtch(clean=True)
        group_key = [[CMEGMatcher.PRODUCT, CMEGMatcher.CLEARED_AS]]
        aggr_func = sum_unique
        dict_keyfunc = self.get_prod_key

        config_exchs = ['cme']
        config_data = {ex: data[cp.sheets[0]] for ex, data in cp.run_config_parse(config_exchs).items()}
        type = cp.properties[config_exchs[0]][0]
        commodity = cp.properties[config_exchs[0]][1]
        config_rowdict = {(cfg_dict[commodity], cfg_dict[type]) for cfg_dict in config_data[config_exchs[0]]}

        exchanges = [CMEGMatcher.CME, CMEGMatcher.CBOT, CMEGMatcher.NYMEX]
        prods_wanted = list()
        for exch in exchanges:
            df = dfs_dict[exch]
            ytd = self.matcher.get_ytd_header(df)
            _, agg_dict = aggregate_todict(df, group_key, ytd, aggr_func, dict_keyfunc)

            for k, row in agg_dict.items():
                if row[ytd] < vol_threshold:
                    continue
                pdnm = row[CMEGMatcher.F_PRODUCT_NAME]
                cf_key = (k[0], k[1][0].upper())
                recorded = cf_key in config_rowdict
                result = {PRODCODE: k[0], TYPE: k[0], PRODUCT: pdnm, RECORDED: recorded}
                prods_wanted.append(result)
                print(result)

            if outpath is not None:
                outdf_cols = [PRODCODE, TYPE, PRODUCT, RECORDED]
                outdf = pd.DataFrame(prods_wanted, columns=outdf_cols)
                XlsxWriter.save_sheets(outpath, {'Products': outdf})


cmeg_checker = CMEGChecker()
cmeg_checker.run_pd_check(outpath='CMEG_checked.xlsx')


# xl_consolidate(test_input, test_output)
# xl = pd.ExcelFile(test_input[0][0])
# summary = xl.parse(test_input[0][1])
# products = xl.parse(test_input[1][1])['commodity_name']
# exp = lambda x: x in products.tolist()
# results = summary[list(filter(summary, 'Globex',  exp))]
# print((summary[list(filter(summary, 'Globex',  exp))].head()))







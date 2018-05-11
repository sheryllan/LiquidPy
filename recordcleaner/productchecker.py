from sortedcontainers import SortedDict

import configparser as cp
from productmatcher import *

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


class CMEGChecker(object):

    def get_prod_code(self, row):
        if not pd.isnull(row[CMEGMatcher.F_GLOBEX]):
            return row[CMEGMatcher.F_GLOBEX]
        elif not pd.isnull(row[CMEGMatcher.F_CLEARING]):
            return row[CMEGMatcher.F_CLEARING]
        else:
            print('no code: {}'.format(row[CMEGMatcher.F_PRODUCT_NAME]))
            return None

    def get_prod_key(self, row):
        if pd.isnull(row[CMEGMatcher.F_PRODUCT_NAME]):
            return None
        pd_code = self.get_prod_code(row)
        if pd_code is not None:
            return pd_code, row[CMEGMatcher.CLEARED_AS]
        return None

    def __is_recorded(self, key, config_keys):
        cf_type = get_cf_type(key[1])
        new_key = (key[0], cf_type)
        return new_key in config_keys

    def check_prod_by(self, agg_dict, threshold, rec_condition, config_keys):
        prods_wanted = list()
        for k, row in agg_dict.items():
            if not rec_condition(row, threshold):
                continue
            pdnm = row[CMEGMatcher.F_PRODUCT_NAME]
            recorded = self.__is_recorded(k, config_keys)
            result = {PRODCODE: k[0], TYPE: k[1], PRODUCT: pdnm, RECORDED: recorded}
            prods_wanted.append(result)
            print(result)
        return prods_wanted

    def check_cme_cbot(self, dfs_dict, config_keys, vol_threshold):
        df_cme, df_cbot = dfs_dict[CMEGMatcher.CME], dfs_dict[CMEGMatcher.CBOT]

        group_key = [[CMEGMatcher.PRODUCT, CMEGMatcher.CLEARED_AS]]
        aggr_func = sum_unique
        dict_keyfunc = self.get_prod_key

        ytd_cme = CMEGMatcher.get_ytd_header(df_cme)
        ytd_cbot = CMEGMatcher.get_ytd_header(df_cbot)
        _, aggdict_cme = aggregate_todict(df_cme, group_key, ytd_cme, aggr_func, dict_keyfunc)
        _, aggdict_cbot = aggregate_todict(df_cbot, group_key, ytd_cbot, aggr_func, dict_keyfunc)

        prods_cme = self.check_prod_by(aggdict_cme, vol_threshold, lambda row, threshold: row[ytd_cme] >= threshold,
                                       config_keys)
        prods_cbot = self.check_prod_by(aggdict_cbot, vol_threshold, lambda row, threshold: row[ytd_cbot] >= threshold,
                                        config_keys)

        return {CMEGMatcher.CME: prods_cme, CMEGMatcher.CBOT: prods_cbot}

    def check_nymex(self, dfs_dict, config_keys, vol_threshold):
        df_nymex = dfs_dict[CMEGMatcher.NYMEX]
        dict_nymex = df_todict(df_nymex, self.get_prod_key)
        ytd_nymex = CMEGMatcher.get_ytd_header(df_nymex)
        prods_nymex = self.check_prod_by(dict_nymex, vol_threshold, lambda row, threshold: row[ytd_nymex] >= threshold,
                                         config_keys)
        return {CMEGMatcher.NYMEX: prods_nymex}

    def run_pd_check(self, dfs_dict, vol_threshold=1000, outpath=None):
        cme = 'cme'
        config_props = cp.PROPERTIES[cme]
        config_keys = get_config_keys(cme, [config_props[1], config_props[0]])

        prods_cme_cbot = self.check_cme_cbot(dfs_dict, config_keys, vol_threshold)
        prods_nymex = self.check_nymex(dfs_dict, config_keys, vol_threshold)
        prods_cmeg = {**prods_cme_cbot, **prods_nymex}

        if outpath is not None:
            outdf_cols = [PRODCODE, TYPE, PRODUCT, RECORDED]
            outdf_dict = {exch: pd.DataFrame(prods, columns=outdf_cols) for exch, prods in prods_cmeg.items()}
            return cp.XlsxWriter.save_sheets(outpath, outdf_dict)
        return prods_cmeg



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

cmechecker = CMEGChecker()
cmechecker.run_pd_check(dict())





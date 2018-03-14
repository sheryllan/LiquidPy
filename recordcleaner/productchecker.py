import pandas as pd
import numpy as np
import configparser as cp
import os
import re


import datascraper as dtsp



# Parse the config files
#cp.parse_save()


reports_path = '/home/slan/Documents/exch_report/'
configs_path = '/home/slan/Documents/config_files/'
# checked_path = '/home/slan/Documents/checked_report/'
checked_path = os.getcwd()

exchanges = ['asx', 'bloomberg', 'cme', 'eurex', 'hkfe', 'ice', 'ose', 'sgx']
report_fmtname = '_Average_Daily_Volume.xlsx'

report_files = {e: e.upper() + report_fmtname for e in exchanges}
# config_files = {e: e + '.xlsx' for e in exchanges}
#
# test_input = [(reports_path + report_files['cme'], 'Summary'), (configs_path + config_files['cme'], 'Config')]
# test_output = test_input[0][0]

def last_word(string):
    words = string.split()
    return words[-1]


# parameter file2sheet is a tuple
def xl_consolidate(file2sheet, dest):
    wrt = cp.XlsxWriter.create_xlwriter(dest, False)
    for fl, sht in file2sheet:
        xl = pd.ExcelFile(fl)
        dt = xl.parse(sht)
        cp.XlsxWriter.to_xlsheet(dt, wrt, sht)
    wrt.save()

def filter(df, col, exp):
    return df[col].map(exp)

#xl_consolidate(test_input, test_output)
# xl = pd.ExcelFile(test_input[0][0])
# summary = xl.parse(test_input[0][1])
# products = xl.parse(test_input[1][1])['commodity_name']
# exp = lambda x: x in products.tolist()
# results = summary[list(filter(summary, 'Globex',  exp))]
# print((summary[list(filter(summary, 'Globex',  exp))].head()))


class CMEChecker(object):
    PATTERN_ADV_YTD = 'ADV Y.T.D'



    def __init__(self, adv_file, prods_file):
        self.adv_file = adv_file
        self.prods_file = prods_file
        self.cols_adv = dtsp.CMEScraper.OUTPUT_COLUMNS
        self.cols_mapping = {self.cols_adv[0]: 'Product Name',
                             self.cols_adv[1]: 'Product Group',
                             self.cols_adv[2]: 'Cleared As'}
        self.cols_prods = list(self.cols_mapping.values()) + ['Globex', 'Sub Group', 'Exchange']



    def __from_adv(self):
        with open(self.adv_file, 'rb') as fh:
            df = pd.read_excel(fh)
        headers = list(df.columns.values)
        ytd = dtsp.find_first_n(headers, lambda x: self.PATTERN_ADV_YTD in x)
        self.cols_adv = self.cols_adv + ytd
        df = df[self.cols_adv]
        return df



    def __from_prods(self, df=None):
        if df is None:
            with open(self.prods_file, 'rb') as fh:
                df = pd.read_excel(fh)
            df.dropna(how='all', inplace=True)
            df.columns = df.iloc[0]
            df.drop(df.head(3).index, inplace=True)
            df.reset_index(drop=0, inplace=True)
        return df[self.cols_prods]

    def __clean_prod_name(self, row):
        product = row[self.cols_prods[0]]
        der_type = row[self.cols_prods[2]]
        return product.replace(der_type, '') if last_word(product) == der_type else product

    def __groupby(self, df, cols):
        if not cols:
            return df
        else:
            gpobj = df.groupby(cols[0])
            group_dict = dict()
            for group in gpobj.groups.keys():
                new_df = gpobj.get_group(group)
                group_dict[group] = self.__groupby(new_df, cols[1:])
            return group_dict

    def match_prod_code(self):
        df_adv = self.__from_adv()
        df_prods = self.__from_prods()
        for index, row in df_prods.iterrows():
            row[self.cols_prods[0]] = self.__clean_prod_name(row)
            df_prods.iloc[index] = row

        gdf_prods = self.__groupby(df_prods, self.cols_prods[1:2])

        gdf_adv = self.__groupby(df_adv, self.cols_adv[1:2])
        self.exhibit([gdf_prods, gdf_adv], [self.cols_prods[0], self.cols_adv[0]])

        print()




    def exhibit(self, gdfs, cols):
        output = os.path.join(checked_path, 'exhibit.xlsx')

        # cdfs = {group: pd.concat([gdf[group][col].sort_values() for gdf, col in zip(gdfs, cols)], axis=1)
        #         for group in gdfs[0].keys()}
        cdfs = dict()
        print([list(gdf.keys()) for gdf in gdfs])
        for group in gdfs[0].keys():
            dfs = []
            for gdf, col in zip(gdfs, cols):
                found_gp = group if group in gdf.keys() \
                    else dtsp.find_first_n(gdf.keys(), lambda x: SearchHelper.match_initials(group, x) or SearchHelper.match_first_n(group, x))[0]
                dfs.append(gdf[found_gp][col].sort_values())
            cc = pd.concat(dfs)
            cdfs[group] = cc


        cp.XlsxWriter.save_sheets(output, cdfs)
        print(output)



class SearchHelper(object):
    vowels = ('a', 'e', 'i', 'o', 'u')

    @staticmethod
    def get_words(string):
        return re.split('[ ,\.\?;:]+', string)

    @staticmethod
    def get_initials(string):
        words = SearchHelper.get_words(string)
        initials = list()
        for word in words:
            if word[0:2] == 'ex':
                initials.append('x')
            elif re.match('[A-Za-z]', word[0]):
                initials.append(word[0])
        return initials

    @staticmethod
    def match_initials(s1, s2):
        return SearchHelper.get_initials(s1) == SearchHelper.get_initials(s2)

    @staticmethod
    def match_first_n(s1, s2, n=2):
        if len(s1) >= n and len(s2) >= n:
            return s1[0:n] == s2[0:n]

    @staticmethod
    def match_sgl_plrl(s1, s2):
        if s1 == s2:
            return True
        if len(s1) > 2 and len(s2) > 2:
            if s1[-1] == 'y' and s1[-2] not in SearchHelper.vowels:
                return s2[-3:] == 'ies'
            if s2[-1] == 'y' and s2[-2] not in SearchHelper.vowels:
                return s1[-3:] == 'ies'
        else:
            return False

    @staticmethod
    def match_sgl_plrl_instring(s1, s2):
        words = SearchHelper.get_words(s1)



# cme_prds_file = os.path.join(checked_path, 'Product_Slate.xls')
cme_prds_file = os.path.join(checked_path, 'Product Slate Export.xls')
cme_adv_file = os.path.join(checked_path, report_files['cme'])

cme = CMEChecker(cme_adv_file, cme_prds_file)
cme.match_prod_code()

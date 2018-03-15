import pandas as pd
import numpy as np
import configparser as cp
import os
import re
import inflect
from math import isnan


from whoosh.fields import *
from whoosh.index import create_in
from whoosh.index import open_dir
from whoosh.query import *
from whoosh import qparser
from whoosh import writing


import datascraper as dtsp

# Parse the config files
# cp.parse_save()


reports_path = '/home/slan/Documents/exch_report/'
configs_path = '/home/slan/Documents/config_files/'
checked_path = '/home/slan/Documents/checked_report/'
# checked_path = os.getcwd()

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


# xl_consolidate(test_input, test_output)
# xl = pd.ExcelFile(test_input[0][0])
# summary = xl.parse(test_input[0][1])
# products = xl.parse(test_input[1][1])['commodity_name']
# exp = lambda x: x in products.tolist()
# results = summary[list(filter(summary, 'Globex',  exp))]
# print((summary[list(filter(summary, 'Globex',  exp))].head()))


class CMEChecker(object):
    PATTERN_ADV_YTD = 'ADV Y.T.D'
    INDEX = 'CME_Product_Index'

    def __init__(self, adv_file, prods_file, out_path=None):
        self.adv_file = adv_file
        self.prods_file = prods_file
        self.cols_adv = dtsp.CMEScraper.OUTPUT_COLUMNS

        self.cols_prods = ['Product Name', 'Product Group', 'Cleared As', 'Clearing', 'Globex', 'Sub Group', 'Exchange']
        self.cols_mapping = {self.cols_adv[0]: self.cols_prods[0],
                             self.cols_adv[1]: self.cols_prods[1],
                             self.cols_adv[2]: self.cols_prods[2]}
        self.index_fields = {'Product_Name': TEXT(stored=True),
                             'Product_Group': KEYWORD(stored=True, scorable=True),
                             'Cleared_As': KEYWORD(stored=True, scorable=True),
                             'Clearing': ID(stored=True, unique=True),
                             'Globex':ID(stored=True, unique=True),
                             'Sub_Group': TEXT(stored=True),
                             'Exchange':KEYWORD(scorable=True)}
        self.col2field = {col: col.replace(' ', '_') for col in self.cols_prods}
        self.out_path = out_path if out_path is not None else os.path.dirname(prods_file)
        self.index = os.path.join(self.out_path, self.INDEX)


    def __from_adv(self, encoding='utf-8'):
        with open(self.adv_file, 'rb') as fh:
            df = pd.read_excel(fh, encoding=encoding)
        headers = list(df.columns.values)
        ytd = dtsp.find_first_n(headers, lambda x: self.PATTERN_ADV_YTD in x)
        self.cols_adv.append(ytd)
        df = df[self.cols_adv]
        return df

    def __from_prods(self, df=None, encoding='utf-8'):
        if df is None:
            with open(self.prods_file, 'rb') as fh:
                df = pd.read_excel(fh, encoding=encoding)
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



    # just for development
    def exhibit(self, gdfs, cols):
        output = os.path.join(checked_path, 'exhibit.xlsx')
        cdfs = dict()
        for gdf, col in zip(gdfs, cols):
            for gp in gdf.keys():
                df_sr = gdf[gp][[col]].sort_values(col).reset_index(drop=True)
                gp_key = dtsp.find_first_n(cdfs.keys(), lambda x: self.__match_pdgp(gp, x))
                if not gp_key:
                    cdfs[gp] = df_sr
                else:
                    if col in cdfs[gp_key].columns:
                        cdfs[gp_key] = pd.merge(cdfs[gp_key], df_sr, how='outer')
                    else:
                        cdfs[gp_key] = pd.concat([cdfs[gp_key], df_sr], axis=1)

        cp.XlsxWriter.save_sheets(output, cdfs)
        print(output)

    def __match_pdgp(self, s1, s2):
        wds1 = SearchHelper.get_words(s1)
        wds2 = SearchHelper.get_words(s2)
        if len(wds1) == 1 and len(wds2) == 1:
            return s1 == s2 or SearchHelper.match_sgl_plrl(wds1[0], wds2[0]) or SearchHelper.match_first_n(wds1[0], wds2[0])
        else:
            return s1 == s2 or SearchHelper.match_initials(s1, s2) or SearchHelper.match_first_n(s1, s2)


    def match_prod_code(self):
        df_adv = self.__from_adv()
        df_prods = self.__from_prods()
        for index, adv_row in df_prods.iterrows():
            adv_row[self.cols_prods[0]] = self.__clean_prod_name(adv_row)
            df_prods.iloc[index] = adv_row

        # gdf_prods = self.__groupby(df_prods, self.cols_prods[1:2])
        # gdf_adv = self.__groupby(df_adv, self.cols_adv[1:2])
        # self.exhibit([gdf_prods, gdf_adv], [self.cols_prods[0], self.cols_adv[0]])

        # gdf_prods = self.__groupby(df_prods, self.cols_prods[1:3])
        # gdf_adv = self.__groupby(df_adv, self.cols_adv[1:3])

        adv_pdgp_col = 'Product Group'
        adv_clas_col = 'Cleared As'
        adv_pd_col = 'Product'
        prods_pdgps = set(df_prods[self.cols_mapping[adv_pdgp_col]])
        #slate_class = set(df_prods[self.cols_mapping[adv_clas_col]])

        adv_row = df_adv.iloc[49]
        pdgp = dtsp.find_first_n(prods_pdgps, lambda x: self.__match_pdgp(x, adv_row[adv_pdgp_col]))

        pdgp_field = self.col2field[self.cols_mapping[adv_pdgp_col]]
        clas_field = self.col2field[self.cols_mapping[adv_clas_col]]
        pd_field = self.col2field[self.cols_mapping[adv_pd_col]]

        # df_prods.rename(columns=self.col2field, inplace=True)
        # ix = self.__setup_ix(self.index, self.index_fields, df_prods, False)
        # docs = list(ix.searcher().documents())
        ix = open_dir(self.index)
        # myquery = And([Term(pdgp_field, pdgp), Term(clas_field, adv_row[adv_clas_col]), Term(pd_field, adv_row[adv_pd_col].lower())])
        #myquery = And([Term(pdgp_field, pdgp), Term(clas_field, adv_row[adv_clas_col])])
        parser = qparser.QueryParser(pd_field, schema=ix.schema, group=qparser.OrGroup)
        print(adv_row[adv_pd_col])
        query = parser.parse(adv_row[adv_pd_col])
        with ix.searcher() as searcher:
            results = searcher.search(query)
            print(results[0])


        print()


    def __create_index(self, index, fields, clean=False):
        if (not clean) and os.path.exists(index):
            return open_dir(index)
        if not os.path.exists(index):
            os.mkdir(index)
        schema = Schema(**fields)
        return create_in(index, schema)

    def __index_from_df(self, ix, df, clean=False):
        wrt = ix.writer()
        records = df.to_dict('records')
        for record in records:
            record = {k: record[k] for k in record if not pd.isnull(record[k])}
            if clean:
                wrt.add_document(**record)
            else:
                wrt.update_document(**record)
        wrt.commit()

    def __setup_ix(self, ix, fields, df, clean=False):
        ix = self.__create_index(ix, fields, clean)
        self.__index_from_df(ix, df, clean)
        return ix

    def __clear_index(self, ix):
        wrt = ix.writer()
        wrt.commit(mergetype=writing.CLEAR)




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
            if word[0:2].lower() == 'ex':
                initials.append(word[1])
            elif re.match('[A-Za-z]', word[0]):
                initials.append(word[0])
        return initials

    @staticmethod
    def match_initials(s1, s2):
        return ''.join(SearchHelper.get_initials(s1)).lower() == s2.lower() \
               or ''.join(SearchHelper.get_initials(s2)).lower() == s1.lower()

    @staticmethod
    def match_first_n(s1, s2, n=2):
        if len(s1) >= n and len(s2) >= n:
            return s1[0:n] == s2[0:n]
        elif len(s1) < n:
            return s1[0:] == s2[0:n]
        elif len(s2) < n:
            return s1[0:n] == s2[0:]
        return False

    @staticmethod
    def match_sgl_plrl(s1, s2):
        p = inflect.engine()
        return p.plural(s1) == s2


    @staticmethod
    def match_sgl_plrl_instring(s1, s2):
        words = SearchHelper.get_words(s1)


cme_prds_file = os.path.join(checked_path, 'Product_Slate.xls')
# cme_prds_file = os.path.join(checked_path, 'Product Slate Export.xls')
cme_adv_file = os.path.join(checked_path, report_files['cme'])

cme = CMEChecker(cme_adv_file, cme_prds_file)
cme.match_prod_code()

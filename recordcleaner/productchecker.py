import pandas as pd
import numpy as np
import configparser as cp
import os
import re
import inflect
from math import isnan


from whoosh.fields import *
from whoosh.analysis import *
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
# checked_path = '/home/slan/Documents/checked_report/'
checked_path = os.getcwd()

exchanges = ['asx', 'bloomberg', 'cme', 'cbot', 'nymex_comex', 'eurex', 'hkfe', 'ice', 'ose', 'sgx']
report_fmtname = 'Web_ADV_Report_{}.xlsx'

report_files = {e: report_fmtname.format(e.upper()) for e in exchanges}


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


class CMEGChecker(object):
    PATTERN_ADV_YTD = 'ADV Y.T.D'
    INDEX_CME = 'CME_Product_Index'
    INDEX_CBOT = 'CBOT_Product_Index'
    CME = 'CME'
    CBOT = 'CBOT'
    NYMEX = 'NYMEX'
    COMEX = 'COMEX'

    PRODUCT = dtsp.CMEGScraper.PRODUCT
    PRODUCT_GROUP = dtsp.CMEGScraper.PRODUCT_GROUP
    CLEARED_AS = dtsp.CMEGScraper.CLEARED_AS
    COLS_ADV = dtsp.CMEGScraper.OUTPUT_COLUMNS

    PRODUCT_NAME = 'Product Name'
    CLEARING = 'Clearing'
    GLOBEX = 'Globex'
    SUB_GROUP = 'Sub Group'
    EXCHANGE = 'Exchange'
    COMMODITY = 'Commodity'
    MATCHED = 'Matched'

    COLS_MAPPING = {PRODUCT: PRODUCT_NAME, PRODUCT_GROUP: PRODUCT_GROUP, CLEARED_AS: CLEARED_AS}
    COLS_PRODS = [PRODUCT_NAME, PRODUCT_GROUP, CLEARED_AS, CLEARING, GLOBEX, SUB_GROUP, EXCHANGE, MATCHED]

    F_PRODUCT_NAME = 'Product_Name'
    F_PRODUCT_GROUP = 'Product_Group'
    F_CLEARED_AS = 'Cleared_As'
    F_CLEARING = CLEARING
    F_GLOBEX = GLOBEX
    F_SUB_GROUP = 'Sub_Group'
    F_EXCHANGE = EXCHANGE
    F_MATCHED = MATCHED

    INDEX_FIELDS = {F_PRODUCT_NAME: TEXT(stored=True, analyzer=FancyAnalyzer()),
                    F_PRODUCT_GROUP: KEYWORD(stored=True, scorable=True),
                    F_CLEARED_AS: KEYWORD(stored=True, scorable=True),
                    F_CLEARING: ID(stored=True, unique=True),
                    F_GLOBEX: ID(stored=True, unique=True),
                    F_SUB_GROUP: TEXT(stored=True),
                    F_EXCHANGE: KEYWORD,
                    F_MATCHED: BOOLEAN(stored=True)}

    COL2FIELD = {PRODUCT_NAME: F_PRODUCT_NAME,
                 PRODUCT_GROUP: F_PRODUCT_GROUP,
                 CLEARED_AS: F_CLEARED_AS,
                 CLEARING: F_CLEARING,
                 GLOBEX: F_GLOBEX,
                 SUB_GROUP: F_SUB_GROUP,
                 EXCHANGE: F_EXCHANGE,
                 MATCHED: F_MATCHED}

    def __init__(self, adv_files, prods_file, out_path=None):
        self.adv_files = adv_files
        self.adv_cme = dtsp.find_first_n(adv_files, lambda x: self.CME.lower() in x.lower())
        self.adv_cbot = dtsp.find_first_n(adv_files, lambda x: self.CBOT.lower() in x.lower())
        self.adv_nymex_comex = dtsp.find_first_n(adv_files, lambda x: self.NYMEX.lower() in x.lower())

        self.prods_file = prods_file
        self.out_path = out_path if out_path is not None else os.path.dirname(prods_file)

        self.index_cme = os.path.join(self.out_path, self.INDEX_CME)
        self.index_cbot = os.path.join(self.out_path, self.INDEX_CBOT)
        self.matched_file = os.path.join(os.getcwd(), 'CME_matched.xlsx')

    def __from_adv(self, filename, cols=None, encoding='utf-8'):
        with open(filename, 'rb') as fh:
            df = pd.read_excel(fh, encoding=encoding)
        headers = list(df.columns.values)
        ytd = dtsp.find_first_n(headers, lambda x: self.PATTERN_ADV_YTD in x)
        cols = self.COLS_ADV if cols is None else cols
        cols = cols + [ytd]
        df = df[cols]
        return df

    def __from_prods(self, filename, df=None, encoding='utf-8'):
        if df is None:
            with open(filename, 'rb') as fh:
                df = pd.read_excel(fh, encoding=encoding)
            df.dropna(axis=0, how='all', inplace=True)
            df.columns = df.iloc[0]
            df.drop(df.head(1).index, inplace=True)
            df.dropna(subset=list(df.columns)[0:4], how='all', inplace=True)
            df.reset_index(drop=0, inplace=True)
            clean_1stcol = pd.Series([self.__clean_prod_name(r) for i, r in df.iterrows()], name=self.PRODUCT_NAME)
            df.update(clean_1stcol)
            df[self.MATCHED] = pd.Series(np.repeat(False, len(df.index)), index=df.index)
        return df[self.COLS_PRODS]

    def __clean_prod_name(self, row):
        product = row[self.PRODUCT_NAME]
        der_type = row[self.CLEARED_AS]
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

    def match_prod_code(self, df_adv, prods_pdgps, ix):
        df_matched = pd.DataFrame(columns=list(df_adv.columns) + ix.schema.names())
        for i, row in df_adv.iterrows():
            with ix.searcher() as searcher:
                pdgp = dtsp.find_first_n(prods_pdgps, lambda x: self.__match_pdgp(x, row[self.PRODUCT_GROUP]))
                grouping_q = And([Term(self.F_PRODUCT_GROUP, pdgp), Term(self.F_CLEARED_AS, row[self.CLEARED_AS]), Term(self.F_MATCHED, False)])
                query = self.__exact_and_query(self.F_PRODUCT_NAME, ix.schema, row[self.PRODUCT])
                results = searcher.search(query, filter=grouping_q, limit=None)
                if results:
                    row_matched = self.__after_hit_results(results, {self.F_MATCHED: True}, ix, row)
                else:
                    query = self.__fuzzy_and_query(self.F_PRODUCT_NAME, ix.schema, row[self.PRODUCT])
                    results = searcher.search(query, filter=grouping_q, limit=None)
                    if results:
                        row_matched = self.__after_hit_results(results, {self.F_MATCHED: True}, ix, row)
                    else:
                        query = self.__exact_or_query(self.F_PRODUCT_NAME, ix.schema, row[self.PRODUCT])
                        results = searcher.search(query, filter=grouping_q, limit=None)
                        if results:
                            row_matched = self.__after_hit_results(results, {self.F_MATCHED: True}, ix, row, False)
                        else:
                            row_matched = row
                df_matched = df_matched.append(row_matched, ignore_index=True)
                if row_matched is row:
                    print('Failed matching {}'.format(row[self.PRODUCT]))
                else:
                    print('Successful matching {} with {}'.format(row[self.PRODUCT], row_matched[self.F_PRODUCT_NAME]))
        return df_matched

    def __chk_prodcode_matched(self, df_prods, df_adv):
        unmatched = [(i, str(entry)) for i, entry in df_adv[self.COMMODITY].iteritems() if str(entry) not in df_prods[self.GLOBEX].astype(str).unique()]
        unmatched = [(i, entry) for i, entry in unmatched if entry not in df_prods[self.CLEARING].astype(str).unique()]
        ytd = dtsp.find_first_n(list(df_adv.columns), lambda x: self.PATTERN_ADV_YTD in x)
        indices = [i for i, _ in unmatched if df_adv.iloc[i][ytd] == 0]
        df_adv.drop(df_adv.index[indices], inplace=True)
        df_adv.reset_index(drop=0, inplace=True)
        return df_adv

    def __after_hit_results(self, results, updates, ix, row, mark=True):
        row_joined, hit = self.__join_best_result(results, row)
        if mark:
            hit.update(updates)
            self.__update_doc(ix, hit)
        return row_joined


    def __join_best_result(self, results, *dfs):
        joined_dict = results[0].fields()
        if dfs is not None:
            for df in dfs:
                joined_dict.update(df)
        return joined_dict, results[0].fields()


    def __exact_and_query(self, field, schema, text):
        parser = qparser.QueryParser(field, schema=schema)
        return parser.parse(text)

    def __fuzzy_and_query(self, field, schema, text, maxdist=2, prefixlength=1):
        parser = qparser.QueryParser(field, schema=schema)
        query = parser.parse(text)
        fuzzy_terms = And([FuzzyTerm(f, t, maxdist=maxdist, prefixlength=prefixlength) for f, t in query.iter_all_terms()])
        return fuzzy_terms

    def __exact_or_query(self, field, schema, text):
        og = qparser.OrGroup.factory(0.9)
        parser = qparser.QueryParser(field, schema=schema, group=og)
        return parser.parse(text)

    def __update_doc(self, ix, doc):
        wrt = ix.writer()
        wrt.update_document(**doc)
        wrt.commit()


    def mark_recorded(self):
        pass

    def run_pd_chck(self, outpath=None, clean=False):
        dfs_adv = {self.CME: self.__from_adv(self.adv_cme, self.COLS_ADV),
                   self.CBOT: self.__from_adv(self.adv_cbot, self.COLS_ADV),
                   self.NYMEX: self.__from_adv(self.adv_nymex_comex, self.COLS_ADV + [self.COMMODITY])}

        df_prods = self.__from_prods(self.prods_file)

        df_ix = df_prods.rename(columns=self.COL2FIELD)
        gdf_exch = {exch: df.reset_index(drop=0) for exch, df in self.__groupby(df_ix, [self.EXCHANGE]).items()}

        df_nymex_comex_prods = gdf_exch[self.NYMEX].append(gdf_exch[self.COMEX], ignore_index=True)
        dfs_adv[self.NYMEX] = self.__chk_prodcode_matched(df_nymex_comex_prods, dfs_adv[self.NYMEX])
        cp.XlsxWriter.save_sheets(self.matched_file, {self.NYMEX: dfs_adv[self.NYMEX]}, override=False)

        ix_cme = self.__setup_ix(self.INDEX_FIELDS, gdf_exch[self.CME], self.index_cme, clean)
        ix_cbot = self.__setup_ix(self.INDEX_FIELDS, gdf_exch[self.CBOT], self.index_cbot, clean)

        pdgp_cme = set(gdf_exch[self.CME][self.COL2FIELD[self.PRODUCT_GROUP]])
        mdf_cme = self.match_prod_code(dfs_adv[self.CME], pdgp_cme, ix_cme)
        pdgp_cbot = set(gdf_exch[self.CBOT][self.COL2FIELD[self.PRODUCT_GROUP]])
        mdf_cbot = self.match_prod_code(dfs_adv[self.CBOT], pdgp_cbot, ix_cbot)

        outpath = self.matched_file if outpath is None else outpath
        cp.XlsxWriter.save_sheets(outpath, {self.CME: mdf_cme, self.CBOT: mdf_cbot}, override=False)


    def __create_index(self, ix_path, fields, clean=False):
        if (not clean) and os.path.exists(ix_path):
            return open_dir(ix_path)
        if not os.path.exists(ix_path):
            os.mkdir(ix_path)
        schema = Schema(**fields)
        return create_in(ix_path, schema)

    def __index_from_df(self, ix, df, clean=False):
        wrt = ix.writer()
        fields = ix.schema.names()
        records = df[fields].to_dict('records')
        for record in records:
            record = {k: record[k] for k in record if not pd.isnull(record[k])}
            if clean:
                wrt.add_document(**record)
            else:
                wrt.update_document(**record)
        wrt.commit()

    def __setup_ix(self, fields, df, ix_path, clean=False):
        ix = self.__create_index(ix_path, fields, clean)
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



cme_prds_file = os.path.join(checked_path, 'Product_Slate.xls')
# cme_prds_file = os.path.join(checked_path, 'Product Slate Export.xls')
cme_adv_files = [os.path.join(checked_path, report_files['cme']),
                 os.path.join(checked_path, report_files['cbot']),
                 os.path.join(checked_path, report_files['nymex_comex'])]

cme = CMEGChecker(cme_adv_files, cme_prds_file)
cme.run_pd_chck(clean=True)

# ix = open_dir(cme.index_cme)
# docs = list(ix.searcher().documents())
# k = len(docs)
# print(pd.DataFrame(docs))

# prod = 'NEW ZEALND DOLLAR'
# pdgp = 'FX'
# ca = 'Futures'
# pdgp_field = 'Product_Group'
# clas_field = 'Cleared_As'
# pd_field = 'Product_Name'
# mcth_field = 'Matched'
# ix = open_dir(cme.index)
# with ix.searcher() as searcher:
#
#     # for doc in searcher.documents():
#     #     print(doc)
#
#     grouping_q = Term(mcth_field, True)
#     # grouping_q = And([Term(pdgp_field, pdgp), Term(clas_field, ca), Term(mcth_field, False)])
#     # grouping_q = qparser.QueryParser(pdgp_field, schema=ix.schema).parse(pdgp)
#     parser = qparser.QueryParser(pd_field, schema=ix.schema)
#     query = parser.parse(prod)
#     ts = query.iter_all_terms()
#     fuzzy_terms = And([FuzzyTerm(f, t, maxdist=2, prefixlength=1) for f, t in ts])
#     # results = searcher.search(query, filter=grouping_q, limit=None)
#     results = searcher.search(grouping_q, limit=None)
#     if results:
#         # hit = results[0].fields()
#         rs = [r.fields() for r in results]
#         print(pd.DataFrame(rs))
#
#     print()

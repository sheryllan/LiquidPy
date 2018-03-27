import pandas as pd
import numpy as np
import configparser as cp
import os
import re
import inflect
import itertools

from whoosh.fields import *
from whoosh.analysis import *
from whoosh.index import create_in
from whoosh.index import open_dir
from whoosh.query import *
from whoosh import qparser
from whoosh import writing

import datascraper as dtsp
from whooshext import *


def last_word(string):
    words = string.split()
    return words[-1]


def filter(df, col, exp):
    return df[col].map(exp)


CRRNCY_MAPPING = {'ad': ('australian dollar', 1.2),
                  'bp': ('british pound', 1.2),
                  'cd': ('canadian dollar', 1.2),
                  'ec': ('euro', 1.2),
                  'efx': ('euro fx', 1.2),
                  'jy': ('japanese yen', 1.2),
                  'jpy': ('japanese yen', 1.2),
                  'ne': ('new zealand dollar', 1.2),
                  'nok': ('norwegian krone', 1.2),
                  'sek': ('swedish krona', 1.2),
                  'sf': ('swiss franc', 1.2),
                  'skr': ('swedish krona', 1.2),
                  'zar': ('south african rand', 1.2),
                  'aud': ('australian dollar', 1.2),
                  'cad': ('canadian dollar', 1.2),
                  'chf': ('swiss franc', 1.2),
                  'eur': ('euro', 1.2),
                  'gbp': ('british pound', 1.2),
                  'pln': ('polish zloty', 1.2),
                  'nkr': ('norwegian krone', 1.2),
                  'inr': ('indian rupee', 1.2),
                  'rmb': ('chinese renminbi', 1.2),
                  'usd': ('us american dollar', 1.2)}

CRRNCY_KEYWORDS = list(set(dtsp.flatten_list(
    [[k.split(' '), v[0].split(' ')] for k, v in CRRNCY_MAPPING.items()], list())))


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
    def match_initials(s1, s2, casesensitive=False):
        if not casesensitive:
            s1 = s1.lower()
            s2 = s2.lower()
        return ''.join(SearchHelper.get_initials(s1)) == s2 \
               or ''.join(SearchHelper.get_initials(s2)) == s1

    @staticmethod
    def match_first_n(s1, s2, n=2, casesensitive=False):
        if not casesensitive:
            s1 = s1.lower()
            s2 = s2.lower()
        if len(s1) >= n and len(s2) >= n:
            return s1[0:n] == s2[0:n]
        elif len(s1) < n:
            return s1[0:] == s2[0:n]
        elif len(s2) < n:
            return s1[0:n] == s2[0:]
        return False

    @staticmethod
    def match_sgl_plrl(s1, s2, casesensitive=False):
        if not casesensitive:
            s1 = s1.lower()
            s2 = s2.lower()
        p = inflect.engine()
        return p.plural(s1) == s2

    @staticmethod
    def match_any_word(s1, s2, stemming=False, casesensitive=False):
        if not casesensitive:
            s1 = s1.lower()
            s2 = s2.lower()
        p = inflect.engine()
        words1 = [(w, 1) for w in SearchHelper.get_words(s1)]
        words2 = [(w, 2) for w in SearchHelper.get_words(s2)]
        if stemming:
            words1 = words1 + [(p.plural(w), 1) for w, i in words1]
        words = set(words1 + words2)
        words = itertools.groupby(words, key=lambda x: x[0])
        for key, group in itertools.groupby(words, key=lambda x: x[0]):
            if len(list(group)) == 2:
                return True
        return False


class CMEGMatcher(object):
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

    COLS_MAPPING = {PRODUCT: PRODUCT_NAME, PRODUCT_GROUP: PRODUCT_GROUP, CLEARED_AS: CLEARED_AS}
    COLS_PRODS = [PRODUCT_NAME, PRODUCT_GROUP, CLEARED_AS, CLEARING, GLOBEX, SUB_GROUP, EXCHANGE]

    F_PRODUCT_NAME = 'Product_Name'
    F_PRODUCT_GROUP = 'Product_Group'
    F_CLEARED_AS = 'Cleared_As'
    F_CLEARING = CLEARING
    F_GLOBEX = GLOBEX
    F_SUB_GROUP = 'Sub_Group'
    F_EXCHANGE = EXCHANGE

    CME_EXACT_MAPPING = {('NIKKEI 225 ($) STOCK', 'Equity Index', 'Futures'): 'Nikkei/USD',
                         ('NIKKEI 225 (YEN) STOCK', 'Equity Index', 'Futures'): 'Nikkei/Yen',
                         ('FT-SE 100', 'Equity Index', 'Futures'): 'FTSE 100 (GBP)',
                         ('AUSTRALIAN DOLLAR', 'FX',
                          'Options'): 'Premium Quoted European Style Options on Australian Dollar/US Dollar Futures',
                         ('BRITISH POUND', 'FX',
                          'Options'): 'Premium Quoted European Style Options on British Pound/US Dollar Futures',
                         ('CANADIAN DOLLAR', 'FX',
                          'Options'): 'Premium Quoted European Style Options on Canadian Dollar/US Dollar Futures',
                         ('EURO FX', 'FX',
                          'Options'): 'Premium Quoted European Style Options on Euro/US Dollar Futures',
                         ('JAPANESE YEN', 'FX',
                          'Options'): 'Premium Quoted European Style Options on Japanese Yen/US Dollar Futures',
                         ('SWISS FRANC', 'FX',
                          'Options'): 'Premium Quoted European Style Options on Swiss Franc/US Dollar Futures',
                         ('BDI', 'FX', 'Futures'): 'CME Bloomberg Dollar Spot Index',
                         ('S.AFRICAN RAND', 'FX', 'Futures'): 'South African Rand',
                         ('CHINESE RENMINBI (CNH)', 'FX', 'Futures'): 'Standard-Size USD/Offshore RMB (CNH)',
                         ('CHF/USD PQO 2pm Fix', 'FX',
                          'Options'): 'Weekly Premium Quoted European Style Options on Swiss Franc/US Dollar Futures - Wk'
                         }

    CME_MULTI_MATCH = [('EURO MIDCURVE', 'Interest Rate', 'Options'),
                       ('AUD/USD PQO 2pm Fix', 'FX', 'Options'),
                       ('GBP/USD PQO 2pm Fix', 'FX', 'Options'),
                       ('JPY/USD PQO 2pm Fix', 'FX', 'Options'),
                       ('EUR/USD PQO 2pm Fix', 'FX', 'Options'),
                       ('CAD/USD PQO 2pm Fix', 'FX', 'Options'),
                       ('CHF/USD PQO 2pm Fix', 'FX', 'Options')]

    CME_SPECIAL_MAPPING = {'midcurve': ('midcurve mc', 1.2),
                           'pqo': ('premium quoted european style options', 1.2),
                           'eow': ('weekly wk', 1.2),
                           'eom': ('monthly', 1.2),
                           'usdzar': ('us dollar south african rand', 1.2),
                           'biotech': ('biotechnology', 1.2),
                           'cross': ('cross', 0.6),
                           'rates': ('rates', 0.6)}

    CME_SPECIAL_KEYWORDS = list(set(dtsp.flatten_list(
        [[k.split(' '), v[0].split(' ')] for k, v in CME_SPECIAL_MAPPING.items()], list())))

    CME_KEYWORD_MAPPING = {**CRRNCY_MAPPING, **CME_SPECIAL_MAPPING}

    CME_KYWRD_EXCLU = CRRNCY_KEYWORDS + CME_SPECIAL_KEYWORDS + \
                      ['nasdaq', 'ibovespa', 'index', 'mini', 'emini',
                       'micro', 'emicro', 'nikkei', 'russell', 'ftse',
                       'european']

    SPLT_FLT = SplitFilter(origin=False, mergewords=True, mergenums=True)
    CME_SP_FLT = SpecialWordFilter(CME_KEYWORD_MAPPING)
    CME_VW_FLT = VowelFilter(CME_KYWRD_EXCLU)

    CME_PDNM_ANA = STD_ANA | SPLT_FLT | CME_SP_FLT | CME_VW_FLT
    INDEX_FIELDS_CME = {F_PRODUCT_NAME: TEXT(stored=True, analyzer=CME_PDNM_ANA),
                        F_PRODUCT_GROUP: ID(stored=True),
                        F_CLEARED_AS: ID(stored=True, unique=True),
                        F_CLEARING: ID(stored=True, unique=True),
                        F_GLOBEX: ID(stored=True, unique=True),
                        F_SUB_GROUP: TEXT(stored=True, analyzer=SimpleAnalyzer()),
                        F_EXCHANGE: ID}

    INDEX_FIELDS_CBOT = {F_PRODUCT_NAME: TEXT(stored=True, analyzer=STD_ANA),
                         F_PRODUCT_GROUP: ID(stored=True),
                         F_CLEARED_AS: ID(stored=True, unique=True),
                         F_CLEARING: ID(stored=True, unique=True),
                         F_GLOBEX: ID(stored=True, unique=True),
                         F_SUB_GROUP: TEXT(stored=True, analyzer=SimpleAnalyzer()),
                         F_EXCHANGE: ID}

    COL2FIELD = {PRODUCT_NAME: F_PRODUCT_NAME,
                 PRODUCT_GROUP: F_PRODUCT_GROUP,
                 CLEARED_AS: F_CLEARED_AS,
                 CLEARING: F_CLEARING,
                 GLOBEX: F_GLOBEX,
                 SUB_GROUP: F_SUB_GROUP,
                 EXCHANGE: F_EXCHANGE}

    def __init__(self, adv_files, prods_file, year, out_path=None):
        self.year = year
        self.adv_files = adv_files
        self.adv_cme = dtsp.find_first_n(adv_files, lambda x: self.CME.lower() in x.lower())
        self.adv_cbot = dtsp.find_first_n(adv_files, lambda x: self.CBOT.lower() in x.lower())
        self.adv_nymex_comex = dtsp.find_first_n(adv_files, lambda x: self.NYMEX.lower() in x.lower())

        self.prods_file = prods_file
        self.out_path = out_path if out_path is not None else os.path.dirname(prods_file)

        self.index_cme = os.path.join(self.out_path, self.INDEX_CME)
        self.index_cbot = os.path.join(self.out_path, self.INDEX_CBOT)
        self.matched_file = os.path.join(os.getcwd(), 'CMEG_matched.xlsx')

    def __from_adv(self, filename, cols=None, encoding='utf-8'):
        with open(filename, 'rb') as fh:
            df = pd.read_excel(fh, encoding=encoding)
        headers = list(df.columns.values)
        ytd = dtsp.find_first_n(headers, lambda x: self.PATTERN_ADV_YTD in x and self.year in x)
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

    def __match_pdgp(self, s1, s2):
        wds1 = SearchHelper.get_words(s1)
        wds2 = SearchHelper.get_words(s2)
        if len(wds1) == 1 and len(wds2) == 1:
            return s1 == s2 or SearchHelper.match_sgl_plrl(wds1[0], wds2[0]) \
                   or SearchHelper.match_first_n(wds1[0], wds2[0])
        else:
            return s1 == s2 or SearchHelper.match_any_word(s1, s2) \
                   or SearchHelper.match_initials(s1, s2) or SearchHelper.match_first_n(s1, s2)

    def match_prod_code(self, df_adv, prods_pdgps, ix):
        df_matched = pd.DataFrame(columns=list(df_adv.columns) + ix.schema.names())
        with ix.searcher() as searcher:
            for i, row in df_adv.iterrows():
                pd_id = (row[self.PRODUCT], row[self.PRODUCT_GROUP], row[self.CLEARED_AS])
                one_or_all = 'all' if pd_id in self.CME_MULTI_MATCH else 'one'
                pdnm = self.CME_EXACT_MAPPING[pd_id] if pd_id in self.CME_EXACT_MAPPING else row[self.PRODUCT]
                pdgp = dtsp.find_first_n(prods_pdgps, lambda x: self.__match_pdgp(x, row[self.PRODUCT_GROUP]))
                grouping_q = And([Term(self.F_PRODUCT_GROUP, pdgp), Term(self.F_CLEARED_AS, row[self.CLEARED_AS])])
                query = self.__exact_and_query(self.F_PRODUCT_NAME, ix.schema, pdnm)
                results = searcher.search(query, filter=grouping_q, limit=None)
                if results:
                    rows_matched = self.__join_results(results, row, how=one_or_all)
                else:
                    query = self.__fuzzy_and_query(self.F_PRODUCT_NAME, ix.schema, row[self.PRODUCT])
                    results = searcher.search(query, filter=grouping_q, limit=None)
                    if results:
                        rows_matched = self.__join_results(results, row, how=one_or_all)
                    else:
                        query = self.__exact_or_query(self.F_PRODUCT_NAME, ix.schema, row[self.PRODUCT])
                        results = searcher.search(query, filter=grouping_q, limit=None)
                        if results:
                            rows_matched = self.__join_results(results, row, how=one_or_all)
                        else:
                            rows_matched = row
                df_matched = df_matched.append(rows_matched, ignore_index=True)
                if rows_matched is row:
                    print('Failed matching {}'.format(row[self.PRODUCT]))
                else:
                    for r in rows_matched:
                        print('Successful matching {} with {}'.format(row[self.PRODUCT], r[self.F_PRODUCT_NAME]))

        return df_matched

    def __get_mtched_prods_by_cmmd(self, df_prods, df_adv):
        prod_dict = {str(row[self.GLOBEX]): row for _, row in df_prods.iterrows()}
        prod_dict.update({str(row[self.CLEARING]): row for _, row in df_prods.iterrows()})
        matched_rows = [{**row, **prod_dict[str(row[self.COMMODITY])]} for _, row in df_adv.iterrows()
                        if str(row[self.COMMODITY]) in prod_dict]
        cols = list(df_adv.columns) + list(df_prods.columns)
        df = pd.DataFrame(matched_rows, columns=cols)
        # unmatched = [(i, str(entry)) for i, entry in df_adv[self.COMMODITY].iteritems() if
        #              str(entry) not in df_prods[self.GLOBEX].astype(str).unique()]
        # unmatched = [(i, entry) for i, entry in unmatched if entry not in df_prods[self.CLEARING].astype(str).unique()]
        # ytd = dtsp.find_first_n(list(df_adv.columns), lambda x: self.PATTERN_ADV_YTD in x)
        # indices = [i for i, _ in unmatched if df_adv.iloc[i][ytd] == 0]
        # df_adv.drop(df_adv.index[indices], inplace=True)
        # df_adv.reset_index(drop=0, inplace=True)
        return df

    def __join_results(self, results, *dfs, how):
        joined_dict = [results[0].fields()] if how == 'one' else [r.fields() for r in results]
        if dfs is not None:
            for df in dfs:
                for jd in joined_dict:
                    jd.update(df)
        return joined_dict

    def __exact_and_query(self, field, schema, text):
        parser = qparser.QueryParser(field, schema=schema)
        return parser.parse(text)

    def __fuzzy_and_query(self, field, schema, text, maxdist=2, prefixlength=1):
        parser = qparser.QueryParser(field, schema=schema)
        query = parser.parse(text)
        fuzzy_terms = And(
            [FuzzyTerm(f, t, maxdist=maxdist, prefixlength=prefixlength) for f, t in query.iter_all_terms()])
        return fuzzy_terms

    def __exact_or_query(self, field, schema, text):
        og = qparser.OrGroup.factory(0.9)
        parser = qparser.QueryParser(field, schema=schema, group=og)
        return parser.parse(text)

    def __update_doc(self, ix, doc):
        wrt = ix.writer()
        wrt.update_document(**doc)
        wrt.commit()
        print(len(list(ix.searcher().documents())))

    def mark_recorded(self):
        pass

    def run_pd_mtch(self, outpath=None, clean=False):
        dfs_adv = {self.CME: self.__from_adv(self.adv_cme, self.COLS_ADV),
                   self.CBOT: self.__from_adv(self.adv_cbot, self.COLS_ADV),
                   self.NYMEX: self.__from_adv(self.adv_nymex_comex, self.COLS_ADV + [self.COMMODITY])}

        df_prods = self.__from_prods(self.prods_file)

        df_ix = df_prods.rename(columns=self.COL2FIELD)
        gdf_exch = {exch: df.reset_index(drop=0) for exch, df in self.__groupby(df_ix, [self.EXCHANGE]).items()}

        df_nymex_comex_prods = gdf_exch[self.NYMEX].append(gdf_exch[self.COMEX], ignore_index=True)
        df_nymex_comex_prods.reset_index(drop=True, inplace=True)
        if 'index' in df_nymex_comex_prods.columns:
            df_nymex_comex_prods.drop(['index'], axis=1, inplace=True)
        dfs_adv[self.NYMEX] = self.__get_mtched_prods_by_cmmd(df_nymex_comex_prods, dfs_adv[self.NYMEX])
        cp.XlsxWriter.save_sheets(self.matched_file, {self.NYMEX: dfs_adv[self.NYMEX]}, override=True)

        ix_cme = self.__setup_ix(self.INDEX_FIELDS_CME, gdf_exch[self.CME], self.index_cme, clean)
        ix_cbot = self.__setup_ix(self.INDEX_FIELDS_CBOT, gdf_exch[self.CBOT], self.index_cbot, clean)

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


checked_path = os.getcwd()

exchanges = ['asx', 'bloomberg', 'cme', 'cbot', 'nymex_comex', 'eurex', 'hkfe', 'ice', 'ose', 'sgx']
report_fmtname = 'Web_ADV_Report_{}.xlsx'

report_files = {e: report_fmtname.format(e.upper()) for e in exchanges}

cme_prds_file = os.path.join(checked_path, 'Product_Slate.xls')
cme_adv_files = [os.path.join(checked_path, report_files['cme']),
                 os.path.join(checked_path, report_files['cbot']),
                 os.path.join(checked_path, report_files['nymex_comex'])]

cme = CMEGMatcher(cme_adv_files, cme_prds_file, '2017')
cme.run_pd_mtch(clean=True)

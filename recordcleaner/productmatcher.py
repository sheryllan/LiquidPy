import pandas as pd
import numpy as np
import configparser as cp
import os
import re
import inflect
import itertools
import datetime
from dateutil.relativedelta import relativedelta

from whoosh.fields import *
from whoosh.analysis import *
from whoosh.query import *
from whoosh import qparser
from whoosh.searching import Hit

import datascraper as dtsp
from whooshext import *


def last_word(string):
    words = string.split()
    return words[-1]


def filter(df, col, exp):
    return df[col].map(exp)


def df_groupby(df, cols):
    if not cols:
        return df
    else:
        gpobj = df.groupby(cols[0])
        group_dict = dict()
        for group in gpobj.groups.keys():
            new_df = gpobj.get_group(group)
            group_dict[group] = df_groupby(new_df, cols[1:])
        return group_dict


STOP_LIST = ['and', 'is', 'it', 'an', 'as', 'at', 'have', 'in', 'yet', 'if', 'from', 'for', 'when',
             'by', 'to', 'you', 'be', 'we', 'that', 'may', 'not', 'with', 'tbd', 'a', 'on', 'your',
             'this', 'of', 'will', 'can', 'the', 'or', 'are']


class Matcher(object):
    vowels = ('a', 'e', 'i', 'o', 'u')

    @staticmethod
    def get_words(string):
        return re.split('[ ,\.\?;:]+', string)

    @staticmethod
    def get_initials(string):
        words = Matcher.get_words(string)
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
        return ''.join(Matcher.get_initials(s1)) == s2 \
               or ''.join(Matcher.get_initials(s2)) == s1

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
    def match_sgl_plrl(s1, s2, casesensitive=False, p=inflect.engine()):
        if not casesensitive:
            s1 = s1.lower()
            s2 = s2.lower()
        return s1 == s2 or p.plural(s1) == s2 or p.singular_noun(s1) == s2

    @staticmethod
    def match_in_string(s_ref, s_sample, one=True, stemming=False, casesensitive=False, engine=inflect.engine()):
        if not casesensitive:
            s_ref = s_ref.lower()
            s_sample = s_sample.lower()

        wds_sample = Matcher.get_words(s_sample)
        wds_ref = Matcher.get_words(s_ref)
        if not one:
            wds_sample = [' '.join(wds_sample)]
            wds_ref = ' '.join(wds_ref)

        found = False
        for w in wds_sample:
            found = w in wds_ref
            if len(w) < 3:
                continue
            if (not found) and stemming:
                found = (engine.plural(w) in wds_ref)
                if not found:
                    sgl = engine.singular_noun(w)
                    found = sgl in wds_ref if sgl else sgl
            if found:
                return found
        return found


class CMEGMatcher(object):
    PATTERN_ADV_YTD = 'ADV Y.T.D'
    INDEX_CME = 'CME_Product_Index'
    INDEX_CBOT = 'CBOT_Product_Index'
    CME = 'CME'
    CBOT = 'CBOT'
    NYMEX = 'NYMEX'
    COMEX = 'COMEX'

    # region columns  & fields
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
    # endregion

    # region CME specific
    CME_EXACT_MAPPING = {
        ('EURO MIDCURVE', 'Interest Rate', 'Options'): 'Eurodollar MC',
        ('NIKKEI 225 ($) STOCK', 'Equity Index', 'Futures'): 'Nikkei/USD',
        ('NIKKEI 225 (YEN) STOCK', 'Equity Index', 'Futures'): 'Nikkei/Yen',
        ('FT-SE 100', 'Equity Index', 'Futures'): 'E-mini FTSE 100 Index (GBP)',
        ('BDI', 'FX', 'Futures'): 'CME Bloomberg Dollar Spot Index',
        ('SKR/USD CROSS RATES', 'FX', 'Futures'): 'Swedish Krona',
        ('NKR/USD CROSS RATE', 'FX', 'Futures'): 'Norwegian Krone',
        ('S.AFRICAN RAND', 'FX', 'Futures'): 'South African Rand',
        ('CHINESE RENMINBI (CNH)', 'FX', 'Futures'): 'Standard-Size USD/Offshore RMB (CNH)',
        ('MILK', 'Ag Products', 'Futures'): 'Class III Milk',
        ('MILK', 'Ag Products', 'Options'): 'Class III Milk'
    }

    CME_NOTFOUND_PRODS = {('AUSTRALIAN DOLLAR', 'FX', 'Options'),
                          ('BRITISH POUND', 'FX', 'Options'),
                          ('CANADIAN DOLLAR', 'FX', 'Options'),
                          ('EURO FX', 'FX', 'Options'),
                          ('JAPANESE YEN', 'FX', 'Options'),
                          ('SWISS FRANC', 'FX', 'Options'),
                          ('JAPANESE YEN (EU)', 'FX', 'Options'),
                          ('SWISS FRANC (EU)', 'FX', 'Options'),
                          ('AUSTRALIAN DLR (EU)', 'FX', 'Options'),
                          ('MLK MID', 'Ag Products', 'Futures')}

    CME_MULTI_MATCH = {('EURO MIDCURVE', 'Interest Rate', 'Options'): {QUERY: 'andmaybe'},
                       ('AUD/USD PQO 2pm Fix', 'FX', 'Options'): {QUERY: 'andmaybe'},
                       ('GBP/USD PQO 2pm Fix', 'FX', 'Options'): {QUERY: 'andmaybe'},
                       ('JPY/USD PQO 2pm Fix', 'FX', 'Options'): {QUERY: 'andmaybe'},
                       ('EUR/USD PQO 2pm Fix', 'FX', 'Options'): {QUERY: 'andmaybe'},
                       ('CAD/USD PQO 2pm Fix', 'FX', 'Options'): {QUERY: 'andmaybe'},
                       ('CHF/USD PQO 2pm Fix', 'FX', 'Options'): {QUERY: 'andmaybe'},
                       ('LV CATL CSO', 'Ag Products', 'Options'): {QUERY: 'and'}}

    CRRNCY_TOKENSUB = {'aud': [TokenSub('australian', 1.5, True, True), TokenSub('dollar', 1, True, True)],
                       'gbp': [TokenSub('british', 1.5, True, True), TokenSub('pound', 1.5, True, True)],
                       'cad': [TokenSub('canadian', 1.5, True, True), TokenSub('dollar', 1, True, True)],
                       'euro': [TokenSub('euro', 1.5, True, True)],
                       'jpy': [TokenSub('japanese', 1.5, True, True), TokenSub('yen', 1.5, True, True)],
                       'nzd': [TokenSub('new', 1.5, True, True), TokenSub('zealand', 1.5, True, True),
                               TokenSub('dollar', 1, True, True)],
                       'nkr': [TokenSub('norwegian', 1.5, True, True), TokenSub('krone', 1.5, True, True)],
                       'sek': [TokenSub('swedish', 1.5, True, True), TokenSub('krona', 1.5, True, True)],
                       'chf': [TokenSub('swiss', 1.5, True, True), TokenSub('franc', 1.5, True, True)],
                       'zar': [TokenSub('south', 1.5, True, True), TokenSub('african', 1.5, True, True),
                               TokenSub('rand', 1.5, True, True)],
                       'pln': [TokenSub('polish', 1.5, True, True), TokenSub('zloty', 1.5, True, True)],
                       'inr': [TokenSub('indian', 1.5, True, True), TokenSub('rupee', 1.5, True, True)],
                       'rmb': [TokenSub('chinese', 1.5, True, True), TokenSub('renminbi', 1.5, True, True)],
                       'usd': [TokenSub('american', 1, True, False), TokenSub('dollar', 1, True, False)],
                       'clp': [TokenSub('chilean', 1.5, True, True), TokenSub('peso', 1.5, True, True)],
                       'mxn': [TokenSub('mexican', 1.5, True, True), TokenSub('peso', 1.5, True, True)],
                       'brl': [TokenSub('brazilian', 1.5, True, True), TokenSub('real', 1.5, True, True)],
                       'huf': [TokenSub('hungarian', 1.5, True, True), TokenSub('forint', 1.5, True, True)]
                       }

    CRRNCY_MAPPING = {'ad': CRRNCY_TOKENSUB['aud'],
                      'bp': CRRNCY_TOKENSUB['gbp'],
                      'cd': CRRNCY_TOKENSUB['cad'],
                      'ec': CRRNCY_TOKENSUB['euro'] +
                            [TokenSub('cross', 0.5, True, False), TokenSub('rates', 0.5, True, False)],
                      'efx': CRRNCY_TOKENSUB['euro'] +
                             [TokenSub('fx', 0.8, True, False)],
                      'jy': CRRNCY_TOKENSUB['jpy'],
                      'jpy': CRRNCY_TOKENSUB['jpy'],
                      'ne': CRRNCY_TOKENSUB['nzd'],
                      'nok': CRRNCY_TOKENSUB['nkr'],
                      'sek': CRRNCY_TOKENSUB['sek'],
                      'sf': CRRNCY_TOKENSUB['chf'],
                      'skr': CRRNCY_TOKENSUB['sek'],
                      'zar': CRRNCY_TOKENSUB['zar'],
                      'aud': CRRNCY_TOKENSUB['aud'],
                      'cad': CRRNCY_TOKENSUB['cad'],
                      'chf': CRRNCY_TOKENSUB['chf'],
                      'eur': CRRNCY_TOKENSUB['euro'],
                      'gbp': CRRNCY_TOKENSUB['gbp'],
                      'pln': CRRNCY_TOKENSUB['pln'],
                      'nkr': CRRNCY_TOKENSUB['nkr'],
                      'inr': CRRNCY_TOKENSUB['inr'],
                      'rmb': CRRNCY_TOKENSUB['rmb'],
                      'usd': CRRNCY_TOKENSUB['usd'],
                      'clp': CRRNCY_TOKENSUB['clp'],
                      'nzd': CRRNCY_TOKENSUB['nzd'],
                      'mxn': CRRNCY_TOKENSUB['mxn'],
                      'brl': CRRNCY_TOKENSUB['brl'],
                      'cnh': CRRNCY_TOKENSUB['rmb'],
                      'huf': CRRNCY_TOKENSUB['huf']}

    CME_SPECIAL_MAPPING = {'mc': [TokenSub('midcurve', 1, True, True)],
                           'pqo': [TokenSub('premium', 1, True, True), TokenSub('quoted', 1, True, True),
                                   TokenSub('european', 1, True, True), TokenSub('style', 1, True, True)],
                           'eow': [TokenSub('weekly', 1, True, True), TokenSub('wk', 1, True, False)],
                           'eom': [TokenSub('monthly', 1, True, True)],
                           'usdzar': CRRNCY_TOKENSUB['usd'] + CRRNCY_TOKENSUB['zar'],
                           'biotech': [TokenSub('biotechnology', 1.5, True, True)],
                           'us': [TokenSub('american', 1, True, False)],
                           'eu': [TokenSub('european', 1.5, True, True)],
                           'nfd': [TokenSub('non', 1.5, True, True), TokenSub('fat', 1.5, True, True),
                                   TokenSub('dry', 1.5, True, True)],
                           'cs': [TokenSub('cash', 1.5, True, True), TokenSub('settled', 1.5, True, True)],
                           'er': [TokenSub('excess', 1.5, True, True), TokenSub('return', 1.5, True, True)],
                           'catl': [TokenSub('cattle', 1.5, True, False)]}

    CME_COMMON_WORDS = ['futures', 'future', 'options', 'option', 'index', 'cross', 'rate', 'rates']

    CME_KEYWORD_MAPPING = {**CRRNCY_MAPPING, **CME_SPECIAL_MAPPING}

    CME_KYWRD_EXCLU = {'nasdaq', 'ibovespa', 'index', 'mini', 'micro', 'nikkei', 'russell', 'ftse', 'swap'}
    # endregion

    # region CBOT specific
    CBOT_EXACT_MAPPING = {('30-YR BOND', 'Interest Rate', 'Futures'): 'U.S. Treasury Bond',
                          ('30-YR BOND', 'Interest Rate', 'Options'): 'U.S. Treasury Bond',
                          ('DOW-UBS COMMOD INDEX', 'Ag Products', 'Futures'): 'Bloomberg Commodity Index',
                          ('DJ_UBS ROLL SELECT INDEX FU', 'Ag Products', 'Futures'):
                              'Bloomberg Roll Select Commodity Index',
                          ('SOYBN NEARBY+2 CAL SPRD', 'Ag Products', 'Options'): 'Consecutive Soybean CSO',
                          ('WHEAT NEARBY+2 CAL SPRD', 'Ag Products', 'Options'): 'Consecutive Wheat CSO',
                          ('CORN NEARBY+2 CAL SPRD', 'Ag Products', 'Options'): 'Consecutive Corn CSO'
                          }

    CBOT_SPECIAL_MAPPING = {'yr': [TokenSub('year', 1, True, False)],
                            'fed': [TokenSub('federal', 1.5, True, True)],
                            't': [TokenSub('treasury', 1, True, True)],
                            'note': [TokenSub('note', 1, True, True)],
                            'dj': [TokenSub('dow', 1, True, True), TokenSub('jones', 1, True, True)],
                            'cso': [TokenSub('calendar', 1.5, True, True), TokenSub('spread', 1, False, False)],
                            'cal': [TokenSub('calendar', 1.5, True, True)],
                            'hrw': [TokenSub('hr', 1.5, True, True), TokenSub('wheat', 1.5, True, True)],
                            'icso': [TokenSub('intercommodity', 1.5, True, True), TokenSub('spread', 1, True, True)],
                            'chi': [TokenSub('chicago', 1, True, True)]
                            }

    CBOT_MULTI_MATCH = {('30-YR BOND', 'Interest Rate', 'Options'): {QUERY: 'andnot', NOTWORDS: 'ultra'},
                        ('10-YR NOTE', 'Interest Rate', 'Options'): {QUERY: 'andnot', NOTWORDS: 'ultra'},
                        ('5-YR NOTE', 'Interest Rate', 'Options'): {QUERY: 'andnot', NOTWORDS: 'ultra'},
                        ('2-YR NOTE', 'Interest Rate', 'Options'): {QUERY: 'andnot', NOTWORDS: 'ultra'},
                        ('FED FUND', 'Interest Rate', 'Options'): {QUERY: 'andmaybe'},
                        ('ULTRA T-BOND', 'Interest Rate', 'Options'): {QUERY: 'andmaybe', ANDEXTRAS: 'ultra bond'},
                        ('Ultra 10-Year Note', 'Interest Rate', 'Options'): {QUERY: 'and'},
                        ('FERTILIZER PRODUCS', 'Ag Products', 'Futures'): {QUERY: 'every'},
                        ('DEC-JULY WHEAT CAL SPRD', 'Ag Products', 'Options'): {QUERY: 'and'},
                        ('JULY-DEC WHEAT CAL SPRD', 'Ag Products', 'Options'): {QUERY: 'and'},
                        ('MAR-JULY WHEAT CAL SPRD', 'Ag Products', 'Options'): {QUERY: 'and'},
                        ('Intercommodity Spread', 'Ag Products', 'Options'):
                            {QUERY: 'orofand', ANDLIST: ['MGEX-Chicago SRW Wheat Spread',
                                                         'KC HRW-Chicago SRW Wheat Intercommodity Spread',
                                                         'MGEX-KC HRW Wheat Intercommodity Spread']}}

    CBOT_COMMON_WORDS = ['futures', 'future', 'options', 'option']

    COL2FIELD = {PRODUCT_NAME: F_PRODUCT_NAME,
                 PRODUCT_GROUP: F_PRODUCT_GROUP,
                 CLEARED_AS: F_CLEARED_AS,
                 CLEARING: F_CLEARING,
                 GLOBEX: F_GLOBEX,
                 SUB_GROUP: F_SUB_GROUP,
                 EXCHANGE: F_EXCHANGE}

    def __init__(self, adv_files=None, prods_file=None, year=(datetime.datetime.now() - relativedelta(years=1)).year,
                 out_path=None):
        dflt_inpath = os.getcwd()
        self.year = year
        adv_files = [os.path.join(dflt_inpath, dtsp.CMEGScraper.DFLT_CME_ADV_XLSX),
                     os.path.join(dflt_inpath, dtsp.CMEGScraper.DFLT_CBOT_ADV_XLSX),
                     os.path.join(dflt_inpath, dtsp.CMEGScraper.DFLT_NYCO_ADV_XLSX)] \
            if adv_files is None else adv_files

        self.adv_cme = dtsp.find_first_n(adv_files, lambda x: self.CME.lower() in x.lower())
        self.adv_cbot = dtsp.find_first_n(adv_files, lambda x: self.CBOT.lower() in x.lower())
        self.adv_nymex_comex = dtsp.find_first_n(adv_files, lambda x: self.NYMEX.lower() in x.lower())

        self.prods_file = os.path.join(dflt_inpath, dtsp.CMEGScraper.DFLT_PROD_SLATE) \
            if prods_file is None else prods_file
        self.out_path = out_path if out_path is not None else os.path.dirname(self.prods_file)

        self.index_cme = os.path.join(self.out_path, self.INDEX_CME)
        self.index_cbot = os.path.join(self.out_path, self.INDEX_CBOT)
        self.matched_file = os.path.join(os.getcwd(), 'CMEG_matched.xlsx')

    def get_ytd_header(self, df):
        headers = list(df.columns.values)
        return dtsp.find_first_n(headers, lambda x: self.PATTERN_ADV_YTD in x and self.year in x)

    def __from_adv(self, filename, cols=None, encoding='utf-8'):
        df = pd.read_excel(filename, encoding=encoding)
        ytd = self.get_ytd_header(df)
        cols = self.COLS_ADV if cols is None else cols
        cols = cols + [ytd]
        df = df[cols]
        return df

    def __from_prods(self, filename, df=None, encoding='utf-8'):
        if df is None:
            df = pd.read_excel(filename, encoding=encoding)
            df.dropna(axis=0, how='all', inplace=True)
            df.columns = df.iloc[0]
            df.drop(df.head(1).index, inplace=True)
            df.dropna(subset=list(df.columns)[0:4], how='all', inplace=True)
            df.reset_index(drop=0, inplace=True)
        return df[self.COLS_PRODS]

    def __match_pdgp(self, s_ref, s_sample):
        return s_ref == s_sample or Matcher.match_in_string(s_ref, s_sample, one=True, stemming=True) \
               or Matcher.match_initials(s_ref, s_sample) or Matcher.match_first_n(s_ref, s_sample)

    def __match_in_string(self, guess, indexed, one=True):
        guess = guess.lower()
        p = inflect.engine()
        for idx in indexed:
            matched = Matcher.match_in_string(guess, idx, one, stemming=True, engine=p)
            if matched:
                return idx
        return None

    def __verify_clearedas(self, prodname, row_clras, clras_set):
        clras = Matcher.get_words(prodname)[-1]
        found_clras = self.__match_in_string(clras, clras_set)
        return found_clras if found_clras is not None else row_clras

    def match_prod_code(self, df_adv, ix, exact_mapping=None, notfound=None, multi_match=None):
        df_matched = pd.DataFrame(columns=list(df_adv.columns) + ix.schema.stored_names())
        with ix.searcher() as searcher:
            lexicons = get_idx_lexicon(searcher, self.F_PRODUCT_GROUP, self.F_CLEARED_AS,
                                       **{self.F_PRODUCT_GROUP: self.F_SUB_GROUP})
            adsearch = AdvSearch(searcher)

            for i, row in df_adv.iterrows():
                results = self.__match_a_row(row, lexicons, exact_mapping, notfound, multi_match, ix.schema, adsearch)
                rows_matched = self.__join_results(results, row)
                df_toadd = pd.DataFrame(rows_matched, columns=df_matched.columns)
                df_matched = df_matched.append(df_toadd, ignore_index=True)
                matched = True if results else False
                self.__prt_match_status(matched, rows_matched)
        return df_matched

    def __match_a_row(self, row, lexicons, exact_mapping, notfound, multi_match, schema, adsearch):
        pd_id = (row[self.PRODUCT], row[self.PRODUCT_GROUP], row[self.CLEARED_AS])
        if notfound is not None and pd_id in notfound:
            return None
        pdnm = exact_mapping[pd_id] if pd_id in exact_mapping else row[self.PRODUCT]
        is_one = True if multi_match is None or pd_id not in multi_match else False
        grouping_q = self.__get_grouping_query(row, pdnm, lexicons)
        qparams = {FIELDNAME: self.F_PRODUCT_NAME,
                   SCHEMA: schema,
                   QSTRING: pdnm}
        min_dist = True

        def callback(val):
            min_dist = val

        if is_one:
            results = self.__search_for_one(adsearch, qparams, grouping_q, callback)
            if results and min_dist:
                results = min_dist_rslt(results, pdnm, self.F_PRODUCT_NAME, schema, minboost=0.2)[0]
        else:
            q_configs = multi_match[pd_id]
            qparams.update(q_configs)
            results = self.__search_for_all(adsearch, qparams, grouping_q, callback)
        return results

    def __get_grouping_query(self, row, pdnm, lexicons):
        prods_pdgps, prods_subgps = lexicons[self.F_PRODUCT_GROUP], lexicons[self.F_SUB_GROUP]
        prods_clras = lexicons[self.F_CLEARED_AS]
        pdgp = dtsp.find_first_n(prods_pdgps, lambda x: self.__match_pdgp(x, row[self.PRODUCT_GROUP]))
        clras = self.__verify_clearedas(pdnm, row[self.CLEARED_AS], prods_clras)
        subgp = self.__match_in_string(pdnm, prods_subgps[pdgp], False)
        return filter_query((self.F_PRODUCT_GROUP, pdgp), (self.F_CLEARED_AS, clras), (self.F_SUB_GROUP, subgp))

    def __search_for_one(self, adsearch, qparams, grouping_q, callback):
        src_and = adsearch.search(*get_query_params('and', **qparams),
                                  lambda: callback(True),
                                  filter=grouping_q,
                                  limit=None)
        src_fuzzy = adsearch.search(*get_query_params('and', **{**qparams, TERMCLASS: FuzzyTerm}),
                                    lambda: callback(False),
                                    filter=grouping_q,
                                    limit=None)
        src_andmaybe = adsearch.search(*get_query_params('andmaybe', **qparams),
                                       lambda: callback(True),
                                       filter=grouping_q,
                                       limit=None)
        return adsearch.chain_search([src_and, src_fuzzy, src_andmaybe])

    def __search_for_all(self, adsearch, qparams, grouping_q, callback):
        query = qparams[QUERY]
        src = adsearch.search(*get_query_params(**qparams),
                              lambda: callback(True),
                              filter=grouping_q,
                              limit=None)

        src_fuzzy = adsearch.search(*get_query_params(**{**qparams, TERMCLASS: FuzzyTerm}),
                                    lambda: callback(False),
                                    filter=grouping_q,
                                    limit=None)

        def chain_condition(r):
            if not r:
                return True
            if r and query != 'every' and len(r) < 2:
                return True
            return False

        return adsearch.chain_search([src, src_fuzzy], chain_condition)

    def __prt_match_status(self, matched, rows_matched):
        for r in rows_matched:
            if not matched:
                print('Failed matching {}'.format(r[self.PRODUCT]))
            else:
                print('Successful matching {} with {}'.format(r[self.PRODUCT], r[self.F_PRODUCT_NAME]))

    def __join_results(self, results, df):
        joined_dicts = []

        def df_to_dict():
            return {k: v for k, v in df.items()}

        results = [results] if isinstance(results, Hit) else results
        if results:
            for r in results:
                df_copy = df_to_dict()
                df_copy.update(r)
                joined_dicts.append(df_copy)
        else:
            joined_dicts.append(df_to_dict())
        return joined_dicts

    def __match_prods_by_commodity(self, df_prods, df_adv):
        prod_dict = {str(row[self.GLOBEX]): row for _, row in df_prods.iterrows()}
        prod_dict.update({str(row[self.CLEARING]): row for _, row in df_prods.iterrows()})
        matched_rows = [{**row, **prod_dict[str(row[self.COMMODITY])]} for _, row in df_adv.iterrows()
                        if str(row[self.COMMODITY]) in prod_dict]
        cols = list(df_adv.columns) + list(df_prods.columns)
        df = pd.DataFrame(matched_rows, columns=cols)
        return df

    def get_cbot_fields(self):
        regtk_exp = '[^\s/\(\)]+'
        regex_tkn = RegexTokenizerExtra(regtk_exp, ignored=False, required=False)
        lwc_flt = LowercaseFilter()
        splt_mrg_flt = SplitMergeFilter(splitcase=True, splitnums=True, ignore_splt=True)

        stp_flt = StopFilter(stoplist=STOP_LIST + self.CBOT_COMMON_WORDS, minsize=1)
        sp_flt = SpecialWordFilter(self.CBOT_SPECIAL_MAPPING)
        vw_flt = VowelFilter(lift_ignore=False)
        multi_flt = MultiFilterFixed(index=vw_flt)
        ana = regex_tkn | splt_mrg_flt | lwc_flt | stp_flt | sp_flt | multi_flt | stp_flt

        return {self.F_PRODUCT_NAME: TEXT(stored=True, analyzer=ana),
                self.F_PRODUCT_GROUP: ID(stored=True, unique=True),
                self.F_CLEARED_AS: ID(stored=True, unique=True),
                self.F_CLEARING: ID(stored=True, unique=True),
                self.F_GLOBEX: ID(stored=True, unique=True),
                self.F_SUB_GROUP: ID(stored=True, unique=True),
                self.F_EXCHANGE: ID}

    def get_cme_fields(self):
        regtk_exp = '[^\s/\(\)]+'
        regex_tkn = RegexTokenizerExtra(regtk_exp, ignored=False, required=False)
        lwc_flt = LowercaseFilter()
        splt_mrg_flt = SplitMergeFilter(mergewords=True, mergenums=True, ignore_mrg=True)

        stp_flt =StopFilter(stoplist=STOP_LIST + self.CME_COMMON_WORDS, minsize=1)
        sp_flt = SpecialWordFilter(self.CME_KEYWORD_MAPPING)
        vw_flt = VowelFilter(self.CME_KYWRD_EXCLU, lift_ignore=False)
        multi_flt = MultiFilterFixed(index=vw_flt)
        ana = regex_tkn | splt_mrg_flt | lwc_flt | stp_flt | sp_flt | multi_flt | stp_flt

        return {self.F_PRODUCT_NAME: TEXT(stored=True, analyzer=ana),
                self.F_PRODUCT_GROUP: ID(stored=True, unique=True),
                self.F_CLEARED_AS: ID(stored=True, unique=True),
                self.F_CLEARING: ID(stored=True, unique=True),
                self.F_GLOBEX: ID(stored=True, unique=True),
                self.F_SUB_GROUP: ID(stored=True, unique=True),
                self.F_EXCHANGE: ID}

    def init_ix_cme_cbot(self, clean=False):
        df_prods = self.__from_prods(self.prods_file)
        df_ix = df_prods.rename(columns=self.COL2FIELD)
        gdf_exch = {exch: df.reset_index(drop=0) for exch, df in df_groupby(df_ix, [self.EXCHANGE]).items()}
        ix_cme = setup_ix(self.get_cme_fields(), gdf_exch[self.CME], self.index_cme, clean)
        ix_cbot = setup_ix(self.get_cbot_fields(), gdf_exch[self.CBOT], self.index_cbot, clean)
        return ix_cme, ix_cbot, gdf_exch

    def match_nymex_comex(self, df_adv, gdf_exch):
        df_nymex_comex_prods = gdf_exch[self.NYMEX].append(gdf_exch[self.COMEX], ignore_index=True)
        df_nymex_comex_prods.reset_index(drop=True, inplace=True)
        if 'index' in df_nymex_comex_prods.columns:
            df_nymex_comex_prods.drop(['index'], axis=1, inplace=True)
        df_adv = self.__match_prods_by_commodity(df_nymex_comex_prods, df_adv)
        return df_adv

    def run_pd_mtch(self, clean=False):
        dfs_adv = {self.CME: self.__from_adv(self.adv_cme, self.COLS_ADV),
                   self.CBOT: self.__from_adv(self.adv_cbot, self.COLS_ADV),
                   self.NYMEX: self.__from_adv(self.adv_nymex_comex, self.COLS_ADV + [self.COMMODITY])}

        ix_cme, ix_cbot, gdf_exch = self.init_ix_cme_cbot(clean)
        mdf_nymex = self.match_nymex_comex(dfs_adv[self.NYMEX], gdf_exch)
        mdf_cme = self.match_prod_code(dfs_adv[self.CME], ix_cme, self.CME_EXACT_MAPPING, self.CME_NOTFOUND_PRODS,
                                       self.CME_MULTI_MATCH)
        mdf_cbot = self.match_prod_code(dfs_adv[self.CBOT], ix_cbot, self.CBOT_EXACT_MAPPING,
                                        multi_match=self.CBOT_MULTI_MATCH)
        return {self.CME: mdf_cme, self.CBOT: mdf_cbot, self.NYMEX: mdf_nymex}

    def save_to_xlsx(self, dfs_dict, outpath=None, override=True):
        outpath = self.matched_file if outpath is None else outpath
        cp.XlsxWriter.save_sheets(outpath, dfs_dict, override=override)
        return outpath


# checked_path = os.getcwd()
#
# exchanges = ['asx', 'bloomberg', 'cme', 'cbot', 'nymex_comex', 'eurex', 'hkfe', 'ice', 'ose', 'sgx']
# report_fmtname = 'Web_ADV_Report_{}.xlsx'
#
# report_files = {e: report_fmtname.format(e.upper()) for e in exchanges}
#
# cmeg_prds_file = os.path.join(checked_path, 'Product_Slate.xls')
# cmeg_adv_files = [os.path.join(checked_path, report_files['cme']),
#                   os.path.join(checked_path, report_files['cbot']),
#                   os.path.join(checked_path, report_files['nymex_comex'])]
#
# cmeg = CMEGMatcher(cmeg_adv_files, cmeg_prds_file, '2017')
#
# cmeg.save_to_xlsx(cmeg.run_pd_mtch(clean=True))



# region TODO: test
# def __match_prods_by_commodity(self, df_prods, df_adv):
#     prod_dict = {str(row[self.GLOBEX]): row for _, row in df_prods.iterrows()}
#     prod_dict.update({str(row[self.CLEARING]): row for _, row in df_prods.iterrows()})
#     matched_rows = [{**row, **prod_dict[str(row[self.COMMODITY])]} for _, row in df_adv.iterrows()
#                     if str(row[self.COMMODITY]) in prod_dict]
#     cols = list(df_adv.columns) + list(df_prods.columns)
#     df = pd.DataFrame(matched_rows, columns=cols)
#     # unmatched = [(i, str(entry)) for i, entry in df_adv[self.COMMODITY].iteritems() if
#     #              str(entry) not in df_prods[self.GLOBEX].astype(str).unique()]
#     # unmatched = [(i, entry) for i, entry in unmatched if entry not in df_prods[self.CLEARING].astype(str).unique()]
#     # ytd = dtsp.find_first_n(list(df_adv.columns), lambda x: self.PATTERN_ADV_YTD in x)
#     # indices = [i for i, _ in unmatched if df_adv.iloc[i][ytd] == 0]
#     # df_adv.drop(df_adv.index[indices], inplace=True)
#     # df_adv.reset_index(drop=0, inplace=True)
#     return df
# endregion

# region unused
# def __clean_prod_name(self, row):
    #     product = row[self.PRODUCT_NAME]
    #     der_type = row[self.CLEARED_AS]
    #     product = product.replace(der_type, '') if last_word(product) == der_type else product
    #     return product.rstrip()

# endregion
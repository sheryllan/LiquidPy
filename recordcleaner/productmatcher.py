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

import datascraper as dtsp
from whooshext import *


def last_word(string):
    words = string.split()
    return words[-1]


def filter(df, col, exp):
    return df[col].map(exp)


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
        words1 = [(w, 1) for w in Matcher.get_words(s1)]
        words2 = [(w, 2) for w in Matcher.get_words(s2)]
        if stemming:
            words1 = words1 + [(p.plural(w), 1) for w, i in words1]
        words = set(words1 + words2)
        words = itertools.groupby(words, key=lambda x: x[0])
        for key, group in itertools.groupby(words, key=lambda x: x[0]):
            if len(list(group)) == 2:
                return True
        return False


class WhooshSnippet(object):
    def __init__(self):
        self.results = None

    @staticmethod
    def get_idx_lexicon(searcher, *fds, **kwfds):
        grouped_docs = searcher.search(Every(), groupedby=fds)
        lexicons = dict()
        for fd in fds:
            gp_dict = {gp: [searcher.stored_fields(docid) for docid in ids]
                     for gp, ids in grouped_docs.groups(fd).items()}
            lexicons.update({fd: set(gp_dict.keys())})
            if fd in kwfds:
                subfd = kwfds[fd]
                subgp_dict = {gp: set([doc[subfd] for doc in docs]) for gp, docs in gp_dict.items()}
                lexicons.update({subfd: subgp_dict})
        return lexicons

    @staticmethod
    def exact_and_query(field, schema, text):
        parser = qparser.QueryParser(field, schema=schema)
        return parser.parse(text)

    @staticmethod
    def fuzzy_and_query(field, schema, text, maxdist=2, prefixlength=1):
        parser = qparser.QueryParser(field, schema=schema)
        query = parser.parse(text)
        fuzzy_terms = And(
            [FuzzyTerm(f, t, maxdist=maxdist, prefixlength=prefixlength)
             if len(t) > maxdist else Term(f, t) for f, t in query.iter_all_terms()])

        return fuzzy_terms

    @staticmethod
    def exact_or_query(field, schema, text):
        og = qparser.OrGroup.factory(0.9)
        parser = qparser.QueryParser(field, schema=schema, group=og)
        return parser.parse(text)

    @staticmethod
    def tokenize_split(field, text, keyfunc, mode='query'):
        tokens = field.analyzer(text, mode=mode)
        and_words, maybe_words = [], []
        for token in tokens:
            (and_words if keyfunc(token) else maybe_words).append(token.text)
        return and_words, maybe_words

    @staticmethod
    def andmaybe_query(fieldname, schema, text, keyfunc):
        field = get_field(schema, fieldname)
        and_words, maybe_words = WhooshSnippet.tokenize_split(field, text, keyfunc)
        and_terms = And([Term(fieldname, w) for w in and_words])
        maybe_terms = Or([Term(fieldname, w) for w in maybe_words])
        return AndMaybe(and_terms, maybe_terms) if and_terms else maybe_terms

    @staticmethod
    def andnot_query(field, schema, and_words, not_words):
        parser = qparser.QueryParser(field, schema=schema)
        q_and = parser.parse(and_words)
        q_not = parser.parse(not_words)
        return AndNot(q_and, q_not)

    @staticmethod
    def filter_query(*args):
        qterms = list()
        for arg in args:
            if all(arg):
                qterms.append(Term(*arg))
        return And(qterms)

    query_dict = {'and': exact_and_query,
                  'or': exact_or_query,
                  'andmaybe': andmaybe_query,
                  'fuzzyand': fuzzy_and_query,
                  'andnot': andnot_query}



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
        ('NIKKEI 225 ($) STOCK', 'Equity Index', 'Futures'): 'Nikkei/USD Futures',
        ('NIKKEI 225 (YEN) STOCK', 'Equity Index', 'Futures'): 'Nikkei/Yen Futures',
        ('BDI', 'FX', 'Futures'): 'CME Bloomberg Dollar Spot Index',
        ('CHINESE RENMINBI (CNH)', 'FX', 'Futures'): 'Standard-Size USD/Offshore RMB (CNH)',
        ('MILK', 'Ag Products', 'Futures'): 'Class III Milk Futures',
        ('MILK', 'Ag Products', 'Options'): 'Class III Milk Options'
    }

    CME_NOTFOUND_PRODS = {('AUSTRALIAN DOLLAR', 'FX', 'Options'),
                          ('BRITISH POUND', 'FX', 'Options'),
                          ('CANADIAN DOLLAR', 'FX', 'Options'),
                          ('EURO FX', 'FX', 'Options'),
                          ('JAPANESE YEN', 'FX', 'Options'),
                          ('SWISS FRANC',  'FX', 'Options'),
                          ('JAPANESE YEN (EU)', 'FX', 'Options'),
                          ('SWISS FRANC (EU)', 'FX', 'Options'),
                          ('AUSTRALIAN DLR (EU)', 'FX', 'Options'),
                          ('MLK MID', 'Ag Products', 'Futures')}

    CME_MULTI_MATCH = {('EURO MIDCURVE', 'Interest Rate', 'Options'): ['andmaybe'],
                       ('AUD/USD PQO 2pm Fix', 'FX', 'Options'): ['andmaybe'],
                       ('GBP/USD PQO 2pm Fix', 'FX', 'Options'): ['andmaybe'],
                       ('JPY/USD PQO 2pm Fix', 'FX', 'Options'): ['andmaybe'],
                       ('EUR/USD PQO 2pm Fix', 'FX', 'Options'): ['andmaybe'],
                       ('CAD/USD PQO 2pm Fix', 'FX', 'Options'): ['andmaybe'],
                       ('CHF/USD PQO 2pm Fix', 'FX', 'Options'): ['andmaybe'],
                       ('LV CATL CSO', 'Ag Products', 'Options'): ['andmaybe']}

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

    CME_SPECIAL_MAPPING = {'pqo': [TokenSub('premium', 1, True, True), TokenSub('quoted', 1, True, True),
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
    CBOT_EXACT_MAPPING = {('DOW-UBS COMMOD INDEX', 'Ag Products', 'Futures'): 'Bloomberg Commodity Index Futures',
                          ('DJ_UBS ROLL SELECT INDEX FU', 'Ag Products', 'Futures'):
                              'Bloomberg Roll Select Commodity Index Futures',
                          ('SOYBN NEARBY+2 CAL SPRD', 'Ag Products', 'Options'): 'Consecutive Soybean CSO',
                          ('WHEAT NEARBY+2 CAL SPRD', 'Ag Products', 'Options'): 'Consecutive Wheat CSO',
                          ('CORN NEARBY+2 CAL SPRD', 'Ag Products', 'Options'): 'Consecutive Corn CSO',
                          ('Intercommodity Spread', 'Ag Products', 'Options'): 'KC HRW-Chicago SRW Wheat Intercommodity Spread Options'
                          }

    CBOT_SPECIAL_MAPPING = {'yr': [TokenSub('year', 1, True, False)],
                            'fed': [TokenSub('federal', 1.5, True, True)],
                            't': [TokenSub('treasury', 1, True, True)],
                            'dj': [TokenSub('dow', 1, True, True), TokenSub('jones', 1, True, True)],
                            'cso': [TokenSub('calendar', 1.5, True, True), TokenSub('spread', 1, True, False)],
                            'cal': [TokenSub('calendar', 1.5, True, True)],
                            'hrw': [TokenSub('hr', 1.5, True, True), TokenSub('wheat', 1.5, True, True)],
                            'icso': [TokenSub('intercommodity', 1.5, True, True), TokenSub('spread', 1, True, True)],
                            }

    CBOT_MULTI_MATCH = {('10-YR NOTE', 'Interest Rate', 'Options'): ['andnot', ('', 'ultra')],
                        ('5-YR NOTE', 'Interest Rate', 'Options'): ['andnot', ('', 'ultra')],
                        ('2-YR NOTE', 'Interest Rate', 'Options'): ['andnot', ('', 'ultra')],
                        ('FED FUND', 'Interest Rate', 'Options'): ['and'],
                        ('ULTRA T-BOND', 'Interest Rate', 'Options'): ['and'],
                        ('Ultra 10-Year Note', 'Interest Rate', 'Options'): ['and'],
                        ('FERTILIZER PRODUCS', 'Ag Products', 'Futures'): ['every'],
                        ('DEC-JULY WHEAT CAL SPRD', 'Ag Products', 'Options'): ['and'],
                        ('JULY-DEC WHEAT CAL SPRD', 'Ag Products', 'Options'): ['and'],
                        ('MAR-JULY WHEAT CAL SPRD', 'Ag Products', 'Options'): ['and'],
                        ('Intercommodity Spread', 'Ag Products', 'Options'): ['andnot', ('', 'wheat-corn')]}

    CBOT_COMMON_WORDS = ['futures', 'future', 'options', 'option']

    # endregion

    REGTK_EXP = '[^\s/\(\)]+'
    REGEX_TKN = RegexTokenizerExtra(REGTK_EXP, ignored=False, required=False)
    SPLT_MRG_FLT = SplitMergeFilter(splitcase=True, splitnums=True, mergewords=True, mergenums=True)
    LWRCS_FLT = LowercaseFilter()

    # region CME index settings
    CME_STP_FLT = StopFilter(stoplist=STOP_LIST + CME_COMMON_WORDS, minsize=1)
    CME_SP_FLT = SpecialWordFilter(CME_KEYWORD_MAPPING)
    CME_VW_FLT = VowelFilter(CME_KYWRD_EXCLU, lift_ignore=False)
    CME_MULT_FLT = MultiFilterFixed(index=CME_VW_FLT)

    CME_PDNM_ANA = REGEX_TKN | SPLT_MRG_FLT | LWRCS_FLT | CME_STP_FLT | CME_SP_FLT | CME_MULT_FLT | CME_STP_FLT
    INDEX_FIELDS_CME = {F_PRODUCT_NAME: TEXT(stored=True, analyzer=CME_PDNM_ANA),
                        F_PRODUCT_GROUP: ID(stored=True, unique=True),
                        F_CLEARED_AS: ID(stored=True, unique=True),
                        F_CLEARING: ID(stored=True, unique=True),
                        F_GLOBEX: ID(stored=True, unique=True),
                        F_SUB_GROUP: ID(stored=True, unique=True),
                        F_EXCHANGE: ID}
    # endregion

    # region cbot index settings
    CBOT_STP_FLT = StopFilter(stoplist=STOP_LIST + CBOT_COMMON_WORDS, minsize=1)
    CBOT_SP_FLT = SpecialWordFilter(CBOT_SPECIAL_MAPPING)
    CBOT_VW_FLT = VowelFilter(lift_ignore=False)
    CBOT_MULT_FLT = MultiFilterFixed(index=CBOT_VW_FLT)

    CBOT_PDNM_ANA = REGEX_TKN | SPLT_MRG_FLT | LWRCS_FLT | CBOT_STP_FLT | CBOT_SP_FLT | CBOT_VW_FLT | CBOT_STP_FLT

    INDEX_FIELDS_CBOT = {F_PRODUCT_NAME: TEXT(stored=True, analyzer=CBOT_PDNM_ANA),
                         F_PRODUCT_GROUP: ID(stored=True, unique=True),
                         F_CLEARED_AS: ID(stored=True, unique=True),
                         F_CLEARING: ID(stored=True, unique=True),
                         F_GLOBEX: ID(stored=True, unique=True),
                         F_SUB_GROUP: ID(stored=True, unique=True),
                         F_EXCHANGE: ID}

    # endregion

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
        product = product.replace(der_type, '') if last_word(product) == der_type else product
        return product.rstrip()

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
        wds1 = Matcher.get_words(s1)
        wds2 = Matcher.get_words(s2)
        if len(wds1) == 1 and len(wds2) == 1:
            return s1 == s2 or Matcher.match_sgl_plrl(wds1[0], wds2[0]) \
                   or Matcher.match_first_n(wds1[0], wds2[0])
        else:
            return s1 == s2 or Matcher.match_any_word(s1, s2) \
                   or Matcher.match_initials(s1, s2) or Matcher.match_first_n(s1, s2)

    def __match_in_string(self, guess, indexed, one=True):
        guess = guess.lower()
        p = inflect.engine()

        for idx in indexed:
            idx_words = Matcher.get_words(idx.lower()) if one else [idx.lower()]
            matched = any(i in guess if len(i) < 3 else i in guess or p.plural(i) in guess
                          for i in idx_words)
            if matched:
                return idx
        return None

    def __verify_clearedas(self, prodname, row_clras, clras_set):
        clras = Matcher.get_words(prodname)[-1]
        found_clras = self.__match_in_string(clras, clras_set)
        return found_clras if found_clras is not None else row_clras

    def __get_query_params(self, row, pd_id, mapping, prods_pdgps, prods_clras, prods_subgps):
        pdnm = mapping[pd_id] if pd_id in mapping else row[self.PRODUCT]
        pdgp = dtsp.find_first_n(prods_pdgps, lambda x: self.__match_pdgp(x, row[self.PRODUCT_GROUP]))
        clras = self.__verify_clearedas(pdnm, row[self.CLEARED_AS], prods_clras)
        subgp = self.__match_in_string(pdnm, prods_subgps[pdgp], False)
        return pdnm, pdgp, clras, subgp

    def match_prod_code(self, df_adv, ix, exact_mapping='', notfound=set(), multi_match=set()):
        df_matched = pd.DataFrame(columns=list(df_adv.columns) + ix.schema.names())
        with ix.searcher() as searcher:
            lexicons = WhooshSnippet.get_idx_lexicon(
                searcher, self.F_PRODUCT_GROUP, self.F_CLEARED_AS, **{self.F_PRODUCT_GROUP: self.F_SUB_GROUP})
            prods_pdgps, prods_clras, prods_subgps = \
                lexicons[self.F_PRODUCT_GROUP], lexicons[self.F_CLEARED_AS], lexicons[self.F_SUB_GROUP]
            schema = ix.schema
            # field_pdnm = {fn: fd for fn, fd in schema.items()}[self.F_PRODUCT_NAME]
            adsearch = AdvSearch(searcher)

            for i, row in df_adv.iterrows():
                pd_id = (row[self.PRODUCT], row[self.PRODUCT_GROUP], row[self.CLEARED_AS])
                results, min_dist, one_or_all, pdnm = None, True, None, None

                def callback(val):
                    min_dist = val

                if pd_id not in notfound:
                    one_or_all = 'all' if pd_id in multi_match else 'one'
                    pdnm, pdgp, clras, subgp = self.__get_query_params(
                        row, pd_id, exact_mapping, prods_pdgps, prods_clras, prods_subgps)
                    grouping_q = WhooshSnippet.filter_query(
                        (self.F_PRODUCT_GROUP, pdgp), (self.F_CLEARED_AS, clras), (self.F_SUB_GROUP, subgp))

                    # and_words, or_words = WhooshSnippet.tokenize_split(field_pdnm, pdnm, lambda x: x.required)
                    qparams1 = (self.F_PRODUCT_NAME, schema, pdnm)
                    qparams2 = (self.F_PRODUCT_NAME, schema, pdnm, lambda x: x.required)


                    if one_or_all == 'one':
                        src_and = adsearch.search(WhooshSnippet.exact_and_query, qparams1, lambda: callback(True),
                                                  filter=grouping_q, limit=None)
                        src_fuzzy = adsearch.search(WhooshSnippet.fuzzy_and_query, qparams1, lambda: callback(False),
                                                    filter=grouping_q, limit=None)
                        src_andmaybe = adsearch.search(WhooshSnippet.andmaybe_query, qparams2, lambda: callback(True),
                                                       filter=grouping_q, limit=None)
                        results = adsearch.chain_search([src_and, src_fuzzy, src_andmaybe])
                    else:
                        q_configs = multi_match[pd_id]


                    # if not and_words or one_or_all == 'all':
                    #     q_or = WhooshSnippet.exact_or_query(*qparams)
                    #     results = adsearch.chain_search(q_and, init=True, callback=lambda: callback(True),
                    #                                     filter=grouping_q, limit=None)\
                    #         .chain_search(q_fuzzy, callback=lambda: callback(False), filter=grouping_q, limit=None)\
                    #         .chain_search(q_or, callback=lambda: callback(True), filter=grouping_q, limit=None)\
                    #         .results
                    # else:
                    #     q_andmaybe = WhooshSnippet.andmaybe_query(self.F_PRODUCT_NAME, and_words, or_words)
                    #     results = adsearch.chain_search(q_andmaybe, init=True, callback=lambda: callback(True),
                    #                                     filter=grouping_q, limit=None)\
                    #         .chain_search(q_fuzzy, callback=lambda: callback(False), filter=grouping_q, limit=None)\
                    #         .results
                if results:
                    if one_or_all == 'one' and min_dist:
                        results = min_dist_rslt(results, pdnm, self.F_PRODUCT_NAME, schema, minboost=0.2)
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




    def __join_results(self, results, *dfs, how='one'):
        joined_dict = [results[0].fields()] if how == 'one' else [r.fields() for r in results]
        if dfs is not None:
            for df in dfs:
                for jd in joined_dict:
                    jd.update(df)
        return joined_dict


    def __match_prods_by_commodity(self, df_prods, df_adv):
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

    def init_ix_cme_cbot(self, clean=False):
        df_prods = self.__from_prods(self.prods_file)
        df_ix = df_prods.rename(columns=self.COL2FIELD)
        gdf_exch = {exch: df.reset_index(drop=0) for exch, df in self.__groupby(df_ix, [self.EXCHANGE]).items()}
        ix_cme = setup_ix(self.INDEX_FIELDS_CME, gdf_exch[self.CME], self.index_cme, clean)
        ix_cbot = setup_ix(self.INDEX_FIELDS_CBOT, gdf_exch[self.CBOT], self.index_cbot, clean)
        return ix_cme, ix_cbot, gdf_exch

    def match_nymex_comex(self, df_adv, gdf_exch):
        df_nymex_comex_prods = gdf_exch[self.NYMEX].append(gdf_exch[self.COMEX], ignore_index=True)
        df_nymex_comex_prods.reset_index(drop=True, inplace=True)
        if 'index' in df_nymex_comex_prods.columns:
            df_nymex_comex_prods.drop(['index'], axis=1, inplace=True)
        df_adv = self.__match_prods_by_commodity(df_nymex_comex_prods, df_adv)
        cp.XlsxWriter.save_sheets(self.matched_file, {self.NYMEX: df_adv}, override=True)

    def run_pd_mtch(self, outpath=None, clean=False):
        dfs_adv = {self.CME: self.__from_adv(self.adv_cme, self.COLS_ADV),
                   self.CBOT: self.__from_adv(self.adv_cbot, self.COLS_ADV),
                   self.NYMEX: self.__from_adv(self.adv_nymex_comex, self.COLS_ADV + [self.COMMODITY])}

        ix_cme, ix_cbot, gdf_exch = self.init_ix_cme_cbot(clean)
        self.match_nymex_comex(dfs_adv[self.NYMEX], gdf_exch)

        mdf_cme = self.match_prod_code(dfs_adv[self.CME], ix_cme, self.CME_EXACT_MAPPING, self.CME_NOTFOUND_PRODS, self.CME_MULTI_MATCH)
        mdf_cbot = self.match_prod_code(dfs_adv[self.CBOT], ix_cbot, self.CBOT_EXACT_MAPPING, multi_match=self.CBOT_MULTI_MATCH)

        outpath = self.matched_file if outpath is None else outpath
        cp.XlsxWriter.save_sheets(outpath, {self.CME: mdf_cme, self.CBOT: mdf_cbot}, override=False)


# checked_path = os.getcwd()
#
# exchanges = ['asx', 'bloomberg', 'cme', 'cbot', 'nymex_comex', 'eurex', 'hkfe', 'ice', 'ose', 'sgx']
# report_fmtname = 'Web_ADV_Report_{}.xlsx'
#
# report_files = {e: report_fmtname.format(e.upper()) for e in exchanges}
#
# cme_prds_file = os.path.join(checked_path, 'Product_Slate.xls')
# cme_adv_files = [os.path.join(checked_path, report_files['cme']),
#                  os.path.join(checked_path, report_files['cbot']),
#                  os.path.join(checked_path, report_files['nymex_comex'])]
#
# cme = CMEGMatcher(cme_adv_files, cme_prds_file, '2017')
# cme.run_pd_mtch(clean=True)

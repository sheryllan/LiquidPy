import re
from whoosh.analysis import *
import unittest as ut
import pandas as pd
import itertools

from whooshext import *
from whoosh.index import open_dir
from whoosh.query import *
from whoosh import qparser

import datascraper as dtsp
from productmatcher import CMEGMatcher


class CMEAnalyzerTests(ut.TestCase):
    STOP_LIST = ['and', 'is', 'it', 'an', 'as', 'at', 'have', 'in', 'yet', 'if', 'from', 'for', 'when',
                 'by', 'to', 'you', 'be', 'we', 'that', 'may', 'not', 'with', 'tbd', 'a', 'on', 'your',
                 'this', 'of', 'will', 'can', 'the', 'or', 'are']

    CRRNCY_TOKENSUB = {'aud': [TokenSub('australian', 1.5, True, True), TokenSub('dollar', 1, True, True)],
                       'gbp': [TokenSub('british', 1.5, True, True), TokenSub('pound', 1.5, True, True)],
                       'cad': [TokenSub('canadian', 1.5, True, True), TokenSub('dollar', 1, True, True)],
                       'euro': [TokenSub('euro', 1.5, True, False)],
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
                       'usd': [TokenSub('us', 1, True, False), TokenSub('dollar', 1, True, False)],
                       'clp': [TokenSub('chilean', 1.5, True, True), TokenSub('peso', 1.5, True, True)],
                       'mxn': [TokenSub('mexican', 1.5, True, True), TokenSub('peso', 1.5, True, True)],
                       'brl': [TokenSub('brazilian', 1.5, True, True), TokenSub('real', 1.5, True, True)],
                       'huf': [TokenSub('hungarian', 1.5, True, True), TokenSub('forint', 1.5, True, True)]
                       }

    CRRNCY_MAPPING = {'ad': CRRNCY_TOKENSUB['aud'],
                      'bp': CRRNCY_TOKENSUB['gbp'],
                      'cd': CRRNCY_TOKENSUB['cad'],
                      'ec': CRRNCY_TOKENSUB['euro'] + [TokenSub('cross', 0.5, True, False),
                                                       TokenSub('rates', 0.5, True, False)],
                      'efx': CRRNCY_TOKENSUB['euro'] + [TokenSub('fx', 0.8, True, False)],
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

    CME_SPECIAL_MAPPING = {'midcurve': [TokenSub('midcurve', 1, True, False), TokenSub('mc', 1.5, True, True)],
                           'pqo': [TokenSub('premium', 1, True, True), TokenSub('quoted', 1, True, True),
                                   TokenSub('european', 1, True, True), TokenSub('style', 1, True, True),
                                   TokenSub('options', 0.5, True, True)],
                           'eow': [TokenSub('weekly', 1, True, True), TokenSub('wk', 1, True, False)],
                           'eom': [TokenSub('monthly', 1, True, True)],
                           'usdzar': CRRNCY_TOKENSUB['usd'] + CRRNCY_TOKENSUB['zar'],
                           'biotech': [TokenSub('biotechnology', 1.5, True, True)],
                           'american': [TokenSub('us', 1, True, False)],
                           'eu': [TokenSub('european', 1.5, True, True)],
                           'nfd': [TokenSub('non', 1.5, True, True), TokenSub('fat', 1.5, True, True),
                                   TokenSub('dry', 1.5, True, True)],
                           'cs': [TokenSub('cash', 1.5, True, True), TokenSub('settled', 1.5, True, True)],
                           'er': [TokenSub('excess', 1.5, True, True), TokenSub('return', 1.5, True, True)]}


    CME_COMMON_WORDS = ['futures', 'options', 'index', 'cross', 'rate', 'rates']

    CRRNCY_KEYWORDS = set(dtsp.flatten_iter(
        [k.split(' ') + [tp.text for tp in v] for k, v in CRRNCY_MAPPING.items()]))

    CME_SPECIAL_KEYWORDS = set(dtsp.flatten_iter(
        [k.split(' ') + [tp.text for tp in v] for k, v in CME_SPECIAL_MAPPING.items()]))

    CME_KEYWORD_MAPPING = {**CRRNCY_MAPPING, **CME_SPECIAL_MAPPING}

    # CME_KYWRD_EXCLU = CRRNCY_KEYWORDS.union(CME_SPECIAL_KEYWORDS).union(
    #     {'nasdaq', 'ibovespa', 'index', 'mini', 'micro', 'nikkei', 'russell', 'ftse', 'swap'})
    CME_KYWRD_EXCLU = {'nasdaq', 'ibovespa', 'index', 'mini', 'micro', 'nikkei', 'russell', 'ftse', 'swap'}


    # def test_spflt(self):
    #     # regex = RegexTokenizer('[^\s/]+')
    #     regex = RegexTokenizerExtra('[^\s/]+', ignored=False, required=False)
    #     # attrflt = TokenAttrFilter(ignored=False, required=False)
    #     lwrflt = LowercaseFilter()
    #     spflt = SpecialWordFilter(self.CME_KEYWORD_MAPPING)
    #     vwflt = VowelFilter(self.CME_KYWRD_EXCLU)
    #     ana = regex | lwrflt | spflt
    #
    #     testcase1 = 'Swiss Franc CHF/USD PQO 2pm Fix'
    #
    #     # out_regex = [(t.text, t.boost, t.ignored, t.required) for t in regex(testcase1, ignored=False, required=False)]
    #     actual1 = [(t.text, t.boost, t.ignored, t.required) for t in ana(testcase1)]
    #     print(actual1)


    # def test_splt_mrg_filter(self):
    #     spltmrgflt = SplitMergeFilter(splitcase=True, splitnums=True, mergewords=True, mergenums=True)
    #     regex = RegexTokenizer('[^\s/]+')
    #     ana = regex | spltmrgflt
    #     testcase1 = 'ciEsta0-8-9 JiO890&9cityETO(MIOm'
    #     testcase5 = 'U.S. Dollar/S.African Rand'
    #     testcase6 = 'FT-SE 100'
    #
    #     actual1 = [(t.text, t.boost)for t in ana(testcase1)]
    #     actual5 = [(t.text, t.boost) for t in ana(testcase5)]
    #     actual6 = [(t.text, t.boost) for t in ana(testcase6)]
    #
    #     expected1 = [('ci', 1/6), ('Esta', 1/6), ('089', 5/12), ('ciEsta', 1/4), ('Ji', 1/10), ('O', 1/10), ('8909', 8/30), ('city', 1/10), ('ETOMIOm', 1/10), ('JiO', 1/6), ('cityETOMIOm', 1/6)]
    #     expected6 = [('FT', 0.25), ('SE', 0.25), ('FTSE', 0.5), ('100', 1.0)]
    #
    #     print(actual1)
    #     print(actual5)
    #     print(actual6)
    #     # self.assertListEqual(expected1, actual1)
    #     self.assertListEqual(expected6, actual6)


    def test_analyzer(self):
        REGEX_TKN = RegexTokenizerExtra('[^\s/]+', ignored=False, required=False)
        SPLT_MRG_FLT = SplitMergeFilter(splitcase=True, splitnums=True, mergewords=True, mergenums=True)
        LWRCS_FLT = LowercaseFilter()

        CME_STP_FLT = StopFilter(stoplist=self.STOP_LIST + self.CME_COMMON_WORDS, minsize=1)
        CME_SP_FLT = SpecialWordFilter(self.CME_KEYWORD_MAPPING)
        CME_VW_FLT = VowelFilter(self.CME_KYWRD_EXCLU)
        CME_MULT_FLT = MultiFilterFixed(index=CME_VW_FLT)

        ana = REGEX_TKN | SPLT_MRG_FLT | LWRCS_FLT | CME_STP_FLT | CME_SP_FLT | CME_MULT_FLT
        # ana = REGEX_TKN | SPLT_MRG_FLT | LWRCS_FLT | CME_SP_FLT | CME_VW_FLT | STP_FLT
        # ana = REGEX_TKN | IntraWordFilter(mergewords=True)

        testcase1 = 'Nikkei/USD'
        testcase2 = ' EOW1 S&P 500'
        testcase3 = ' AUD/USD PQO 2pm Fix'
        testcase4 = 'E-mini NASDAQ Biotechnology Index'
        testcase5 = 'Nikkei/USD Futures'
        testcase6 = 'S.AFRICAN RAND'
        testcase7 = 'Chilean Peso/US Dollar (CLP/American Dollar) Futures'

        # result1 = [t.text for t in ana(testcase1)]
        # result2 = [t.text for t in ana(testcase2)]
        # result3 = [t.text for t in ana(testcase3)]
        # result4 = [t.text for t in ana(testcase4, mode='index')]
        # result5 = [t.text for t in ana(testcase5, mode='index')]
        # result6 = [t.text for t in ana(testcase6, mode='index')]
        result7 = [(t.text, t.boost, t.ignored, t.required) for t in ana(testcase7, mode='index')]


        # print(result6)
        # print(result5)
        print(result7)
        # print(result4)
        # expected1 = ['e', 'micro', 'emicro', 'aud', 'australian', 'dollar', 'usd', 'us', 'american', 'dollar']
        # expected4 = ['e', 'mini', 'emini', 'nasdaq', 'biotechnology', 'btchnlgy', 'index']
        #
        # self.assertListEqual(expected1, result1)
        # self.assertListEqual(expected4, result4)


    # def test_composite_filter(self):
    #     spltflt = SplitFilter(origin=False, mergewords=True, mergenums=True)
    #     cf = SpecialWordFilter(self.CME_KEYWORD_MAPPING) & VowelFilter(self.CME_KYWRD_EXCLU)
    #     ana = STD_ANA | spltflt | cf
    #
    #     testcase4 = 'E-mini NASDAQ Biotechnology Index'
    #     result4 = [t.text for t in ana(testcase4)]
    #
    #     expected4 = ['e', 'mini', 'emini', 'nasdaq', 'biotechnology', 'index']
    #
    #     self.assertListEqual(expected4, result4)

    # def test_ana_query_mode(self):
    #     # SPLT_FLT = SplitFilter(delims='[&/\(\)\.-]', splitcase=True, splitnums=True, mergewords=True, mergenums=True)
    #     # CME_SP_FLT = SpecialWordFilter(self.CME_KEYWORD_MAPPING)
    #     # CME_VW_FLT = VowelFilter(self.CME_KYWRD_EXCLU)
    #     # CME_PDNM_ANA = RegexTokenizer('[^\s/]+') | SPLT_FLT
    #
    #     # testcase = 'EOW1 E-MINI RUSSELL 2000 WE'
    #     # result = [t.text for t in CME_PDNM_ANA(testcase, mode='index')]
    #
    #     # print(result)
    #
    #     F_PRODUCT_NAME = 'Product_Name'
    #
    #     ix = open_dir('CME_Product_Index')
    #     pdnm = 'Nikkei/USD Futures'
    #     query = self.__exact_and_query(F_PRODUCT_NAME, ix.schema, pdnm)
    #     print(query.terms())
    #
    # def __exact_and_query(self, field, schema, text):
    #     parser = qparser.QueryParser(field, schema=schema)
    #     return parser.parse(text)
    #
    # def __exact_or_query(self, field, schema, text):
    #     og = qparser.OrGroup.factory(0.9)
    #     parser = qparser.QueryParser(field, schema=schema, group=og)
    #     return parser.parse(text)

    # def test_min_dist_rslt(self):
    #     f_pn = 'Product_Name'
    #     f_pg = 'Product_Group'
    #     f_ca = 'Cleared_As'
    #
    #     cmeg = CMEGMatcher()
    #     index_cme = cmeg.INDEX_CME
    #     # cmeg.init_ix_cme_cbot(True)
    #
    #     ######## case 1 ##########
    #     qstring = 'S&P MATERIALS SECTOR'
    #     pdgp = 'Equity Index'
    #     ca = 'Futures'
    #
    #     ix = open_dir(index_cme)
    #     grouping_q = And([Term(f_pg, pdgp), Term(f_ca, ca)])
    #     query = self.__exact_or_query(f_pn, ix.schema, qstring)
    #     searcher = ix.searcher()
    #     results = searcher.search(query, filter=grouping_q, limit=None)
    #
    #     fields = {x[0]: x[1] for x in ix.schema.items()}
    #     best_result = min_dist_rslt(results, qstring, f_pn, fields[f_pn])
    #     print(best_result.fields())
    #     searcher.close()
    #
    #     ######## case 2 ##########
    #     qstring = 'EC/AD CROSS RATES'
    #     pdgp = 'FX'
    #     ca = 'Futures'
    #
    #     ix = open_dir('CME_Product_Index')
    #     grouping_q = And([Term(f_pg, pdgp), Term(f_ca, ca)])
    #     query = self.__exact_or_query(f_pn, ix.schema, qstring)
    #     searcher = ix.searcher()
    #     results = searcher.search(query, filter=grouping_q, limit=None)
    #
    #     fields = {x[0]: x[1] for x in ix.schema.items()}
    #     best_result = min_dist_rslt(results, qstring, f_pn, fields[f_pn])
    #     expected = {f_pn: 'Euro/Australian Dollar',
    #                 f_pg: 'FX',
    #                 f_ca: 'Futures',
    #                 'Clearing': 'CA',
    #                 'Globex': 'EAD',
    #                 'Sub_Group': 'Cross Rates'}
    #     print(best_result.fields())
    #     print(expected)
    #     self.assertEqual(expected, best_result.fields())
    #     searcher.close()

    # def test_treemap(self):
    #     tm = TreeMap()
    #     head = tm.add(((0, 1), 'jiowf'))
    #     head = tm.add(((0, 0.5), 'kf2m2['), head)
    #     head = tm.add(((6, 4), 'j2093vj'), head)
    #     head = tm.add(((4, 9.65), 'jwuf0wu'), head)
    #
    #     items = tm.get_items(head)
    #     expected = [((0, 0.5), 'kf2m2['),
    #                 ((0, 1), 'jiowf'),
    #                 ((4, 9.65), 'jwuf0wu'),
    #                 ((6, 4), 'j2093vj')]
    #
    #     print(items)
    #     self.assertListEqual(expected, items)





import re
from whoosh.analysis import *
import unittest as ut
import pandas as pd

from whooshext import *
from whoosh.index import open_dir
from whoosh.query import *
from whoosh import qparser

import datascraper as dtsp
from productmatcher import CMEGMatcher


class CMEAnalyzerTests(ut.TestCase):
    CRRNCY_MAPPING = {'ad': [TokenSub('australian', 1.5, True), TokenSub('dollar', 1, True)],
                      'bp': [TokenSub('british', 1.5, True), TokenSub('pound', 1.5, True)],
                      'cd': [TokenSub('canadian', 1.5, True), TokenSub('dollar', 1, True)],
                      'ec': [TokenSub('euro', 1.5, True), TokenSub('cross', 0.5, True),
                             TokenSub('rates', 0.5, True)],
                      'efx': [TokenSub('euro', 1.5, True), TokenSub('fx', 0.8, True)],
                      'jy': [TokenSub('japanese', 1.5, True), TokenSub('yen', 1.5, True)],
                      'jpy': [TokenSub('japanese', 1.5, True), TokenSub('yen', 1.5, True)],
                      'ne': [TokenSub('new', 1.5, True), TokenSub('zealand', 1.5, True),
                             TokenSub('dollar', 1, True)],
                      'nok': [TokenSub('norwegian', 1.5, True), TokenSub('krone', 1.5, True)],
                      'sek': [TokenSub('swedish', 1.5, True), TokenSub('krona', 1.5, True)],
                      'sf': [TokenSub('swiss', 1.5, True), TokenSub('franc', 1.5, True)],
                      'skr': [TokenSub('swedish', 1.5, True), TokenSub('krona', 1.5, True)],
                      'zar': [TokenSub('south', 1.5, True), TokenSub('african', 1.5, True),
                              TokenSub('rand', 1.5, True)],
                      'aud': [TokenSub('australian', 1.5, True), TokenSub('dollar', 1, True)],
                      'cad': [TokenSub('canadian', 1.5, True), TokenSub('dollar', 1, True)],
                      'chf': [TokenSub('swiss', 1.5, True), TokenSub('franc', 1.5, True)],
                      'eur': [TokenSub('euro', 1.5, True)],
                      'gbp': [TokenSub('british', 1.5, True), TokenSub('pound', 1.5, True)],
                      'pln': [TokenSub('polish', 1.5, True), TokenSub('zloty', 1.5, True)],
                      'nkr': [TokenSub('norwegian', 1.5, True), TokenSub('krone', 1.5, True)],
                      'inr': [TokenSub('indian', 1.5, True), TokenSub('rupee', 1.5, True)],
                      'rmb': [TokenSub('chinese', 1.5, True), TokenSub('renminbi', 1.5, True)],
                      'usd': [TokenSub('us', 0.75, True), TokenSub('american', 0.75, True),
                              TokenSub('dollar', 0.5, True)]}

    # CRRNCY_KEYWORDS = list(set(dtsp.flatten_list(
    #     [k.split(' ') + [tp[0] for tp in v] for k, v in CRRNCY_MAPPING.items()], list())))

    CME_SPECIAL_MAPPING = {'midcurve': [TokenSub('midcurve', 1, True), TokenSub('mc', 1.5, True)],
                           'pqo': [TokenSub('premium', 1, True), TokenSub('quoted', 1, True),
                                   TokenSub('european', 1, True), TokenSub('style', 1, True),
                                   TokenSub('options', 0.5, True)],
                           'eow': [TokenSub('weekly', 1, True), TokenSub('wk', 1, True)],
                           'eom': [TokenSub('monthly', 1, True)],
                           'usdzar': [TokenSub('us', 0.75, True), TokenSub('american', 0.75, True),
                                      TokenSub('dollar', 0.5, True), TokenSub('south', 1, True),
                                      TokenSub('african', 1, True), TokenSub('rand', 1, True)],
                           'biotech': [TokenSub('biotechnology', 1.5, True)],
                           '$': [TokenSub('us', 1, True), TokenSub('american', 1, True), TokenSub('dollar', 1, True)],
                           'eu': [TokenSub('european', 1.5, True)]}

    CME_COMMON_WORDS = ['futures', 'options', 'index', 'cross', 'rate', 'rates']

    # CME_SPECIAL_KEYWORDS = list(set(dtsp.flatten_list(
    #     [k.split(' ') + [tp[0] for tp in v] for k, v in CME_SPECIAL_MAPPING.items()], list())))

    CME_KEYWORD_MAPPING = {**CRRNCY_MAPPING, **CME_SPECIAL_MAPPING}

    CME_KYWRD_EXCLU = ['nasdaq', 'ibovespa', 'index', 'mini',
                       'micro', 'nikkei', 'russell', 'ftse',
                       ]

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
        REGEX_TKN = RegexTokenizer('[^\s/]+')
        SPLT_MRG_FLT = SplitMergeFilter(splitcase=True, splitnums=True, mergewords=True, mergenums=True)

        LWRCS_FLT = LowercaseFilter()
        STP_FLT = StopFilter(stoplist=STOP_LIST + self.CME_COMMON_WORDS, minsize=1)
        CME_SP_FLT = SpecialWordFilter(self.CME_KEYWORD_MAPPING)
        CME_VW_FLT = VowelFilter(self.CME_KYWRD_EXCLU)

        ana = REGEX_TKN | SPLT_MRG_FLT | LWRCS_FLT | CME_SP_FLT | CME_VW_FLT | STP_FLT
        # ana = REGEX_TKN | IntraWordFilter(mergewords=True)

        testcase1 = 'Nikkei/USD'
        testcase2 = ' EOW1 S&P 500'
        testcase3 = ' AUD/USD PQO 2pm Fix'
        testcase4 = 'E-mini NASDAQ Biotechnology Index'
        testcase5 = 'NIKKEI 225 ($) STOCK'
        testcase6 = 'S.AFRICAN RAND'

        # result1 = [t.text for t in ana(testcase1)]
        # result2 = [t.text for t in ana(testcase2)]
        # result3 = [t.text for t in ana(testcase3)]
        # result4 = [t.text for t in ana(testcase4)]
        # result5 = [t.text for t in ana(testcase5, mode='query')]
        result6 = [t.text for t in ana(testcase6, mode='index')]

        print(result6)
        # print(result5)
        # print(result1)
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
    #     SPLT_FLT = SplitFilter(delims='[&/\(\)\.-]', splitcase=True, splitnums=True, mergewords=True, mergenums=True)
    #     CME_SP_FLT = SpecialWordFilter(self.CME_KEYWORD_MAPPING)
    #     CME_VW_FLT = VowelFilter(self.CME_KYWRD_EXCLU)
    #     CME_PDNM_ANA = RegexTokenizer('[^\s/]+') | SPLT_FLT
    #
    #     testcase = 'EOW1 E-MINI RUSSELL 2000 WE'
    #     result = [t.text for t in CME_PDNM_ANA(testcase, mode='index')]
    #
    #     print(result)

        # F_PRODUCT_NAME = 'Product_Name'
        #
        # ix = open_dir('CME_Product_Index')
        # pdnm = 'EOW1 E-MINI RUSSELL 2000 WE'
        # query = self.__exact_and_query(F_PRODUCT_NAME, ix.schema, pdnm)
        # print()

    def __exact_and_query(self, field, schema, text):
        parser = qparser.QueryParser(field, schema=schema)
        return parser.parse(text)

    def __exact_or_query(self, field, schema, text):
        og = qparser.OrGroup.factory(0.9)
        parser = qparser.QueryParser(field, schema=schema, group=og)
        return parser.parse(text)

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

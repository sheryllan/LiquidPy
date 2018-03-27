import re
from whoosh.analysis import *
import unittest as ut
import pandas as pd

from whooshext import *
from whoosh.index import open_dir
from whoosh.query import *
from whoosh import qparser

import datascraper as dtsp


class CMEAnalyzerTests(ut.TestCase):
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

    CME_SPECIAL_MAPPING = {'midcurve': ('midcurve mc', 1.2),
                           'pqo': ('premium quoted european style options', 1.2),
                           'eow': ('weekly wk', 1.2),
                           'eom': ('monthly', 1.2),
                           'usdzar': ('us dollar south african rand', 1.2),
                           'biotech': ('biotechnology', 1.2),
                           '$': ('us dollar', 1.2)
                           }

    CME_COMMON_WORDS = {'cross': ('cross', 0.6),
                        'rates': ('rates', 0.6),
                        'rate': ('rate', 0.6),
                        'futures': ('futures', 0),
                        'options': ('options', 0),
                        'index': ('index', 0)}

    CME_SPECIAL_KEYWORDS = list(set(dtsp.flatten_list(
        [[k.split(' '), v[0].split(' ')] for k, v in CME_SPECIAL_MAPPING.items()], list())))

    CME_KEYWORD_MAPPING = {**CRRNCY_MAPPING, **CME_SPECIAL_MAPPING, **CME_COMMON_WORDS}

    CME_KYWRD_EXCLU = CRRNCY_KEYWORDS + CME_SPECIAL_KEYWORDS + \
                      ['nasdaq', 'ibovespa', 'index', 'mini', 'emini',
                       'micro', 'emicro', 'nikkei', 'russell', 'ftse',
                       'european']

    def test_analyzer(self):
        spltflt = SplitFilter(delims='[&/\(\)-]', origin=False, mergewords=True, mergenums=True)
        ana = STD_ANA | spltflt | SpecialWordFilter(self.CME_KEYWORD_MAPPING) | VowelFilter(self.CME_KYWRD_EXCLU)

        testcase1 = 'Nikkei/USD'
        testcase2 = ' EOW1 S&P 500'
        testcase3 = ' AUD/USD PQO 2pm Fix'
        testcase4 = 'E-mini NASDAQ Biotechnology Index'
        testcase5 = 'NIKKEI 225 ($) STOCK'

        result1 = [t.text for t in ana(testcase1)]
        # result2 = [t.text for t in ana(testcase2)]
        # result3 = [t.text for t in ana(testcase3)]
        # result4 = [t.text for t in ana(testcase4)]
        # result5 = [t.text for t in ana(testcase5)]

        # print(result5)
        print(result1)
        # expected1 = ['e', 'micro', 'emicro', 'aud', 'australian', 'dollar', 'usd', 'us', 'american', 'dollar']
        # expected4 = ['e', 'mini', 'emini', 'nasdaq', 'biotechnology', 'btchnlgy', 'index']
        #
        # self.assertListEqual(expected1, result1)
        # self.assertListEqual(expected4, result4)

    def test_composite_filter(self):
        spltflt = SplitFilter(origin=False, mergewords=True, mergenums=True)
        cf = SpecialWordFilter(self.CME_KEYWORD_MAPPING) & VowelFilter(self.CME_KYWRD_EXCLU)
        ana = STD_ANA | spltflt | cf

        testcase4 = 'E-mini NASDAQ Biotechnology Index'
        result4 = [t.text for t in ana(testcase4)]

        expected4 = ['e', 'mini', 'emini', 'nasdaq', 'biotechnology', 'index']

        self.assertListEqual(expected4, result4)


    # def test_ana_query_mode(self):
    #     # SPLT_FLT = SplitFilter(origin=False, mergewords=True, mergenums=True)
    #     # CME_SP_FLT = SpecialWordFilter(self.CME_KEYWORD_MAPPING)
    #     # CME_VW_FLT = VowelFilter(self.CME_KYWRD_EXCLU)
    #     # CME_PDNM_ANA = STD_ANA | SPLT_FLT | MultiFilter(index=CME_SP_FLT & CME_VW_FLT, query=CME_SP_FLT)
    #     #
    #     # testcase = 'EOW1 E-MINI RUSSELL 2000 WE'
    #     # result = [t.text for t in CME_PDNM_ANA(testcase)]
    #     #
    #     # print(result)
    #
    #     F_PRODUCT_NAME = 'Product_Name'
    #
    #     ix = open_dir('CME_Product_Index')
    #     pdnm = 'EOW1 E-MINI RUSSELL 2000 WE'
    #     query = self.__exact_and_query(F_PRODUCT_NAME, ix.schema, pdnm)
    #     print()


    def __exact_and_query(self, field, schema, text):
        parser = qparser.QueryParser(field, schema=schema)
        return parser.parse(text)

    def __exact_or_query(self, field, schema, text):
        og = qparser.OrGroup.factory(0.9)
        parser = qparser.QueryParser(field, schema=schema, group=og)
        return parser.parse(text)
    #
    # def test_min_dist_rslt(self):
    #     ix = open_dir('CME_Product_Index')
    #     qstring = 'EC/AD CROSS RATES'
    #     f_pn = 'Product_Name'
    #     f_pg = 'Product_Group'
    #     f_ca = 'Cleared_As'
    #     pdgp = 'FX'
    #     ca = 'Futures'
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


    def test_treemap(self):
        tm = TreeMap()
        head = tm.add(((0, 1), 'jiowf'))
        head = tm.add(((0, 0.5), 'kf2m2['), head)
        head = tm.add(((6, 4), 'j2093vj'), head)
        head = tm.add(((4, 9.65), 'jwuf0wu'), head)

        items = tm.get_items(head)
        expected = [((0, 0.5), 'kf2m2['),
                    ((0, 1), 'jiowf'),
                    ((4, 9.65), 'jwuf0wu'),
                    ((6, 4), 'j2093vj')]

        print(items)
        self.assertListEqual(expected, items)



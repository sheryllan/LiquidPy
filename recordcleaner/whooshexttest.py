import re
from whoosh.analysis import *
import unittest as ut
import pandas as pd
import itertools

from whooshext import *
from whoosh.index import open_dir
from whoosh.query import *
from whoosh import qparser
from whoosh.searching import Searcher

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
                       'usd': [TokenSub('american', 1, True, False), TokenSub('dollar', 1, True, False)],
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

    CME_SPECIAL_MAPPING = {'midcurve': [TokenSub('mc', 1.5, True, True)],
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

    CRRNCY_KEYWORDS = set(dtsp.flatten_iter(
        [k.split(' ') + [tp.text for tp in v] for k, v in CRRNCY_MAPPING.items()]))

    CME_SPECIAL_KEYWORDS = set(dtsp.flatten_iter(
        [k.split(' ') + [tp.text for tp in v] for k, v in CME_SPECIAL_MAPPING.items()]))

    CME_KEYWORD_MAPPING = {**CRRNCY_MAPPING, **CME_SPECIAL_MAPPING}

    # CME_KYWRD_EXCLU = CRRNCY_KEYWORDS.union(CME_SPECIAL_KEYWORDS).union(
    #     {'nasdaq', 'ibovespa', 'index', 'mini', 'micro', 'nikkei', 'russell', 'ftse', 'swap'})
    CME_KYWRD_EXCLU = {'nasdaq', 'ibovespa', 'index', 'mini', 'micro', 'nikkei', 'russell', 'ftse', 'swap'}

    REGTK_EXP = '[^\s/\(\)]+'


    # def test_spflt(self):
    #     # regex = RegexTokenizer(self.REGTK_EXP)
    #     regex = RegexTokenizerExtra(self.REGTK_EXP, ignored=False, required=False)
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
    #     regex = RegexTokenizer(self.REGTK_EXP)
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


    # def test_analyzer(self):
    #     REGEX_TKN = RegexTokenizerExtra(self.REGTK_EXP, ignored=False, required=False)
    #     SPLT_MRG_FLT = SplitMergeFilter(splitcase=True, splitnums=True, mergewords=True, mergenums=True)
    #     LWRCS_FLT = LowercaseFilter()
    #
    #     CME_STP_FLT = StopFilter(stoplist=self.STOP_LIST + self.CME_COMMON_WORDS, minsize=1)
    #     CME_SP_FLT = SpecialWordFilter(self.CME_KEYWORD_MAPPING)
    #     CME_VW_FLT = VowelFilter(self.CME_KYWRD_EXCLU)
    #     CME_MULT_FLT = MultiFilterFixed(index=CME_VW_FLT)
    #
    #     ana = REGEX_TKN | SPLT_MRG_FLT | LWRCS_FLT | CME_STP_FLT | CME_SP_FLT | CME_MULT_FLT | CME_STP_FLT
    #     # ana = REGEX_TKN | SPLT_MRG_FLT | LWRCS_FLT | CME_STP_FLT | CME_SP_FLT | CME_MULT_FLT
    #     # ana = REGEX_TKN | SPLT_MRG_FLT | LWRCS_FLT | CME_SP_FLT | CME_VW_FLT | STP_FLT
    #     # ana = REGEX_TKN | IntraWordFilter(mergewords=True)
    #
    #     testcase1 = 'Nikkei/USD'
    #     testcase2 = ' EOW1 S&P 500'
    #     testcase3 = ' AUD/USD PQO 2pm Fix'
    #     testcase4 = 'E-mini NASDAQ Biotechnology Index'
    #     testcase5 = 'Nikkei/USD Futures'
    #     testcase6 = 'S.AFRICAN RAND'
    #     testcase7 = 'Chilean Peso/US Dollar (CLP/American Dollar) Futures'
    #     testcase8 = '(CLP/USD) Chilean Peso/US Dollar American'
    #     # testcase9 = 'EURO MIDCURVE'
    #     # testcase9 = 'BRAZIL REAL'
    #
    #     # result1 = [t.text for t in ana(testcase1)]
    #     # result2 = [t.text for t in ana(testcase2)]
    #     # result3 = [t.text for t in ana(testcase3)]
    #     # result4 = [t.text for t in ana(testcase4, mode='index')]
    #     # result5 = [t.text for t in ana(testcase5, mode='index')]
    #     # result6 = [t.text for t in ana(testcase6, mode='index')]
    #     result7 = [(t.text, t.boost, t.ignored, t.required) for t in ana(testcase7, mode='index')]
    #     result8 = [(t.text, t.boost, t.ignored, t.required) for t in ana(testcase8, mode='index')]
    #
    #     # result9 = [(t.text, t.boost, t.ignored, t.required) for t in ana(testcase9, mode='query')]
    #
    #
    #     # print(result6)
    #     # print(result5)
    #     print(result7)
    #     print(result8)
    #     # print(result9)
    #     # print(result4)
    #     # expected1 = ['e', 'micro', 'emicro', 'aud', 'australian', 'dollar', 'usd', 'us', 'american', 'dollar']
    #     # expected4 = ['e', 'mini', 'emini', 'nasdaq', 'biotechnology', 'btchnlgy', 'index']
    #     #
    #     # self.assertListEqual(expected1, result1)
    #     # self.assertListEqual(expected4, result4)


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

    def test_ana_query_mode(self):
        F_PRODUCT_NAME = 'Product_Name'
        checked_path = os.getcwd()
        cmeg_prds_file = os.path.join(checked_path, 'Product_Slate.xls')
        cmeg = CMEGMatcher(prods_file=cmeg_prds_file)

        ix_cme, ix_cbot, gdf_exch = cmeg.init_ix_cme_cbot(True)
        # field_pdnm = {x[0]: x[1] for x in ix.schema.items()}[F_PRODUCT_NAME]
        #
        # pdnm1 = 'GBP/USD PQO 2pm Fix'
        # pdnm2 = 'E-MINI EURO FX'
        #
        # rcd1 = 'Weekly Premium Quoted European Style Options on British Pound/US Dollar Futures - Wk 3'
        # rcd2 = 'E-mini Euro FX Futures'
        #
        # ana = field_pdnm.analyzer
        # tks2_index = [(t.text, t.boost, t.ignored, t.required) for t in ana(rcd2, mode='index')]
        # tks2_query = [(t.text, t.boost, t.ignored, t.required) for t in ana(pdnm2, mode='query')]
        # print(tks2_index)
        # print(tks2_query)
        # # tks_query = [t.text for t in ana(pdnm, mode='query')]
        # # print(tks_query)
        # # and_words, or_words = WhooshSnippet.tokenize_split(field_pdnm, pdnm1, lambda x: x.required)
        # # query = WhooshSnippet.andmaybe_query(F_PRODUCT_NAME, and_words, or_words)
        # # with ix.searcher() as searcher:
        # #     results = searcher.search(query)
        # #     if results:
        # #         for r in results:
        # #             print(r)
        #
        # and_words2, or_words2 = WhooshSnippet.tokenize_split(field_pdnm, pdnm2, lambda x: x.required)
        # query2 = WhooshSnippet.andmaybe_query(F_PRODUCT_NAME, and_words2, or_words2)
        #
        # with ix.searcher() as searcher:
        #     results = searcher.search(query2)
        #     if results:
        #         for r in results:
        #             print(r)

        field_pdnm = {x[0]: x[1] for x in ix_cbot.schema.items()}[F_PRODUCT_NAME]
        ana = field_pdnm.analyzer
        pdnm1 = '2-YR NOTE'
        rcd1 = '2-Year T-Note Weekly Options Wk 1'

        tks1_index = [(t.text, t.boost, t.ignored, t.required) for t in ana(rcd1, mode='index')]
        tks1_query = [(t.text, t.boost, t.ignored, t.required) for t in ana(pdnm1, mode='query')]
        print(tks1_index)
        print(tks1_query)

        pdnm2 = 'ULTRA T-BOND'
        tks2_query = [(t.text, t.boost, t.ignored, t.required) for t in ana(pdnm2, mode='query')]
        print(tks2_query)



    # def test_match_prod_code(self):
    #     ix = open_dir('CME_Product_Index')
    #     cmeg = CMEGMatcher()
    #     df = pd.DataFrame([{
    #         cmeg.PRODUCT: 'E-MINI EURO FX',
    #         cmeg.PRODUCT_GROUP: 'FX',
    #         cmeg.CLEARED_AS: 'Futures',
    #         'ADV Y.T.D 2017': 4205
    #     }])
    #
    #     cmeg.match_prod_code(df, ix)
    #
    #     pdnm2 = 'E-MINI EURO FX'
    #     field_pdnm = {x[0]: x[1] for x in ix.schema.items()}[cmeg.F_PRODUCT_NAME]
    #     ana = field_pdnm.analyzer
    #     tks2_query = [(t.text, t.boost, t.ignored, t.required) for t in ana(pdnm2, mode='query')]
    #
    #     and_words2, or_words2 = WhooshSnippet.tokenize_split(field_pdnm, pdnm2, lambda x: x.required)
    #     query2 = WhooshSnippet.andmaybe_query(cmeg.F_PRODUCT_NAME, and_words2, or_words2)
    #
    #     pdnm3 = 'AUD/USD PQO 2pm Fix'
    #     and_words3, or_words3 = WhooshSnippet.tokenize_split(field_pdnm, pdnm3, lambda x: x.required)
    #     parser = qparser.QueryParser(field_pdnm, schema=ix.schema)
    #     mt_dfl_query = parser.multitoken_query('default', and_words3, field_pdnm, Term, 1)
    #     mt_and_query = parser.multitoken_query('and', and_words3, field_pdnm, Term, 1)
    #     mt_or_query = parser.multitoken_query('or', or_words3, field_pdnm, Term, 1)
    #
    #     print(mt_dfl_query)


    # def test_index_grouping(self):
    #     ix = open_dir('CBOT_Product_Index')
    #     f_pdgp = CMEGMatcher.F_PRODUCT_GROUP
    #     f_clas = CMEGMatcher.F_CLEARED_AS
    #     f_sbgp = CMEGMatcher.F_SUB_GROUP
    #     with ix.searcher() as searcher:
    #         lexicons = WhooshSnippet.get_idx_lexicon(
    #             searcher, f_pdgp, f_clas, **{f_pdgp: f_sbgp})
    #         prods_pdgps, prods_clras, prods_subgps = \
    #             lexicons[f_pdgp], lexicons[f_clas], lexicons[f_sbgp]
    #         print(prods_pdgps)
    #         print(prods_clras)
    #         print(prods_subgps)
    #
    #         # results = searcher.search(Every(), groupedby=[F_PRODUCT_GROUP, F_CLEARED_AS])
    #         # pdgp_dict = {gp: [searcher.stored_fields(docid) for docid in ids]
    #         #              for gp, ids in results.groups(F_PRODUCT_GROUP).items()}
    #         # pdgps = list(pdgp_dict.keys())
    #         # clas = list(results.groups(F_CLEARED_AS).keys())
    #         # subgps = {gp: set([doc[F_SUB_GROUP] for doc in docs]) for gp, docs in pdgp_dict.items()}
    #         #
    #         # print(pdgps)
    #         # print(clas)
    #         # print(subgps)


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





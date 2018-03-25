import re
from whoosh.analysis import *
# import unittest as ut

from whooshext import *
import datascraper as dtsp

# class CMEAnalyzerTests(ut.TestCase):
#     @classmethod
#     def setUpClass(cls):
#         cls.exclusions = CurrencyConverter.get_cnvtd_kws() + \
#                   ['nasdq', 'ibovespa', 'index', 'mini',
#                    'micro', 'nikkei', 'russell', 'ftse',
#                    'european', 'futures', 'options']
#
#     def currency_pair_test(self):
#         spltflt = SplitFilter(origin=False, mergewords=True, mergenums=True)
#         ana = STD_ANA | spltflt | VowelFilter(self.exclusions) | CurrencyConverter()
#
#         testcase1 = ' E-MICRO AUD/USD'
#         testcase2 = ' E-MICRO EUR/USD'
#         testcase3 = ' E-MICRO CAD/USD'
#
#         result1 = [t.text for t in ana(testcase1)]
#         result2 = [t.text for t in ana(testcase2)]
#         result3 = [t.text for t in ana(testcase3)]
#
#         # print(result1)
#         expected1 = ['e', 'micro', 'emicro', 'aud', 'australian', 'dollar', 'usd', 'us', 'american', 'dollar']
#
#         # self.assertListEqual(expected1, result1)

CRRNCY_MAPPING = {'ad': 'australian dollar',
                  'bp': 'british pound',
                  'cd': 'canadian dollar',
                  'ec': 'euro cross rate',
                  'efx': 'euro fx',
                  'jy': 'japanese yen',
                  'jpy': 'japanese yen',
                  'ne': 'new zealand dollar',
                  'nok': 'norwegian krone',
                  'sek': 'swedish krona',
                  'sf': 'swiss franc',
                  'skr': 'swedish krona',
                  'zar': 'south african rand',
                  'aud': 'australian dollar',
                  'cad': 'canadian dollar',
                  'eur': 'euro',
                  'gbp': 'british pound',
                  'pln': 'polish zloty',
                  'nkr': 'norwegian krone',
                  'inr': 'indian rupee',
                  'rmb': 'chinese renminbi',
                  'usd': 'us american dollar'}

CRRNCY_KEYWORDS = list(set(dtsp.flatten_list([v.split(' ') for v in CRRNCY_MAPPING.values()], list())))
CME_SPECIAL_MAPPING = {'midcurve': 'mc',
                       'pqo': 'premium quoted european style options',
                       'eow': 'weekly',
                       'eom': 'monthly',
                       'eu': 'european'}

CME_KEYWORD_MAPPING = {**CRRNCY_MAPPING, **CME_SPECIAL_MAPPING}
KYWRD_EXCLU = CRRNCY_KEYWORDS + \
              ['nasdq', 'ibovespa', 'index', 'mini',
               'micro', 'nikkei', 'russell', 'ftse',
               'european', 'premium', 'quoted', 'style',
               'futures', 'options']

spltflt = SplitFilter(origin=False, mergewords=True, mergenums=True)
ana = STD_ANA | spltflt | SpecialWordFilter(CME_KEYWORD_MAPPING) | VowelFilter(KYWRD_EXCLU)

testcase1 = ' E-MICRO AUD/USD'
testcase2 = ' EOW1 E-MINI RUSSELL 2000 WE'
testcase3 = ' AUD/USD PQO 2pm Fix'

result1 = [t.text for t in ana(testcase1)]
result2 = [t.text for t in ana(testcase2)]
result3 = [t.text for t in ana(testcase3)]

print(result1)
print(result2)
print(result3)

import unittest as ut
import re
import pandas as pd

from commonlib.commonfuncs import *


A_PRODUCT_NAME = 'Product Name'
A_PRODUCT_GROUP = 'Product Group'
A_CLEARED_AS = 'Cleared As'
A_COMMODITY = 'Commodity'
PATTERN_ADV_YTD = 'ADV Y.T.D'

F_PRODUCT_NAME = 'P_Product_Name'
F_PRODUCT_GROUP = 'P_Product_Group'
F_CLEARED_AS = 'P_Cleared_As'
F_CLEARING = 'P_Clearing'
F_GLOBEX = 'P_Globex'
F_SUB_GROUP = 'P_Sub_Group'
F_EXCHANGE = 'P_Exchange'


class PublicFuncTests(ut.TestCase):
    def get_mtobjs1(self):
        string = """PETROCHEMICALS                                               94               63.68            48.3%
                          56.57             67.0%                      75                  85          -12.1%"""
        pattern = '([A-Za-z0-9/\(\)\.%&$,-]|(?<! ) (?! ))+'
        return re.finditer(pattern, string)

    def test_map_recursive(self):
        mobjs = list(self.get_mtobjs1())
        dmobjs = list(zip(mobjs, mobjs))
        actual = list(map_recursive(lambda x: x.group(), dmobjs))

        expected = [('PETROCHEMICALS', 'PETROCHEMICALS'),
                    ('94', '94'),
                    ('63.68', '63.68'),
                    ('48.3%', '48.3%'),
                    ('56.57', '56.57'),
                    ('67.0%', '67.0%'),
                    ('75', '75'),
                    ('85', '85'),
                    ('-12.1%', '-12.1%')]

        self.assertListEqual(expected, actual)

    def test_flatten_iter(self):
        iter1 = [(1, ('abf', 0)), 4, (8, ('bey', 9))]
        actual1 = list(flatten_iter(iter1, 0))
        expected1 = [(1, 1), (2, 'abf'), (2, 0), (0, 4), (1, 8), (2, 'bey'), (2, 9)]
        self.assertListEqual(expected1, actual1)

    def test_hierarch_groupby(self):
        ytd = 'ADV Y.T.D 2017'
        cols = [A_PRODUCT_NAME, A_PRODUCT_GROUP,
                A_CLEARED_AS, ytd,
                F_CLEARED_AS, F_CLEARING,
                F_GLOBEX, F_PRODUCT_GROUP,
                F_PRODUCT_NAME, F_SUB_GROUP]
        row_emin_sp500 = pd.Series(['E-MINI S&P500', 'Equity Index',
                                    'Futures', 1593256, 'Futures',
                                    'ES', 'ES', 'Equities',
                                    'E-mini S&P 500 Futures', 'US Index'],
                                   index=cols)
        row_ad_cd = pd.Series(['AD/CD CROSS RATES', 'FX',
                               'Futures', 0, 'Futures',
                               'AC', 'ACD', 'FX',
                               'Australian Dollar/Canadian Dollar Futures', 'Cross Rates'],
                              index=cols)
        row_emin_russell2000 = pd.Series(['E-MINI RUSSELL 2000', 'Equity Index',
                                          'Options', 0, 'Options',
                                          'RTO', 'RTO', 'Equities',
                                          'E-mini  Russell 2000 Options', 'US Index'],
                                         index=cols)
        test_dict = {(1593256, 'ES'): row_emin_sp500,
                     (0, 'ACD'): row_ad_cd,
                     (0, 'RTO'): row_emin_russell2000}

        keyfuncs = [lambda x: x[ytd], lambda x: (x[F_PRODUCT_NAME], x[F_CLEARED_AS]),
                    lambda x: x[F_GLOBEX]]
        out_dict = pdck.hierarch_groupby(test_dict, keyfuncs, True)
        print(out_dict)



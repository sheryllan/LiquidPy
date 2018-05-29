import unittest as ut
import pandas as pd

from productmatcher import MatchHelper
from productchecker import ProductKey


import productchecker as pdck


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

class MatcherTester(ut.TestCase):
    def test_match_in_string(self):
        tc_sample1 = 'Cross Rates'
        tc_ref1 = 'SKR/USD CROSS RATE'

        tc_sample2 = 'Equity Index'
        tc_ref2 = 'Equities'

        # test all
        self.assertTrue(MatchHelper.match_in_string(tc_ref1, tc_sample1, False, True))

        # test any
        self.assertTrue(MatchHelper.match_in_string(tc_ref2, tc_sample2, stemming=True))
        self.assertFalse(MatchHelper.match_in_string(tc_ref2, tc_sample2, False, True))

        # test stemming
        self.assertFalse(MatchHelper.match_in_string(tc_ref2, tc_sample2, True, False))

    def test_hierarch_groupby(self):
        ytd = 'ADV Y.T.D 2017'
        cols = [F_PRODUCT_NAME, F_PRODUCT_GROUP,
                F_CLEARED_AS, ytd,
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


class MainTests(ut.TestCase):
    def test_ProductKey(self):
        # case1 = ProductKey('GC', 'Futures')
        # print(case1)

        case2 = ProductKey(type='Options')
        self.assertTrue((None, 'option') == case2)

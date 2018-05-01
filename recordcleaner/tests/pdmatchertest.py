import unittest as ut
import pandas as pd

from productmatcher import Matcher
from productmatcher import CMEGMatcher

import productchecker as pdck


class MatcherTester(ut.TestCase):
    def test_match_in_string(self):
        tc_sample1 = 'Cross Rates'
        tc_ref1 = 'SKR/USD CROSS RATE'

        tc_sample2 = 'Equity Index'
        tc_ref2 = 'Equities'

        # test all
        self.assertTrue(Matcher.match_in_string(tc_ref1, tc_sample1, False, True))

        # test any
        self.assertTrue(Matcher.match_in_string(tc_ref2, tc_sample2, stemming=True))
        self.assertFalse(Matcher.match_in_string(tc_ref2, tc_sample2, False, True))

        # test stemming
        self.assertFalse(Matcher.match_in_string(tc_ref2, tc_sample2, True, False))

    def test_hierarch_groupby(self):
        ytd = 'ADV Y.T.D 2017'
        cols = [CMEGMatcher.PRODUCT, CMEGMatcher.PRODUCT_GROUP,
                CMEGMatcher.CLEARED_AS, ytd,
                CMEGMatcher.F_CLEARED_AS, CMEGMatcher.F_CLEARING,
                CMEGMatcher.F_GLOBEX, CMEGMatcher.F_PRODUCT_GROUP,
                CMEGMatcher.F_PRODUCT_NAME, CMEGMatcher.F_SUB_GROUP]
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

        keyfuncs = [lambda x: x[ytd], lambda x: (x[CMEGMatcher.PRODUCT], x[CMEGMatcher.CLEARED_AS]),
                    lambda x: x[CMEGMatcher.F_GLOBEX]]
        out_dict = pdck.hierarch_groupby(test_dict, keyfuncs, True)
        print(out_dict)

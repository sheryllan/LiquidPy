import unittest as ut

from productchecker import ProductKey
from productmatcher import MatchHelper


class MatcherTester(ut.TestCase):
    def test_match_in_string(self):
        tc_sample1 = 'Cross Rates'
        tc_ref1 = 'SKR/USD CROSS RATE'

        tc_sample2 = 'Equity Index'
        tc_ref2 = 'Equities'

        # test all
        self.assertTrue(MatchHelper.match_in_string(tc_ref1, tc_sample1, False, True))
        #
        # test any
        self.assertTrue(MatchHelper.match_in_string(tc_ref2, tc_sample2, stemming=True))
        self.assertFalse(MatchHelper.match_in_string(tc_ref2, tc_sample2, False, True))

        # test stemming
        self.assertFalse(MatchHelper.match_in_string(tc_ref2, tc_sample2, True, False))


class MainTests(ut.TestCase):
    def test_ProductKey(self):
        # case1 = ProductKey('GC', 'Futures')
        # print(case1)

        case2 = ProductKey(type='Options')
        print(str(case2))
        is_in_by_eq = case2 in [(None, 'option'), ('san', 'do')]
        self.assertTrue(is_in_by_eq)
        is_in_by_hash = case2 in {(None, 'option'): 'k', ('san', 'do'): 'v'}
        self.assertTrue(is_in_by_hash)

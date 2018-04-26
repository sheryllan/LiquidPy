import unittest as ut
from productmatcher import Matcher


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

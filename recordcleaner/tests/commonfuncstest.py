import unittest as ut
import re

from commonlib.commonfuncs import *


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


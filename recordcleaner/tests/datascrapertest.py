import unittest as ut
from datascraper import OSEScraper


class OSEScraperTests(ut.TestCase):
    def test_get_col_header(self):
        lines = ['                                 取 引 高 報 告              2017年',
                 '                                  Yearly Trading Volume 2017',
                 '                                        (立会日数 247 日）',
                 '                                         （Trading Days 247)',
                 '\n',
                 '          種   別                    数 量（単位）                       金 額（円)               年末建玉（単位）',
                 '           Type                Trading Volume(units)         Trading Value(yen)      Open Interest(units)',
                 'JPX-Nikkei Index 400 Futures                 7,669,469         11,167,497,625,807                    140,190',
                 '        (一日平均）                                  31,050              45,212,540,995            -']
        expected = ['Type', 'Trading Volume(units)', 'Trading Value(yen)', 'Open Interest(units)']
        actual = [m.group() for m in OSEScraper.get_col_header(lines)]

        self.assertListEqual(expected, actual)
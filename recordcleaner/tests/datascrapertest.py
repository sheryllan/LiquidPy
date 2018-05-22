import unittest as ut

from datascraper import *
import re


class MainTests(ut.TestCase):
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

    def test_match_min_split(self):
        line1 = '                                                                                           February 2018' \
                '                                                                                      Page 7 of 17'
        line2 = '                                                      FEB 2018             FEB 2017            % CHG' \
                '               JAN 2018            % CHG              Y.T.D 2018          Y.T.D 2017            % CHG'

        actual1 = match_min_split(line1)
        actual2 = match_min_split(line2)

        expected1 = None
        expected_str2 = ['FEB 2018', 'FEB 2017', '% CHG', 'JAN 2018', '% CHG', 'Y.T.D 2018', 'Y.T.D 2017', '% CHG']
        expected_cords2 = [(54, 62), (75, 83), (95, 100), (115, 123), (135, 140), (154, 164), (174, 184), (196, 201)]
        self.assertEqual(expected1, actual1)
        self.assertListEqual(expected_str2, actual2[0])
        self.assertListEqual(expected_cords2, actual2[1])

    def test_match_tabular_line(self):
        line1 = '                                                      FEB 2018             FEB 2017            % CHG' \
                '               JAN 2018            % CHG              Y.T.D 2018          Y.T.D 2017            % CHG'
        line2 = 'JPX-Nikkei Index 400 Futures                 7,669,469         11,167,497,625,807                    140,190'

        colname_func = lambda x: re.search('[A-Za-z]+', x)
        actual1 = match_tabular_line(line1, colname_func=colname_func)
        actual2 = match_tabular_line(line2, colname_func=colname_func)

        expected_str1 = ['FEB 2018', 'FEB 2017', '% CHG', 'JAN 2018', '% CHG', 'Y.T.D 2018', 'Y.T.D 2017', '% CHG']
        expected_cords1 = [(54, 62), (75, 83), (95, 100), (115, 123), (135, 140), (154, 164), (174, 184), (196, 201)]
        expected2 = None

        self.assertListEqual(expected_str1, actual1[0])
        self.assertListEqual(expected_cords1, actual1[1])
        self.assertEqual(expected2, actual2)


class TxtFormatterTests(ut.TestCase):
    def testcase1(self):
        line_longer = """                                                      FEB 2018             FEB 2017            
                % CHG               JAN 2018            % CHG              Y.T.D 2018          Y.T.D 2017            % CHG"""
        line_shorter = """                                                          ADV                  ADV
                                                     ADV                                      ADV                 ADV"""
        pattern = '(\S+( \S+)*)+'
        rlonger = list(re.finditer(pattern, line_longer))
        rshorter = list(re.finditer(pattern, line_shorter))
        return rlonger, rshorter

    def testcase2(self):
        line_longer = '           Type                Trading Volume(units)         Trading Value(yen)      Open Interest(units)'
        line_shorter = '                        31,050            45,212,540,995            -'
        pattern = '(\S+( \S+)*)+'
        rlonger = list(re.finditer(pattern, line_longer))
        rshorter = list(re.finditer(pattern, line_shorter))
        return rlonger, rshorter

    def test_align_by_min_tot_diff_rightaligned(self):
        expected = [('FEB 2018', 'ADV'),
                    ('FEB 2017', 'ADV'),
                    ('% CHG', None),
                    ('JAN 2018', 'ADV'),
                    ('% CHG', None),
                    ('Y.T.D 2018', 'ADV'),
                    ('Y.T.D 2017', 'ADV'),
                    ('% CHG', None)]

        actual = list(map_recursive(lambda x: x.group() if x else None,
                                    TxtFormatter.align_by_min_tot_offset(*self.testcase1(), 'right')))
        self.assertListEqual(expected, actual)

    def test_align_by_min_tot_diff_combined(self):
        expected = [('Type', '31,050'),
                    ('Trading Volume(units)', '45,212,540,995'),
                    ('Trading Value(yen)', '-'),
                    ('Open Interest(units)', None)]
        actual = list(map_recursive(lambda x: x.group() if x else None,
                                    TxtFormatter.align_by_min_tot_offset(*self.testcase2())))
        self.assertListEqual(expected, actual)

    def test_merge_2rows(self):
        def merge_headers(hl, hs):
            headers = list()
            if hs:
                hs = hs.rstrip()
                hs = hs.lstrip()
                headers.append(hs)
            if hl:
                hl = hl.rstrip()
                hl = hl.lstrip()
                headers.append(hl)
            return ' '.join(headers)

        expected = ['ADV FEB 2018',
                    'ADV FEB 2017',
                    '% CHG',
                    'ADV JAN 2018',
                    '% CHG',
                    'ADV Y.T.D 2018',
                    'ADV Y.T.D 2017',
                    '% CHG']

        merged, aligned_mtobjs = TxtFormatter.merge_2rows(*self.testcase1(), merge_headers, 'right')
        self.assertListEqual(expected, merged)


if __name__ == "__main__":
    ut.main()
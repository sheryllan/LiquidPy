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

        actual1 = TabularTxtParser.match_min_split(line1)
        actual2 = list(TabularTxtParser.match_min_split(line2))

        expected1 = None
        expected2 = [((54, 62), 'FEB 2018'),
                     ((75, 83), 'FEB 2017'),
                     ((95, 100), '% CHG'),
                     ((115, 123), 'JAN 2018'),
                     ((135, 140), '% CHG'),
                     ((154, 164), 'Y.T.D 2018'),
                     ((174, 184), 'Y.T.D 2017'),
                     ((196, 201), '% CHG')]
        self.assertEqual(expected1, actual1)
        self.assertListEqual(expected2, actual2)

    def test_match_tabular_line(self):
        line1 = '                                                      FEB 2018             FEB 2017            % CHG' \
                '               JAN 2018            % CHG              Y.T.D 2018          Y.T.D 2017            % CHG'
        line2 = 'JPX-Nikkei Index 400 Futures                 7,669,469         11,167,497,625,807                    140,190'

        colname_func = lambda x: re.search('[A-Za-z]+', x)
        actual1 = TabularTxtParser.match_tabular_line(line1, colname_func=colname_func)
        actual2 = TabularTxtParser.match_tabular_line(line2, colname_func=colname_func)

        expected1 = [((54, 62), 'FEB 2018'),
                     ((75, 83), 'FEB 2017'),
                     ((95, 100), '% CHG'),
                     ((115, 123), 'JAN 2018'),
                     ((135, 140), '% CHG'),
                     ((154, 164), 'Y.T.D 2018'),
                     ((174, 184), 'Y.T.D 2017'),
                     ((196, 201), '% CHG')]
        expected2 = None

        self.assertListEqual(expected1, actual1)
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
        line_shorter = '                            31,050       45,212,540,995            -'
        pattern = '(\S+( \S+)*)+'
        rlonger = list(re.finditer(pattern, line_longer))
        rshorter = list(re.finditer(pattern, line_shorter))
        return rlonger, rshorter

    def to_cords(self, testcase):
        longer, shorter = testcase
        cords_longer = [l.span() for l in longer]
        cords_shorter = [s.span() for s in shorter]
        return cords_longer, cords_shorter

    def test_align_by_min_tot_diff_rightaligned(self):
        # expected = [('FEB 2018', 'ADV'),
        #             ('FEB 2017', 'ADV'),
        #             ('% CHG', None),
        #             ('JAN 2018', 'ADV'),
        #             ('% CHG', None),
        #             ('Y.T.D 2018', 'ADV'),
        #             ('Y.T.D 2017', 'ADV'),
        #             ('% CHG', None)]

        aligned_idxes = [0, 1, 3, 5, 6]
        cords_longer, cords_shorter = self.to_cords(self.testcase1())
        expected = [(cs, None) for cs in cords_longer]
        for i, il in enumerate(aligned_idxes):
            expected[il] = (cords_longer[il], cords_shorter[i])

        actual = list(TabularTxtParser.align_by_min_tot_offset(cords_longer, cords_shorter, TabularTxtParser.RIGHT))
        self.assertListEqual(expected, actual)

    def test_align_by_min_tot_diff_combined(self):
        # expected = [('Type', '31,050'),
        #             ('Trading Volume(units)', '45,212,540,995'),
        #             ('Trading Value(yen)', '-'),
        #             ('Open Interest(units)', None)]

        aligned_idxes = [0, 1, 2]
        cords_longer, cords_shorter = self.to_cords(self.testcase2())
        expected = [(cs, None) for cs in cords_longer]
        for i, il in enumerate(aligned_idxes):
            expected[il] = (cords_longer[il], cords_shorter[i])

        actual = list(TabularTxtParser.align_by_min_tot_offset(cords_longer, cords_shorter))
        self.assertListEqual(expected, actual)

    def test_align_by_min_tot_diff_multi_cross(self):
        cords_longer = [(0, 3), (10, 19), (30, 32), (36, 39), (52, 53), (54, 57), (66, 70)]
        cords_shorter = [(8, 10), (12, 15), (17, 22), (40, 41), (60, 64), (69, 72)]
        actual = list(TabularTxtParser.align_by_min_tot_offset(cords_longer, cords_shorter, TabularTxtParser.LEFT))

        aligned_idxes = [0, 1, 2, 3, 5, 6]
        expected = [(cs, None) for cs in cords_longer]
        for i, il in enumerate(aligned_idxes):
            expected[il] = (cords_longer[il], cords_shorter[i])

        self.assertListEqual(expected, actual)

    def test_merge_2rows(self):
        def merge_headers(h1, h2):
            h1 = h1.strip()
            h2 = h2.strip()
            return ' '.join(filter(None, [h1, h2]))

        expected = [(((58, 61), (54, 62)), 'ADV FEB 2018'),
                    (((79, 82), (75, 83)), 'ADV FEB 2017'),
                    ((None, (112, 117)), '% CHG'),
                    (((136, 139), (132, 140)), 'ADV JAN 2018'),
                    ((None, (152, 157)), '% CHG'),
                    (((177, 180), (171, 181)), 'ADV Y.T.D 2018'),
                    (((197, 200), (191, 201)), 'ADV Y.T.D 2017'),
                    ((None, (213, 218)), '% CHG')]

        rlonger, rshorter = self.testcase1()
        str_longer, str_shorter = rlonger[0].string, rshorter[0].string
        cords_longer, cords_shorter = [m.span() for m in rlonger], [m.span() for m in rshorter]
        actual = list(TabularTxtParser.merge_2rows(str_shorter, str_longer, cords_shorter, cords_longer,
                                                   merge_headers, TabularTxtParser.RIGHT))

        self.assertListEqual(expected, actual)


if __name__ == "__main__":
    ut.main()

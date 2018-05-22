import math
import tempfile
from datetime import datetime
from subprocess import Popen, PIPE

import jaconv
import pandas as pd
from PyPDF2 import PdfFileReader
from dateutil.relativedelta import relativedelta

from commonlib.websourcing import *
from configparser import XlsxWriter

PDF_SUFFIX = '.pdf'
TXT_SUFFIX = '.txt'
XLSX_SUFFIX = '.xlsx'


def last_year():
    return (datetime.now() - relativedelta(years=1)).year


def first_nonna_index(df):
    return pd.notna(df).all(1).nonzero()[0][0]


def filter_df(df, filterfunc):
    cols = filterfunc(df.columns)
    return df[cols]


def set_df_col(df):
    header_index = first_nonna_index(df)
    if header_index != 0:
        df.columns = df.iloc[header_index]
        df.drop(header_index, inplace=True)
    return df


def clean_df(df, nonna_subset):
    df.dropna(subset=nonna_subset, how='all', inplace=True)
    df.reset_index(drop=0, inplace=True)
    return df


def run_pdftotext(pdf, txt=None, encoding='utf-8', **kwargs):
    # txt = re.sub('\.pdf$', pdf, '.txt') if txt is None else txt
    txt = '-' if txt is None else txt
    args = ['pdftotext', pdf, txt, '-layout']
    args.extend(flatten_iter(('-' + str(k), str(v)) for k, v in kwargs.items()))
    out, err = Popen(args, stdout=PIPE).communicate()
    if out:
        out = out.decode(encoding)
        out = out.splitlines()
    if err is not None:
        raise RuntimeError(err)
    print('\n[*] Successfully convert {} to {}'.format(pdf, 'stdout' if txt == '-' else txt))
    return out if txt == '-' else txt


class TxtFormatter(object):
    LEFT = 'left'
    RIGHT = 'right'
    CENTRE = 'centre'

    alignment_metric = {LEFT: lambda x: x[0],
                        RIGHT: lambda x: x[1],
                        CENTRE: lambda x: (x[0] + x[1]) / 2}

    alignment_func = {LEFT: lambda cord1, cord2: abs(cord1[0] - cord2[0]),
                      RIGHT: lambda cord1, cord2: abs(cord1[1] - cord2[1]),
                      CENTRE: lambda cord1, cord2: abs((cord1[0] + cord1[1]) / 2 - (cord2[0] + cord2[1]) / 2)}

    @classmethod
    def which_alignment(cls, cord1, cord2):
        left = abs(cord1[0] - cord2[0])
        right = abs(cord1[1] - cord2[1])
        centre = abs((cord1[0] + cord1[1]) / 2 - (cord2[0] + cord2[1]) / 2)
        offsets = {left: TxtFormatter.LEFT, right: TxtFormatter.RIGHT, centre: TxtFormatter.CENTRE}
        return offsets[min(offsets)]

    @classmethod
    def mtobjs_to_cords(cls, mtobjs):
        return [mtobj.span() for mtobj in mtobjs]

    @classmethod
    def sort_mtobjs(cls, robjs):
        return sorted(robjs, key=lambda x: x.start())

    # r1, r2 are dictionaries with (x0, x1) as keys
    # @classmethod
    # def merge_2rows(cls, rlonger, rshorter, alignment, atol, merge):
    #     if len(rlonger) < len(rshorter):
    #         rlonger, rshorter = swap(rlonger, rshorter)
    #     func = cls.alignment_metric[alignment]
    #     rlonger = list(sorted(rlonger.items(), key=lambda x: func(x[0])))
    #     rshorter = list(sorted(rshorter.items(), key=lambda x: func(x[0])))
    #     merged = dict()
    #     i, j = 0, 0
    #     while i < len(rlonger):
    #         clonger, vlonger = rlonger[i]
    #         if j < len(rshorter):
    #             cshorter, vshorter = rshorter[j]
    #             while abs(func(clonger) - func(cshorter)) > atol:
    #                 if func(clonger) < func(cshorter):
    #                     merged[clonger] = vlonger
    #                     i += 1
    #                     clonger, vlonger = rlonger[i]
    #                 else:
    #                     merged[cshorter] = vshorter
    #                     j += 1
    #                     cshorter, vshorter = rshorter[j]
    #             merged[clonger] = merge(vshorter, vlonger)
    #             i += 1
    #             j += 1
    #         else:
    #             merged[clonger] = vlonger
    #             i += 1
    #     return [v for k, v in sorted(merged.items(), key=lambda x: func(x[0]))]


    @classmethod
    def merge_2rows(cls, row_longer, row_shorter, cords_longer, cords_shorter, mergfunc, alignment='centre'):
        aligned_cords = list(TxtFormatter.align_by_min_tot_offset(cords_longer, cords_shorter, alignment))
        return [mergfunc( if cl is not None else cl, cs.group() if cs else cs)
                for cl, cs in aligned_cords], aligned_cords


    @classmethod
    def sqr_offset(cls, cols1, cols2, alignment='centre'):
        func = TxtFormatter.alignment_func[alignment]
        result = 0
        for c1, c2 in zip(cols1, cols2):
            result += func(c1, c2)**2
        return result


    @classmethod
    def align_by_mse(cls, robjs_baseline, robjs_instance, alignment='centre'):
        if not robjs_baseline or not robjs_instance:
            return None
        min_offset = math.inf
        aligned_cols = None
        col_num = len(robjs_instance)
        for i in range(0, len(robjs_baseline) - col_num + 1):
            cords_cols = [(h.start(), h.end()) for h in robjs_baseline[i: i + col_num]]
            cords_instance = [(d.start(), d.end()) for d in robjs_instance]
            offset = cls.sqr_offset(cords_cols, cords_instance, alignment)
            if offset < min_offset:
                min_offset = offset
                aligned_cols = {obj_base.group(): obj_inst.group()
                                for obj_base, obj_inst in zip(robjs_baseline[i: i + col_num], robjs_instance)}
        return aligned_cols

    @classmethod
    def __get_turning_idx(cls, point_ref, points, dist_func, start=0, end=None):
        end = len(points) if end is None else end
        min_dist = math.inf
        i = start
        for i in range(start, end):
            distance = abs(dist_func(point_ref, points[i]))
            if distance >= min_dist:
                return i - 1
            else:
                min_dist = distance
        return i

    @classmethod
    def __distances_to_point(cls, cord_point, cords, dist_func, start=0, end=None):
        end = len(cords) if end is None else end
        for i in range(start, end):
            yield abs(dist_func(cord_point, cords[i]))


    # take as many indexes as possible in prev_idxes until i_turning
    @classmethod
    def __rearrange_prev_idxes(cls, prev_idxes, i_turning):
        if not prev_idxes:
            return prev_idxes, i_turning
        il_curr, is_curr = i_turning, len(prev_idxes)
        if il_curr > prev_idxes[is_curr - 1]:
            return prev_idxes, il_curr
        il_curr = il_curr if il_curr >= is_curr else is_curr

        new_idxes = [idx if idx <= il_curr else il_curr- is_curr + i for i, idx in enumerate(prev_idxes)]
        return new_idxes, il_curr





    @classmethod
    def align_by_min_tot_offset(cls, cords1, cords2, alignment='centre'):
        cords1, cords2 = list(sorted(cords1)), list(sorted(cords2))
        cords_longer, cords_shorter, swapped = (swap(cords1, cords2), True) \
            if len(cords2) > len(cords1) else (cords1, cords2, False)

        def get_dist_matrix(points_shorter, points_longer, alignment):
            func = TxtFormatter.alignment_func[alignment]
            end = len(points_longer) - len(points_shorter)
            for ps in points_shorter:
                end += 1
                yield list(TxtFormatter.__distances_to_point(ps, points_longer, func, end=end))

        def get_discontinuous_idx(arr, from_idx, stop_idx, step, decreasing=True):
            for i in range(from_idx, stop_idx, step):
                if not (arr[i] - arr[i - 1] == 1 if decreasing else arr[i - 1] - arr[i] == 1):
                    return i
            return None



        dist_matrix, aligned_idexes = list(), list()
        for distances in get_dist_matrix(cords_shorter, cords_longer, alignment):
            dist_matrix.append(distances)
            aligned_idexes.append(index_culmulative(distances, min))
        if not verify_non_decreasing(aligned_idexes):
            raise ValueError('Error with mapping of the sorted coordinates: mapped indexes should be non-decreasing')

        for is_curr, il_curr in enumerate(aligned_idexes[1:]):
            is_prev = is_curr - 1
            il_prev = aligned_idexes[is_prev]
            if il_curr != il_prev:
                continue

            il_left, is_left = il_prev - 1, is_prev - 1
            diff_left = dist_matrix[is_curr][il_curr] + dist_matrix[is_prev][il_prev] - dist_matrix[is_prev][il_left]
            while 0 < is_left <= il_left == aligned_idexes[is_left]:
                diff_left += dist_matrix[is_left][il_left] - dist_matrix[is_left][il_left - 1]
                il_left -= 1
                is_left -= 1

            il_right, is_right = il_curr + 1, is_curr + 1
            diff_right = dist_matrix[is_prev][il_prev] + dist_matrix[is_curr][il_curr] - dist_matrix[is_curr][il_right]
            while len(aligned_idexes) > is_right and il_right >= is_right and il_right == aligned_idexes[is_right]:
                diff_right += dist_matrix[is_right][il_right] - dist_matrix[is_right][il_right + 1]



        # func = TxtFormatter.alignment_func[alignment]
        # num_remaining = len(cords_shorter)
        # aligned_idxes = list()
        # for is_curr, cs in enumerate(cords_shorter):
        #     num_remaining -= 1
        #     il_end = len(cords_longer) - num_remaining
        #     il_turning = TxtFormatter.__find_turning_idx(cs, cords_longer, func, end=il_end)
        #     aligned_idxes, il_turning = TxtFormatter.__rearrange_prev_idxes(aligned_idxes, il_turning)
        #     aligned_idxes.append(il_turning)
        #
        # return ((cords_longer[i], cords_shorter[aligned_idxes.index(i)] if i in aligned_idxes else None)
        #         for i in range(0, len(cords_longer)))



        # By sorting both coordinates, it can be inferred that if j = min_index(dist_matrix[i]),
        # then min_index(dist_matrix[i - 1]) <= j




        # il_end = len(cords_longer) - len(cords_shorter)
        # dist_matrix = list()
        # for cs in cords_shorter:
        #     il_end += 1
        #     dist_matrix.append(list(TxtFormatter.__distances_to_point(cs, cords_longer, func, end=il_end)))
        # By sorting both coordinates, it can be inferred that if j = min_index(dist_matrix[i]),
        # then min_index(dist_matrix[i - 1]) <= j


        # # Find the last cross-aligned index in cords_longer and cords_shorter by the min of dist_matrix
        # il_turning, is_turning = len(cords_longer) - 1, len(cords_shorter) - 1
        # aligned_idxes = [index_culmulative(distances, min) for distances in dist_matrix]
        # for i, j in enumerate(aligned_idxes[::-1]):
        #     if j < il_turning:
        #         il_turning = j
        #     else:
        #         is_turning = len(cords_shorter) - i
        #         il_turning = aligned_idxes[is_turning]
        #         break
        #
        # def is_idx_fit(ishorter, ilonger):
        #     return il_turning > ilonger and is_turning - ishorter <= il_turning - ilonger
        #
        # if il_turning < is_turning:
        #     il_turning = is_turning
        #
        # aligned_reidxes = range(il_turning - is_turning, il_turning)
        # for is_curr, il_curr in enumerate(range(il_turning - is_turning, il_turning)):
        #     if aligned_idxes[is_curr] >= il_curr + 1:
        #         break






ASCII_PATTERN = '([A-Za-z0-9/\(\)\.%&$,-]|(?<! ) (?! ))+'


def get_safe_phrase_pattern(pattern):
    pattern_phrase = '(^|(?<=((?<!\S) ))){}($|(?=( (?!\S))))'
    return pattern_phrase.format(pattern)


def get_char_phrases(string, p_separator=' {2,}|^ | $', p_chars=ASCII_PATTERN):
    return [match for match in re.split(p_separator, string) if re.match(p_chars, match)]


def match_min_split(line, p_separator='^ +| {2,}| +$', min_splits=3):
    sep_cords = (match.span() for match in re.finditer(p_separator, line))
    match_indices = list(flatten_iter(sep_cords))
    match_indices = match_indices[1:] if match_indices[0] == 0 else [0] + match_indices
    match_indices = match_indices[:-1] if match_indices[-1] == len(line) else match_indices + [len(line)]
    match_cords = [cord for cord in group_every_n(match_indices, 2, tuple) if len(cord) == 2]
    match_strings = [line[cord[0]: cord[1]] for cord in match_cords]
    return (match_strings, match_cords) if len(match_strings) >= min_splits else None


def match_tabular_line(line, p_separator='^ +| {2,}| +$', min_splits=3, colname_func=lambda x: re.match(ASCII_PATTERN, x)):
    match_strings, match_cords = match_min_split(line, p_separator, min_splits)
    return (match_strings, match_cords) if all(colname_func(s) for s in match_strings) else None


class OSEScraper(object):
    URL_OSE = 'http://www.jpx.co.jp'
    URL_VOLUME = URL_OSE + '/english/markets/statistics-derivatives/trading-volume/01.html'

    TABLE_TITLE = 'Year'

    TYPE = 'Type'
    VOLUME = 'Trading Volume(units)'
    DAILY_AVG = 'Daily Average'

    def find_report_url(self, year=last_year()):
        year_str = str(year)
        tbparser = HtmlTableParser(self.URL_VOLUME, self.filter_table)
        headers = tbparser.get_tb_headers()

        # find in headers for the column of the year
        col_idx = headers.index(year_str)
        tds = tbparser.get_td_rows(lambda x: x[col_idx])

        pattern = r'^(?=.*{}.*\.pdf).*$'.format(year_str)
        file_url = find_link(tds, pattern)
        return self.URL_OSE + file_url

    # returns the table in which first tr has first th of which text = title
    def filter_table(self, tables):
        return [tbl for tbl in tables if tbl.find(TR_TAB).find(TH_TAB, text=self.TABLE_TITLE)][0]

    @staticmethod
    def get_ascii_words(string, pattern_ascii='([A-Za-z0-9/\(\)\.%&$,-]|(?<! ) (?! ))+'):
        pattern_words = '(^|(?<=((?<!\S) ))){}($|(?=( (?!\S))))'
        pattern_ascii_words = pattern_words.format(pattern_ascii)
        return list(re.finditer(pattern_ascii_words, string))

    @staticmethod
    def is_header(line, pattern_data='([A-Za-z0-9/\(\)\.%&$,-]|(?<! ) (?! ))+',
                  pattern_header='[A-Za-z]+', min_colnum=4):
        matches = OSEScraper.get_ascii_words(line, pattern_data)
        if len(matches) < min_colnum:
            return False
        isheader = all(re.search(pattern_header, m.group()) for m in matches)
        if isheader:
            return matches

    @staticmethod
    def get_col_header(lines, pattern_data='([A-Za-z0-9/\(\)\.%&$,-]|(?<! ) (?! ))+',
                       pattern_header='[A-Za-z]+', min_colnum=4):
        for line in lines:
            isheader = OSEScraper.is_header(line, pattern_data, pattern_header, min_colnum)
            if isheader:
                return isheader
        return None

    def parse_from_txt(self, lines=None, alignment='centre'):
            header_mtchobjs, data_row = None, dict()
            lines = iter(lines)
            line = next(lines, None)

            while line is not None and not header_mtchobjs:
                ln_convt = jaconv.z2h(line, kana=False, digit=True, ascii=True)
                header_mtchobjs = self.is_header(ln_convt)
                line = next(lines, None)
            headers = [h.group() for h in header_mtchobjs]
            results = pd.DataFrame(columns=headers)
            while line is not None:
                ln_convt = jaconv.z2h(line, kana=False, digit=True, ascii=True)
                data_piece = OSEScraper.get_ascii_words(ln_convt)
                aligned_cols = TxtFormatter.align_by_mse(header_mtchobjs, data_piece, alignment)
                if aligned_cols and any(c in data_row for c in aligned_cols):
                    results = results.append(pd.DataFrame([data_row], columns=headers))
                    data_row = aligned_cols
                elif aligned_cols:
                    data_row.update(aligned_cols)
                line = next(lines, None)
            if data_row:
                results = results.append(pd.DataFrame([data_row], columns=headers))
            return results

    def parse_by_pages(self, pdf_name, end_page, start_page=1, outpath=None, sheetname=None):
        tables = list()
        for page in range(start_page, end_page + 1):
            txt = run_pdftotext(pdf_name, f=page, l=page)
            df = self.parse_from_txt(txt)
            tables.append(df)
        results = pd.concat(tables)
        if outpath:
            sheetname = 'Sheet1' if sheetname is None else sheetname
            return XlsxWriter.save_sheets(outpath, {sheetname: results})
        return results

    def run_scraper(self, year=last_year()):
        dl_url = self.find_report_url(year)
        f_pdf = download(dl_url, tempfile.NamedTemporaryFile())
        num_pages = PdfFileReader(f_pdf).getNumPages()
        results = self.parse_by_pages(f_pdf.name, num_pages)
        f_pdf.close()
        return results

    def filter_adv(self, df):
        df_adv = pd.DataFrame(columns=df.columns.values)
        last_type = None
        for i, row in df.iterrows():
            if pd.isnull(row[OSEScraper.TYPE]):
                continue
            if OSEScraper.DAILY_AVG not in row[OSEScraper.TYPE]:
                last_type = row[OSEScraper.TYPE]
            elif last_type is not None and not pd.isnull(last_type):
                row[OSEScraper.TYPE] = last_type
                df_adv = df_adv.append(row, ignore_index=True)
                last_type = None
        return df_adv



# download_path = os.getcwd()
# # # download_path = '/home/slan/Documents/downloads/'
# cme = CMEGScraper(download_path)
# cme.run_scraper()
# cme.download_to_xlsx_adv()


# ose = OSEScraper()
# rp_url = ose.find_report_url()
# df_all = ose.run_scraper()
# ose.filter_adv(df_all)
# output = run_pdftotext_cmd('OSE_Average_Daily_Volume.pdf', f=1, l=1)
#
# df = ose.parse_from_txt('OSE_Average_Daily_Volume.txt')
# XlsxWriter.save_sheets('OSE_adv_parsed.xlsx', {'sheet1': df})
# ose.download_adv()
# ose.parse_pdf_adv()
# ose.tabula_parse()

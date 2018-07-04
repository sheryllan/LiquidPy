import math
from datetime import datetime
from subprocess import Popen, PIPE
import logging

import pandas as pd
from PyPDF2 import PdfFileReader
from dateutil.relativedelta import relativedelta

from commonlib.websourcing import *

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
    df.reset_index(drop=True, inplace=True)
    return df


def rename_filter(data, col_mapping=None, outcols=None):
    def process(d):
        renamed = rename_mapping(d, col_mapping)
        return select_mapping(renamed, outcols, False)

    if isinstance(data, pd.DataFrame):
        return process(data)
    elif nontypes_iterable(data):
        return (process(d) for d in data)


ASCII_PATTERN = '([A-Za-z0-9/\(\)\.%&$,-]|(?<! ) (?! ))+'
LETTER_PATTERN = '[A-Za-z]+'


def get_char_phrases(string, p_separator='^ +| {2,}| +$', p_chars=ASCII_PATTERN):
    return [match for match in re.split(p_separator, string) if re.match(p_chars, match)]


def slice_str(string, span):
    if not string or not span:
        return ''
    return string[span[0]: span[1]]


def run_pdftotext(pdf, txt=None, encoding='utf-8', **kwargs):
    txt = '-' if txt is None else txt
    args = ['pdftotext', pdf, txt, '-layout']
    args.extend(flatten_iter(('-' + str(k), str(v)) for k, v in kwargs.items()))
    out, err = Popen(args, stdout=PIPE).communicate()
    if out:
        out = out.decode(encoding)
        out = out.splitlines()
    if err is not None:
        raise RuntimeError(err)

    return out if txt == '-' else txt


def text_to_num(values):
    pattern = '^-?[\d\.,]+%?$'
    for i, value in enumerate(values):
        result = value.strip()
        if re.match(pattern, result):
            result = result.replace('%', '').replace(',', '')
            result = float(result) if '.' in result else int(result)
        yield result


class PdfParser(object):
    def __init__(self, f_pdf):
        self.f_pdf = f_pdf
        self.pdf_reader = PdfFileReader(f_pdf)
        self.logger = logging.getLogger(__name__)

    def get_num_pages(self):
        return self.pdf_reader.getNumPages()

    def pdftotext_bypages(self, start_page=1, end_page=None):
        tot_pages = self.get_num_pages()
        end_page = tot_pages if end_page is None or end_page > tot_pages else end_page
        pdf = self.f_pdf.name
        for page in range(start_page, end_page + 1):
            yield run_pdftotext(pdf, f=page, l=page)
            self.logger.debug('Successfully convert page#{} in pdf {} to {}'.format(page, pdf, 'stdout'))


class TabularTxtParser(object):
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
        offsets = {left: cls.LEFT, right: cls.RIGHT, centre: cls.CENTRE}
        return offsets[min(offsets)]

    @classmethod
    def mtobjs_to_cords(cls, mtobjs):
        return [mtobj.span() for mtobj in mtobjs]

    @classmethod
    def sort_mtobjs(cls, robjs):
        return sorted(robjs, key=lambda x: x.start())

    @classmethod
    def merge_2rows(cls, row1, row2, cords1, cords2, mergfunc, alignment='centre'):
        aligned_cords = cls.align_by_min_tot_offset(cords1, cords2, alignment)
        return (((c1, c2), mergfunc(slice_str(row1, c1), slice_str(row2, c2))) for c1, c2 in aligned_cords)

    @classmethod
    def distances_to_point(cls, cord_point, cords, dist_func, start=0, end=None):
        end = len(cords) if end is None else end
        for i in range(start, end):
            yield abs(dist_func(cord_point, cords[i]))

    @classmethod
    def __get_dist_matrix(cls, points_shorter, points_longer, alignment):
        func = cls.alignment_func[alignment]
        end = len(points_longer) - len(points_shorter)
        for ps in points_shorter:
            end += 1
            yield list(cls.distances_to_point(ps, points_longer, func, end=end))

    @classmethod
    def __move_adj_idxes(cls, i_left, i_right, aligned_idexes, dist_matrix):
        il_left, is_left = i_left
        il_right, is_right = i_right
        reidx, dist = dict(), 0

        while 0 <= is_left < il_left <= aligned_idexes[is_left]:
                reidx.update({is_left: il_left - 1})
                dist += dist_matrix[is_left][il_left - 1]
                il_left, is_left = il_left - 1, is_left - 1

        len_l, len_s = len(dist_matrix[-1]), len(aligned_idexes)
        while 0 < len_s - is_right < len_l - il_right and il_right >= aligned_idexes[is_right]:
                reidx.update({is_right: il_right + 1})
                dist += dist_matrix[is_right][il_right + 1]
                il_right, is_right = il_right + 1, is_right + 1

        return dist if reidx else math.inf, reidx

    @classmethod
    def __remap_idxes(cls, i_curr, i_prev, dist_matrix, aligned_idxes):
        il_curr, is_curr = i_curr
        il_prev, is_prev = i_prev

        dist_left, reidx_left = cls.__move_adj_idxes((il_prev, is_prev), (il_curr, is_curr + 1), aligned_idxes, dist_matrix)
        dist_right, reidx_right = cls.__move_adj_idxes((il_prev, is_prev - 1), (il_curr, is_curr), aligned_idxes, dist_matrix)

        if not (reidx_left or reidx_right):
            raise ValueError('Invalid indexes for remapping: cannot move a position on either side of the points')

        is_set = set().union(reidx_left.keys(), reidx_right.keys())
        dist_left = dist_left + sum(dist_matrix[i][aligned_idxes[i]] for i in is_set if i not in reidx_left)
        dist_right = dist_right + sum(dist_matrix[i][aligned_idxes[i]] for i in is_set if i not in reidx_right)
        return reidx_left if dist_left <= dist_right else reidx_right

    @classmethod
    def __aligned_idxes_to_cords(cls, aligned_idxes, cords_longer, cords_shorter):
        iter_il = iter(range(len(cords_longer)))

        def gen_misaligned_cords():
            for il in iter_il:
                if il == aligned_il:
                    break
                yield (cords_longer[il], None)

        for i, aligned_il in enumerate(aligned_idxes):
            yield from gen_misaligned_cords()
            yield (cords_longer[aligned_il], cords_shorter[i])

        yield from gen_misaligned_cords()

    @classmethod
    def align_by_min_tot_offset(cls, cords1, cords2, alignment='centre'):
        cords1, cords2 = list(sorted(cords1)), list(sorted(cords2))
        cords_longer, cords_shorter, swapped = (*swap(cords1, cords2), True) \
            if len(cords2) > len(cords1) else (cords1, cords2, False)

        dist_matrix, aligned_idxes = list(), list()
        for distances in cls.__get_dist_matrix(cords_shorter, cords_longer, alignment):
            dist_matrix.append(distances)
            aligned_idxes.append(index_culmulative(distances, min))
        if not verify_non_decreasing(aligned_idxes):
            raise ValueError('Error with mapping of the sorted coordinates: mapped indexes should be non-decreasing')

        for is_curr in range(1, len(aligned_idxes)):
            il_curr = aligned_idxes[is_curr]
            is_prev = is_curr - 1
            il_prev = aligned_idxes[is_prev]
            if il_curr != il_prev:
                continue

            reidx = cls.__remap_idxes((il_curr, is_curr), (il_prev, is_prev), dist_matrix, aligned_idxes)
            for is_new, il_new in reidx.items():
                aligned_idxes[is_new] = il_new

        for cords in cls.__aligned_idxes_to_cords(aligned_idxes, cords_longer, cords_shorter):
            if swapped:
                cords = swap(*cords)
            yield cords

    @classmethod
    def match_min_split(cls, line, p_separator='^ +| {2,}| +$', min_splits=3):
        sep_cords = (match.span() for match in re.finditer(p_separator, line))
        match_indices = list(flatten_iter(sep_cords))
        if len(match_indices) < 2 * (min_splits - 1):
            return None
        match_indices = match_indices[1:] if match_indices[0] == 0 else [0] + match_indices
        match_indices = match_indices[:-1] if match_indices[-1] == len(line) else match_indices + [len(line)]
        match_cords = [cord for cord in group_every_n(match_indices, 2, tuple) if len(cord) == 2]
        if len(match_cords) < min_splits:
            return None
        return ((cord, slice_str(line, cord)) for cord in match_cords)

    @classmethod
    def match_tabular_line(cls, line, p_separator='^ +| {2,}| +$', min_splits=3,
                           colname_func=lambda x: re.match(ASCII_PATTERN, x)):
        matches = cls.match_min_split(line, p_separator, min_splits)
        if matches is None:
            return None
        matches = list(matches)
        return matches if matches and all(colname_func(s) for s in map(lambda x: x[1], matches)) else None

    @classmethod
    def match_tabular_header(cls, line, p_separator=None, min_splits=None):
        p_separator = '^ +| {2,}| +$' if p_separator is None else p_separator
        min_splits = 3 if min_splits is None else min_splits
        return TabularTxtParser.match_tabular_line(line, p_separator, min_splits, colname_func=lambda x: re.search(LETTER_PATTERN, x))




    @classmethod
    def sqr_offset(cls, cols1, cols2, alignment='centre'):
        func = cls.alignment_func[alignment]
        result = 0
        for c1, c2 in zip(cols1, cols2):
            result += func(c1, c2) ** 2
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


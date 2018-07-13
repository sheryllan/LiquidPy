import math
from datetime import datetime
from subprocess import Popen, PIPE
from itertools import chain
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


def whole_pattern(pattern=ASCII_PATTERN):
    return '^{}$'.format(pattern)


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


def text_to_num(value):
    pattern = '^-?[\d\.,]+%?$'
    if not isinstance(value, str):
        raise TypeError('Input must be an instance of str')

    result = value.strip()
    if re.match(pattern, result):
        result = result.replace('%', '').replace(',', '')
        result = float(result) if '.' in result else int(result)

    return result


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

    @staticmethod
    def which_alignment(cord1, cord2):
        left = abs(cord1[0] - cord2[0])
        right = abs(cord1[1] - cord2[1])
        centre = abs((cord1[0] + cord1[1]) / 2 - (cord2[0] + cord2[1]) / 2)
        offsets = {left: TabularTxtParser.LEFT, right: TabularTxtParser.RIGHT, centre: TabularTxtParser.CENTRE}
        return offsets[min(offsets)]

    @staticmethod
    def mtobjs_to_cords(mtobjs):
        return [mtobj.span() for mtobj in mtobjs]

    @staticmethod
    def sort_mtobjs(robjs):
        return sorted(robjs, key=lambda x: x.start())

    @staticmethod
    def distances_to_point(cord_point, cords, dist_func, start=0, end=None):
        end = len(cords) if end is None else end
        for i in range(start, end):
            yield abs(dist_func(cord_point, cords[i]))

    @staticmethod
    def __get_dist_matrix(points_shorter, points_longer, alignment):
        func = TabularTxtParser.alignment_func[alignment]
        end = len(points_longer) - len(points_shorter)
        for ps in points_shorter:
            end += 1
            yield list(TabularTxtParser.distances_to_point(ps, points_longer, func, end=end))

    @staticmethod
    def __move_adj_idxes(i_left, i_right, aligned_idexes, dist_matrix):
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

    @staticmethod
    def __remap_idxes(i_curr, i_prev, dist_matrix, aligned_idxes):
        il_curr, is_curr = i_curr
        il_prev, is_prev = i_prev

        dist_left, reidx_left = TabularTxtParser.__move_adj_idxes((il_prev, is_prev), (il_curr, is_curr + 1),
                                                                  aligned_idxes, dist_matrix)
        dist_right, reidx_right = TabularTxtParser.__move_adj_idxes((il_prev, is_prev - 1), (il_curr, is_curr),
                                                                    aligned_idxes, dist_matrix)

        if not (reidx_left or reidx_right):
            raise ValueError('Invalid indexes for remapping: cannot move a position on either side of the points')

        is_set = set().union(reidx_left.keys(), reidx_right.keys())
        dist_left = dist_left + sum(dist_matrix[i][aligned_idxes[i]] for i in is_set if i not in reidx_left)
        dist_right = dist_right + sum(dist_matrix[i][aligned_idxes[i]] for i in is_set if i not in reidx_right)
        return reidx_left if dist_left <= dist_right else reidx_right

    @staticmethod
    def __aligned_idxes_to_cords(aligned_idxes, cords_longer, cords_shorter):
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

    @staticmethod
    def align_cords_by_min_dist(cords1, cords2, alignment=CENTRE, sort=False):
        cords1, cords2 = list(cords1), list(cords2)
        if cords1 and cords2:
            if sort:
                cords1, cords2 = sorted(cords1), sorted(cords2)
            cords_longer, cords_shorter, swapped = (*swap(cords1, cords2), True) \
                if len(cords2) > len(cords1) else (cords1, cords2, False)

            dist_matrix, aligned_idxes = list(), list()
            for distances in TabularTxtParser.__get_dist_matrix(cords_shorter, cords_longer, alignment):
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

                reidx = TabularTxtParser.__remap_idxes((il_curr, is_curr), (il_prev, is_prev), dist_matrix, aligned_idxes)
                for is_new, il_new in reidx.items():
                    aligned_idxes[is_new] = il_new

            for cords in TabularTxtParser.__aligned_idxes_to_cords(aligned_idxes, cords_longer, cords_shorter):
                if swapped:
                    cords = swap(*cords)
                yield cords
        else:
            return iter(())

    @staticmethod
    def __misalign_heading(cords1, cords2, margin):
        if cords1[0][0] > cords2[0][0]:
            ref_start, head_cords = cords1[0][0], cords2
            yieldfunc = lambda x: (None, x)
        else:
            ref_start, head_cords = cords2[0][0], cords1
            yieldfunc = lambda x: (x, None)

        for cords in slicing_gen(head_cords, lambda x: x[1] - ref_start >= margin):
            yield yieldfunc(cords)

    @staticmethod
    def __misalign_tailing(cords1, cords2, margin):
        if cords1[-1][1] > cords2[-1][1]:
            ref_end, tail_cords = cords2[-1][1], cords1
            yieldfunc = lambda x: (x, None)
        else:
            ref_end, tail_cords = cords1[-1][1], cords2
            yieldfunc = lambda x: (None, x)

        for cords in slicing_gen(tail_cords, lambda x: ref_end - x[0] >= margin):
            yield yieldfunc(cords)


    @staticmethod
    def algin_cords_by_margin(cords1, cords2, margin=0, alignment=CENTRE, sort=False):
        if sort:
            cords1, cords2 = sorted(cords1), sorted(cords2)
        cords1_cpy, cords2_cpy = list(cords1), list(cords2)
        if cords1_cpy and cords2_cpy:
            heads = list(TabularTxtParser.__misalign_heading(cords1_cpy, cords2_cpy, margin))
            tails = list(TabularTxtParser.__misalign_tailing(cords1_cpy, cords2_cpy, margin))
            middle = TabularTxtParser.align_cords_by_min_dist(cords1_cpy, cords2_cpy, alignment, not sort)
            yield from chain(heads, middle, tails)
        else:
            return iter(())

    @staticmethod
    def align_txt_by_min_dist(matches1, matches2, align_method=None, alignment=CENTRE, c_selector=None, defaultval=None):
        align_method = TabularTxtParser.align_cords_by_min_dist if align_method is None else align_method
        match_dict1 = {m[0]: m[1] for m in matches1} if not isinstance(matches1, dict) else matches1
        match_dict2 = {m[0]: m[1] for m in matches2} if not isinstance(matches2, dict) else matches2
        cords1, cords2 = match_dict1.keys(), match_dict2.keys()
        c_selector = (lambda x: x[0] if len(cords1) >= len(cords2) else x[1]) if c_selector is None else c_selector
        aligned_cords = align_method(cords1, cords2, alignment=alignment)
        return {c_selector(cords): (match_dict1.get(cords[0], defaultval), match_dict2.get(cords[1], defaultval))
                for cords in aligned_cords}

    @staticmethod
    def match_min_split(line, p_separator='^ +| {2,}| +$', min_splits=3):
        sep_cords = (match.span() for match in re.finditer(p_separator, line))
        match_indices = list(flatten_iter(sep_cords))
        if not match_indices:
            return None
        if min_splits is not None and len(match_indices) < 2 * (min_splits - 1):
            return None
        match_indices = match_indices[1:] if match_indices[0] == 0 else [0] + match_indices
        match_indices = match_indices[:-1] if match_indices[-1] == len(line) else match_indices + [len(line)]
        match_cords = [cord for cord in group_every_n(match_indices, 2, tuple) if len(cord) == 2]

        if min_splits is not None and len(match_cords) < min_splits:
            return None
        return ((cord, slice_str(line, cord)) for cord in match_cords)

    @staticmethod
    def match_tabular_line(line, p_separator=None, min_splits=None,
                           verify_func=lambda x: re.match(whole_pattern(ASCII_PATTERN), x)):
        p_separator = '^ +| {2,}| +$' if p_separator is None else p_separator

        matches = TabularTxtParser.match_min_split(line, p_separator, min_splits)

        if matches is None:
            return None
        matches = list(matches)
        return matches if verify_func is None or all(verify_func(s) for s in map(lambda x: x[1], matches)) else None


    @staticmethod
    def match_tabular_header(line, p_separator=None, min_splits=None):
        return TabularTxtParser.match_tabular_line(line, p_separator, min_splits,
                                                   verify_func=lambda x: re.search(LETTER_PATTERN, x))




    # @staticmethod
    # def merge_2rows(row1, row2, mergfunc, matchfunc=None, c_seletor=None, alignment=CENTRE):
    #     matchfunc = TabularTxtParser.match_tabular_line if matchfunc is None else matchfunc
    #     matches1, matches2 = matchfunc(row1), matchfunc(row2)
    #     if matches1 is None or matches2 is None:
    #         return None
    #
    #     aligned = TabularTxtParser.align_txt_by_min_dist(matches1, matches2, alignment=alignment,
    #                                                      c_selector=c_seletor, defaultval='')
    #     return {cords: mergfunc(*aligned[cords]) for cords in aligned}

    # @classmethod
    # def sqr_offset(cls, cols1, cols2, alignment='centre'):
    #     func = cls.alignment_func[alignment]
    #     result = 0
    #     for c1, c2 in zip(cols1, cols2):
    #         result += func(c1, c2) ** 2
    #     return result
    #
    # @classmethod
    # def align_by_mse(cls, robjs_baseline, robjs_instance, alignment='centre'):
    #     if not robjs_baseline or not robjs_instance:
    #         return None
    #     min_offset = math.inf
    #     aligned_cols = None
    #     col_num = len(robjs_instance)
    #     for i in range(0, len(robjs_baseline) - col_num + 1):
    #         cords_cols = [(h.start(), h.end()) for h in robjs_baseline[i: i + col_num]]
    #         cords_instance = [(d.start(), d.end()) for d in robjs_instance]
    #         offset = cls.sqr_offset(cords_cols, cords_instance, alignment)
    #         if offset < min_offset:
    #             min_offset = offset
    #             aligned_cols = {obj_base.group(): obj_inst.group()
    #                             for obj_base, obj_inst in zip(robjs_baseline[i: i + col_num], robjs_instance)}
    #     return aligned_cols


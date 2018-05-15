import math
import os
import tempfile
from datetime import datetime
from subprocess import Popen, PIPE

import jaconv
import pandas as pd
from PyPDF2 import PdfFileReader
from PyPDF2.generic import Destination
from dateutil.relativedelta import relativedelta

from configparser import XlsxWriter
from commonlib.websourcing import *

PDF_SUFFIX = '.pdf'
TXT_SUFFIX = '.txt'
XLSX_SUFFIX = '.xlsx'


def last_year():
    return (datetime.now() - relativedelta(years=1)).year


def run_pdftotext_cmd(pdf, txt=None, encoding='utf-8', **kwargs):
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


class TxtHelper(object):
    alignment_metric = {'left': lambda x: x[0],
                        'right': lambda x: x[1],
                        'centre': lambda x: (x[0] + x[1]) / 2}

    alignment_func = {'left': lambda cord1, cord2: abs(cord1[0] - cord2[0]),
                      'right': lambda cord1, cord2: abs(cord1[1] - cord2[1]),
                      'centre': lambda cord1, cord2: abs((cord1[0] + cord1[1]) / 2 - (cord2[0] + cord2[1]) / 2)}

    @classmethod
    def which_alignment(cls, cord1, cord2):
        left = abs(cord1[0] - cord2[0])
        right = abs(cord1[1] - cord2[1])
        centre = abs((cord1[0] + cord1[1]) / 2 - (cord2[0] + cord2[1]) / 2)
        offsets = {left: 'left', right: 'right', centre: 'centre'}
        return offsets[min(offsets)]

    # r1, r2 are dictionaries with (x0, x1) as keys
    @classmethod
    def merge_2rows(cls, rlonger, rshorter, alignment, atol, merge):
        if len(rlonger) < len(rshorter):
            rlonger, rshorter = swap(rlonger, rshorter)
        func = cls.alignment_metric[alignment]
        rlonger = list(sorted(rlonger.items(), key=lambda x: func(x[0])))
        rshorter = list(sorted(rshorter.items(), key=lambda x: func(x[0])))
        merged = dict()
        i, j = 0, 0
        while i < len(rlonger):
            clonger, vlonger = rlonger[i]
            if j < len(rshorter):
                cshorter, vshorter = rshorter[j]
                while abs(func(clonger) - func(cshorter)) > atol:
                    if func(clonger) < func(cshorter):
                        merged[clonger] = vlonger
                        i += 1
                        clonger, vlonger = rlonger[i]
                    else:
                        merged[cshorter] = vshorter
                        j += 1
                        cshorter, vshorter = rshorter[j]
                merged[clonger] = merge(vshorter, vlonger)
                i += 1
                j += 1
            else:
                merged[clonger] = vlonger
                i += 1
        return [v for k, v in sorted(merged.items(), key=lambda x: func(x[0]))]


    @classmethod
    def calc_sqr_offset(cls, cols1, cols2, alignment='centre'):
        func = TxtHelper.alignment_func[alignment]
        result = 0
        for c1, c2 in zip(cols1, cols2):
            result += func(c1, c2)**2
        return result


    @classmethod
    def align_by_mse(cls, headers, data, alignment='centre'):
        if not data:
            return None
        min_offset = math.inf
        aligned_cols = None
        col_num = len(data)
        for i in range(0, len(headers) - col_num + 1):
            cords_cols = [(h.start(), h.end()) for h in headers[i: i + col_num]]
            cords_data = [(d.start(), d.end()) for d in data]
            offset = cls.calc_sqr_offset(cords_cols, cords_data, alignment)
            if offset < min_offset:
                min_offset = offset
                aligned_cols = {header.group(): dt.group() for header, dt in zip(headers[i: i + col_num], data)}
        return aligned_cols


class CMEGScraper(object):
    URL_CME_ADV = 'http://www.cmegroup.com/daily_bulletin/monthly_volume/Web_ADV_Report_CME.pdf'
    URL_CBOT_ADV = 'http://www.cmegroup.com/daily_bulletin/monthly_volume/Web_ADV_Report_CBOT.pdf'
    URL_NYMEX_COMEX_ADV = 'http://www.cmegroup.com/daily_bulletin/monthly_volume/Web_ADV_Report_NYMEX_COMEX.pdf'
    URL_PRODSLATE = 'http://www.cmegroup.com/CmeWS/mvc/ProductSlate/V1/Download.xls'

    PRODUCT = 'Product'
    PRODUCT_GROUP = 'Product Group'
    CLEARED_AS = 'Cleared As'
    OUTPUT_COLUMNS = [PRODUCT, PRODUCT_GROUP, CLEARED_AS]

    CME = 'CME'
    CBOT = 'CBOT'
    NYMEX = 'NYMEX'
    COMEX = 'COMEX'

    def default_adv_basenames(self):
        basename_cme = rreplace(os.path.basename(self.URL_CME_ADV), PDF_SUFFIX, '', 1)
        basename_cbot = rreplace(os.path.basename(self.URL_CBOT_ADV), PDF_SUFFIX, '', 1)
        basename_nymex = rreplace(os.path.basename(self.URL_NYMEX_COMEX_ADV), PDF_SUFFIX, '', 1)
        return basename_cme, basename_cbot, basename_nymex

    def default_adv_xlsx(self):
        basename_cme, basename_cbot, basename_nymex = self.default_adv_basenames()
        adv_xlsx_cme = basename_cme + XLSX_SUFFIX
        adv_xlsx_cbot = basename_cbot + XLSX_SUFFIX
        adv_xlsx_nymex = basename_nymex + XLSX_SUFFIX
        return adv_xlsx_cme, adv_xlsx_cbot, adv_xlsx_nymex

    # returns a dictionary with key: full group name, and value: (asset, instrument)
    def get_pdf_product_groups(self, sections):
        prev_level = sections[0][0]
        asset = sections[0][1]
        result = dict()
        for level, title in sections[1:]:
            if level < prev_level:
                asset = title
            else:
                instrument = title
                result[('{} {}'.format(asset, instrument))] = (asset, instrument)
            prev_level = level
        return result

    def read_pdf_metadata(self, path):
        with open(path, mode='rb') as fh:
            fr = PdfFileReader(fh)
            outlines = fr.getOutlines()
            flat_outlines = flatten_iter(outlines, True, (str, Destination))
            # self.report_name = ' '.join([o.title for _, o in flat_outlines[0:2]]).replace('/', '-')
            return self.get_pdf_product_groups([(l, o.title) for l, o in flat_outlines[2:]])

    def parse_from_txt(self, pdf_path, lines):
        product_groups = self.read_pdf_metadata(pdf_path)
        pattern_data = '^ ?((\S+ )+ {2,}){2,}.*$'
        pattern_headers = '^ {10,}((\S+ )+ {2,}){2,}.*$'
        if lines:
            header_line = find_first_n(lines, lambda x: re.match(pattern_headers, x) is not None, 2)
            df = pd.DataFrame(columns=self.__get_output_headers(header_line))
            group, clearing = None, None
            for line in lines:
                if 'total' in line.lower():
                    break
                line = line.rstrip()
                if line.lstrip() in product_groups:
                    group, clearing = product_groups[line.lstrip()]
                elif re.match(pattern_data, line) is not None:
                    df = self.__append_to_df(df, line, group, clearing)
            return df
        else:
            return None

    def to_xlsx_adv(self, pdfpath, txt, outpath=None, sheetname=None):
        table = self.parse_from_txt(pdfpath, txt)
        if outpath is not None:
            sheetname = 'Sheet1' if sheetname is None else sheetname
            XlsxWriter.save_sheets(outpath, {sheetname: table})
        return table

    def download_to_xlsx_adv(self, url, outpath=None, sheetname=None):
        f_pdf = download(url, tempfile.NamedTemporaryFile())
        txt = run_pdftotext_cmd(f_pdf.name)
        result = self.to_xlsx_adv(f_pdf.name, txt, outpath, sheetname)
        f_pdf.close()
        return result

    def run_scraper(self, path_prods=None, path_advs=None):
        prods_file = tempfile.NamedTemporaryFile() if path_prods is None else path_prods
        prods_file = download(self.URL_PRODSLATE, prods_file)

        df_cme = self.download_to_xlsx_adv(self.URL_CME_ADV, outpath=path_advs, sheetname=self.CME)
        df_cbot = self.download_to_xlsx_adv(self.URL_CBOT_ADV, outpath=path_advs, sheetname=self.CBOT)
        df_nymex = self.download_to_xlsx_adv(self.URL_NYMEX_COMEX_ADV, outpath=path_advs, sheetname=self.NYMEX)

        adv_dict = {self.CME: df_cme, self.CBOT: df_cbot, self.NYMEX: df_nymex}
        return prods_file.name, adv_dict

    def __get_output_headers(self, pdf_headers, pattern=None):
        pattern = '(\S+( \S+)*)+' if pattern is None else pattern
        heading_cols = self.OUTPUT_COLUMNS[0:1]
        tailing_cols = self.OUTPUT_COLUMNS[1:3]
        to_cord = lambda mobj: (mobj.start(), mobj.end())
        to_group = lambda mobj: mobj.group()
        headers = [to_dict(re.finditer(pattern, string), to_cord, to_group) for string in pdf_headers]
        alignment = 'right'
        h1 = headers[0]
        for h2 in headers[1:]:
            h1 = TxtHelper.merge_2rows(h1, h2, alignment, 2, self.__merge_headers)
        return heading_cols + h1 + tailing_cols

    def __merge_headers(self, *headers):
        headers = [h.lstrip() for h in headers]
        headers = [h.rstrip() for h in headers]
        return ' '.join(headers)

    def __parse_line(self, line, **kwargs):
        kw_pattern = 'pattern'
        kw_extras = 'extras'
        pattern = kwargs[kw_pattern] if kw_pattern in kwargs else '(\S+( \S+)*)+'
        values = [v[0] for v in re.findall(pattern, line)]
        values = self.__text_to_num(values)
        return values if kw_extras not in kwargs else values + list(kwargs[kw_extras])

    def __append_to_df(self, df, line, *args):
        line_parsed = self.__parse_line(line, extras=list(args))
        df = df.append(pd.Series(line_parsed, index=df.columns), ignore_index=True)
        return df

    def __text_to_num(self, values):
        pattern = '^-?[\d\.,]+%?$'
        for i, value in enumerate(values):
            if re.match(pattern, value):
                value = value.replace('%', '')
                value = value.replace(',', '')
                values[i] = float(value) if '.' in value else int(value)
        return values


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
                aligned_cols = TxtHelper.align_by_mse(header_mtchobjs, data_piece, alignment)
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
            txt = run_pdftotext_cmd(pdf_name, f=page, l=page)
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

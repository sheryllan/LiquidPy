import collections
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import subprocess
import tempfile

import pandas as pd
import tabula
from bs4 import BeautifulSoup
from PyPDF2 import PdfFileReader

import configparser


USER_AGENT = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7; X11; Linux x86_64) ' \
             'Gecko/2009021910 Firefox/3.0.7 Chrome/23.0.1271.64 Safari/537.11'
TABLE_TAB = 'table'
TR_TAB = 'tr'
TH_TAB = 'th'
TD_TAB = 'td'
A_TAB = 'a'
HREF_ATTR = 'href'


def flatten_list(items, flat_list, level):
    if isinstance(items, list):
        for sublist in items:
            flatten_list(sublist, flat_list, level + 1)
    else:
        flat_list.append((level, items))
    return flat_list


def to_list(x):
    return [x] if not isinstance(x, collections.Iterable) else list(x)


def find_first_n(arry, condition, n=1):
    result = list()
    for a in arry:
        if n == 0:
            return result
        if condition(a):
            result.append(a)
            n -= 1
    return result


def download(url, filename=None, mode='wb'):
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})

    try:
        response = urllib.request.urlopen(request)
        if filename is not None and isinstance(filename, str):
            with open(filename, mode) as fh:
                print(('\n[*] Downloading from: {}'.format(url)))
                fh.write(response.read())
                print('\n[*] Successfully downloaded to ' + filename)
        elif filename is None:
            filename = tempfile.NamedTemporaryFile()
            print(('\n[*] Downloading from: {}'.format(url)))
            filename.write(response.read())
            print('\n[*] Successfully downloaded to ' + filename.name)

        return filename
    except urllib.error.HTTPError as e:
        print(e.fp.read())


def make_soup(url):
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    html = urllib.request.urlopen(request)
    soup = BeautifulSoup(html, 'html.parser')
    return soup

def swap(a, b):
    tmp = a
    a = b
    b = tmp
    return a, b

def to_dict(items, tkey, tval):
    return {tkey(x): tval(x) for x in items}

class TxtHelper(object):
    alignment_metric = {'left': lambda x: x[0],
                        'right': lambda x: x[1],
                        'centre': lambda x: (x[0] + x[1])/2}

    alignment_func = {'left': lambda cord1, cord2: abs(cord1[0] - cord2[0]),
                      'right': lambda cord1, cord2: abs(cord1[1] - cord2[1]),
                      'centre': lambda cord1, cord2: abs((cord1[0] + cord1[1])/2 - (cord2[0] + cord2[1])/2)}

    @classmethod
    def which_alignment(cls, cord1, cord2):
        left = abs(cord1[0] - cord2[0])
        right = abs(cord1[1] - cord2[1])
        centre = abs((cord1[0] + cord1[1])/2 - (cord2[0] + cord2[1])/2)
        offsets = {left: 'left', right: 'right', centre: 'centre'}
        return offsets[min(offsets)]

    # r1, r2 are dictionaries with (x0, x1) as keys
    @classmethod
    def merge_2rows(cls, rlonger, rshorter, alignment, atol, sep=' '):
        if len(rlonger) < len(rshorter):
            rlonger, rshorter = swap(rlonger, rshorter)
        func = cls.alignment_metric[alignment]
        rlonger = list(sorted(rlonger.items(), key=lambda x: func(x[0])))
        rshorter = list(sorted(rshorter.items(), key=lambda x: func(x[0])))
        merged = dict()
        j = 0
        for i in range(0, len(rlonger)):
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
                merged[clonger] = sep.join([vlonger, vshorter])
                j += 1
            else:
                merged[clonger] = vlonger
        return [v for k, v in sorted(merged.items(), key=lambda x: func(x[0]))]



class CMEScraper(object):
    URL_ADV = 'http://www.cmegroup.com/daily_bulletin/monthly_volume/Web_ADV_Report_CMEG.pdf'
    PDF_ADV = 'CME_Average_Daily_Volume.pdf'
    TXT_ADV = 'CME_Average_Daily_Volume.txt'
    XLSX_ADV = 'CME_Average_Daily_Volume.xlsx'

    URL_PRODSLATE = 'http://www.cmegroup.com/CmeWS/mvc/ProductSlate/V1/Download.xls'
    XLS_PRODSLATE = 'Product_Slate.xls'

    OUTPUT_COLUMNS = ['Product', 'Product Group', 'Cleared As']
    COLUMN_MAPPING = {OUTPUT_COLUMNS[0]: 'Product Name',
                      OUTPUT_COLUMNS[1]: 'Product Group',
                      OUTPUT_COLUMNS[2]: 'Sub Group'}

    def __init__(self, download_path):
        self.download_path = download_path
        self.pdf_path_adv = os.path.join(self.download_path, self.PDF_ADV)
        self.xls_path_prodslate = os.path.join(self.download_path, self.XLS_PRODSLATE)
        self.txt_path_adv = os.path.join(self.download_path, self.TXT_ADV)
        self.xlsx_path_adv = os.path.join(self.download_path, self.XLSX_ADV)

        self.pdf_path_prodslate = os.path.join(self.download_path, self.XLS_PRODSLATE)

        self.adv_headers = []
        self.report_name = None
        self.product_groups = None

    def download_adv(self, path=None):
        return download(self.URL_ADV, path)

    def download_prodslate(self, path=None):
        path = self.pdf_path_prodslate if path is None else path
        download(self.URL_PRODSLATE, path)

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

    def read_pdf_metadata(self, path=None):
        path = self.pdf_path_adv if path is None else path
        with open(path, mode='rb') as fh:
            fr = PdfFileReader(fh)
            outlines = fr.getOutlines()
            flat_outlines = flatten_list(outlines, list(), 0)
            self.report_name = ' '.join([o.title for l, o in flat_outlines[0:2]]).replace('/', '-')
            self.product_groups = self.get_pdf_product_groups([(l, o.title) for l, o in flat_outlines[2:]])

    def parse_from_txt(self, path=None):
        self.read_pdf_metadata()
        path = self.txt_path_adv if path is None else path
        with open(path) as fh:
            lines = fh.readlines()
            pattern_data = '^((([\w\(\)\.%&,-]+) )+ {2,})+.*$'
            pattern_headers = '^ +((([\w\(\)\.%&,-]+) )+ {2,}){2,}.*$'
        if lines:
            header_line = find_first_n(lines, lambda x: re.match(pattern_headers, x) is not None, 2)
            self.adv_headers = self.__get_output_headers(header_line)
            df = pd.DataFrame(columns=self.adv_headers)
            for line in lines:
                if 'total' in line.lower():
                    break
                line = line.rstrip()
                if line in self.product_groups:
                    group, clearing = self.product_groups[line]
                elif re.match(pattern_data, line) is not None:
                    df = self.__append_to_df(df, line, group, clearing)
            return df
        else:
            return None

    def txt_to_xlsx_adv(self, inpath=None, outpath=None):
        table = self.parse_from_txt(inpath)
        outpath = self.xlsx_path_adv if outpath is None else outpath
        return configparser.XlsxWriter.save_sheets(outpath, {self.report_name: table})

    def get_adv_xlsx(self, outpath=None):
        outpath = self.xlsx_path_adv if outpath is None else outpath
        f_pdf = self.download_adv()
        f_txt = tempfile.NamedTemporaryFile()
        try:
            out, err = subprocess.Popen(["pdftotext", "-layout", f_pdf.name, f_txt.name]).communicate()
            if err is not None:
                raise RuntimeError(err)
            self.txt_to_xlsx_adv(f_txt.name, outpath)
        finally:
            f_pdf.close()
            f_txt.close()


    def __get_output_headers(self, pdf_headers, pattern=None):
        pattern = '([\w\(\)\.%&,-]+( [\w\(\)\.%&,-]+)*)+' if pattern is None else pattern
        heading_cols = self.OUTPUT_COLUMNS[0:1]
        tailing_cols = self.OUTPUT_COLUMNS[1:3]
        to_cord = lambda mobj: (mobj.start(), mobj.end())
        to_group = lambda mobj: mobj.group()
        headers = [to_dict(re.finditer(pattern, string), to_cord, to_group) for string in pdf_headers]
        alignment = 'right'
        h1 = headers[0]
        for h2 in headers[1:]:
            h1 = TxtHelper.merge_2rows(h1, h2, alignment, 2)
        return heading_cols + h1 + tailing_cols

    def __parse_line(self, line, **kwargs):
        kw_pattern = 'pattern'
        kw_extras = 'extras'
        kw_sep = 'sep'
        line = line.rstrip()
        pattern = kwargs[kw_pattern] if kw_pattern in kwargs else '(?<! )+ {2,}|(?<=[0-9%,-])+ +?(?=[0-9%,-])+'
        sep = kwargs[kw_sep] if kw_sep in kwargs else '\t'
        repl = re.sub(pattern, sep, line)
        values = self.__text_to_num(repl.split(sep))
        return values if kw_extras not in kwargs else values + list(kwargs[kw_extras])

    def __append_to_df(self, df, line, *args):
        line_parsed = self.__parse_line(line, extras=list(args))
        df = df.append(pd.Series(line_parsed, index=list(df)), ignore_index=True)
        return df

    def __text_to_num(self, values):
        pattern = '^-?[\d\.,]+%?$'
        for i, value in enumerate(values):
            if re.match(pattern, value):
                value = value.replace('%', '')
                value = value.replace(',', '')
                values[i] = float(value) if '.' in value else int(value)
        return values

    # def tabula_parse(self):
    #     tabula.convert_into(self.pdf_path_adv, 'CME_ADV.csv', output_format='csv', pages='all')





class OSEScraper(object):
    URL_OSE = 'http://www.jpx.co.jp'
    URL_VOLUME = URL_OSE + '/english/markets/statistics-derivatives/trading-volume/01.html'

    PDF_ADV = 'OSE_Average_Daily_Volume.pdf'

    YEAR_INTERESTED = '2017'
    TABLE_TITLE = 'Year'

    def __init__(self, download_path):
        self.download_path = download_path
        self.full_path_adv = os.path.join(self.download_path, self.PDF_ADV)

    def find_report_url(self):
        soup = make_soup(self.URL_VOLUME)

        tables = soup.find_all(TABLE_TAB)
        table = self.filter_table(tables, self.TABLE_TITLE)
        ths = table.find(TR_TAB).find_all(TH_TAB)

        # find in headers for the column of the year
        col_idx = self.get_indexes(ths, lambda x: x.text == self.YEAR_INTERESTED)[0]

        trs = table.find_all(TR_TAB)
        td_rows = [tr for tr in trs if tr.find_all(TD_TAB)]
        tds = [tr.find_all(TD_TAB)[col_idx] for tr in td_rows]

        links = [str(td.find(href=True)[HREF_ATTR]) for td in tds]
        pattern = r'^(?=.*{}.*\.pdf).*$'.format(self.YEAR_INTERESTED)
        file_url = find_first_n(links, lambda x: re.match(pattern, x))
        return self.URL_OSE + file_url

    def download_adv(self):
        download(self.find_report_url(), self.full_path_adv)

    def get_indexes(self, arry, condition):
        return [i for i, a in enumerate(arry) if condition(a)]

    # returns the table in which first tr has first th of which text = title
    def filter_table(self, tables, title):
        return [tbl for tbl in tables if tbl.find(TR_TAB).find(TH_TAB, text=title)][0]


    def tabula_parse(self):
        # tabula.convert_into(self.full_path_adv, 'OSE_ADV.csv', output_format='csv', pages='all')
        df = tabula.read_pdf(self.full_path_adv, pages=2)
        print()


#download_path = '/Users/sheryllan/Downloads/'
download_path = '/home/slan/Documents/downloads/'
cme = CMEScraper(download_path)
cme.get_adv_xlsx()
# cme.tabula_parse()
# cme.download_adv()
# cme.download_prodslate()
# tb = cme.parse_pdf_adv()

# ose = OSE(download_path)
# ose.download_adv()
# ose.parse_pdf_adv()
# ose.tabula_parse()

import urllib2
import os
import sys
import collections
import pandas as pd
from bs4 import BeautifulSoup
import re


from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.layout import LAParams, LTTextBox, LTTextLine, LTTextBoxHorizontal, LTTextLineHorizontal
from pdfminer.converter import PDFPageAggregator
from pdfminer.pdfpage import PDFPage

user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7; X11; Linux x86_64) ' \
             'Gecko/2009021910 Firefox/3.0.7 Chrome/23.0.1271.64 Safari/537.11'


def flatten_list(items):
    flat_list = list()
    if isinstance(items, collections.Iterable):
        for sublist in items:
            if isinstance(sublist, list):
                flat_list = flat_list + sublist
            else:
                flat_list.append(sublist)
    else:
        flat_list = items
    return flat_list


def to_list(x):
    return [x] if not isinstance(x, collections.Iterable) else list(x)


def download(url, filename):
    request = urllib2.Request(url, headers={'User-Agent': user_agent})

    try:
        response = urllib2.urlopen(request)
        with open(filename, 'wb') as fh:
            print('\n[*] Downloading: {}'.format(os.path.basename(filename)))
            fh.write(response.read())
            print ('\n[*] Successful')
    except urllib2.HTTPError, e:
        print e.fp.read()

def make_soup(url):
    request = urllib2.Request(url, headers={'User-Agent': user_agent})
    html = urllib2.urlopen(request)
    soup = BeautifulSoup(html, 'html.parser')
    return soup

class PDFHelper(object):
    TEXT_ELEMENTS = [
        LTTextBox,
        LTTextBoxHorizontal,
        LTTextLine,
        LTTextLineHorizontal
    ]

    @staticmethod
    def get_text_from_ltobj(obj):
        if any(isinstance(obj, ele) for ele in PDFHelper.TEXT_ELEMENTS):
            return obj.get_text()
        else:
            raise ValueError('No text found in the given LTObject')

    @staticmethod
    def extract_pdf_sections(doc, levels=None):
        return [(level, title) for (level, title, dest, a, structelem) in doc.get_outlines()
                if level in levels] if levels is not None else [(level, title) for (level, title, dest, a, structelem) in doc.get_outlines()]

    @staticmethod
    def setup_pdfdocument(fh):
        parser = PDFParser(fh)
        doc = PDFDocument(parser)
        return doc

    @staticmethod
    def setup_interpreter():
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        return PDFPageInterpreter(rsrcmgr, device), device

    @staticmethod
    def match_ltobjs_by_coordinate(headers, contents, coordinate):
        if len(headers) != len(contents):
            raise ValueError('Invalid input: counts of columns do not match.')
        headers_sorted = sorted(headers, key=lambda o: o.bbox[coordinate])
        contents_sorted = sorted(contents, key=lambda o: o.bbox[coordinate])
        result = list()
        for header, content in zip(headers_sorted, contents_sorted):
            abs_tol = header.width / 2
            if abs(header.bbox[coordinate] - content.bbox[coordinate]) <= abs_tol:
                result.append((header, content))
            else:
                raise ValueError('Incorrect input: misaligned columns')
        return result

    @staticmethod
    def unzip_ltobjs_inorder(ltobjs, sort_key):
        cols_ordered = [obj for obj in sorted(ltobjs, key=sort_key)]
        headers, contents = zip(*cols_ordered)
        return list(headers), list(contents)


    # returns a single dictionary
    @staticmethod
    def zip_dicts_by_keys(dicts, keys=None):
        keys = dicts[0].keys() if keys is None else keys
        try:
            return {k: flatten_list([dt[k] for dt in dicts]) for k in keys}
        except KeyError:
            value, tracestack = sys.exc_info()
            raise KeyError, ('Unable to zip dictionaries: missing key(s)', value), tracestack


    # returns a list of dictionary with (y0, y1) as keys and the text as values
    @staticmethod
    def ltobjs_to_dict(ltobjs, crd, format=None):
        format = format if format is not None else lambda x: x
        result = list()
        for obj in ltobjs:
            if isinstance(obj, collections.Iterable):
                crd_txt_dict = {tuple([o.bbox[c] for c in to_list(crd)]): format(PDFHelper.get_text_from_ltobj(o))
                                for o in obj}
            else:
                crd_txt_dict = {
                    tuple([obj.bbox[c] for c in to_list(crd)]): format(PDFHelper.get_text_from_ltobj(obj))}
            result.append(crd_txt_dict)
        return result





class CME(object):
    URL_ADV = 'http://www.cmegroup.com/daily_bulletin/monthly_volume/Web_ADV_Report_CMEG.pdf'
    PDF_ADV = 'CME_Average_Daily_Volume.pdf'

    URL_PRODSLATE = 'http://www.cmegroup.com/CmeWS/mvc/ProductSlate/V1/Download.xls'
    XLS_PRODSLATE = 'Product_Slate.xls'

    # metadata in adv.pdf
    adv_pdf_levels = [3, 4]
    adv_date_no = 1
    adv_headers_no = range(3, 11)
    adv_contents_no = range(11, 19)
    adv_products_no = 19

    def __init__(self, download_path):
        self.download_path = download_path
        self.full_path_adv = os.path.join(self.download_path, self.PDF_ADV)
        self.full_path_prodslate = os.path.join(self.download_path, self.XLS_PRODSLATE)

    def download_adv(self):
        download(self.URL_ADV, self.full_path_adv)

    def download_prodslate(self):
        download(self.URL_PRODSLATE, self.full_path_prodslate)

    def parse_pdf_adv(self):
        with open(self.full_path_adv, 'rb') as infile:
            doc = PDFHelper.setup_pdfdocument(infile)
            interpreter, device = PDFHelper.setup_interpreter()
            table_df = pd.DataFrame()
            for i, page in enumerate(PDFPage.create_pages(document=doc)):
                interpreter.process_page(page)
                ltobjs = list(device.get_result())
                header_objs = self.__get_header_objs(ltobjs)
                content_objs = self.__get_content_objs(ltobjs)
                column_objs = PDFHelper.match_ltobjs_by_coordinate(header_objs, content_objs, 2)
                header_objs, content_objs = PDFHelper.unzip_ltobjs_inorder(column_objs, lambda x: x[0].bbox[0])
                contents = PDFHelper.ltobjs_to_dict(content_objs, [1, 3], self.format_text)
                product_obj = ltobjs[self.adv_products_no]
                products = PDFHelper.ltobjs_to_dict([product_obj], [1, 3], self.format_text)

                table_dict = PDFHelper.zip_dicts_by_keys(products + contents, contents[0].keys())
                table = self.__sort_tabledict_by_row_crd(table_dict)
                headers = self.get_output_header(header_objs)
                tmp = pd.DataFrame(table, columns=headers)
                print(tmp)
                table_df.append(tmp, ignore_index=True)
                print(table_df)





    # returns a dictionary with key: full group name, and value: (asset, instrument)
    def get_product_group(self, sections):
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

    def get_output_header(self, header_objs):
        return ['Product'] + [self.get_adv_header(obj) for obj in header_objs]

    def get_adv_header(self, ltobj):
        return str(self.format_text(PDFHelper.get_text_from_ltobj(ltobj)))

    def format_text(self, text):
        return (' '.join(str(text).split('\n'))).rstrip()

    # returns the date string at the top of every page
    def get_adv_todate(self, ltobjs):
        return PDFHelper.get_text_from_ltobj(ltobjs[self.adv_date_no]).rstrip()

    def __get_content_objs(self, ltobjs):
        return [ltobjs[ind] for ind in self.adv_contents_no]

    def __get_header_objs(self, ltobjs):
        return [ltobjs[ind] for ind in self.adv_headers_no]

    def __sort_tabledict_by_row_crd(self, table_dict):
        return [v for k, v in sorted(table_dict.items(), key=lambda x: x[0][0], reverse=True)]

    # parameter columns is a ordered list of entries [header, dict{(y0, y1): text}]
    # def zip_cols_by_key(self, columns):
    #     cols_no_header = [content for header, content in columns]
    #     if any(len(col) != len(cols_no_header[0]) for col in cols_no_header):
    #         raise ValueError('Unable to zip columns: row numbers inconsistent')
    #     rows = PDFHelper.__zip_dicts_by_equal_keys(cols_no_header)
    #     return [v for k, v in (sorted(rows, key=rows.items()[0][0])).items()]




class OSE(object):
    URL_OSE = 'http://www.jpx.co.jp'
    URL_VOLUME = URL_OSE + '/english/markets/statistics-derivatives/trading-volume/01.html'

    PDF_ADV = 'OSE_Average_Daily_Volume.pdf'

    TABLE_TAB = 'table'
    TR_TAB = 'tr'
    TH_TAB = 'th'
    TD_TAB = 'td'
    A_TAB = 'a'
    HREF_ATTR = 'href'
    YEAR_INTERESTED = '2017'
    TABLE_TITLE = 'Year'


    def __init__(self, download_path):
        self.download_path = download_path
        self.full_path_adv = os.path.join(self.download_path, self.PDF_ADV)



    def find_report_url(self):
        soup = make_soup(self.URL_VOLUME)

        tables = soup.find_all(self.TABLE_TAB)
        table = self.filter_table(tables, self.TABLE_TITLE)
        ths = table.find(self.TR_TAB).find_all(self.TH_TAB)

        # find in headers for the column of the year
        col_idx = self.get_indexes(ths, lambda x: x.text == self.YEAR_INTERESTED)[0]

        trs = table.find_all(self.TR_TAB)
        td_rows = [tr for tr in trs if tr.find_all(self.TD_TAB)]
        tds = [tr.find_all(self.TD_TAB)[col_idx] for tr in td_rows]

        links = [str(td.find(href=True)[self.HREF_ATTR]) for td in tds]
        pattern = r'^(?=.*{}.*\.pdf).*$'.format(self.YEAR_INTERESTED)
        file_url = self.find_first(links, lambda x: re.match(pattern, x))
        return self.URL_OSE + file_url

    def download_adv(self):
        download(self.find_report_url(), self.full_path_adv)



    def get_indexes(self, arry, condition):
        return [i for i, a in enumerate(arry) if condition(a)]

    # returns the table in which first tr has first th of which text = title
    def filter_table(self, tables, title):
        return [tbl for tbl in tables if tbl.find(self.TR_TAB).find(self.TH_TAB, text=title)][0]

    def find_first(self, arry, condition):
        for a in arry:
            if condition(a):
                return a
        return None

    def parse_pdf_adv(self):
        with open(self.full_path_adv, 'rb') as infile:
            doc = PDFHelper.setup_pdfdocument(infile)
            outlines = PDFHelper.extract_pdf_sections(doc)

            interpreter, device = PDFHelper.setup_interpreter()
            table_df = pd.DataFrame()
            for i, page in enumerate(PDFPage.create_pages(document=doc)):
                interpreter.process_page(page)
                ltobjs = list(device.get_result())
                # header_objs = self.__get_header_objs(ltobjs)
                # content_objs = self.__get_content_objs(ltobjs)
                # column_objs = PDFHelper.match_ltobjs_by_coordinate(header_objs, content_objs, 2)
                # header_objs, content_objs = PDFHelper.unzip_ltobjs_inorder(column_objs, lambda x: x[0].bbox[0])
                # contents = PDFHelper.ltobjs_to_dict(content_objs, [1, 3], self.format_text)
                # product_obj = ltobjs[self.adv_products_no]
                # products = PDFHelper.ltobjs_to_dict([product_obj], [1, 3], self.format_text)
                #
                # table_dict = PDFHelper.zip_dicts_by_keys(products + contents, contents[0].keys())
                # table = self.__sort_tabledict_by_row_crd(table_dict)
                # headers = self.get_output_header(header_objs)
                # tmp = pd.DataFrame(table, columns=headers)
                # print(tmp)
                # table_df.append(tmp, ignore_index=True)
                # print(table_df)
                pass









#download_path = '/Users/sheryllan/Downloads/'
download_path = '/home/slan/Documents/downloads/'
cme = CME(download_path)
# cme.download_adv()
# cme.download_prodslate()
#tb = cme.parse_pdf_adv()

ose = OSE(download_path)
#ose.download_adv()
ose.parse_pdf_adv()


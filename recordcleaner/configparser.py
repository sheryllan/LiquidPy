from bs4 import BeautifulSoup
import pandas as pd
import openpyxl
import os
from StyleFrame import StyleFrame


class XmlParser(object):
    @staticmethod
    # returns a list of dictionaries
    def fltr_attrs(tags, attrs=None):
        if attrs is None:
            return tags
        return [{k: v for k, v in list(tag.attrs.items()) if k in attrs} for tag in tags]

    @staticmethod
    def parse(path, tags=None):
        with open(path, 'r') as input:
            raw_xml = input.read()
        soup = BeautifulSoup(raw_xml, 'html.parser')
        return soup if tags is None else soup.find_all(tags)


class XlsxWriter(object):

    @staticmethod
    def load_xlsx(filepath):
        return openpyxl.load_workbook(filepath) if os.path.isfile(filepath) else openpyxl.Workbook()

    @staticmethod
    def __config_xlwriter(wrt, wb):
        wrt.book = wb
        wrt.sheets = dict((ws.title, ws) for ws in wb.worksheets)
        return wrt

    @staticmethod
    def create_xlwriter(path, override=True):
        wrt = pd.ExcelWriter(path, engine='openpyxl')
        return wrt if override else XlsxWriter.__config_xlwriter(wrt, XlsxWriter.load_xlsx(path))

    @staticmethod
    def to_xlsheet(data, wrt, sheet, columns=None, col_width=None):
        df = pd.DataFrame(data) if columns is None else pd.DataFrame(data, columns=columns)
        sf = StyleFrame(df)
        if col_width is None:
            col_width = {col: max(df[col].astype(str).map(len).max(), len(str(col))) + 1 for col in df.columns}
        sf.set_column_width_dict(col_width)
        sf.to_excel(excel_writer=wrt, sheet_name=sheet)

    # @staticmethod
    # def save_sheets(path, sheet2data, columns=None, override=True):
    #     wrt = XlsxWriter.create_xlwriter(path, override)
    #     for sheet, data in list(sheet2data.items()):
    #         XlsxWriter.to_xlsheet(data, wrt, sheet, columns)
    #     wrt.save()
    #     return path

    @staticmethod
    def save_sheets(path, sheet2data, columns=None, sheet2colwidth=None):
        wrt = StyleFrame.ExcelWriter(path)
        for sheet, data in list(sheet2data.items()):
            width = None if sheet2colwidth is None else sheet2colwidth[sheet]
            XlsxWriter.to_xlsheet(data, wrt, sheet, columns, width)
        wrt.save()
        return path


base_path = '/home/slan/Documents/config_files/'
exchanges = ['asx', 'bloomberg', 'cme', 'eurex', 'hkfe', 'ice', 'ose', 'sgx']
source_files = {e: e + '.xml' for e in exchanges}
dest_files = {e: e + '.xlsx' for e in exchanges}

properties = {'asx': ['type', 'commodity_code', 'reactor_name'],
              'bloomberg': ['reactor_symbol', 'exchange_symbol', 'isin', 'market_code'],
              'cme': ['type', 'commodity_name', 'reactor_name'],
              'eurex': ['type', 'commodity', 'reactor_name'],
              'hkfe': ['type', 'market', 'commodity', 'name', 'reactor_name'],
              'ice': ['type', 'market', 'commodity', 'name'],
              'ose': ['type', 'symbol', 'reactor_name'],
              'sgx': ['symbo', 'reactor_name', 'typecode']
              }
param_keys = ['tags', 'fields', 'sheet']
tags = ['product']
sheets = ['Config']


def parse_save():
    for exch in exchanges:
        src_path = base_path + source_files[exch]
        dest_path = base_path + dest_files[exch]
        columns = properties[exch]
        data_parsed = XmlParser.fltr_attrs(XmlParser.parse(src_path, tags), columns)
        dict = {sheet: data_parsed for sheet in sheets}
        XlsxWriter.save_sheets(dest_path, dict, columns, True)

#parse_save()
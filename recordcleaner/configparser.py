import pandas as pd
import openpyxl
from openpyxl.utils import get_column_letter
import os
import tempfile
from utils import *


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
    def create_xlwriter(path, override=True):
        wrt = pd.ExcelWriter(path, engine='openpyxl')

        def __config_xlwriter(wrt, wb):
            wrt.book = wb
            wrt.sheets = dict((ws.title, ws) for ws in wb.worksheets)
            return wrt

        return wrt if override else __config_xlwriter(wrt, XlsxWriter.load_xlsx(path))

    @staticmethod
    def to_xlsheet(data, wrt, sheet, columns=None):
        df = pd.DataFrame(data) if columns is None else pd.DataFrame(data, columns=columns)
        df.to_excel(wrt, sheet, index=False)

    @staticmethod
    def auto_size_cols(ws):
        i = 0
        while i < ws.max_column:
            max_len = max([len(str(row.value)) for row in list(ws.columns)[i]])
            ws.column_dimensions[get_column_letter(i + 1)].width = max_len + 2
            i += 1

    @staticmethod
    def save_sheets(path, sheet2data, columns=None, override=True, auto_size=True):
        wrt = XlsxWriter.create_xlwriter(path, override)
        for sheet, data in list(sheet2data.items()):
            XlsxWriter.to_xlsheet(data, wrt, sheet, columns)
            if auto_size:
                XlsxWriter.auto_size_cols(wrt.sheets[sheet])
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

repo_url = 'http://stash.liquid-capital.liquidcap.com/projects/PPT/repos/reactor/browse/files/TNG/products'



def get_raw_file(filename):
    full_url = '/'.join([repo_url, filename])
    f_temp = tempfile.NamedTemporaryFile()
    download(full_url, f_temp)


def run_config_parse(exchs=None, dest=None, save=False):
    exchs = exchanges if exchs is None else exchs
    dest = tempfile.NamedTemporaryFile() if dest is None else dest
    results = dict()
    for exch in exchs:
        src_path = base_path + source_files[exch]
        columns = properties[exch]
        data_parsed = XmlParser.fltr_attrs(XmlParser.parse(src_path, tags), columns)
        sheet2data = {sheet: data_parsed for sheet in sheets}
        if save:
            outpath = XlsxWriter.save_sheets(dest, sheet2data, columns, True)
            results.update({exch: outpath})
        else:
            results.update({exch: sheet2data})
        return results


def parse_save(save=True):
    for exch in exchanges:
        src_path = base_path + source_files[exch]
        dest_path = base_path + dest_files[exch]
        columns = properties[exch]
        data_parsed = XmlParser.fltr_attrs(XmlParser.parse(src_path, tags), columns)
        sheet2data = {sheet: data_parsed for sheet in sheets}
        if save:
            return XlsxWriter.save_sheets(dest_path, sheet2data, columns, True)
        else:
            return sheet2data

# parse_save()
get_raw_file('asx.xml')

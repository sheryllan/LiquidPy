from bs4 import BeautifulSoup
import pandas as pd
import openpyxl
import os


class InputParams(object):
    def __init__(self, path, tags, attrs):
        self.path = path
        self.tags = tags
        self.attrs = attrs


class OutputParams(object):
    def __init__(self, path, sheets):
        self.path = path
        self.sheets = sheets


class XlsxParser(object):


    @classmethod
    # returns a list of dictionaries
    def fltr_attrs(cls, tags, attrs=None):
        if attrs is None:
            return tags
        return [{k: v for k, v in tag.attrs.items() if k in attrs} for tag in tags]

    @classmethod
    def parse(cls, path, tags=None):
        with open(path, 'r') as input:
            raw_xml = input.read()
        soup = BeautifulSoup(raw_xml, 'html.parser')
        return soup if tags is None else soup.find_all(tags)

    # @classmethod
    # def dict_to_df(cls, dict, columns=None):
    #     return pd.DataFrame(dict, columns=columns)

    @classmethod
    def create_xlsx(cls, filepath):
        if not os.path.isfile(filepath):
            wb = openpyxl.Workbook()
            wb.save(filepath)
        else:
            wb = openpyxl.load_workbook(filepath)
        return wb

    # @classmethod
    # def df_to_xlsx(cls, df, outputs, override=True):
    #     dest = outputs.path
    #     wb = cls.create_xlsx(dest)
    #     wrt = pd.ExcelWriter(dest, engine='openpyxl')
    #     if not override:
    #         wrt.book = wb
    #         wrt.sheets = dict((ws.title, ws) for ws in wb.worksheets)
    #     df.to_excel(wrt, outputs.sheet, index=False)
    #     wrt.save()

    @classmethod
    def __config_xlwriter(cls, wrt, wb):
        wrt.book = wb
        wrt.sheets = dict((ws.title, ws) for ws in wb.worksheets)
        return wrt

    @classmethod
    def df_to_xlsx(cls, df, wrt, sheet, override=True):
        if not override:
            cls.__config_xlwriter()
        df.to_excel(wrt, sheet, index=False)

    @classmethod
    def dict_to_xlsx(cls, dict, headers, wrt, sheet, override=True):
        df = pd.DataFrame(dict, headers)
        cls.df_to_xlsx(df, wrt, sheet, override)

    @classmethod
    def parse_to_save_sheets(cls, inParams, outParams):
        for ip in inParams:
            parsed = cls.fltr_attrs(cls.parse(inputs.path, inputs.tags), inputs.attrs)
            wrt = pd.ExcelWriter(outputs.path, engine='openpyxl')



base_path = '/home/slan/Documents/config_files/'
exchanges = ['asx', 'bloomberg', 'cme', 'eurex', 'hkfe', 'ice', 'ose', 'sgx']
source_files = {e: e + '.xml' for e in exchanges}
dest_files = {e: e + '.xlsx' for e in exchanges}

properties = {'asx': ['type', 'commodity_code', 'reactor_name'],
              'bloomberg': ['reactor_symbol', 'exchange_symbol', 'ISIN', 'market_code'],
              'cme': ['type', 'commodity_code', 'reactor_name'],
              'eurex': ['type', 'commodity', 'reactor_name'],
              'hkfe': ['type', 'market', 'commodity', 'name', 'reactor_name'],
              'ice': ['type', 'market', 'commodity', 'name'],
              'ose': ['type', 'symbol', 'reactor_name'],
              'sgx': ['symbol', 'reactor_name', 'typecode']
              }
param_keys = ['tags', 'fields', 'sheet']
tags = 'product'
sheet = 'Config'

a = 'fdfs'
for aa in a:
    print(isinstance([1,2,3], list))
# for exch in exchanges:
#     src = base_path + source_files[exch]
#     dest = base_path + dest_files[exch]
#     params = {param_keys[0]: tags, param_keys[1]: properties[exch], param_keys[2]: sheet}
#     XlsxParser.parse_to_save(src, dest, params)

import pandas as pd
import tempfile
from commonlib.websourcing import *
from commonlib.iohelper import XlsxWriter


BASE_PATH = '/home/slan/Documents/config_files/'
EXCHANGES = ['asx', 'cme', 'eurex', 'hkfe', 'ice', 'ose', 'sgx']
source_files = {e: e + '.xml' for e in EXCHANGES}

PROPERTIES = {'asx': ['type', 'commodity_code', 'reactor_name'],
              'cme': ['type', 'commodity_name', 'reactor_name'],
              'eurex': ['type', 'commodity', 'reactor_name'],
              'hkfe': ['type', 'market', 'commodity', 'name', 'reactor_name'],
              'ice': ['type', 'market', 'commodity', 'name'],
              'ose': ['type', 'symbol', 'reactor_name', 'description'],
              'sgx': ['typecode', 'symbo', 'reactor_name']
              }

TYPE_COLNUM = 0
TAGS = ['product']
REPO_URL = 'http://stash.liquid-capital.liquidcap.com/projects/PPT/repos/reactor/browse/files/TNG/products'

INSTRUMENT_TYPES = {'F': 'Futures',
                    'O': 'Options',
                    'S': 'Strategies',
                    'E': 'Equities'}


def get_default_destfiles():
    return {e: BASE_PATH + e + '.xlsx' for e in EXCHANGES}


def get_src_file(exch):
    return BASE_PATH + source_files[exch]


def get_raw_file(filename):
    full_url = '/'.join([REPO_URL, filename])
    f_temp = tempfile.NamedTemporaryFile()
    download(full_url, f_temp)


def parse_config(exch, outpath=None, sheetname=None):
    src_path = get_src_file(exch)
    columns = PROPERTIES[exch]
    data_parsed = fltr_attrs(make_soup(src_path).find_all(TAGS), columns)
    df_parsed = pd.DataFrame(data_parsed, columns=columns)
    type_col = columns[TYPE_COLNUM]
    df_parsed[type_col] = df_parsed[type_col].apply(lambda x: INSTRUMENT_TYPES[x])
    if outpath is not None:
        sheetname = 'configs' if sheetname is None else sheetname
        return XlsxWriter.save_sheets(outpath, {sheetname: df_parsed}, columns, True)
    return df_parsed


def run_all_configs_parse(dest_files=None, sheetname=None):
    results = dict()
    for exch in EXCHANGES:
        outpath = dest_files[exch] if dest_files is not None else None
        df_parsed = parse_config(exch, outpath, sheetname)
        results.update({exch: df_parsed})
    return results


run_all_configs_parse(get_default_destfiles())

# get_raw_file('asx.xml')

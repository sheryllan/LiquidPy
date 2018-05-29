import pandas as pd
import tempfile
from commonlib.websourcing import *
from commonlib.iohelper import XlsxWriter


BASE_PATH = '/home/slan/Documents/config_files/'
EXCHANGES = ['asx', 'cme', 'eurex', 'hkfe', 'ice', 'ose', 'sgx']
source_files = {e: e + '.xml' for e in EXCHANGES}

asx = 'asx'
cme = 'cme'
eurex = 'eurex'
hkfe = 'hkfe'
ice = 'ice'
ose = 'ose'
sgx = 'sgx'


CF_TYPE = 'type'
CF_TYPECODE = 'typecode'
CF_COMMODITY_CODE = 'commodity_code'
CF_COMMODITY_NAME = 'commodity_name'
CF_COMMODITY = 'commodity'
CF_REACTOR_NAME = 'reactor_name'
CF_NAME = 'name'
CF_SYMBO = 'symbo'
CF_SYMBOL = 'symbol'
CF_DESCRIPTION = 'description'


ATTR_NAMES = {asx: [CF_TYPE, CF_COMMODITY_CODE, CF_REACTOR_NAME],
              cme: [CF_TYPE, CF_COMMODITY_NAME, CF_REACTOR_NAME],
              eurex: [CF_TYPE, CF_COMMODITY, CF_REACTOR_NAME],
              hkfe: [CF_TYPE, CF_NAME, CF_COMMODITY, CF_REACTOR_NAME],
              ice: [CF_TYPE, CF_NAME, CF_COMMODITY, CF_REACTOR_NAME],
              ose: [CF_TYPE, CF_SYMBOL, CF_REACTOR_NAME, CF_DESCRIPTION],
              sgx: [CF_TYPECODE, CF_SYMBO, CF_REACTOR_NAME]
              }


CO_TYPE = 'type'
CO_PRODCODE = 'prod_code'
CO_REACTOR_NAME = 'reactor_name'
CO_DESCRIPTION = 'description'

CF_OUTCOLS_MAPPING = {CF_TYPE: CO_TYPE,
                      CF_TYPECODE: CO_TYPE,
                      CF_COMMODITY_CODE: CO_PRODCODE,
                      CF_COMMODITY_NAME: CO_PRODCODE,
                      CF_NAME: CO_PRODCODE,
                      CF_SYMBO: CO_PRODCODE,
                      CF_SYMBOL: CO_PRODCODE,
                      CF_REACTOR_NAME: CO_REACTOR_NAME,
                      CF_DESCRIPTION: CO_DESCRIPTION}


TAG_PRODUCT = 'product'
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


def parse_config(exch, tag=TAG_PRODUCT, columns=None, mapping_cols=CF_OUTCOLS_MAPPING, to_df=False):
    src_path = get_src_file(exch)
    columns = ATTR_NAMES[exch] if columns is None else columns
    data_parsed = fltr_attrs(make_soup(src_path).find_all(tag), columns, mapping_cols)
    config_data = list(dict_updated(d, {CO_TYPE: INSTRUMENT_TYPES.get(d[CO_TYPE], d[CO_TYPE])}) for d in data_parsed)
    if to_df:
        renamed_cols = (mapping_cols.get(c, c) for c in columns)
        config_data = pd.DataFrame(config_data, columns=renamed_cols)
    return config_data


def run_all_configs_parse(dest_files=None, sheetname=None):
    results = dict()
    for exch in EXCHANGES:
        outpath = dest_files[exch] if dest_files is not None else None
        df_parsed = parse_config(exch)
        results.update({exch: df_parsed})
    return results


# run_all_configs_parse(get_default_destfiles())

# get_raw_file('asx.xml')

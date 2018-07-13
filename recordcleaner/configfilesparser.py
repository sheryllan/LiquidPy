import os

import paramiko

from commonlib.websourcing import *

XML_SUFFIX = '.xml'

asx = 'asx'
cme = 'cme'
eurex = 'eurex'
hkfe = 'hkfe'
ice = 'ice'
ose = 'ose'
sgx = 'sgx'
EXCHANGES = [asx, cme, eurex, hkfe, ice, ose, sgx]


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


REACTOR_BOX = 'lcmint-core1'
USERNAME = 'rsprod'
CONFIG_PATH = '/opt/reactor/base/data/tng-cat/products'


def get_default_destfiles():
    return {e: e + '.xlsx' for e in EXCHANGES}


def get_src_file(exch, pkey_path=os.path.expanduser('~/.ssh/id_rsa')):
    logger = logging.getLogger(__name__)
    logger.debug('SSH to remote config files for the symbol information')
    with paramiko.SSHClient() as ssh_client:
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(hostname=REACTOR_BOX, username=USERNAME, key_filename=pkey_path)
        logger.debug('Connected to {}@{}'.format(USERNAME, REACTOR_BOX))
        file_path = os.path.join(CONFIG_PATH, exch + XML_SUFFIX)
        sftp_client = ssh_client.open_sftp()
        with sftp_client.open(file_path) as remote_file:
            logger.debug('Reading the configs for {} from the remote file {}'.format(exch, file_path))
            return remote_file.read()


def parse_config(exch, tag=TAG_PRODUCT, attrs=None, mapping_cols=CF_OUTCOLS_MAPPING, to_df=False):
    logger = logging.getLogger(__name__)
    src = get_src_file(exch)
    attrs = ATTR_NAMES[exch] if attrs is None else attrs
    data_parsed = fltr_attrs(make_soup(src).find_all(tag), attrs, mapping_cols)
    config_data = (mapping_updated(d, {CO_TYPE: INSTRUMENT_TYPES.get(d[CO_TYPE], d[CO_TYPE])}) for d in data_parsed)
    if to_df:
        renamed_cols = (mapping_cols.get(c, c) for c in attrs)
        config_data = pd.DataFrame(list(config_data), columns=renamed_cols)
    logger.debug('Configs for {} parsed into {}'.format(exch, type(config_data)))
    return config_data


# def get_raw_file(filename):
#     full_url = '/'.join([REPO_URL, filename])
#     f_temp = tempfile.NamedTemporaryFile()
#     download(full_url, f_temp)
#
# def run_all_configs_parse(dest_files=None, sheetname=None):
#     results = dict()
#     for exch in EXCHANGES:
#         outpath = dest_files[exch] if dest_files is not None else None
#         df_parsed = parse_config(exch)
#         results.update({exch: df_parsed})
#     return results


# run_all_configs_parse(get_default_destfiles())

# parse_config('cme')

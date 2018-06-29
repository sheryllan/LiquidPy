import os
from dotenv import load_dotenv


if os.getenv('DIR') is None:
    os.environ.setdefault('DIR', os.getcwd())


def cwd_full_path(filename):
    return os.path.join(os.getenv('DIR'), filename)


envfile = cwd_full_path('envfile.sh')
load_dotenv(dotenv_path=envfile)

HTTPS_HEADER = 'https://'
OUTDIR = os.getenv('OUTDIR')
ICINGA_CA_CRT = cwd_full_path(os.getenv('CA_CRT'))
ICINGA_HOST = os.getenv('ICINGA_HOST')
ICINGA_API_PORT = os.getenv('ICINGA_API_PORT')
ICINGA_API_PCR = os.getenv('ICINGA_API_PCR')
ICINGA_API_USER = os.getenv('ICINGA_API_USER')
ICINGA_API_PSW = os.getenv('ICINGA_API_PSW')


def get_icinga_api_url(child_dir):
    host = '{}:{}'.format(ICINGA_HOST, ICINGA_API_PORT)
    pcr_path = os.path.join(host, child_dir)
    return HTTPS_HEADER + pcr_path


class SettingBase(object):
    OUTPATH = None
    VOLLIM = 0
    ICINGA = False


class CMEGSetting(SettingBase):
    OUTFILE = 'CMEG_checked.xlsx'
    MATCH_FILE = 'CMEG_matched.xlsx'
    OUTPATH = os.path.join(OUTDIR, OUTFILE)
    MATCH_OUTPATH = os.path.join(OUTDIR, MATCH_FILE)
    VOLLIM = 1000
    SVC_CME = os.getenv('SVC_CME')
    SVC_CBOT = os.getenv('SVC_CBOT')
    SVC_NYMEX = os.getenv('SVC_NYMEX')


import os
from dotenv import load_dotenv


if os.getenv('DIR') is None:
    os.environ.setdefault('DIR', os.getcwd())

envfile = 'envfile.sh'
load_dotenv(dotenv_path=envfile)


class SettingBase(object):
    OUTDIR = os.getenv('OUTDIR')
    ICINGA_HOST = os.getenv('ICINGA_HOST')
    ICINGA_API_PORT = os.getenv('ICINGA_API_PORT')
    ICINGA_PCR = os.getenv('ICINGA_PCR')

    OUTPATH = None
    VOLLIM = 0
    TO_ICINGA = 'TO_ICINGA'


class CMEGSetting(SettingBase):
    OUTFILE = 'CMEG_checked.xlsx'
    MATCH_FILE = 'CMEG_matched.xlsx'
    OUTPATH = os.path.join(SettingBase.OUTDIR, OUTFILE)
    MATCH_OUTPATH = os.path.join(SettingBase.OUTDIR, MATCH_FILE)
    VOLLIM = 1000
    SERVICE = os.getenv('SVC_CME')

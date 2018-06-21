import os
from dotenv import load_dotenv


if os.getenv('DIR') is None:
    os.environ.setdefault('DIR', os.getcwd())

envfile = 'envfile.sh'
load_dotenv(dotenv_path=envfile)


class SettingBase(object):
    OUTDIR = os.getenv('OUTDIR')
    CA_CRT = os.getenv('CA_CRT')
    OUTPATH = None
    VOLLIM = 0


class CMEGSetting(SettingBase):
    OUTFILE = 'CMEG_checked.xlsx'
    MATCH_FILE = 'CMEG_matched.xlsx'
    OUTPATH = os.path.join(SettingBase.OUTDIR, OUTFILE)
    MATCH_OUTPATH = os.path.join(SettingBase.OUTDIR, MATCH_FILE)
    VOLLIM = 1000
    SVC_CME = os.getenv('SVC_CME')
    SVC_CBOT = os.getenv('SVC_CBOT')
    SVC_NYMEX = os.getenv('SVC_NYMEX')


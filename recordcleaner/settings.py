import os
from dotenv import load_dotenv


if os.getenv('DIR') is None:
    os.environ.setdefault('DIR', os.getcwd())

envfile = 'envfile.sh'
load_dotenv(dotenv_path=envfile)

OUTDIR = os.getenv('OUTDIR')


class SettingBase(object):
    OUTDIR = OUTDIR


class CMEGSetting(SettingBase):
    OUTFILE = 'CMEG_checked.xlsx'
    MATCH_FILE = 'CMEG_matched.xlsx'
    OUTPATH = os.path.join(SettingBase.OUTDIR, OUTFILE)
    MATCH_OUTPATH = os.path.join(SettingBase.OUTDIR, MATCH_FILE)
    VOLLIM = 1000

import os
from dotenv import load_dotenv


if os.getenv('DIR') is None:
    os.environ.setdefault('DIR', os.getcwd())
    DIR = os.getenv('DIR')


def cwd_full_path(filename):
    return os.path.join(DIR, filename)


envfile = cwd_full_path('envfile.sh')
load_dotenv(dotenv_path=envfile)

OUTDIR = os.getenv('OUTDIR')
ICINGA_CA_CRT = cwd_full_path('ca.crt')


class SettingBase(object):
    COUTPATH = None
    SOUTPATH = None
    VOLLIM = 0
    LOGLEVEL = 'DEBUG'
    LOGFILE = None


class CMEGSetting(SettingBase):
    COUTFILE = 'CMEG_checked.xlsx'
    SOUTFILE = 'CMEG_all.xlsx'
    COUTPATH = os.path.join(OUTDIR, COUTFILE)
    SOUTPATH = os.path.join(OUTDIR, SOUTFILE)
    VOLLIM = 1000
    SVC_CME = 'cme_check'
    SVC_CBOT = 'cbot_check'
    SVC_NYMEX = 'nymex_check'


class OSESetting(SettingBase):
    COUTFILE = 'OSE_checked.xlsx'
    SOUTFILE = 'OSE_all.xlsx'
    COUTPATH = os.path.join(OUTDIR, COUTFILE)
    SOUTPATH = os.path.join(OUTDIR, SOUTFILE)
    VOLLIM = 1000
    REPORT = 'monthly'
    SVC_OSE = 'ose_check'




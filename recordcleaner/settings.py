import os
from commonlib.commonfuncs import *

if os.getenv('DIR') is None:
    os.environ.setdefault('DIR', os.getcwd())
DIR = os.getenv('DIR')

OUTDIR = os.getenv('OUTDIR', os.path.join(DIR, 'results'))

ANNUAL = 'annual'
MONTHYLY = 'monthly'


def cwd_full_path(filename):
    return os.path.join(DIR, filename)


class SettingBase(object):
    EXCH = None
    VOLLIM = 0
    REPORT = MONTHYLY
    LOGLEVEL = 'DEBUG'
    LOGFILE = None
    DFLT_RTIME = {ANNUAL: (last_n_year(),), MONTHYLY: (last_n_year(0), last_n_month())}

    @classmethod
    def rtime(cls):
        return cls.DFLT_RTIME[cls.REPORT]

    @classmethod
    def coutpath(cls):
        filename = '{}_{}_checked.xlsx'.format(cls.EXCH, fmt_date(*cls.rtime()))
        return os.path.join(OUTDIR, filename)

    @classmethod
    def soutpath(cls):
        filename = '{}_{}_all.xlsx'.format(cls.EXCH, fmt_date(*cls.rtime()))
        return os.path.join(OUTDIR, filename)


class CMEGSetting(SettingBase):
    EXCH = 'CMEG'
    VOLLIM = 1000
    SVC_CME = 'cme_check'
    SVC_CBOT = 'cbot_check'
    SVC_NYMEX = 'nymex_check'


class OSESetting(SettingBase):
    EXCH = 'OSE'
    VOLLIM = 1000
    SVC_OSE = 'ose_check'
    DFLT_RTIME = {ANNUAL: (last_n_year(),),
                  MONTHYLY: (last_n_year(0), last_n_month(2) if date.today().day <= 5 else last_n_month())}


class EUREXSetting(SettingBase):
    EXCH = 'EUREX'
    VOLLIM = 1000
    SVC_EUREX = 'eurex_check'

import os
from dotenv import load_dotenv
from datetime import date
from dateutil.relativedelta import relativedelta

if os.getenv('DIR') is None:
    os.environ.setdefault('DIR', os.getcwd())
    DIR = os.getenv('DIR')


def cwd_full_path(filename):
    return os.path.join(DIR, filename)


envfile = cwd_full_path('envfile.sh')
load_dotenv(dotenv_path=envfile)

OUTDIR = os.getenv('OUTDIR')


def last_year():
    return (date.today() + relativedelta(years=-1)).year


def last_month():
    return (date.today() + relativedelta(months=-1)).month


def this_year():
    return date.today().year


def this_month():
    return date.today().month


def format_coutpath(exch):
    filename = '{}_checked.xlsx'.format(exch)
    return os.path.join(OUTDIR, filename)


def format_soutpath(exch):
    filename = '{}_all.xlsx'.format(exch)
    return os.path.join(OUTDIR, filename)


ANNUAL = 'annual'
MONTHYLY = 'monthly'


class SettingBase(object):
    DFLT_RTIMES = {ANNUAL: (last_year()), MONTHYLY: (this_year(), last_month())}

    COUTPATH = None
    SOUTPATH = None
    VOLLIM = 0
    REPORT = 'monthly'
    RTIME = DFLT_RTIMES[REPORT]
    LOGLEVEL = 'DEBUG'
    LOGFILE = None


class CMEGSetting(SettingBase):
    EXCH = 'CMEG'
    COUTPATH = format_coutpath(EXCH)
    SOUTPATH = format_soutpath(EXCH)
    VOLLIM = 1000
    SVC_CME = 'cme_check'
    SVC_CBOT = 'cbot_check'
    SVC_NYMEX = 'nymex_check'


class OSESetting(SettingBase):
    EXCH = 'OSE'
    COUTPATH = format_coutpath(EXCH)
    SOUTPATH = format_soutpath(EXCH)
    VOLLIM = 1000
    SVC_OSE = 'ose_check'


class EUREXSetting(SettingBase):
    EXCH = 'EUREX'
    COUTPATH = format_coutpath(EXCH)
    SOUTPATH = format_soutpath(EXCH)
    VOLLIM = 1000
    SVC_OSE = 'eurex_check'



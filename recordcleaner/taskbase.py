import argparse
import socket
from tabulate import tabulate

from productchecker import *
from settings import *


class IcingaHelper(object):
    PROCESS_CHECK_URL = get_icinga_api_url(ICINGA_API_PCR)

    TYPE = 'type'
    FILTER = 'filter'
    SVCNAME = 'service.name'
    HOSTNAME = 'host.name'
    EXIT_STATUS = 'exit_status'
    PLUGIN_OUPUT = 'plugin_output'
    CHECK_SOURCE = 'check_source'
    PERF_DATA = 'performance_data'

    SERVICE = 'Service'
    HOST = 'Host'

    PerfData = namedtuple_with_defaults(namedtuple('PerfData',
                                                   ['label', 'value', 'warn', 'crit', 'min', 'max']),
                                        {'warn': '', 'crit': '', 'min': '', 'max': ''})


    @staticmethod
    def to_json(typecode, fltname, poutput=None, exitcode=0, checksource=socket.gethostname(), perfdata=None):
        json_dict = {}
        ctype = IcingaHelper.SERVICE if typecode else IcingaHelper.HOST

        json_dict.update({IcingaHelper.TYPE: ctype})
        json_dict.update({IcingaHelper.FILTER: '{}==\"{}\"'.format(IcingaHelper.SVCNAME if typecode else IcingaHelper.HOST, fltname)})
        json_dict.update({IcingaHelper.EXIT_STATUS: exitcode})
        json_dict.update({IcingaHelper.CHECK_SOURCE: checksource})

        if poutput:
            json_dict.update({IcingaHelper.PLUGIN_OUPUT: poutput})
        if perfdata:
            json_dict.update({IcingaHelper.PERF_DATA: perfdata})

        return json.dumps(json_dict)

    @staticmethod
    def format_perf_data(data):
        for d in data:
            if d.label is None or d.value is None:
                raise ValueError('the field \"label\" and \"value\" of PerfData must not be None')
            yield '{}={};{};{};{};{}'.format(d.label, d.value, d.warn, d.crit, d.min, d.max).rstrip(';')


class TaskBase(object):
    OUTPATH = 'outpath'
    VOLLIM = 'vollim'
    ICINGA = 'icinga'

    PRODCODE = 'Prodcode'
    PRODTYPE = 'Prodtype'
    PRODNAME = 'Prodname'
    PRODGROUP = 'Prodgroup'
    VOLUME = 'Volume'

    PERF_RECORDED = 'recorded'
    PERF_UNRECORDED = 'unrecorded'
    PERF_TOT_PRODS = 'prods_tot'
    PERF_FLT_PERCENTAGE = 'flt_pct'

    ICINGA_OUTCOLS = [PRODNAME, RECORDED, PRODCODE, PRODTYPE, VOLUME]

    def __init__(self, settings):
        self.dflt_args = {self.ICINGA: settings.ICINGA, self.OUTPATH: settings.OUTPATH, self.VOLLIM: settings.VOLLIM}
        self.aparser = argparse.ArgumentParser()
        self.aparser.add_argument('-icg', '--' + self.ICINGA,  action='store_true', help='set it to enable results transfer to icinga')
        self.aparser.add_argument('-o', '--' + self.OUTPATH, type=str, help='the output path of the check results')
        self.aparser.add_argument('-v', '--' + self.VOLLIM, type=int, help='the volume threshold to filter out products')
        self.services = None
        self.voltype = ''
        self.task_args = None

        self._tot_exch = None
        self._tot_checked = None
        self._exch_prods = None
        self._checked_prods = None

    def get_count(self, data):
        raise NotImplementedError("Please implement this method")

    def scrape(self):
        raise NotImplementedError("Please implement this method")

    def check(self):
        raise NotImplementedError("Please implement this method")

    def run_scraping(self):
        self._exch_prods = self.scrape()
        self._tot_exch = self.get_count(self._exch_prods)

    def run_checking(self):
        self._checked_prods = self.check()
        self._tot_checked = self.get_count(self._checked_prods)

    def set_task_args(self, **kwargs):
        self.task_args = dict(self.dflt_args)
        stdin_args = {k: v for k, v in vars(self.aparser.parse_args()).items() if v is not None and k not in kwargs}
        self.task_args.update(stdin_args)
        self.task_args.update(kwargs)

    def row_tolist(self, d):
        return list(d.values()) if isinstance(d, dict) else to_iter(d)

    def delimit_grouped_data(self, data, fltcols=None, gcol=GROUP, delimit=None):
        first = peek_iter(data)
        if first is not None and gcol in first:
            for key, subitems in groupby(data, key=lambda x: x[gcol]):
                yield key
                for sub in subitems:
                    yield select_mapping(sub, fltcols)
                if delimit is not None:
                    yield delimit
        else:
            for d in data:
                yield select_mapping(d, fltcols)

    def tabulate_rows(self, data, outcols, tablefmt='simple', numalign='right'):
        table = [self.row_tolist(gd) for gd in self.delimit_grouped_data(data, outcols, delimit='')]
        return tabulate(table, outcols, tablefmt, numalign=numalign)

    def format_plugin_output(self, exch, voltype, data, outcols=ICINGA_OUTCOLS, tablefmt='simple', numalign='right'):
        vollim = self.task_args[self.VOLLIM]
        title = '{} products for which {} is higher than {}:'.format(exch, voltype, vollim)
        details = self.tabulate_rows(data, outcols, tablefmt, numalign)
        return '\n'.join([title, details])

    def format_perfdata(self, data, prods_tot, checked_tot):
        cnt_rcd = count_unique(data, RECORDED)
        flt_pct = '{}%'.format(round(checked_tot / prods_tot * 100), 2)

        recorded = IcingaHelper.PerfData(self.PERF_RECORDED, cnt_rcd[True])
        unrecorded = IcingaHelper.PerfData(self.PERF_UNRECORDED, cnt_rcd[False])
        prods_tot = IcingaHelper.PerfData(self.PERF_TOT_PRODS, prods_tot)
        flt_pct = IcingaHelper.PerfData(self.PERF_FLT_PERCENTAGE, flt_pct)

        return list(IcingaHelper.format_perf_data([recorded, unrecorded, prods_tot, flt_pct]))

    def to_json_data(self, data, exch, voltype, service, exit_code=0, tablefmt='simple', numalign='right'):
        poutput = self.format_plugin_output(exch, voltype, data, tablefmt=tablefmt, numalign=numalign)
        perf_data = self.format_perfdata(data, self._tot_exch[exch], self._tot_checked[exch])
        return IcingaHelper.to_json(True, service, poutput, exit_code, perfdata=perf_data)

    def post_pcr(self, data):
        url = IcingaHelper.PROCESS_CHECK_URL
        auth = (ICINGA_API_USER, ICINGA_API_PSW)
        cert = ICINGA_CA_CRT
        return http_post(url, data, auth, cert)

    def send_to_icinga(self, exit_status):
        raise NotImplementedError("Please implement this method")

    def run(self, **kwargs):
        self.set_task_args(**kwargs)
        print('Results output to {}'.format(self.task_args[self.OUTPATH]))
        exit_status = (0, None)
        try:
            self.run_scraping()
            self.run_checking()
        except Exception as e:
            print(str(e))
            exit_status = (1, e)

        if self.task_args[self.ICINGA]:
            self.send_to_icinga(exit_status)









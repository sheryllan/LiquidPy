import argparse
import json
import socket

from tabulate import tabulate

from productchecker import *


def to_tbstring(data, dtypes, cols=None, tablefmt='simple'):
    if nontypes_iterable(data, dtypes):
        for d in data:
            yield to_tbstring(d, dtypes, cols, tablefmt)
    elif isinstance(data, dtypes):
        try:
            table = select_mapping(data, cols)
        except AttributeError as e:
            raise ValueError('Invalid type of row: must be either pandas Series or dict').with_traceback(
                e.__traceback__)

        if isinstance(table, pd.Series):
            return tabulate([table.tolist()], tablefmt)
        elif isinstance(table, dict):
            return tabulate([table.values()], tablefmt)
        else:
            return tabulate([list(table)], tablefmt)
    else:
        raise TypeError('Inconsistent function input: data parameter must contain data of dtype')


def tabulate_rows(data, outcols=None, grouping=None):
    first, data = peek_iter(data)
    if first is None:
        return ''

    if grouping is not None:
        for key, subitems in groupby(data, key=grouping):
            yield tabulate([key], 'plain')
            yield from to_tbstring(subitems, type(first), outcols)
    else:
        yield from to_tbstring(data, type(first), outcols)


class IcingaHelper(object):
    TYPE = 'type'
    FILTER = 'filter'
    SVCNAME = 'service.name'
    HOSTNAME = 'host.name'
    EXIT_STATUS =  'exit_status'
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
        json_dict.update({IcingaHelper.FILTER: {IcingaHelper.SVCNAME: fltname} if typecode else {IcingaHelper.HOST: fltname}})
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

    ICINGA_OUTCOLS = {RECORDED, PRODCODE, PRODTYPE, PRODNAME}

    def __init__(self, settings):
        self.dflt_args = {self.ICINGA: False, self.OUTPATH: settings.OUTPATH, self.VOLLIM: settings.VOLLIM}
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

    def format_plugin_output(self, exch, voltype, vollim, details):
        title = '{} products for which {} is higher than {}'.format(exch, voltype, vollim)
        return '\n'.join([title, details])

    def plugin_output_details(self, data):
        return '\n'.join(tabulate_rows(data, self.ICINGA_OUTCOLS))

    def format_perfdata(self, data, prods_tot, checked_tot):
        cnt_rcd = count_unique(data, RECORDED)

        recorded = IcingaHelper.PerfData(self.PERF_RECORDED, cnt_rcd[True])
        unrecorded = IcingaHelper.PerfData(self.PERF_UNRECORDED, cnt_rcd[False])
        prods_tot = IcingaHelper.PerfData(self.PERF_TOT_PRODS, prods_tot)
        flt_pct = '{}%'.format(checked_tot / prods_tot * 100)
        flt_pct = IcingaHelper.PerfData(self.PERF_FLT_PERCENTAGE, flt_pct)

        return list(IcingaHelper.format_perf_data([recorded, unrecorded, prods_tot, flt_pct]))

    def send_to_icinga(self, exit_status, exmsg=None):
        if not exit_status:
            vollim = self.task_args[self.VOLLIM]
            for exch, data in self._checked_prods.items():
                po_details = self.plugin_output_details(data)
                perf_data = self.format_perfdata(data, self._exch_prods[exch], self._checked_prods[exch])
                poutput = self.format_plugin_output(exch, self.voltype, vollim, po_details)

                IcingaHelper.to_json(IcingaHelper.SERVICE, self.services[exch], poutput, exit_status, perfdata=perf_data)
        else:
            poutput = exmsg
            for exch, service in self.services.items():
                IcingaHelper.to_json(IcingaHelper.SERVICE, service, poutput, exit_status)

    def run(self, **kwargs):
        self.set_task_args(**kwargs)
        print('Results output to {}'.format(self.task_args[self.OUTPATH]))
        exit_status = 0
        try:
            self.run_scraping()
            self.run_checking()
        except Exception:
            exit_status = 1

        if self.task_args[self.ICINGA]:
            self.send_to_icinga(exit_status)









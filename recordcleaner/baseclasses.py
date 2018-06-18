import os
import argparse
import pandas as pd
import socket
import json
from itertools import groupby
from tabulate import tabulate

from commonlib.commonfuncs import *
from productchecker import RECORDED, GROUP


PRODCODE = 'Prodcode'
PRODTYPE = 'Prodtype'
PRODNAME = 'Prodname'
PRODGROUP = 'Prodgroup'
VOLUME = 'Volume'


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

    ICINGA_OUTCOLS = {RECORDED, PRODCODE, PRODTYPE, PRODNAME}


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



class TaskBase(object):
    OUTPATH = 'outpath'
    VOLLIM = 'vollim'

    def __init__(self, settings):
        self.dft_check_params = {self.OUTPATH: settings.OUTPATH, self.VOLLIM: settings.VOLLIM}
        self.aparser = argparse.ArgumentParser()
        self.aparser.add_argument('-o', '--' + self.OUTPATH, type=str, help='the output path of the check results')
        self.aparser.add_argument('-v', '--' + self.VOLLIM, type=int, help='the volume threshold to filter out products')
        self.to_icinga = os.getenv(settings.TO_ICINGA, False)

    def check(self, **kwargs):
        raise NotImplementedError("Please Implement this method")

    def get_args(self, **kwargs):
        args = dict(self.dft_check_params)
        stdin_args = {k: v for k, v in vars(self.aparser.parse_args()).items() if v is not None and k not in kwargs}
        args.update(stdin_args)
        args.update(kwargs)
        return args

    def format_plugin_output(self, data, exch, voltype, vollim, grouping=None):
        title = '{} products for which {} is higher than {}'.format(exch, voltype, vollim)
        details = tabulate_rows(data, IcingaHelper.ICINGA_OUTCOLS, grouping)
        poutput = '\n'.join(details)
        return '\n'.join([title, poutput])

    def run(self, **kwargs):
        args = self.get_args(**kwargs)
        print('Results output to {}'.format(args[self.OUTPATH]))
        results = self.check(**args)
        if self.to_icinga:

            for exch in results:




    # def sendto_icinga(self, results, args):


import argparse
import socket
import sys

from tabulate import tabulate

from commonlib.iohelper import LogWriter
from productchecker import *
from settings import *

ARG_ICINGA = 'icinga'
ARG_COUTPATH = 'coutpath'
ARG_SOUTPATH = 'soutpath'
ARG_VOLLIM = 'vollim'
ARG_LOGLEVEL = 'loglevel'
ARG_LOGFILE = 'logfile'
ARG_REPORT = 'report'
ARG_RTIME = 'rtime'


class MetaBase(type):
    def __init__(cls, *args):
        super().__init__(*args)

        # Explicit name mangling
        logger_attribute_name = '_' + cls.__name__ + '__logger'

        # Logger name derived accounting for inheritance for the bonus marks
        # logger_name = '.'.join([c.__name__ for c in cls.mro()[-2::-1]])
        logger_name = cls.mro()[-2].__name__

        setattr(cls, logger_attribute_name, logging.getLogger(logger_name))


class IcingaHelper(object):
    HTTPS_HEADER = 'https://'
    ICINGA_HOST = 'lcldn-icinga1'
    ICINGA_API_PORT = 5665
    ICINGA_API_PCR = 'v1/actions/process-check-result'
    ICINGA_API_USER = 'icinga'
    ICINGA_API_PSW = 'icinga2002'
    ICINGA_CA_CRT = cwd_full_path('ca.crt')

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

    LOGGER = logging.getLogger(__name__)

    @staticmethod
    def get_icinga_api_url(child_dir):
        host = '{}:{}'.format(IcingaHelper.ICINGA_HOST, IcingaHelper.ICINGA_API_PORT)
        pcr_path = os.path.join(host, child_dir)
        return IcingaHelper.HTTPS_HEADER + pcr_path


    @staticmethod
    def to_json(is_service, fltname, poutput=None, exitcode=0, checksource=socket.gethostname(), perfdata=None):
        json_dict = {}
        ctype = IcingaHelper.SERVICE if is_service else IcingaHelper.HOST

        json_dict.update({IcingaHelper.TYPE: ctype})
        json_dict.update({IcingaHelper.FILTER: '{}==\"{}\"'.format(IcingaHelper.SVCNAME if is_service else IcingaHelper.HOST, fltname)})
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

    @staticmethod
    def post_pcr(data):
        url = IcingaHelper.get_icinga_api_url(IcingaHelper.ICINGA_API_PCR)
        auth = (IcingaHelper.ICINGA_API_USER, IcingaHelper.ICINGA_API_PSW)
        cert = IcingaHelper.ICINGA_CA_CRT
        IcingaHelper.LOGGER.info('Posting data to {}, user: {}, certificate: {}'.format(url, auth, cert))
        return http_post(url, data, auth, cert)


class IcingaCheckHandler(object):
    PERF_RECORDED = 'recorded'
    PERF_UNRECORDED = 'unrecorded'
    PERF_TOT_PRODS = 'prods_tot'
    PERF_FLT_PERCENTAGE = 'flt_pct'
    PERF_TOT_GROUPS = 'groups_tot'
    PERF_GROUP_UNREC = 'gp_unrec'

    def __init__(self, df_checked, df_scraped, exch, voltype, vollim):
        self.logger = logging.getLogger(__name__)
        self._df_checked = pd.DataFrame(df_checked)
        self._df_scraped = pd.DataFrame(df_scraped)
        self._exch = exch
        self._voltype = voltype
        self._vollim = vollim
        self._groups = None
        self._cnt_checked, self._cnt_groups = {}, {}

    def get_count(self, x):
        return len(x)

    def is_group_ok(self, df_group):
        return any(x for x in df_group[RECORDED].values)

    @property
    def groups(self):
        if self._groups is None:
            self._groups = {i: self.is_group_ok(self._df_checked.loc[[i]])
                            for i in unique_gen(self._df_checked.index.get_level_values(0))}

        return self._groups

    @property
    def checked_tot(self):
        return self.get_count(self._df_checked)

    @property
    def scraped_tot(self):
        return self.get_count(self._df_scraped)

    @property
    def group_tot(self):
        return self.get_count(self.groups)

    @property
    def cnt_checked(self):
        if not self._cnt_checked:
            self._cnt_checked = count_unique(self._df_checked)
        return self._cnt_checked

    @property
    def cnt_groups(self):
        if not self._cnt_groups:
            self._cnt_groups = count_unique(self.groups.values(), None)
        return self._cnt_groups

    def tabulate_rows(self, outcols, tablefmt='simple', numalign='right'):
        outcols = to_iter(outcols)
        df = self._df_checked[outcols]

        if self.group_tot != self.checked_tot:
            table = list()
            for g in self.groups:
                table = table + [[g]] + df.loc[g].values.tolist() + [[]]
        else:
            table = df.values.tolist()

        return tabulate(table, outcols, tablefmt, numalign=numalign)

    def format_plugin_output(self, outcols, tablefmt='simple', numalign='right'):
        title = '{} products for which {} is higher than {}:'.format(self._exch, self._voltype, self._vollim)
        details = self.tabulate_rows(outcols, tablefmt, numalign)
        return '\n'.join([title, details])

    def format_perfdata(self):
        flt_pct = '{}%'.format(round(self.checked_tot / self.scraped_tot * 100), 2)

        recorded = IcingaHelper.PerfData(self.PERF_RECORDED, self.cnt_checked.get(True, 0))
        unrecorded = IcingaHelper.PerfData(self.PERF_UNRECORDED, self.cnt_checked.get(False, 0))
        prods_tot = IcingaHelper.PerfData(self.PERF_TOT_PRODS, self.scraped_tot)
        flt_pct = IcingaHelper.PerfData(self.PERF_FLT_PERCENTAGE, flt_pct)
        perf_data = [recorded, unrecorded, prods_tot, flt_pct]
        if self.group_tot != self.checked_tot:
            groups_tot = IcingaHelper.PerfData(self.PERF_TOT_GROUPS, self.group_tot)
            gp_unrec = IcingaHelper.PerfData(self.PERF_GROUP_UNREC, self.cnt_groups.get(False, 0))
            perf_data.extend([groups_tot, gp_unrec])

        return list(IcingaHelper.format_perf_data(perf_data))

    def to_json_data(self, service, outcols, tablefmt='simple', numalign='right'):
        poutput = self.format_plugin_output(outcols, tablefmt, numalign)
        perf_data = self.format_perfdata()
        return IcingaHelper.to_json(True, service, poutput, self.exit_code(), perfdata=perf_data)

    def exit_code(self):
        ok = all(x for x in self.groups.values())
        return 0 if ok else 2


class TaskBase(object, metaclass=MetaBase):
    PRODCODE = 'Prodcode'
    PRODTYPE = 'Prodtype'
    PRODNAME = 'Prodname'
    VOLUME = 'Volume'

    DFLT_OUTCOLS = [PRODNAME, RECORDED, PRODCODE, PRODTYPE, VOLUME]

    ROOT_FMT = "%(levelname)s:%(module)s[%(name)s.%(funcName)s]:%(lineno)d: %(message)s"

    def __init__(self, settings, scraper, checker):
        self.scraper, self.checker = scraper, checker
        self.aparser = argparse.ArgumentParser()
        self.aparser.add_argument('-icg', '--' + ARG_ICINGA,
                                  action='store_true',
                                  help='set it to enable results transfer to icinga')
        self.aparser.add_argument('-co', '--' + ARG_COUTPATH,
                                  type=str,
                                  nargs='?', const=settings.coutpath(),
                                  help='the output path of the check results')
        self.aparser.add_argument('-so', '--' + ARG_SOUTPATH,
                                  nargs='?', const=settings.soutpath(),
                                  type=str,
                                  help='the output path of the matching results')
        self.aparser.add_argument('-v', '--' + ARG_VOLLIM,
                                  nargs='?', default=settings.VOLLIM,
                                  type=int,
                                  help='the volume threshold to filter out products')
        self.aparser.add_argument('-rp', '--' + ARG_REPORT,
                                  nargs='?', default=settings.REPORT,
                                  type=str,
                                  help='the type of report to evaluate')
        self.aparser.add_argument('-rt', '--' + ARG_RTIME,
                                  nargs='*', default=settings.rtime(),
                                  type=int,
                                  help='set the year(yyyy)(and month(mm) if applicable) for the report')
        self.aparser.add_argument('-ll', '--' + ARG_LOGLEVEL,
                                  type=str,
                                  nargs='?', default=settings.LOGLEVEL,
                                  help='level name or number: DEBUG(10), INFO(20), WARNING(30), ERROR(40) or CRITICAL(50)')
        self.aparser.add_argument('-lf', '--' + ARG_LOGFILE,
                                  nargs='?', default=settings.LOGFILE,
                                  type=str,
                                  help='the path to log file')

        self.services = None
        self.task_args = None

        self._exch_prods = None
        self._checked_prods = None

        self.outcols = self.DFLT_OUTCOLS

    @property
    def voltype(self):
        rtime = self.task_args[ARG_RTIME]
        return 'ADV {}({})'.format(self.task_args[ARG_REPORT], fmt_date(*rtime))

    def run_scraper(self):
        self.__logger.info('Start running scraping')
        self._exch_prods = self.scraper.run(**self.task_args)
        self.__logger.info('Finish running scraping')

    def run_checker(self):
        self.__logger.info('Start running checking')
        self._checked_prods = self.checker.run(self._exch_prods, self.outcols, **self.task_args)
        self.__logger.info('Finish running checking')
        
    def get_rtime(self, value):
        if value is None:
            return self.aparser.get_default(ARG_RTIME)
        return list(map(int, to_iter(value, ittype=iter)))

    def set_task_args(self, **kwargs):
        self.task_args = vars(self.aparser.parse_args())
        if ARG_RTIME in kwargs:
            kwargs[ARG_RTIME] = self.get_rtime(kwargs[ARG_RTIME])
        self.task_args.update(**kwargs)

    def set_logger(self):
        level = self.task_args[ARG_LOGLEVEL]
        logging.basicConfig(level=level,
                            format=self.ROOT_FMT,
                            stream=sys.stdout)

        logfile = self.task_args[ARG_LOGFILE]
        if logfile:
            fh = logging.FileHandler(logfile)
            fh.setFormatter(logging.Formatter('%(asctime)s ' + self.ROOT_FMT, datefmt="%Y-%m-%d %H:%M:%S"))
            logging.getLogger().addHandler(fh)

        sys.stderr = LogWriter(self.__logger.warning)

    def send_to_icinga(self, exit_status, handler_type=IcingaCheckHandler):
        exit_code, ex = exit_status
        if not exit_code and ex is None:
            for exch, data in self._checked_prods.items():
                handler = handler_type(data, self._exch_prods[exch], exch, self.voltype, self.task_args[ARG_VOLLIM])
                service = self.services[exch]
                json_data = handler.to_json_data(service, self.DFLT_OUTCOLS)
                IcingaHelper.post_pcr(json_data)
        elif exit_code:
            for exch, service in self.services.items():
                service = self.services[exch]
                msg = format_ex_str(ex)
                json_data = IcingaHelper.to_json(True, service, msg, exit_code)
                IcingaHelper.post_pcr(json_data)

    def run(self, **kwargs):
        self.set_task_args(**kwargs)
        exit_status = (0, None)
        self.set_logger()

        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                self.run_scraper()
                self.run_checker()
            except Warning as w:
                self.__logger.warning(str(w))
            except Exception as e:
                self.__logger.error('Failed running the task', exc_info=True)
                exit_status = (2, e)
                print(format_ex_str(e))

        if self.task_args[ARG_ICINGA]:
            self.__logger.info('Sending results to icinga')
            self.send_to_icinga(exit_status)


class ScraperBase(object, metaclass=MetaBase):

    def validate_rtime(self, rtime):
        if rtime[0] < 1970 or rtime[0] > last_n_year(0):
            raise ValueError('Invalid rtime: the year is out of range (< 1970 or > {})'.format(last_n_year(0)))
        if rtime[1:] and rtime[1] < 0 or rtime[1] > 12:
            raise ValueError('Invalid rtime: the month is out of range (< 0 or > 12)')

    def scrape_args(self, kwargs):
        report = kwargs[ARG_REPORT]
        rtime = kwargs[ARG_RTIME]
        return {ARG_REPORT: report, ARG_RTIME: rtime}

    def scrape(self, report, rtime, **kwargs):
        raise NotImplementedError("Please implement this method")

    def run(self, **kwargs):
        args = self.scrape_args(kwargs)
        self.__logger.info('Running scraping with parameters: {}'.format(args))
        scraped = self.scrape(**args)

        outpath = kwargs.get(ARG_SOUTPATH, None)
        if outpath:
            XlsxWriter.save_sheets(outpath, scraped)
            self.__logger.info('Scraper results output to {}'.format(outpath))

        return scraped


class CheckerBase(object, metaclass=MetaBase):

    def __init__(self, exch_cf):
        self._config_dict = None
        self.exch_cf = exch_cf

    @property
    def config_dict(self):
        if self._config_dict is None:
            self._config_dict = get_config_dict(self.exch_cf)
            self.__logger.info('Successfully retrieve pcaps configurations for {}'.format(self.exch_cf))
        return self._config_dict

    @property
    def cols_mapping(self):
        raise NotImplementedError('Please implement this property')

    def check_args(self, kwargs):
        return {ARG_VOLLIM: kwargs[ARG_VOLLIM]}

    def check(self, data, vollim, **kwargs):
        raise NotImplementedError("Please implement this method")

    def run(self, data, outcols=None, **kwargs):
        validate_precheck(data)

        args = self.check_args(kwargs)
        self.__logger.info('Running checking with parameters: {}'.format(args))
        checked = self.check(data, **args)
        postcheck(checked, self.cols_mapping, outcols, self.__logger)

        outpath = kwargs.get(ARG_COUTPATH, None)
        if outpath:
            XlsxWriter.save_sheets(outpath, checked)
            self.__logger.info('Checker results output to {}'.format(outpath))
        return checked

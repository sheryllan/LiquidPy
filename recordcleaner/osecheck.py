import jaconv
from tempfile import TemporaryDirectory, NamedTemporaryFile

from baseclasses import *
from settings import OSESetting

from extrawhoosh.indexing import *
from extrawhoosh.analysis import *
from extrawhoosh.query import *
from extrawhoosh.searching import *


ARG_REPORT = 'report'
ARG_RTIME = 'rtime'

OSE = 'OSE'
YEARLY_TIME = (last_year(),)
ADV_YEARLY = 'ADV yearly({})'.format(*YEARLY_TIME)
MONTHLY_TIME = (this_year(), last_month())
ADV_MONTHLY = 'ADV monthly({})'.format(fmt_date(*MONTHLY_TIME))

PRODUCT_NAME = 'Product_Name'
PRODUCT_TYPE = 'Product_Type'
PRODUCT_CODE = 'Product_Code'
TRADING_VOLUME = 'Trading_Volume'
PRODTYPES = {'Futures', 'Options'}


class OSEScraper(ScraperBase):
    URL_OSE = 'http://www.jpx.co.jp'
    URL_ANNUAL_VOLUME = URL_OSE + '/english/markets/statistics-derivatives/trading-volume/01.html'
    URL_MONTHLY_VOLUME = URL_OSE + '/english/markets/statistics-derivatives/trading-volume/index.html'

    TABLE_TITLE = 'Year'

    TYPE = 'Type'
    TRADING_VOLUME_UNITS = 'Trading Volume(units)'
    DAILY_AVG = 'Daily Average'
    JNET_MKT = 'J-NET Market'

    COLS_MAPPING = {TYPE: PRODUCT_NAME, TRADING_VOLUME_UNITS: TRADING_VOLUME}
    OUTCOLS = [PRODUCT_NAME, PRODUCT_TYPE, TRADING_VOLUME]

    def __init__(self):
        super().__init__()

    def find_annual_report_url(self, year=last_year()):
        year_str = str(year)
        table = HtmlTableParser.get_tables_by_th(self.URL_ANNUAL_VOLUME, self.TABLE_TITLE)[0]
        headers = HtmlTableParser.get_tb_headers(table)

        # find in headers for the column of the year
        col_idx = headers.index(year_str)
        tds = HtmlTableParser.select_tds_by_index(table, column=col_idx)

        pattern = r'^(?=.*{}.*\.pdf).*$'.format(year_str)
        file_url = find_link(tds, pattern)
        return self.URL_OSE + file_url

    def find_monthly_report_url(self, year=this_year(), month=last_month()):
        year_str = str(year)
        table = HtmlTableParser.get_tables_by_th(self.URL_MONTHLY_VOLUME)[0]
        tr = HtmlTableParser.select_trs(table, text=year_str)[0]
        tds = HtmlTableParser.select_tds_by_index(tr, column=month)

        pattern = r'^(?=.*{}.*\.pdf).*$'.format(fmt_date(year, month))
        file_url = find_link(tds, pattern)
        return self.URL_OSE + file_url

    def get_report_url(self, report, rtime):
        funcs = {'annual': self.find_annual_report_url,
                 'monthly': self.find_monthly_report_url}
        return funcs[report](*rtime)

    def parse_tabular_lines(self, lines, header_matches, alignment=TabularTxtParser.CENTRE):
        coldict = {m[1]: None for m in header_matches}
        for line in lines:
            if not line:
                continue
            line = '  ' + jaconv.z2h(line, kana=False, digit=True, ascii=True)
            data_matches = TabularTxtParser.match_tabular_line(line, verify_func=None)
            if data_matches:
                data_matches = filter(lambda x: re.match(whole_pattern(ASCII_PATTERN), x[1]), data_matches)
                aligned_cols = TabularTxtParser.align_txt_by_min_dist(header_matches, data_matches,
                                                                      alignment=alignment).values()
                new_coldict = select_mapping({k: text_to_num(v) for k, v in aligned_cols}, coldict.keys())
                if any(coldict[col] is not None and new_coldict[col] is not None for col in new_coldict):
                    yield coldict
                    coldict = new_coldict
                else:
                    mapping_updated(coldict, new_coldict, insert=False, condition=lambda k, v: v is not None)

            if all(v is not None for v in coldict.values()):
                yield coldict
                coldict = {m[1]: None for m in header_matches}

    def parse_from_txt(self, lines=None, alignment=TabularTxtParser.CENTRE):

        def filter_adv_lines():
            prodname = ''
            for line in self.parse_tabular_lines(lines, header_matches, alignment):
                typeval = line[self.TYPE]
                if typeval is None:
                    continue
                if all(t not in typeval for t in [self.DAILY_AVG, self.JNET_MKT]):
                    prodname = typeval
                elif self.DAILY_AVG in typeval:
                    record = dict(line)
                    record[self.TYPE] = prodname
                    yield record

        lines = iter(lines)
        for line in lines:
            ln_convt = jaconv.z2h(line, kana=False, digit=True, ascii=True)
            header_matches = TabularTxtParser.match_tabular_header(ln_convt, min_splits=4,
                                                                   extra_verfunc=lambda matches: self.TYPE in [m[1] for m in matches])
            if header_matches:
                headers = [h[1] for h in header_matches]
                return pd.DataFrame(list(filter_adv_lines()), columns=headers)

    def normalise_data(self, data):

        def get_prodtype(name):
            if not any(t in name for t in PRODTYPES):
                return 'Futures'
            return find_first_n(name.split(), lambda x: x in PRODTYPES)

        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        df = rename_mapping(data, self.COLS_MAPPING)
        df[PRODUCT_TYPE] = df[PRODUCT_NAME].map(get_prodtype)
        return select_mapping(df, self.OUTCOLS, False)

    def scrape_args(self, kwargs):
        report = kwargs.get(ARG_REPORT, 'monthly')
        rtime = kwargs.get(ARG_RTIME, MONTHLY_TIME)
        return {ARG_REPORT: report, ARG_RTIME: rtime}

    def scrape(self, report, rtime):
        dl_url = self.get_report_url(report, rtime)
        self.__logger.info(('Downloading from: {}'.format(dl_url)))
        with download(dl_url, NamedTemporaryFile()) as f_pdf:
            pdfparser = PdfParser(f_pdf)
            self.__logger.info('Parsing tables from the pdf')
            tables = [self.parse_from_txt(page) for page in pdfparser.pdftotext_bypages()]
        tbdata = pd.concat(tables, ignore_index=True)
        return {OSE: self.normalise_data(tbdata)}


class OSEMatcher(object):
    OSE_COMMON_WORDS = ['futures', 'future', 'options', 'option']

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_ix_fields(self):
        regtk_exp = '[^\s/\(\)]+'
        regex_tkn = RegexTokenizerExtra(regtk_exp, ignored=False, required=False)
        lwc_flt = LowercaseFilter()
        splt_mrg_flt = SplitMergeFilter(splitcase=True, splitnums=True, ignore_splt=True, ignore_mrg=True)
        stp_flt_norm = StopFilter()
        stp_flt_spcl = StopFilter(stoplist=self.OSE_COMMON_WORDS)

        ana = regex_tkn | splt_mrg_flt | lwc_flt | stp_flt_norm | stp_flt_spcl
        return {PRODUCT_NAME: TEXT(stored=True, analyzer=ana),
                PRODUCT_TYPE: ID(stored=True, unique=True),
                TRADING_VOLUME: NUMERIC(stored=True)}

    def init_ix(self, df, ix_name, clean=True):
        self.logger.debug('Creating index {}'.format(ix_name))
        return setup_ix(self.get_ix_fields(), df, ix_name, clean)

    def __search_for_one(self, searcher, qparams, grouping_q):
        src_and = search_func(searcher,
                              *get_query_params('and', **qparams),
                              filter=grouping_q,
                              limit=None)
        src_fuzzy = search_func(searcher,
                                *get_query_params('and', **{**qparams, TERMCLASS: FuzzyTerm}),
                                filter=grouping_q,
                                limit=None)
        src_andmaybe = search_func(searcher,
                                   *get_query_params('andmaybe', **qparams),
                                   filter=grouping_q,
                                   limit=None)
        return chain_search([src_and, src_fuzzy, src_andmaybe])

    def match_by_prodname(self, data, ix, searcher=None):

        def search():
            pdnm, pdtype = data[PRODUCT_NAME], data[PRODUCT_TYPE]
            qparams = {FIELDNAME: PRODUCT_NAME, SCHEMA: ix.schema, QSTRING: pdnm}
            grouping_q = filter_query((PRODUCT_TYPE, pdtype))
            results = self.__search_for_one(searcher, qparams, grouping_q)
            return next(min_dist_rslt(results, pdnm, PRODUCT_NAME, ix.schema, minboost=0.2)).fields() \
                if results else None

        if searcher is None:
            with ix.searcher() as searcher:
                return search()
        else:
            return search()


class OSEChecker(CheckerBase):
    CONFIG_MAPPING = {ProductKey('NK225M', 'Futures'): 'Nikkei 225 mini',
                      ProductKey('NK225', 'Futures'): 'Nikkei 225 Futures',
                      ProductKey('NK225', 'Options'): 'Nikkei 225 Options',
                      ProductKey('NK225W', 'Options'): 'Nikkei 225 Weekly Options',
                      ProductKey('JN400', 'Futures'): 'JPX-Nikkei Index 400 Futures',
                      ProductKey('JN400', 'Options'): 'JPX-Nikkei Index 400 Options',
                      ProductKey('JGBL', 'Futures'): 'JGB Futures',
                      ProductKey('JGBL', 'Options'): 'Options on 10-yearJGB Futures',
                      ProductKey('TOPIX', 'Futures'): 'TOPIX Futures',
                      ProductKey('TOPIXM', 'Futures'): 'mini-TOPIX Futures'}

    COLS_MAPPING = {PRODUCT_NAME: TaskBase.PRODNAME,
                    PRODUCT_TYPE: TaskBase.PRODTYPE,
                    PRODUCT_CODE: TaskBase.PRODCODE,
                    TRADING_VOLUME: TaskBase.VOLUME}

    def __init__(self):
        super().__init__(ose)
        self.matcher = OSEMatcher()

    @property
    def cols_mapping(self):
        return self.COLS_MAPPING

    def __match_productkey(self, key, ix, searcher=None):
        if key.type not in PRODTYPES:
            return None

        if key not in self.CONFIG_MAPPING:
            self.__logger.warning(('No pre-defined mapping found for config key {},'
                                   'potentially unmatched with indexed products')
                                  .format(key))
            prodname = self.config_dict[key][CF_DESCRIPTION]
        else:
            prodname = self.CONFIG_MAPPING[key]
        config_data = {PRODUCT_NAME: prodname, PRODUCT_TYPE: key.type}
        match = self.matcher.match_by_prodname(config_data, ix, searcher)
        if not match:
            raise ValueError('Unable to match config record{}'.format(key))
        self.__logger.debug('Successfully match {} with {}'.format(key, match[PRODUCT_NAME]))
        return match

    def mark_recorded(self, df):
        labled_df = df.set_index(PRODUCT_NAME, drop=False)
        labled_df[RECORDED] = False
        labled_df[PRODUCT_CODE] = None
        with TemporaryDirectory() as ixfolder:
            ix = self.matcher.init_ix(df, ixfolder)
            with ix.searcher() as searcher:
                for key in self.config_dict:
                    match = self.__match_productkey(key, ix, searcher)
                    if match:
                        labled_df.loc[match[PRODUCT_NAME], RECORDED] = True
                        labled_df.loc[match[PRODUCT_NAME], PRODUCT_CODE] = key.prod_code
        return labled_df

    def check(self, data, vollim, **kwargs):
        for exch in data:
            df = df_lower_limit(data[exch], TRADING_VOLUME, vollim)
            data[exch] = self.mark_recorded(df)
        return data


class OSETask(TaskBase):
    VOLTYPES = {'annual': ADV_YEARLY, 'monthly': ADV_MONTHLY}

    def __init__(self):
        super().__init__(OSESetting, OSEScraper(), OSEChecker())
        self.aparser.add_argument('-rp', '--report',
                                  nargs='?', default=OSESetting.REPORT,
                                  type=str,
                                  help='the type of report to evaluate')
        self.services = {OSE: OSESetting.SVC_OSE}

    def run_scraper(self):
        self.voltype = self.task_args[ARG_REPORT]
        super().run_scraper()


if __name__ == '__main__':
    task = OSETask()
    results = task.run()

from tempfile import NamedTemporaryFile
from datetime import datetime

from baseclasses import *


EUREX = 'EUREX'

PRODUCT_GROUP = 'Product_Group'
PRODUCT_NAME = 'Product_Name'
PRODUCT_TYPE = 'Product_Type'
PRODUCT_CODE = 'Product_Code'
TRADING_VOLUME = 'Trading_Volume'

PRODTYPES = {'Futures', 'Options'}


class EUREXScraper(ScraperBase):
    URL_EUREX = 'http://www.eurexchange.com'
    URL_MONTHLY = URL_EUREX + '/exchange-en/market-data/statistics/monthly-statistics'
    REPORT_NAME = 'statistics overview'

    TRADED_CONTRACTS = 'Traded Contracts'
    AVDAILY_MONTH = 'Daily average for month'

    OUTCOLS = [PRODUCT_NAME, PRODUCT_CODE, PRODUCT_GROUP, PRODUCT_TYPE, TRADING_VOLUME]

    def __init__(self):
        super().__init__()

    def find_report_url(self, year, month):
        soup = make_soup(self.URL_MONTHLY)
        pattern = r'^(?=.*{}.*\.xls).*$'.format(fmt_date(year, month))

        text_navs = soup.find_all(text=re.compile(self.REPORT_NAME, flags=re.IGNORECASE))
        lis = (nav.find_parent(LI_TAG) for nav in text_navs)
        link = find_link(lis, pattern)

        return self.URL_EUREX + link

    def __parse_data_rows(self, df, outcols):

        def get_prodtype(name):
            if not any(t in name for t in PRODTYPES):
                return 'Futures'
            return MatchHelper.find_first_in_string(name, PRODTYPES, stemming=True)

        # known_cols = [c for c in df.columns if not re.match('^unnamed', c, flags=re.IGNORECASE)]
        # unknown_cols = [c for c in df.columns if re.match('^unnamed', c, flags=re.IGNORECASE)]
        known_cols = list(df.columns[df.columns.notna()])
        # columns = [i if pd.isna(df.columns[i]) else df.columns[i] for i in range(len(df.columns))]
        # df.columns = columns
        # unkown_cols = [c for c in df.columns if isinstance(c, int)]

        prod_group = None
        for i, row in df.iterrows():
            if pd.notnull(row.iloc[0]) and 'sum' in str(row.iloc[0]).lower():
                break

            if row[known_cols].isnull().all():
                prod_group = find_first_n(row, pd.notnull)
            elif 'sum' in str(find_first_n(row, pd.notnull)).lower():
                continue
            else:
                name_code = find_first_n(row[row.index.isna()], pd.notna, n=2)
                if len(name_code) != 2:
                    raise ValueError('Missing not null column value(s) for product name/code, found: {}'.format(name_code))
                group_type = [prod_group, get_prodtype(prod_group)]
                row_extra = pd.Series(name_code + group_type, [PRODUCT_NAME, PRODUCT_CODE, PRODUCT_GROUP, PRODUCT_TYPE])
                yield pd.concat([row_extra, row[known_cols]])[outcols]

    def get_table(self, df, min_colnum=10):

        def rename_adv_col():
            prev_col = df.columns[0]
            start = 1
            for i, curr_col in enumerate(df.columns[start:]):
                if curr_col == self.AVDAILY_MONTH and prev_col == self.TRADED_CONTRACTS:
                    df.columns.values[start + i] = TRADING_VOLUME
                    break
                prev_col = curr_col

        df = set_first_valid_header(df, df[df.notna().sum(axis=1) >= min_colnum])
        df.reset_index(drop=True, inplace=True)
        rename_adv_col()
        return pd.DataFrame(self.__parse_data_rows(df, self.OUTCOLS))

    def validate_report_rtime(self, report, rtime):
        self.validate_rtime(rtime)
        if report != MONTHYLY:
            raise ValueError('Invalid report: only {} report is available'.format(MONTHYLY))
        if len(rtime) < 2:
            raise ValueError('Invalid rtime: both year and month must be provided')

        if rtime[0] != last_n_year(0):
            raise ValueError('Invalid rtime: year must be this year')

        soup = make_soup(self.URL_MONTHLY)

        pattern = r'\b\d{{2}}.?(\w+).?{}\b'.format(last_n_year(0))
        dates = soup.find_all(text=re.compile(pattern, re.IGNORECASE))
        months = {datetime.strptime(d, '%d %b %Y').month for d in dates}

        if rtime[1] not in months:
            raise ValueError('Invalid rtime: month not available')

    def scrape(self, report, rtime, **kwargs):
        self.validate_report_rtime(report, rtime)
        with NamedTemporaryFile() as xls_file:
            fn = download(self.find_report_url(*rtime), xls_file).name
            df = pd.read_excel(fn, sheet_name=1)
            return {EUREX: self.get_table(df)}


class EUREXChecker(CheckerBase):
    COLS_MAPPING = {PRODUCT_NAME: TaskBase.PRODNAME,
                    PRODUCT_TYPE: TaskBase.PRODTYPE,
                    PRODUCT_CODE: TaskBase.PRODCODE,
                    TRADING_VOLUME: TaskBase.VOLUME}

    def __init__(self):
        super().__init__(eurex)


    @property
    def cols_mapping(self):
        return self.COLS_MAPPING

    def mark_recorded(self, df):
        prod_keys = df.apply(lambda x: (x[PRODUCT_CODE], x[PRODUCT_TYPE]), axis=1)
        df = set_check_index(df, prod_keys, drop=False)
        return mark_recorded(df, self.config_dict)

    def check(self, data, vollim, **kwargs):
        for exch in data:
            df = df_lower_limit(data[exch], TRADING_VOLUME, vollim)
            data[exch] = self.mark_recorded(df)
        return data


class EUREXTask(TaskBase):
    def __init__(self):
        super().__init__(EUREXSetting, EUREXScraper(), EUREXChecker())
        self.services = {EUREX: EUREXSetting.SVC_EUREX}


if __name__ == '__main__':
    task = EUREXTask()
    task.run()
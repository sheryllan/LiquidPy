from tempfile import NamedTemporaryFile

from baseclasses import *


ARG_REPORT = 'report'
ARG_RTIME = 'rtime'

YEARLY_TIME = (last_year(),)
ADV_YEARLY = 'ADV yearly({})'.format(*YEARLY_TIME)
MONTHLY_TIME = (this_year(), last_month())
ADV_MONTHLY = 'ADV monthly({})'.format(fmt_date(*MONTHLY_TIME))

EUREX = 'EUREX'
PRODUCT_GROUP = 'Product_Group'
PRODUCT_NAME = 'Product_Name'
PRODUCT_TYPE = 'Product_Type'
PRODUCT_CODE = 'Product_Code'

PRODTYPES = {'Futures', 'Options'}


class EUREXScraper(ScraperBase):
    URL_EUREX = 'http://www.eurexchange.com'
    URL_MONTHLY = URL_EUREX + '/exchange-en/market-data/statistics/monthly-statistics'
    REPORT_NAME = 'statistics overview'

    TRADED_CONTRACTS = 'Traded Contracts'
    AVDAILY_MONTH = 'Daily average for month'

    def __init__(self):
        super().__init__()

    def find_report_url(self, year=this_year(), month=last_month()):
        soup = make_soup(self.URL_MONTHLY)
        pattern = r'^(?=.*{}.*\.xls).*$'.format(fmt_date(year, month))

        text_navs = soup.find_all(text=re.compile(self.REPORT_NAME, flags=re.IGNORECASE))
        lis = (nav.find_parent(LI_TAG) for nav in text_navs)
        link = find_link(lis, pattern)

        return self.URL_EUREX + link

    def __parse_data_rows(self, df, outcols):
        known_cols = [c for c in df.columns if not re.match('^unnamed', c, flags=re.IGNORECASE)]
        unknown_cols = [c for c in df.columns if re.match('^unnamed', c, flags=re.IGNORECASE)]
        prod_group = None
        for i, row in df.iterrows():
            if pd.notnull(row.iloc[0]) and 'sum' in str(row.iloc[0]).lower():
                break

            if row[known_cols].isnull().all():
                prod_group = find_first_n(row, pd.notnull)
            elif 'sum' in str(find_first_n(row, pd.notnull)).lower():
                continue
            else:
                name_code = find_first_n(row[unknown_cols], pd.notnull, n=2)
                if len(name_code) != 2:
                    raise ValueError('Missing not null column value(s) for product name/code, found: {}'.format(name_code))
                group_type = [prod_group, MatchHelper.find_first_in_string(prod_group, PRODTYPES, stemming=True)]
                row_extra = pd.Series(name_code + group_type, [PRODUCT_NAME, PRODUCT_CODE, PRODUCT_GROUP, PRODUCT_TYPE])
                yield pd.concat([row_extra, row[known_cols]])[outcols]

    def get_table(self, df, min_colnum=10):
        def set_headers():
            for i, row in df.iterrows():
                indices = pd.notna(row).nonzero()[0]
                if len(indices) >= min_colnum:
                    df.rename(columns={row.index[j]: row.iloc[j] for j in indices}, inplace=True)
                    df.drop(df.index[df.index <= i], inplace=True)
                    df.reset_index(drop=True, inplace=True)
                    break

        def rename_adv_col():
            prev_col = df.columns[0]
            start = 1
            for i, curr_col in enumerate(df.columns[start:]):
                if curr_col == self.AVDAILY_MONTH and prev_col == self.TRADED_CONTRACTS:
                    df.columns.values[start + i] = ADV_MONTHLY
                    break
                prev_col = curr_col

        set_headers()
        rename_adv_col()

        outcols = [PRODUCT_NAME, PRODUCT_CODE, PRODUCT_GROUP, PRODUCT_TYPE, ADV_MONTHLY]
        return pd.DataFrame(self.__parse_data_rows(df, outcols))

    def scrape_args(self, kwargs):
        rtime = kwargs.get(ARG_RTIME, MONTHLY_TIME)
        return {ARG_RTIME: rtime}

    def scrape(self, report, rtime, **kwargs):
        with NamedTemporaryFile() as xls_file:
            fn = download(self.find_report_url(*rtime), xls_file).name
            df = pd.read_excel(pd.ExcelFile(fn, on_demand=True), sheet_name=1)
            return {EUREX: self.get_table(df)}


class EUREXChecker(CheckerBase):
    COLS_MAPPING = {PRODUCT_NAME: TaskBase.PRODNAME,
                    PRODUCT_TYPE: TaskBase.PRODTYPE,
                    PRODUCT_CODE: TaskBase.PRODCODE,
                    ADV_MONTHLY: TaskBase.VOLUME}


    @property
    def cols_mapping(self):
        pass

    def check(self, data, vollim, **kwargs):
        pass


scp = EUREXScraper()
scp.scrape()
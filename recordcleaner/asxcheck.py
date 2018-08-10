from tempfile import NamedTemporaryFile, TemporaryDirectory
import zipfile

from baseclasses import *

ASX = 'ASX'

PRODUCT_GROUP = 'Product_Group'
PRODUCT_NAME = 'Product_Name'
PRODUCT_TYPE = 'Product_Type'
PRODUCT_CODE = 'Product_Code'
TRADING_VOLUME = 'Trading_Volume'


class ASXScraper(ScraperBase):

    class ASXHistDataScraper(object):
        URL_WEEK_ZIP = 'https://www.asxhistoricaldata.com/data/week{}.zip'
        URL_JAN_JUN_ZIP = 'https://www.asxhistoricaldata.com/data/{}jan-june.zip'
        URL_JUL_DEC_ZIP = 'https://www.asxhistoricaldata.com/data/{}july-december.zip'
        URL_LISTED_ALL = 'https://www.asx.com.au/asx/research/ASXListedCompanies.csv'

        E_TICKER = 'Ticker'
        E_DATE = 'Date'
        E_OPEN = 'Open'
        E_HIGH = 'High'
        E_LOW = 'Low'
        E_CLOSE = 'Close'
        E_VOLUME = 'Volume'

        E_HEADERS = [E_TICKER, E_DATE, E_OPEN, E_HIGH, E_LOW, E_CLOSE, E_VOLUME]

        L_CMP_NAME = 'Company name'
        L_ASX_CODE = 'ASX code'
        L_GICS_GROUP = 'GICS industry group'

        COL_MAPPING = {L_CMP_NAME: PRODUCT_NAME,
                       E_VOLUME: TRADING_VOLUME}

        OUTCOLS = [PRODUCT_NAME, PRODUCT_CODE, PRODUCT_GROUP, PRODUCT_TYPE, TRADING_VOLUME]

        def get_weekzip_url(self, wdate):
            dstring = fmt_date(wdate.year, wdate.month, wdate.day, '%Y%m%d')
            return self.URL_WEEK_ZIP.format(dstring)

        def get_month_weekdays(self, year, month, weekday=4, fullweek=False, weekday_start=0):
            first = find_first_n((date(year, month, d) for d in range(1, 8)), lambda x: x.weekday() == weekday)
            weekdays = [last_n_week(i, first) for i in range(0, -5, -1)]

            if weekdays[-1].day > 7:
                raise ValueError('The day of last weekday is greater than 7')

            max_day = (weekday + 7 - weekday_start) % 7
            if weekdays[-1].day > max_day:
                weekdays = weekdays[:-1]

            if fullweek and weekdays[-1].month != month:
                weekdays = weekdays[:-1]

            return weekdays

        def all_listed_companies(self, sep=',', index_col=L_ASX_CODE):
            with NamedTemporaryFile('w+t') as fh:
                download(self.URL_LISTED_ALL, fh, decode_unicode=True)
                fh.seek(0)
                line, offset = fh.readline(), 0
                while sep not in line:
                    offset = offset + len(line)
                    line = fh.readline()
                fh.seek(offset, 0)
                return pd.read_csv(fh, index_col=index_col)

        def get_daily(self, year, month, sep=','):
            from glob import glob
            for wd in self.get_month_weekdays(year, month):
                url = self.get_weekzip_url(wd)

                with NamedTemporaryFile() as zip:
                    download(url, zip)
                    with zipfile.ZipFile(zip, 'r') as zip_ref:
                        with TemporaryDirectory() as zdir:
                            zip_ref.extractall(zdir)
                            txtfiles = glob(os.path.join(zdir, '**/*.txt'), recursive=True)

                            for txt in txtfiles:
                                yield pd.read_csv(txt, sep, names=self.E_HEADERS, index_col=self.E_TICKER)

        def get_table(self, year, month, sep=','):

            volumes, days = pd.Series(), 0
            for daily in self.get_daily(year, month, sep):
                volumes = volumes.add(daily[self.E_VOLUME], fill_value=0)
                days += 1
            volumes = (volumes / days).rename(self.E_VOLUME)

            listed_all = self.all_listed_companies()
            table = listed_all.join(volumes, how='inner')
            table[PRODUCT_TYPE] = 'Equities'
            return table

    def validate_report_rtime(self, report, rtime):
        self.validate_rtime(rtime)
        if report != MONTHYLY:
            raise ValueError('Invalid report: only {} report is available'.format(MONTHYLY))
        if len(rtime) < 2:
            raise ValueError('Invalid rtime: both year and month must be provided')




    def scrape(self, report, rtime, **kwargs):
        self.validate_rtime(rtime)
        df = self.get_equity_table(*rtime)
        return {ASX: rename_filter(df, self.COL_MAPPING, self.OUTCOLS)}



import requests
url = 'https://www.asxoptions.com/bhp/?view=360&vol=20&pcr=9&submit=View&code=bhp'
r = requests.get(url)
t = r.content

if __name__ == '__main__':
    s = ASXScraper()
    s.get_equity_table(2018, 7)
    print()

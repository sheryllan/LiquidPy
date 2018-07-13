from datascraper import *
import jaconv
import tempfile

from productchecker import *
from baseclasses import CheckerBase



OSE = 'OSE'
PRODUCT_NAME = 'Product Name'
ADV_YEARLY = 'ADV Yearly'


class OSEScraper(object):
    URL_OSE = 'http://www.jpx.co.jp'
    URL_VOLUME = URL_OSE + '/english/markets/statistics-derivatives/trading-volume/01.html'

    TABLE_TITLE = 'Year'

    TYPE = 'Type'
    VOLUME = 'Trading Volume(units)'
    DAILY_AVG = 'Daily Average'
    JNET_MKT = 'J-NET Market'

    COLS_MAPPING = {TYPE: PRODUCT_NAME, VOLUME: ADV_YEARLY}
    OUTCOLS = [PRODUCT_NAME, ADV_YEARLY]

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def find_report_url(self, year=last_year()):
        year_str = str(year)
        table = HtmlTableParser.get_tables_by_th(self.URL_VOLUME, self.TABLE_TITLE)[0]
        headers = HtmlTableParser.get_tb_headers(table)

        # find in headers for the column of the year
        col_idx = headers.index(year_str)
        tds = HtmlTableParser.get_td_rows(table, lambda x: x[col_idx])

        pattern = r'^(?=.*{}.*\.pdf).*$'.format(year_str)
        file_url = find_link(tds, pattern)
        return self.URL_OSE + file_url

    def parse_from_txt(self, lines=None, alignment='centre'):

            def update_coldict(coldict, new_coldict):
                for k, v in new_coldict.items():
                    if k in coldict and v is not None:
                        coldict[k] = v

            def parse_lines():
                coldict = {m[1]: None for m in header_matches}
                for line in lines:
                    if not line:
                        continue
                    line = '  ' + jaconv.z2h(line, kana=False, digit=True, ascii=True)
                    data_matches = TabularTxtParser.match_tabular_line(line, verify_func=None)
                    if data_matches:
                        data_matches = list(
                            filter(lambda x: re.match(whole_pattern(ASCII_PATTERN), x[1]), data_matches))
                        aligned_cols = TabularTxtParser.align_txt_by_min_dist(header_matches, data_matches,
                                                                              alignment=alignment).values()
                        aligned_cols = select_mapping({k: v for k, v in aligned_cols}, coldict.keys())
                        if any(coldict[col] is not None and aligned_cols[col] is not None for col in aligned_cols):
                            yield coldict
                            coldict = aligned_cols
                        else:
                            update_coldict(coldict, aligned_cols)

                    if all(v is not None for v in coldict.values()):
                        yield coldict
                        coldict = {m[1]: None for m in header_matches}

            def filter_adv_lines():
                prodname = ''
                a = list(parse_lines())
                for line in a:
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
                header_matches = TabularTxtParser.match_tabular_header(ln_convt, min_splits=4)
                if header_matches:
                    headers = [h[1] for h in header_matches]
                    return pd.DataFrame(list(filter_adv_lines()), columns=headers)

    def run_scraper(self, year=last_year(), outpath=None):
        dl_url = self.find_report_url(year)
        self.logger.info(('Downloading from: {}'.format(dl_url)))
        with download(dl_url, tempfile.NamedTemporaryFile()) as f_pdf:
            pdfparser = PdfParser(f_pdf)
            self.logger.info('Parsing tables from the pdf')
            tables = [self.parse_from_txt(page) for page in pdfparser.pdftotext_bypages()]

        df = pd.concat(tables, ignore_index=True)
        df = rename_filter(df, self.COLS_MAPPING, self.OUTCOLS)
        self.logger.info('Finished scraping')
        if outpath:
            XlsxWriter.save_sheets(outpath, {OSE: df})
            self.logger.info('Scraper results saved to {}'.format(outpath))
        return df


class OSEChecker(CheckerBase):
    def __init__(self):
        super().__init__(ose)
        


s = OSEScraper()
ddd = s.run_scraper()

print(ddd)
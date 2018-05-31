from datascraper import *
import jaconv
import tempfile
from commonlib.iohelper import XlsxWriter


class OSEScraper(object):
    URL_OSE = 'http://www.jpx.co.jp'
    URL_VOLUME = URL_OSE + '/english/markets/statistics-derivatives/trading-volume/01.html'

    TABLE_TITLE = 'Year'

    TYPE = 'Type'
    VOLUME = 'Trading Volume(units)'
    DAILY_AVG = 'Daily Average'

    def find_report_url(self, year=last_year()):
        year_str = str(year)
        tbparser = HtmlTableParser(self.URL_VOLUME, self.filter_table)
        headers = tbparser.get_tb_headers()

        # find in headers for the column of the year
        col_idx = headers.index(year_str)
        tds = tbparser.get_td_rows(lambda x: x[col_idx])

        pattern = r'^(?=.*{}.*\.pdf).*$'.format(year_str)
        file_url = find_link(tds, pattern)
        return self.URL_OSE + file_url

    # returns the table in which first tr has first th of which text = title
    def filter_table(self, tables):
        return [tbl for tbl in tables if tbl.find(TR_TAB).find(TH_TAB, text=self.TABLE_TITLE)][0]

    @staticmethod
    def get_ascii_words(string, pattern_ascii='([A-Za-z0-9/\(\)\.%&$,-]|(?<! ) (?! ))+'):
        pattern_words = '(^|(?<=((?<!\S) ))){}($|(?=( (?!\S))))'
        pattern_ascii_words = pattern_words.format(pattern_ascii)
        return list(re.finditer(pattern_ascii_words, string))

    @staticmethod
    def is_header(line, pattern_data='([A-Za-z0-9/\(\)\.%&$,-]|(?<! ) (?! ))+',
                  pattern_header='[A-Za-z]+', min_colnum=4):
        matches = OSEScraper.get_ascii_words(line, pattern_data)
        if len(matches) < min_colnum:
            return False
        isheader = all(re.search(pattern_header, m.group()) for m in matches)
        if isheader:
            return matches

    @staticmethod
    def get_col_header(lines, pattern_data='([A-Za-z0-9/\(\)\.%&$,-]|(?<! ) (?! ))+',
                       pattern_header='[A-Za-z]+', min_colnum=4):
        for line in lines:
            isheader = OSEScraper.is_header(line, pattern_data, pattern_header, min_colnum)
            if isheader:
                return isheader
        return None

    def parse_from_txt(self, lines=None, alignment='centre'):
            header_mtchobjs, data_row = None, dict()
            lines = iter(lines)
            line = next(lines, None)

            while line is not None and not header_mtchobjs:
                ln_convt = jaconv.z2h(line, kana=False, digit=True, ascii=True)
                header_mtchobjs = self.is_header(ln_convt)
                line = next(lines, None)
            headers = [h.group() for h in header_mtchobjs]
            results = pd.DataFrame(columns=headers)
            while line is not None:
                ln_convt = jaconv.z2h(line, kana=False, digit=True, ascii=True)
                data_piece = OSEScraper.get_ascii_words(ln_convt)
                aligned_cols = TabularTxtParser.align_by_mse(header_mtchobjs, data_piece, alignment)
                if aligned_cols and any(c in data_row for c in aligned_cols):
                    results = results.append(pd.DataFrame([data_row], columns=headers))
                    data_row = aligned_cols
                elif aligned_cols:
                    data_row.update(aligned_cols)
                line = next(lines, None)
            if data_row:
                results = results.append(pd.DataFrame([data_row], columns=headers))
            return results

    def parse_by_pages(self, pdf_name, end_page, start_page=1, outpath=None, sheetname=None):
        tables = list()
        for page in range(start_page, end_page + 1):
            txt = run_pdftotext(pdf_name, f=page, l=page)
            df = self.parse_from_txt(txt)
            tables.append(df)
        results = pd.concat(tables)
        if outpath:
            sheetname = 'Sheet1' if sheetname is None else sheetname
            return XlsxWriter.save_sheets(outpath, {sheetname: results})
        return results

    def run_scraper(self, year=last_year()):
        dl_url = self.find_report_url(year)
        f_pdf = download(dl_url, tempfile.NamedTemporaryFile())
        num_pages = PdfFileReader(f_pdf).getNumPages()
        results = self.parse_by_pages(f_pdf.name, num_pages)
        f_pdf.close()
        return results

    def filter_adv(self, df):
        df_adv = pd.DataFrame(columns=df.columns.values)
        last_type = None
        for i, row in df.iterrows():
            if pd.isnull(row[OSEScraper.TYPE]):
                continue
            if OSEScraper.DAILY_AVG not in row[OSEScraper.TYPE]:
                last_type = row[OSEScraper.TYPE]
            elif last_type is not None and not pd.isnull(last_type):
                row[OSEScraper.TYPE] = last_type
                df_adv = df_adv.append(row, ignore_index=True)
                last_type = None
        return df_adv

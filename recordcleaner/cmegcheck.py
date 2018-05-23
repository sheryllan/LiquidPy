from PyPDF2.generic import Destination
from whoosh.searching import Hit

from datascraper import *
from extrawhoosh.analysis import *
from extrawhoosh.indexing import *
from extrawhoosh.query import *
from extrawhoosh.searching import *
from productchecker import *

PRODUCT_NAME = 'Product Name'
PRODUCT_GROUP = 'Product Group'
CLEARED_AS = 'Cleared As'
PATTERN_ADV_YTD = 'ADV Y.T.D'

CLEARING = 'Clearing'
GLOBEX = 'Globex'
SUB_GROUP = 'Sub Group'
EXCHANGE = 'Exchange'
COMMODITY = 'Commodity'

CME = 'CME'
CBOT = 'CBOT'
NYMEX = 'NYMEX'
COMEX = 'COMEX'

ADV_COLS = {CME: [PRODUCT_NAME, PRODUCT_GROUP, CLEARED_AS],
            CBOT: [PRODUCT_NAME, PRODUCT_GROUP, CLEARED_AS],
            NYMEX: [PRODUCT_NAME, PRODUCT_GROUP, CLEARED_AS, COMMODITY]}
PRODS_COLS = [PRODUCT_NAME, PRODUCT_GROUP, CLEARED_AS, CLEARING, GLOBEX, SUB_GROUP, EXCHANGE]


def validate_df(dfs_dict, cols_dict):
    for key, df in dfs_dict.items():
        if not all(c in cols_dict[key] for c in df.columns):
            raise ValueError('Input {} dataframe is missing necessary columns'.format(key))


def get_ytd_header(df, year=None):
    year = last_year() if not year else year
    header = list(df.columns.values)
    return find_first_n(header, lambda x: PATTERN_ADV_YTD in x and str(year) in x)


def get_filtered_adv_cols(dfs, years=None):
    return {list(df.columns) if years is not None else exch: ADV_COLS[exch] + [get_ytd_header(df, y) for y in years]
            for exch, df in dfs.items()}


class CMEGScraper(object):
    URL_CME_ADV = 'http://www.cmegroup.com/daily_bulletin/monthly_volume/Web_ADV_Report_CME.pdf'
    URL_CBOT_ADV = 'http://www.cmegroup.com/daily_bulletin/monthly_volume/Web_ADV_Report_CBOT.pdf'
    URL_NYMEX_COMEX_ADV = 'http://www.cmegroup.com/daily_bulletin/monthly_volume/Web_ADV_Report_NYMEX_COMEX.pdf'
    URL_PRODSLATE = 'http://www.cmegroup.com/CmeWS/mvc/ProductSlate/V1/Download.xls'
    ALIGNMENT_ADV = TxtFormatter.RIGHT

    def default_adv_basenames(self):
        basename_cme = rreplace(os.path.basename(self.URL_CME_ADV), PDF_SUFFIX, '', 1)
        basename_cbot = rreplace(os.path.basename(self.URL_CBOT_ADV), PDF_SUFFIX, '', 1)
        basename_nymex = rreplace(os.path.basename(self.URL_NYMEX_COMEX_ADV), PDF_SUFFIX, '', 1)
        return basename_cme, basename_cbot, basename_nymex

    def default_adv_xlsx(self):
        basename_cme, basename_cbot, basename_nymex = self.default_adv_basenames()
        adv_xlsx_cme = basename_cme + XLSX_SUFFIX
        adv_xlsx_cbot = basename_cbot + XLSX_SUFFIX
        adv_xlsx_nymex = basename_nymex + XLSX_SUFFIX
        return adv_xlsx_cme, adv_xlsx_cbot, adv_xlsx_nymex

    # def get_pdf_product_groups(self, sections):
    #     prev_level = sections[0][0]
    #     asset = sections[0][1]
    #     result = dict()
    #     for level, title in sections[1:]:
    #         if level < prev_level:
    #             asset = title
    #         else:
    #             instrument = title
    #             result[('{} {}'.format(asset, instrument))] = (asset, instrument)
    #         prev_level = level
    #     return result

    # returns a dictionary with key: full group name, and value: (asset, instrument)
    def get_prod_groups_from_pdf(self, fh):
        fr = PdfFileReader(fh)
        outlines = fr.getOutlines()
        flat_outlines = flatten_iter(outlines, 0, (str, Destination))
        sections = [(l, o.title) for l, o in flat_outlines if l > 1]

        def parse_flatten_sections():
            prev_level = sections[0][0]
            asset = sections[0][1]
            result = dict()
            for level, title in sections[1:]:
                if level < prev_level:
                    asset = title
                else:
                    instrument = title
                    result[('{} {}'.format(asset, instrument))] = (asset, instrument)
                prev_level = level
            return result

        return parse_flatten_sections()

    def parse_table_from_txt(self, exch, pdf, lines):
        product_groups = self.get_prod_groups_from_pdf(pdf)
        group, clearedas = None, None
        lines = iter(lines)
        header_dict = self.__parse_adv_header(lines)
        if not header_dict:
            return None
        cols_benchmark, adv_header = list(header_dict.keys()), list(header_dict.values())

        tbl_rows = list()
        for line in lines:
            if match_tabular_header(line) or 'total' in line.lower():
                break
            line = line.strip()
            matches = match_tabular_line(line, min_splits=len(header_dict))
            if matches:
                prod_data = [matches[0], group, clearedas]
                match_dict = {m[0]: m[1] for m in matches}
                aligned = TxtFormatter.align_by_min_tot_offset(match_dict.keys(), cols_benchmark, CMEGScraper.ALIGNMENT_ADV)
                adv_data = [match_dict[cords[0]] for cords in aligned]
                tbl_rows.append(prod_data + adv_data)
            elif line in product_groups:
                group, clearedas = product_groups[line]

        return pd.DataFrame.from_records(tbl_rows, columns=ADV_COLS[exch] + adv_header)

    def download_parse_prods(self, outpath=None):
        with tempfile.NamedTemporaryFile() as prods_file:
            xls = pd.ExcelFile(download(self.URL_PRODSLATE, prods_file).name, on_demand=True)
            df_prods = pd.read_excel(xls)
            nonna_subset = [PRODUCT_NAME, PRODUCT_GROUP, CLEARED_AS]
            df_clean = clean_df(set_df_col(df_prods), nonna_subset)
            if outpath is not None:
                sheetname = xls.sheet_names[0]
                return XlsxWriter.save_sheets(outpath, {sheetname: df_clean})
            return df_clean

    def download_parse_adv(self, exch, url, outpath=None):
        f_pdf = download(url, tempfile.NamedTemporaryFile())
        num_pages = get_pdf_num_pages(f_pdf)
        table = pdftotext_parse_by_pages(f_pdf.name, lambda x: self.parse_table_from_txt(exch, f_pdf, x), num_pages)
        f_pdf.close()
        if outpath is not None:
            exch = 'Sheet1' if exch is None else exch
            return XlsxWriter.save_sheets(outpath, {exch: table})
        return table

    def run_scraper(self, path_prods=None, path_advs=None, years=None):
        df_prods = self.download_parse_prods(path_prods)
        df_cme = self.download_parse_adv(CME, self.URL_CME_ADV, outpath=path_advs)
        df_cbot = self.download_parse_adv(CBOT, self.URL_CBOT_ADV, outpath=path_advs)
        df_nymex = self.download_parse_adv(NYMEX, self.URL_NYMEX_COMEX_ADV, outpath=path_advs)

        adv_dict = {CME: df_cme, CBOT: df_cbot, NYMEX: df_nymex}
        out_cols = get_filtered_adv_cols(adv_dict, years)
        fltred_advs = {exch: df_adv[out_cols[exch]] for exch, df_adv in adv_dict.items()}
        return df_prods, fltred_advs

    def __parse_adv_header(self, lines, p_separator=None, min_splits=None):
        lines = iter(lines)
        for line in lines:
            matches = match_tabular_header(line, p_separator, min_splits)
            if matches is not None:
                line1, line2 = next(lines, ''), line
                cords1 = [m[0] for m in matches]
                cords2 = [m[0] for m in match_tabular_header(line2, p_separator, min_splits)]

                c_selector = 0 if len(cords1) >= len(cords2) else 1
                merged = TxtFormatter.merge_2rows(line1, line2, cords1, cords2, self.__merge_headers, self.ALIGNMENT_ADV)
                return {m[0][c_selector]: m[1] for m in merged}
        return None

    def __merge_headers(self, h1, h2):
        h1 = h1.strip()
        h2 = h2.strip()
        return ' '.join(filter(None, [h1, h2]))

    def __parse_line(self, line, **kwargs):
        kw_pattern = 'pattern'
        kw_extras = 'extras'
        pattern = kwargs[kw_pattern] if kw_pattern in kwargs else '(\S+( \S+)*)+'
        values = [v[0] for v in re.findall(pattern, line)]
        values = self.__text_to_num(values)
        return values if kw_extras not in kwargs else values + list(kwargs[kw_extras])

    def __append_to_df(self, df, line, *args):
        line_parsed = self.__parse_line(line, extras=list(args))
        df = df.append(pd.Series(line_parsed, index=df.columns), ignore_index=True)
        return df

    def __text_to_num(self, values):
        pattern = '^-?[\d\.,]+%?$'
        for i, value in enumerate(values):
            if re.match(pattern, value):
                value = value.replace('%', '')
                value = value.replace(',', '')
                values[i] = float(value) if '.' in value else int(value)
        return values


class CMEGMatcher(object):
    F_PRODUCT_NAME = 'Product_Name'
    F_PRODUCT_GROUP = 'Product_Group'
    F_CLEARED_AS = 'Cleared_As'
    F_CLEARING = CLEARING
    F_GLOBEX = GLOBEX
    F_SUB_GROUP = 'Sub_Group'
    F_EXCHANGE = EXCHANGE

    COL2FIELD = {PRODUCT_NAME: F_PRODUCT_NAME,
                 PRODUCT_GROUP: F_PRODUCT_GROUP,
                 CLEARED_AS: F_CLEARED_AS,
                 CLEARING: F_CLEARING,
                 GLOBEX: F_GLOBEX,
                 SUB_GROUP: F_SUB_GROUP,
                 EXCHANGE: F_EXCHANGE}

    # region CME specific
    CME_EXACT_MAPPING = {
        ('EURO MIDCURVE', 'Interest Rate', 'Options'): 'Eurodollar MC',
        ('NIKKEI 225 ($) STOCK', 'Equity Index', 'Futures'): 'Nikkei/USD',
        ('NIKKEI 225 (YEN) STOCK', 'Equity Index', 'Futures'): 'Nikkei/Yen',
        ('FT-SE 100', 'Equity Index', 'Futures'): 'E-mini FTSE 100 Index (GBP)',
        ('BDI', 'FX', 'Futures'): 'CME Bloomberg Dollar Spot Index',
        ('SKR/USD CROSS RATES', 'FX', 'Futures'): 'Swedish Krona',
        ('NKR/USD CROSS RATE', 'FX', 'Futures'): 'Norwegian Krone',
        ('S.AFRICAN RAND', 'FX', 'Futures'): 'South African Rand',
        ('CHINESE RENMINBI (CNH)', 'FX', 'Futures'): 'Standard-Size USD/Offshore RMB (CNH)',
        ('MILK', 'Ag Products', 'Futures'): 'Class III Milk',
        ('MILK', 'Ag Products', 'Options'): 'Class III Milk'
    }

    CME_NOTFOUND_PRODS = {('AUSTRALIAN DOLLAR', 'FX', 'Options'),
                          ('BRITISH POUND', 'FX', 'Options'),
                          ('CANADIAN DOLLAR', 'FX', 'Options'),
                          ('EURO FX', 'FX', 'Options'),
                          ('JAPANESE YEN', 'FX', 'Options'),
                          ('SWISS FRANC', 'FX', 'Options'),
                          ('JAPANESE YEN (EU)', 'FX', 'Options'),
                          ('SWISS FRANC (EU)', 'FX', 'Options'),
                          ('AUSTRALIAN DLR (EU)', 'FX', 'Options'),
                          ('MLK MID', 'Ag Products', 'Futures')}

    CME_MULTI_MATCH = {('EURO MIDCURVE', 'Interest Rate', 'Options'): {QUERY: 'andmaybe'},
                       ('AUD/USD PQO 2pm Fix', 'FX', 'Options'): {QUERY: 'andmaybe'},
                       ('GBP/USD PQO 2pm Fix', 'FX', 'Options'): {QUERY: 'andmaybe'},
                       ('JPY/USD PQO 2pm Fix', 'FX', 'Options'): {QUERY: 'andmaybe'},
                       ('EUR/USD PQO 2pm Fix', 'FX', 'Options'): {QUERY: 'andmaybe'},
                       ('CAD/USD PQO 2pm Fix', 'FX', 'Options'): {QUERY: 'andmaybe'},
                       ('CHF/USD PQO 2pm Fix', 'FX', 'Options'): {QUERY: 'andmaybe'},
                       ('LV CATL CSO', 'Ag Products', 'Options'): {QUERY: 'and'}}

    CRRNCY_TOKENSUB = {'aud': [TokenSub('australian', 1.5, True, True), TokenSub('dollar', 1, True, True)],
                       'gbp': [TokenSub('british', 1.5, True, True), TokenSub('pound', 1.5, True, True)],
                       'cad': [TokenSub('canadian', 1.5, True, True), TokenSub('dollar', 1, True, True)],
                       'euro': [TokenSub('euro', 1.5, True, True)],
                       'jpy': [TokenSub('japanese', 1.5, True, True), TokenSub('yen', 1.5, True, True)],
                       'nzd': [TokenSub('new', 1.5, True, True), TokenSub('zealand', 1.5, True, True),
                               TokenSub('dollar', 1, True, True)],
                       'nkr': [TokenSub('norwegian', 1.5, True, True), TokenSub('krone', 1.5, True, True)],
                       'sek': [TokenSub('swedish', 1.5, True, True), TokenSub('krona', 1.5, True, True)],
                       'chf': [TokenSub('swiss', 1.5, True, True), TokenSub('franc', 1.5, True, True)],
                       'zar': [TokenSub('south', 1.5, True, True), TokenSub('african', 1.5, True, True),
                               TokenSub('rand', 1.5, True, True)],
                       'pln': [TokenSub('polish', 1.5, True, True), TokenSub('zloty', 1.5, True, True)],
                       'inr': [TokenSub('indian', 1.5, True, True), TokenSub('rupee', 1.5, True, True)],
                       'rmb': [TokenSub('chinese', 1.5, True, True), TokenSub('renminbi', 1.5, True, True)],
                       'usd': [TokenSub('american', 1, True, False), TokenSub('dollar', 1, True, False)],
                       'clp': [TokenSub('chilean', 1.5, True, True), TokenSub('peso', 1.5, True, True)],
                       'mxn': [TokenSub('mexican', 1.5, True, True), TokenSub('peso', 1.5, True, True)],
                       'brl': [TokenSub('brazilian', 1.5, True, True), TokenSub('real', 1.5, True, True)],
                       'huf': [TokenSub('hungarian', 1.5, True, True), TokenSub('forint', 1.5, True, True)]
                       }

    CRRNCY_MAPPING = {'ad': CRRNCY_TOKENSUB['aud'],
                      'bp': CRRNCY_TOKENSUB['gbp'],
                      'cd': CRRNCY_TOKENSUB['cad'],
                      'ec': CRRNCY_TOKENSUB['euro'] +
                            [TokenSub('cross', 0.5, True, False), TokenSub('rates', 0.5, True, False)],
                      'efx': CRRNCY_TOKENSUB['euro'] +
                             [TokenSub('fx', 0.8, True, False)],
                      'jy': CRRNCY_TOKENSUB['jpy'],
                      'jpy': CRRNCY_TOKENSUB['jpy'],
                      'ne': CRRNCY_TOKENSUB['nzd'],
                      'nok': CRRNCY_TOKENSUB['nkr'],
                      'sek': CRRNCY_TOKENSUB['sek'],
                      'sf': CRRNCY_TOKENSUB['chf'],
                      'skr': CRRNCY_TOKENSUB['sek'],
                      'zar': CRRNCY_TOKENSUB['zar'],
                      'aud': CRRNCY_TOKENSUB['aud'],
                      'cad': CRRNCY_TOKENSUB['cad'],
                      'chf': CRRNCY_TOKENSUB['chf'],
                      'eur': CRRNCY_TOKENSUB['euro'],
                      'gbp': CRRNCY_TOKENSUB['gbp'],
                      'pln': CRRNCY_TOKENSUB['pln'],
                      'nkr': CRRNCY_TOKENSUB['nkr'],
                      'inr': CRRNCY_TOKENSUB['inr'],
                      'rmb': CRRNCY_TOKENSUB['rmb'],
                      'usd': CRRNCY_TOKENSUB['usd'],
                      'clp': CRRNCY_TOKENSUB['clp'],
                      'nzd': CRRNCY_TOKENSUB['nzd'],
                      'mxn': CRRNCY_TOKENSUB['mxn'],
                      'brl': CRRNCY_TOKENSUB['brl'],
                      'cnh': CRRNCY_TOKENSUB['rmb'],
                      'huf': CRRNCY_TOKENSUB['huf']}

    CME_SPECIAL_MAPPING = {'mc': [TokenSub('midcurve', 1, True, True)],
                           'pqo': [TokenSub('premium', 1, True, True), TokenSub('quoted', 1, True, True),
                                   TokenSub('european', 1, True, True), TokenSub('style', 1, True, True)],
                           'eow': [TokenSub('weekly', 1, True, True), TokenSub('wk', 1, True, False)],
                           'eom': [TokenSub('monthly', 1, True, True)],
                           'usdzar': CRRNCY_TOKENSUB['usd'] + CRRNCY_TOKENSUB['zar'],
                           'biotech': [TokenSub('biotechnology', 1.5, True, True)],
                           'us': [TokenSub('american', 1, True, False)],
                           'eu': [TokenSub('european', 1.5, True, True)],
                           'nfd': [TokenSub('non', 1.5, True, True), TokenSub('fat', 1.5, True, True),
                                   TokenSub('dry', 1.5, True, True)],
                           'cs': [TokenSub('cash', 1.5, True, True), TokenSub('settled', 1.5, True, True)],
                           'er': [TokenSub('excess', 1.5, True, True), TokenSub('return', 1.5, True, True)],
                           'catl': [TokenSub('cattle', 1.5, True, False)]}

    CME_COMMON_WORDS = ['futures', 'future', 'options', 'option', 'index', 'cross', 'rate', 'rates']

    CME_KEYWORD_MAPPING = {**CRRNCY_MAPPING, **CME_SPECIAL_MAPPING}

    CME_KYWRD_EXCLU = {'nasdaq', 'ibovespa', 'index', 'mini', 'micro', 'nikkei', 'russell', 'ftse', 'swap'}
    # endregion

    # region CBOT specific
    CBOT_EXACT_MAPPING = {('30-YR BOND', 'Interest Rate', 'Futures'): 'U.S. Treasury Bond',
                          ('30-YR BOND', 'Interest Rate', 'Options'): 'U.S. Treasury Bond',
                          ('DOW-UBS COMMOD INDEX', 'Ag Products', 'Futures'): 'Bloomberg Commodity Index',
                          ('DJ_UBS ROLL SELECT INDEX FU', 'Ag Products', 'Futures'):
                              'Bloomberg Roll Select Commodity Index',
                          ('SOYBN NEARBY+2 CAL SPRD', 'Ag Products', 'Options'): 'Consecutive Soybean CSO',
                          ('WHEAT NEARBY+2 CAL SPRD', 'Ag Products', 'Options'): 'Consecutive Wheat CSO',
                          ('CORN NEARBY+2 CAL SPRD', 'Ag Products', 'Options'): 'Consecutive Corn CSO'
                          }

    CBOT_SPECIAL_MAPPING = {'yr': [TokenSub('year', 1, True, False)],
                            'fed': [TokenSub('federal', 1.5, True, True)],
                            't': [TokenSub('treasury', 1, True, True)],
                            'note': [TokenSub('note', 1, True, True)],
                            'dj': [TokenSub('dow', 1, True, True), TokenSub('jones', 1, True, True)],
                            'cso': [TokenSub('calendar', 1.5, True, True), TokenSub('spread', 1, False, False)],
                            'cal': [TokenSub('calendar', 1.5, True, True)],
                            'hrw': [TokenSub('hr', 1.5, True, True), TokenSub('wheat', 1.5, True, True)],
                            'icso': [TokenSub('intercommodity', 1.5, True, True), TokenSub('spread', 1, True, True)],
                            'chi': [TokenSub('chicago', 1, True, True)]
                            }

    CBOT_MULTI_MATCH = {('30-YR BOND', 'Interest Rate', 'Options'): {QUERY: 'andnot', NOTWORDS: 'ultra'},
                        ('10-YR NOTE', 'Interest Rate', 'Options'): {QUERY: 'andnot', NOTWORDS: 'ultra'},
                        ('5-YR NOTE', 'Interest Rate', 'Options'): {QUERY: 'andnot', NOTWORDS: 'ultra'},
                        ('2-YR NOTE', 'Interest Rate', 'Options'): {QUERY: 'andnot', NOTWORDS: 'ultra'},
                        ('FED FUND', 'Interest Rate', 'Options'): {QUERY: 'andmaybe'},
                        ('ULTRA T-BOND', 'Interest Rate', 'Options'): {QUERY: 'andmaybe', ANDEXTRAS: 'ultra bond'},
                        ('Ultra 10-Year Note', 'Interest Rate', 'Options'): {QUERY: 'and'},
                        ('FERTILIZER PRODUCS', 'Ag Products', 'Futures'): {QUERY: 'every'},
                        ('DEC-JULY WHEAT CAL SPRD', 'Ag Products', 'Options'): {QUERY: 'and'},
                        ('JULY-DEC WHEAT CAL SPRD', 'Ag Products', 'Options'): {QUERY: 'and'},
                        ('MAR-JULY WHEAT CAL SPRD', 'Ag Products', 'Options'): {QUERY: 'and'},
                        ('Intercommodity Spread', 'Ag Products', 'Options'):
                            {QUERY: 'orofand', ANDLIST: ['MGEX-Chicago SRW Wheat Spread',
                                                         'KC HRW-Chicago SRW Wheat Intercommodity Spread',
                                                         'MGEX-KC HRW Wheat Intercommodity Spread']}}

    CBOT_COMMON_WORDS = ['futures', 'future', 'options', 'option']
    # endregion

    STOP_LIST = ['and', 'is', 'it', 'an', 'as', 'at', 'have', 'in', 'yet', 'if', 'from', 'for', 'when',
                 'by', 'to', 'you', 'be', 'we', 'that', 'may', 'not', 'with', 'tbd', 'a', 'on', 'your',
                 'this', 'of', 'will', 'can', 'the', 'or', 'are']

    def __match_pdgp(self, s_ref, s_sample):
        return s_ref == s_sample or MatchHelper.match_in_string(s_ref, s_sample, one=True, stemming=True) \
               or MatchHelper.match_initials(s_ref, s_sample) or MatchHelper.match_first_n(s_ref, s_sample)

    def __match_in_string(self, guess, indexed, one=True):
        guess = guess.lower()
        p = inflect.engine()
        for idx in indexed:
            matched = MatchHelper.match_in_string(guess, idx, one, stemming=True, engine=p)
            if matched:
                return idx
        return None

    def __verify_clearedas(self, prodname, row_clras, clras_set):
        clras = MatchHelper.get_words(prodname)[-1]
        found_clras = self.__match_in_string(clras, clras_set)
        return found_clras if found_clras is not None else row_clras

    def match_by_prodname(self, df_adv, ix, exact_mapping=None, notfound=None, multi_match=None):
        df_matched = pd.DataFrame(columns=list(df_adv.columns) + ix.schema.stored_names())
        with ix.searcher() as searcher:
            lexicons = get_idx_lexicon(searcher, CMEGMatcher.F_PRODUCT_GROUP, CMEGMatcher.F_CLEARED_AS,
                                       **{CMEGMatcher.F_PRODUCT_GROUP: CMEGMatcher.F_SUB_GROUP})

            for i, row in df_adv.iterrows():
                results = self.__match_a_row(row, lexicons, exact_mapping, notfound, multi_match, ix.schema, searcher)
                rows_matched = self.__join_results(results, row)
                df_toadd = pd.DataFrame(rows_matched, columns=df_matched.columns)
                df_matched = df_matched.append(df_toadd, ignore_index=True)
                matched = True if results else False
                self.__prt_match_status(matched, rows_matched)
        return df_matched

    def __match_a_row(self, row, lexicons, exact_mapping, notfound, multi_match, schema, searcher):
        pd_id = (row[PRODUCT_NAME], row[PRODUCT_GROUP], row[CLEARED_AS])
        if notfound is not None and pd_id in notfound:
            return None
        pdnm = exact_mapping[pd_id] if pd_id in exact_mapping else row[PRODUCT_NAME]
        is_one = True if multi_match is None or pd_id not in multi_match else False
        grouping_q = self.__get_grouping_query(row, pdnm, lexicons)
        qparams = {FIELDNAME: CMEGMatcher.F_PRODUCT_NAME,
                   SCHEMA: schema,
                   QSTRING: pdnm}
        min_dist = True

        def callback(val):
            min_dist = val

        if is_one:
            results = self.__search_for_one(searcher, qparams, grouping_q, callback)
            if results and min_dist:
                results = min_dist_rslt(results, pdnm, CMEGMatcher.F_PRODUCT_NAME, schema, minboost=0.2)[0]
        else:
            q_configs = multi_match[pd_id]
            qparams.update(q_configs)
            results = self.__search_for_all(searcher, qparams, grouping_q, callback)
        return results

    def __get_grouping_query(self, row, pdnm, lexicons):
        prods_pdgps, prods_subgps = lexicons[CMEGMatcher.F_PRODUCT_GROUP], lexicons[CMEGMatcher.F_SUB_GROUP]
        prods_clras = lexicons[CMEGMatcher.F_CLEARED_AS]
        pdgp = find_first_n(prods_pdgps, lambda x: self.__match_pdgp(x, row[PRODUCT_GROUP]))
        clras = self.__verify_clearedas(pdnm, row[CLEARED_AS], prods_clras)
        subgp = self.__match_in_string(pdnm, prods_subgps[pdgp], False)
        return filter_query((CMEGMatcher.F_PRODUCT_GROUP, pdgp),
                            (CMEGMatcher.F_CLEARED_AS, clras),
                            (CMEGMatcher.F_SUB_GROUP, subgp))

    def __search_for_one(self, searcher, qparams, grouping_q, callback):
        src_and = search_func(searcher,
                              *get_query_params('and', **qparams),
                              lambda: callback(True),
                              filter=grouping_q,
                              limit=None)
        src_fuzzy = search_func(searcher,
                                *get_query_params('and', **{**qparams, TERMCLASS: FuzzyTerm}),
                                lambda: callback(False),
                                filter=grouping_q,
                                limit=None)
        src_andmaybe = search_func(searcher,
                                   *get_query_params('andmaybe', **qparams),
                                   lambda: callback(True),
                                   filter=grouping_q,
                                   limit=None)
        return chain_search([src_and, src_fuzzy, src_andmaybe])

    def __search_for_all(self, searcher, qparams, grouping_q, callback):
        query = qparams[QUERY]
        src = search_func(searcher,
                          *get_query_params(**qparams),
                          lambda: callback(True),
                          filter=grouping_q,
                          limit=None)

        src_fuzzy = search_func(searcher,
                                *get_query_params(**{**qparams, TERMCLASS: FuzzyTerm}),
                                lambda: callback(False),
                                filter=grouping_q,
                                limit=None)

        def chain_condition(r):
            if not r:
                return True
            if r and query != 'every' and len(r) < 2:
                return True
            return False

        return chain_search([src, src_fuzzy], chain_condition)

    def __prt_match_status(self, matched, rows_matched):
        for r in rows_matched:
            if not matched:
                print('Failed matching {}'.format(r[PRODUCT_NAME]))
            else:
                print('Successful matching {} with {}'.format(r[PRODUCT_NAME], r[CMEGMatcher.F_PRODUCT_NAME]))

    def __join_results(self, results, df):
        joined_dicts = []

        def df_to_dict():
            return {k: v for k, v in df.items()}

        results = [results] if isinstance(results, Hit) else results
        if results:
            for r in results:
                df_copy = df_to_dict()
                df_copy.update(r)
                joined_dicts.append(df_copy)
        else:
            joined_dicts.append(df_to_dict())
        return joined_dicts

    def __match_prods_by_commodity(self, df_prods, df_adv):
        prod_dict = {str(row[GLOBEX]): row for _, row in df_prods.iterrows()}
        prod_dict.update({str(row[CLEARING]): row for _, row in df_prods.iterrows()})
        matched_rows = [{**row, **prod_dict[str(row[COMMODITY])]} for _, row in df_adv.iterrows()
                        if str(row[COMMODITY]) in prod_dict]
        cols = list(df_adv.columns) + list(df_prods.columns)
        df = pd.DataFrame(matched_rows, columns=cols)
        return df

    def get_cbot_fields(self):
        regtk_exp = '[^\s/\(\)]+'
        regex_tkn = RegexTokenizerExtra(regtk_exp, ignored=False, required=False)
        lwc_flt = LowercaseFilter()
        splt_mrg_flt = SplitMergeFilter(splitcase=True, splitnums=True, ignore_splt=True)

        stp_flt = StopFilter(stoplist=CMEGMatcher.STOP_LIST + CMEGMatcher.CBOT_COMMON_WORDS, minsize=1)
        sp_flt = SpecialWordFilter(CMEGMatcher.CBOT_SPECIAL_MAPPING)
        vw_flt = VowelFilter(lift_ignore=False)
        multi_flt = MultiFilterFixed(index=vw_flt)
        ana = regex_tkn | splt_mrg_flt | lwc_flt | stp_flt | sp_flt | multi_flt | stp_flt

        return {CMEGMatcher.F_PRODUCT_NAME: TEXT(stored=True, analyzer=ana),
                CMEGMatcher.F_PRODUCT_GROUP: ID(stored=True, unique=True),
                CMEGMatcher.F_CLEARED_AS: ID(stored=True, unique=True),
                CMEGMatcher.F_CLEARING: ID(stored=True, unique=True),
                CMEGMatcher.F_GLOBEX: ID(stored=True, unique=True),
                CMEGMatcher.F_SUB_GROUP: ID(stored=True, unique=True),
                CMEGMatcher.F_EXCHANGE: ID}

    def get_cme_fields(self):
        regtk_exp = '[^\s/\(\)]+'
        regex_tkn = RegexTokenizerExtra(regtk_exp, ignored=False, required=False)
        lwc_flt = LowercaseFilter()
        splt_mrg_flt = SplitMergeFilter(mergewords=True, mergenums=True, ignore_mrg=True)

        stp_flt = StopFilter(stoplist=CMEGMatcher.STOP_LIST + CMEGMatcher.CME_COMMON_WORDS, minsize=1)
        sp_flt = SpecialWordFilter(CMEGMatcher.CME_KEYWORD_MAPPING)
        vw_flt = VowelFilter(CMEGMatcher.CME_KYWRD_EXCLU, lift_ignore=False)
        multi_flt = MultiFilterFixed(index=vw_flt)
        ana = regex_tkn | splt_mrg_flt | lwc_flt | stp_flt | sp_flt | multi_flt | stp_flt

        return {CMEGMatcher.F_PRODUCT_NAME: TEXT(stored=True, analyzer=ana),
                CMEGMatcher.F_PRODUCT_GROUP: ID(stored=True, unique=True),
                CMEGMatcher.F_CLEARED_AS: ID(stored=True, unique=True),
                CMEGMatcher.F_CLEARING: ID(stored=True, unique=True),
                CMEGMatcher.F_GLOBEX: ID(stored=True, unique=True),
                CMEGMatcher.F_SUB_GROUP: ID(stored=True, unique=True),
                CMEGMatcher.F_EXCHANGE: ID}

    def __prepfor_index(self, df_prods):
        df_ix = df_prods.rename(columns=CMEGMatcher.COL2FIELD)
        return {exch: df.reset_index(drop=0) for exch, df in df_groupby(df_ix, [EXCHANGE]).items()}

    def init_ix_cme_cbot(self, gdf_exch, ixname_cme, ixname_cbot, clean=False):

        ix_cme = setup_ix(self.get_cme_fields(), gdf_exch[CME], ixname_cme, clean)
        ix_cbot = setup_ix(self.get_cbot_fields(), gdf_exch[CBOT], ixname_cbot, clean)
        return ix_cme, ix_cbot

    def match_nymex_comex(self, df_adv, gdf_exch):
        df_nymex_comex_prods = gdf_exch[NYMEX].append(gdf_exch[COMEX], ignore_index=True)
        df_nymex_comex_prods.reset_index(drop=True, inplace=True)
        if 'index' in df_nymex_comex_prods.columns:
            df_nymex_comex_prods.drop(['index'], axis=1, inplace=True)
        df_adv = self.__match_prods_by_commodity(df_nymex_comex_prods, df_adv)
        return df_adv

    def run_pd_mtch(self, dfs_adv, df_prods, ix_names, clean=False):
        validate_df({'Products': df_prods}, {'Product': PRODS_COLS})
        validate_df(dfs_adv, ADV_COLS)
        gdf_exch = self.__prepfor_index(df_prods)
        ix_cme, ix_cbot = self.init_ix_cme_cbot(gdf_exch, *ix_names, clean)
        mdf_nymex = self.match_nymex_comex(dfs_adv[NYMEX], gdf_exch)
        mdf_cme = self.match_by_prodname(dfs_adv[CME], ix_cme, CMEGMatcher.CME_EXACT_MAPPING,
                                         CMEGMatcher.CME_NOTFOUND_PRODS, CMEGMatcher.CME_MULTI_MATCH)
        mdf_cbot = self.match_by_prodname(dfs_adv[CBOT], ix_cbot, CMEGMatcher.CBOT_EXACT_MAPPING,
                                          multi_match=CMEGMatcher.CBOT_MULTI_MATCH)
        return {CME: mdf_cme, CBOT: mdf_cbot, NYMEX: mdf_nymex}


class CMEGChecker(object):

    def get_prod_code(self, row):
        if not pd.isnull(row[CMEGMatcher.F_GLOBEX]):
            return row[CMEGMatcher.F_GLOBEX]
        elif not pd.isnull(row[CMEGMatcher.F_CLEARING]):
            return row[CMEGMatcher.F_CLEARING]
        else:
            print('no code: {}'.format(row[CMEGMatcher.F_PRODUCT_NAME]))
            return None

    def get_prod_key(self, row):
        if pd.isnull(row[CMEGMatcher.F_PRODUCT_NAME]):
            return None
        pd_code = self.get_prod_code(row)
        if pd_code is not None:
            return ProductKey(pd_code, row[CLEARED_AS])
        return None

    def __is_recorded(self, key, config_keys):
        cf_type = get_cf_type(key[1])
        new_key = (key[0], cf_type)
        return new_key in config_keys

    def check_prod_by(self, agg_dict, threshold, rec_condition, config_keys):
        prods_wanted = list()
        for k, row in agg_dict.items():
            if not rec_condition(row, threshold):
                continue
            pdnm = row[CMEGMatcher.F_PRODUCT_NAME]
            recorded = self.__is_recorded(k, config_keys)
            result = {PRODCODE: k[0], TYPE: k[1], PRODUCT: pdnm, RECORDED: recorded}
            prods_wanted.append(result)
            print(result)
        return prods_wanted

    def check_cme_cbot(self, dfs_dict, config_keys, vol_threshold):
        df_cme, df_cbot = dfs_dict[CME], dfs_dict[CBOT]

        group_key = [[PRODUCT_NAME, CLEARED_AS]]
        aggr_func = sum_unique
        dict_keyfunc = self.get_prod_key

        ytd_cme = get_ytd_header(df_cme)
        ytd_cbot = get_ytd_header(df_cbot)
        _, aggdict_cme = aggregate_todict(df_cme, group_key, ytd_cme, aggr_func, dict_keyfunc)
        _, aggdict_cbot = aggregate_todict(df_cbot, group_key, ytd_cbot, aggr_func, dict_keyfunc)

        prods_cme = self.check_prod_by(aggdict_cme, vol_threshold, lambda row, threshold: row[ytd_cme] >= threshold,
                                       config_keys)
        prods_cbot = self.check_prod_by(aggdict_cbot, vol_threshold, lambda row, threshold: row[ytd_cbot] >= threshold,
                                        config_keys)

        return {CME: prods_cme, CBOT: prods_cbot}

    def check_nymex(self, dfs_dict, config_keys, vol_threshold):
        df_nymex = dfs_dict[NYMEX]
        dict_nymex = df_todict(df_nymex, self.get_prod_key)
        ytd_nymex = get_ytd_header(df_nymex)
        prods_nymex = self.check_prod_by(dict_nymex, vol_threshold, lambda row, threshold: row[ytd_nymex] >= threshold,
                                         config_keys)
        return {NYMEX: prods_nymex}

    def run_pd_check(self, dfs_dict, vol_threshold=1000, outpath=None):
        cme = 'cme'
        config_props = cp.PROPERTIES[cme]
        config_keys = get_config_keys(cme, [config_props[1], config_props[0]])

        prods_cme_cbot = self.check_cme_cbot(dfs_dict, config_keys, vol_threshold)
        prods_nymex = self.check_nymex(dfs_dict, config_keys, vol_threshold)
        prods_cmeg = {**prods_cme_cbot, **prods_nymex}

        if outpath is not None:
            outdf_cols = [PRODCODE, TYPE, PRODUCT, RECORDED]
            outdf_dict = {exch: pd.DataFrame(prods, columns=outdf_cols) for exch, prods in prods_cmeg.items()}
            return XlsxWriter.save_sheets(outpath, outdf_dict)
        return prods_cmeg

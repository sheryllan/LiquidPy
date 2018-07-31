from tempfile import NamedTemporaryFile, TemporaryDirectory
from PyPDF2.generic import Destination

from baseclasses import *
from datascraper import *
from datascraper import TabularTxtParser as tp
from extrawhoosh.analysis import *
from extrawhoosh.indexing import *
from extrawhoosh.query import *
from extrawhoosh.searching import *
from settings import CMEGSetting


ARG_CLEAN_MATCH = 'clean_match'

A_PRODUCT_NAME = 'Product Name'
A_PRODUCT_GROUP = 'Product Group'
A_CLEARED_AS = 'Cleared As'
A_COMMODITY = 'Commodity'
A_ADV = 'Trading Volume'

F_PRODUCT_NAME = 'P_Product_Name'
F_PRODUCT_GROUP = 'P_Product_Group'
F_CLEARED_AS = 'P_Cleared_As'
F_CLEARING = 'P_Clearing'
F_GLOBEX = 'P_Globex'
F_SUB_GROUP = 'P_Sub_Group'
F_EXCHANGE = 'P_Exchange'

CME = 'CME'
CBOT = 'CBOT'
NYMEX = 'NYMEX'
COMEX = 'COMEX'


class CMEGScraper(ScraperBase):
    URL_CME_ADV = 'http://www.cmegroup.com/daily_bulletin/monthly_volume/Web_ADV_Report_CME.pdf'
    URL_CBOT_ADV = 'http://www.cmegroup.com/daily_bulletin/monthly_volume/Web_ADV_Report_CBOT.pdf'
    URL_NYMEX_COMEX_ADV = 'http://www.cmegroup.com/daily_bulletin/monthly_volume/Web_ADV_Report_NYMEX_COMEX.pdf'
    URL_PRODSLATE = 'http://www.cmegroup.com/CmeWS/mvc/ProductSlate/V1/Download.xls'

    P_PRODUCT_NAME = 'Product Name'
    P_PRODUCT_GROUP = 'Product Group'
    P_CLEARED_AS = 'Cleared As'
    P_CLEARING = 'Clearing'
    P_GLOBEX = 'Globex'
    P_SUB_GROUP = 'Sub Group'
    P_EXCHANGE = 'Exchange'

    COL2FIELD = {P_PRODUCT_NAME: F_PRODUCT_NAME,
                 P_PRODUCT_GROUP: F_PRODUCT_GROUP,
                 P_CLEARED_AS: F_CLEARED_AS,
                 P_CLEARING: F_CLEARING,
                 P_GLOBEX: F_GLOBEX,
                 P_SUB_GROUP: F_SUB_GROUP,
                 P_EXCHANGE: F_EXCHANGE}

    MUST_COLS = [A_PRODUCT_NAME, A_PRODUCT_GROUP, A_CLEARED_AS]

    ADV_OUTCOLS = {CME: MUST_COLS + [A_ADV],
                   CBOT: MUST_COLS + [A_ADV],
                   NYMEX: MUST_COLS + [A_COMMODITY, A_ADV]}

    PRODS_OUTCOLS = [F_PRODUCT_NAME, F_PRODUCT_GROUP, F_CLEARED_AS, F_CLEARING, F_GLOBEX, F_SUB_GROUP, F_EXCHANGE]

    def __init__(self):
        super().__init__()
        self.matcher = CMEGMatcher()

    class CMEGPdfParser(PdfParser):
        def __init__(self, f_pdf):
            super().__init__(f_pdf)
            self.prod_groups = self.__get_prod_groups()

        # returns a dictionary with key: full group name, and value: (asset, instrument)
        def __get_prod_groups(self):
            outlines = self.pdf_reader.getOutlines()
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

        def parse_adv_headers(self, lines, matchfunc, alignment=tp.RIGHT):
            def merge_headers(h1, h2):
                h1 = h1.strip()
                h2 = h2.strip()
                return ' '.join(filter(None, [h1, h2]))

            def merge_2rows(row1, row2, mergfunc):
                matches1, matches2 = matchfunc(row1), matchfunc(row2)
                if matches1 is None or matches2 is None:
                    return None
                aligned = TabularTxtParser.align_txt_by_min_dist(matches1, matches2, alignment=alignment, defaultval='')
                return {cords: mergfunc(*aligned[cords]) for cords in aligned}

            for line in lines:
                line1, line2 = line, next(lines, '')
                merged = merge_2rows(line1, line2, merge_headers)
                if merged is not None:
                    return merged
            return None

        def parse_table_from_txt(self, lines, alignment=tp.RIGHT):
            lines = iter(lines)
            match_headers = lambda x: tp.match_tabular_header(x, min_splits=3)
            header_dict = self.parse_adv_headers(lines, match_headers)
            if not header_dict:
                return None
            headers = [A_PRODUCT_NAME, A_PRODUCT_GROUP, A_CLEARED_AS] + list(header_dict.values())

            def parse_data_lines():
                group, clearedas = None, None
                for line in lines:
                    if not line:
                        continue
                    if match_headers(line) or 'total' in line.lower():
                        break
                    line = line.strip()
                    matches = tp.match_tabular_line(line, min_splits=len(header_dict))
                    if matches:
                        prod_data = [matches[0][1],  group,  clearedas]
                        aligned_txt = tp.align_txt_by_min_dist(matches[1:], header_dict, alignment=alignment).values()
                        adv_data = [text_to_num(t[0]) for t in aligned_txt]
                        yield prod_data + adv_data
                    elif line in self.prod_groups:
                        group, clearedas = self.prod_groups[line]

            return pd.DataFrame(list(parse_data_lines()), columns=headers)

    def get_prods_table(self):
        with NamedTemporaryFile() as prods_file:
            xls = pd.ExcelFile(download(self.URL_PRODSLATE, prods_file).name, on_demand=True)
            df_prods = pd.read_excel(xls)
            nonna_subset = [A_PRODUCT_NAME, A_PRODUCT_GROUP, A_CLEARED_AS]
            return clean_df(set_df_col(df_prods), nonna_subset)

    def get_adv_table(self, url):
        self.__logger.info(('Downloading from: {}'.format(url)))
        with download(url, NamedTemporaryFile()) as f_pdf:
            pdf_parser = self.CMEGPdfParser(f_pdf)
            self.__logger.info('Parsing tables from the pdf')
            tables = [pdf_parser.parse_table_from_txt(page) for page in pdf_parser.pdftotext_bypages()]
        return pd.concat(tables, ignore_index=True)

    def get_vol_col(self, df, report, rtime):
        pattern = 'ADV Y.T.D {}'.format(fmt_date(*rtime)) \
            if report == ANNUAL else 'ADV {}'.format(fmt_date(*rtime, fmt='%b %Y'))
        vol_col = find_first_n(df.columns, lambda x: re.match(pattern, x, re.IGNORECASE))
        if not vol_col:
            raise ValueError('No matched volume column found with pattern: {}'.format(pattern))
        return vol_col

    def scrape_args(self, kwargs):
        clean_match = kwargs.get(ARG_CLEAN_MATCH, True)
        return {**super().scrape_args(kwargs), ARG_CLEAN_MATCH: clean_match}

    def validate_rtime(self, rtime):
        year = rtime[0]
        if year > this_year() or year < this_year() - 1:
            raise ValueError('Invalid rtime: year must be this or the last year')
        if rtime[1:]:
            month = rtime[1]
            if month > last_month() or month < last_month() - 1:
                raise ValueError('Invalid rtime: month must be last month or the month before last')
            if year == last_year() and month < last_month():
                raise ValueError('Invalid rtime: month must be last month for last year')

    def scrape(self, report, rtime, **kwargs):
        self.validate_rtime(rtime)
        clean_match = kwargs[ARG_CLEAN_MATCH]
        df_prods = self.get_prods_table()
        df_prods = df_prods.rename(columns=CMEGScraper.COL2FIELD)[self.PRODS_OUTCOLS]
        self.__logger.debug('Renamed and filtered product slate dataframe columns to {}'.format(list(df_prods.columns)))

        df_cme = self.get_adv_table(self.URL_CME_ADV)
        df_cbot = self.get_adv_table(self.URL_CBOT_ADV)
        df_nymex = self.get_adv_table(self.URL_NYMEX_COMEX_ADV)

        dfs_adv = {
            CME: rename_filter(df_cme, {self.get_vol_col(df_cme, report, rtime): A_ADV}, self.ADV_OUTCOLS[CME]),
            CBOT: rename_filter(df_cbot, {self.get_vol_col(df_cbot, report, rtime): A_ADV}, self.ADV_OUTCOLS[CBOT]),
            NYMEX: rename_filter(df_nymex, {self.get_vol_col(df_nymex, report, rtime): A_ADV},
                                 self.ADV_OUTCOLS[NYMEX])}

        with TemporaryDirectory() as ixfolder_cme, TemporaryDirectory() as ixfolder_cbot:
            return self.matcher.run(df_prods, dfs_adv, (ixfolder_cme, ixfolder_cbot), clean_match)


class CMEGMatcher(object):
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
                           'gi': [TokenSub('growth', 1, True, True)],
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

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)


    # region Private methods for grouping
    def __match_pdgp(self, s_ref, s_sample):
        return s_ref == s_sample or MatchHelper.match_in_string(s_ref, s_sample, one=True, stemming=True) \
               or MatchHelper.match_initials(s_ref, s_sample) or MatchHelper.match_first_n(s_ref, s_sample)

    def __verify_clearedas(self, prodname, row_clras, clras_set):
        clras = MatchHelper.get_words(prodname)[-1]
        found_clras = MatchHelper.match_in_lexicon(clras, clras_set)
        return found_clras if found_clras is not None else row_clras

    def __get_grouping_query(self, row, pdnm, lexicons):
        prods_pdgps, prods_subgps = lexicons[F_PRODUCT_GROUP], lexicons[F_SUB_GROUP]
        prods_clras = lexicons[F_CLEARED_AS]
        pdgp = find_first_n(prods_pdgps, lambda x: self.__match_pdgp(x, row[A_PRODUCT_GROUP]))
        clras = self.__verify_clearedas(pdnm, row[A_CLEARED_AS], prods_clras)
        subgp = MatchHelper.match_in_lexicon(pdnm, prods_subgps[pdgp], one=False)
        return filter_query((F_PRODUCT_GROUP, pdgp),
                            (F_CLEARED_AS, clras),
                            (F_SUB_GROUP, subgp))

    # endregion

    # region Private methods for matching a row
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

    def __match_a_row(self, row, lexicons, exact_mapping, notfound, multi_match, schema, searcher):
        pd_id = (row[A_PRODUCT_NAME], row[A_PRODUCT_GROUP], row[A_CLEARED_AS])
        if notfound is not None and pd_id in notfound:
            return []
        pdnm = exact_mapping[pd_id] if pd_id in exact_mapping else row[A_PRODUCT_NAME]
        is_one = True if multi_match is None or pd_id not in multi_match else False
        grouping_q = self.__get_grouping_query(row, pdnm, lexicons)
        qparams = {FIELDNAME: F_PRODUCT_NAME,
                   SCHEMA: schema,
                   QSTRING: pdnm}
        min_dist = True

        def callback(val):
            min_dist = val

        if is_one:
            results = self.__search_for_one(searcher, qparams, grouping_q, callback)
            if results and min_dist:
                results = [next(min_dist_rslt(results, pdnm, F_PRODUCT_NAME, schema, minboost=0.2))]
        else:
            q_configs = multi_match[pd_id]
            qparams.update(q_configs)
            results = self.__search_for_all(searcher, qparams, grouping_q, callback)
        return results

    def __prt_match_status(self, matched, row):
        if not matched:
            self.logger.debug('Failed matching {}'.format(row[A_PRODUCT_NAME]))
        else:
            self.logger.debug('Successful matching {} with {}'.format(row[A_PRODUCT_NAME], row[F_PRODUCT_NAME]))

    def __join_row_results(self, row, results):
        for result in results:
            row_result = pd.Series(result.fields())
            yield pd.concat([row, row_result])

    # endregion

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

        return {F_PRODUCT_NAME: TEXT(stored=True, analyzer=ana),
                F_PRODUCT_GROUP: ID(stored=True, unique=True),
                F_CLEARED_AS: ID(stored=True, unique=True),
                F_CLEARING: ID(stored=True, unique=True),
                F_GLOBEX: ID(stored=True, unique=True),
                F_SUB_GROUP: ID(stored=True, unique=True),
                F_EXCHANGE: ID}

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

        return {F_PRODUCT_NAME: TEXT(stored=True, analyzer=ana),
                F_PRODUCT_GROUP: ID(stored=True, unique=True),
                F_CLEARED_AS: ID(stored=True, unique=True),
                F_CLEARING: ID(stored=True, unique=True),
                F_GLOBEX: ID(stored=True, unique=True),
                F_SUB_GROUP: ID(stored=True, unique=True),
                F_EXCHANGE: ID}

    def init_ix_cme_cbot(self, gdf_exch, ixname_cme, ixname_cbot, clean=False):
        self.logger.debug('Creating index {}'.format(ixname_cme))
        ix_cme = setup_ix(self.get_cme_fields(), gdf_exch[CME], ixname_cme, clean)
        self.logger.debug('Creating index {}'.format(ixname_cbot))
        ix_cbot = setup_ix(self.get_cbot_fields(), gdf_exch[CBOT], ixname_cbot, clean)
        return ix_cme, ix_cbot

    def match_by_prodcode(self, df_prods, df_adv, on_prods=F_CLEARING, on_adv=A_COMMODITY):
        df_adv[on_adv] = df_adv[on_adv].astype(str)
        return df_adv.merge(df_prods, left_on=on_adv, right_on=on_prods)

    def match_by_prodname(self, df_adv, ix, exact_mapping=None, notfound=None, multi_match=None):
        with ix.searcher() as searcher:
            lexicons = get_idx_lexicon(searcher, F_PRODUCT_GROUP, F_CLEARED_AS, **{F_PRODUCT_GROUP: F_SUB_GROUP})
            for i, row in df_adv.iterrows():
                results = self.__match_a_row(row, lexicons, exact_mapping, notfound, multi_match, ix.schema,
                                             searcher)
                matched = True if results else False
                for row_result in self.__join_row_results(row, results):
                    self.__prt_match_status(matched, row_result)
                    yield row_result

    def run(self, df_prods, dfs_adv, ix_names, clean=True):
        self.logger.info('Running {} with clean={}'.format(self.__class__.__name__, clean))
        gdf_exch = {exch: df.reset_index(drop=True) for exch, df in df_prods.groupby(F_EXCHANGE)}

        df_nymex_comex_prods = pd.concat([gdf_exch[NYMEX], gdf_exch[COMEX]], ignore_index=True)
        data_nymex = self.match_by_prodcode(df_nymex_comex_prods, dfs_adv[NYMEX])

        ix_cme, ix_cbot = self.init_ix_cme_cbot(gdf_exch, *ix_names, clean)
        data_cme = pd.DataFrame(self.match_by_prodname(dfs_adv[CME], ix_cme, CMEGMatcher.CME_EXACT_MAPPING,
                                                       CMEGMatcher.CME_NOTFOUND_PRODS, CMEGMatcher.CME_MULTI_MATCH))
        data_cbot = pd.DataFrame(self.match_by_prodname(dfs_adv[CBOT], ix_cbot, CMEGMatcher.CBOT_EXACT_MAPPING,
                                                        multi_match=CMEGMatcher.CBOT_MULTI_MATCH))
        self.logger.info('Finish running matching')

        return {CME: data_cme, CBOT: data_cbot, NYMEX: data_nymex}


class CMEGChecker(CheckerBase):
    PRODUCT_CODE = 'Product_Code'

    COLS_MAPPING = {PRODUCT_CODE: TaskBase.PRODCODE,
                    F_CLEARED_AS: TaskBase.PRODTYPE,
                    F_PRODUCT_NAME: TaskBase.PRODNAME,
                    A_ADV: TaskBase.VOLUME}

    def __init__(self):
        super().__init__(cme)

    @property
    def cols_mapping(self):
        return self.COLS_MAPPING

    def __prod_code(self, row):
        if not pd.isnull(row[F_GLOBEX]):
            return row[F_GLOBEX]
        elif not pd.isnull(row[F_CLEARING]):
            return row[F_CLEARING]
        else:
            self.__logger.warning('no code: {}'.format(row[F_PRODUCT_NAME]))
            return None

    def __prod_key(self, row):
        if pd.isnull(row[F_PRODUCT_NAME]):
            return None
        if row[self.PRODUCT_CODE] is not None:
            return row[self.PRODUCT_CODE], row[F_CLEARED_AS]
        return None

    def set_prodcode_col(self, dfs):
        for exch, df in dfs.items():
            df[self.PRODUCT_CODE] = df.apply(self.__prod_code, axis=1)

    def get_group_keys(self, data):
        return data[A_PRODUCT_NAME].astype(str) + ' ' + data[A_CLEARED_AS].astype(str)

    def mark_filter_prods(self, data, lower_limit, fcol):
        df = dfgroupby_aggr(data, data.index.get_level_values(GROUP), A_ADV, sum_mapping) \
            if data.index.names[0] == GROUP else data
        mark_recorded(df, self.config_dict)
        return df_lower_limit(df, fcol, lower_limit)

    def check(self, data, vollim, **kwargs):
        self.set_prodcode_col(data)

        data[CME] = set_check_index(data[CME], get_prod_keys(data[CME], self.__prod_key), self.get_group_keys(data[CME]))
        data[CBOT] = set_check_index(data[CBOT], get_prod_keys(data[CBOT], self.__prod_key), self.get_group_keys(data[CBOT]))
        data[NYMEX] = set_check_index(data[NYMEX], get_prod_keys(data[NYMEX], self.__prod_key))

        return {exch: self.mark_filter_prods(data[exch], vollim, A_ADV) for exch in data}


class CMEGTask(TaskBase):
    def __init__(self):
        super().__init__(CMEGSetting, CMEGScraper(), CMEGChecker())
        self.services = {CME: CMEGSetting.SVC_CME,
                         CBOT: CMEGSetting.SVC_CBOT,
                         NYMEX: CMEGSetting.SVC_NYMEX}


if __name__ == '__main__':
    task = CMEGTask()
    task.run()

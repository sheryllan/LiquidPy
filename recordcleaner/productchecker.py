import pandas as pd
import numpy as np
import configparser as cp
import os
import re
import inflect
import itertools
import math

from productmatcher import *



# Parse the config files
# cp.parse_save()


# reports_path = '/home/slan/Documents/exch_report/'
# configs_path = '/home/slan/Documents/config_files/'
# # checked_path = '/home/slan/Documents/checked_report/'
# checked_path = os.getcwd()
#
# EXCHANGES = ['asx', 'bloomberg', 'cme', 'cbot', 'nymex_comex', 'eurex', 'hkfe', 'ice', 'ose', 'sgx']
# REPORT_FMTNAME = 'Web_ADV_Report_{}.xlsx'
#
# REPORT_FILES = {e: REPORT_FMTNAME.format(e.upper()) for e in EXCHANGES}



# parameter file2sheet is a tuple
def xl_consolidate(file2sheet, dest):
    wrt = cp.XlsxWriter.create_xlwriter(dest, False)
    for fl, sht in file2sheet:
        xl = pd.ExcelFile(fl)
        dt = xl.parse(sht)
        cp.XlsxWriter.to_xlsheet(dt, wrt, sht)
    wrt.save()


class CMEGChecker(object):
    EXCHANGES = ['cme', 'cbot', 'nymex_comex']
    REPORT_FMTNAME = 'Web_ADV_Report_{}.xlsx'
    PRODSLAT_FILE = 'Product_Slate.xls'

    def __init__(self, checked_path=None):
        self.report_files = [self.REPORT_FMTNAME.format(e.upper()) for e in self.EXCHANGES]
        self.checked_path = checked_path if checked_path is not None else os.getcwd()
        self.cmeg_prds_file = os.path.join(self.checked_path, self.PRODSLAT_FILE)
        self.cmeg_adv_files = [os.path.join(self.checked_path, f) for f in self.report_files]
        self.matcher = CMEGMatcher(self.cmeg_adv_files, self.cmeg_prds_file, '2017', self.checked_path)

    def id_rows(self, df):
        groups = df_groupby(df, [[self.matcher.PRODUCT, self.matcher.CLEARED_AS]])
        dict_globex = dict()
        dict_pdnm = dict()
        header_ytd = self.matcher.get_ytd_header(df)
        for group, subdf in groups.items():
            tot_ytd = sum(subdf[header_ytd].unique())
            cleared_as = group[1]
            for _, row in subdf.iterrows():
                if pd.isnull(row[self.matcher.F_PRODUCT_NAME]):
                    continue
                row.update(pd.Series([tot_ytd], index=[header_ytd]))
                pd_code = self.get_prod_code(row)
                if row[self.matcher.F_PRODUCT_NAME] in dict_pdnm:
                    print()
                    print('In group: {}'.format(group))
                    print('duplicate PRODUCT_NAME: {}'.format(row[self.matcher.F_PRODUCT_NAME]))
                dict_pdnm.update({row[self.matcher.F_PRODUCT_NAME]: row})
                if pd_code is not None:
                    if (pd_code, cleared_as) in dict_globex:
                        print('duplicate (pd_code, cleared_as): {}'.format((pd_code, cleared_as)))
                    dict_globex.update({(pd_code, cleared_as): row})

            # subdf.drop_duplicates(self.matcher.F_PRODUCT_NAME, 'first', inplace=True)
            # new_df = pd.DataFrame({header_ytd: [tot_ytd] * subdf.shape[0]})
            # subdf.update(new_df)
            # dict_globex.update({self.get_prod_code(row): row for _, row in subdf.iterrows() if self.get_prod_code(row)})
        return groups, dict_globex, dict_pdnm

    def get_prod_code(self, row):
        if not pd.isnull(row[self.matcher.F_GLOBEX]):
            return row[self.matcher.F_GLOBEX]
        elif not pd.isnull(row[self.matcher.F_CLEARING]):
            return row[self.matcher.F_CLEARING]
        else:
            print('no code: {}'.format(row[self.matcher.F_PRODUCT_NAME]))
            return None

    def run_pd_check(self):
        dfs_dict = self.matcher.run_pd_mtch(clean=True)
        self.id_rows(dfs_dict[self.matcher.CME])




cmeg = CMEGChecker()
# cmeg.run_pd_check()
matched_file = os.path.join(cmeg.checked_path, 'CMEG_matched.xlsx')
with open(matched_file, 'rb') as fh:
    df_cme = pd.read_excel(fh, sheet_name=cmeg.matcher.CME)
    # df_cbot = pd.read_excel(fh, sheet_name=cmeg.matcher.CBOT)
    # df_nymex = pd.read_excel(fh, sheet_name=cmeg.matcher.NYMEX)

cme_groups, cme_dict, cme_dict_pdnm = cmeg.id_rows(df_cme)
print()

# xl_consolidate(test_input, test_output)
# xl = pd.ExcelFile(test_input[0][0])
# summary = xl.parse(test_input[0][1])
# products = xl.parse(test_input[1][1])['commodity_name']
# exp = lambda x: x in products.tolist()
# results = summary[list(filter(summary, 'Globex',  exp))]
# print((summary[list(filter(summary, 'Globex',  exp))].head()))





# ix = open_dir(cme.index_cme)
# # print(ix.schema.items())
# docs = list(ix.searcher().documents())
# print(len(docs))

# #
# # for row in docs:
# #     if 'Eurodollar' in row[cme.F_PRODUCT_NAME]:
# #         print(row)
# # for i, row in pd.DataFrame(docs).iterrows():
# #     if 'Cash-Settled' in row[cme.F_PRODUCT_NAME]:
# #         print(row[cme.F_PRODUCT_NAME], row[cme.F_CLEARED_AS])
#
# prod = 'EURODOLLARS'
# pdgp = 'Interest Rate'
# ca = 'Futures'
#
# ix = open_dir(cme.index_cme)
# with ix.searcher() as searcher:
#
#     # for doc in searcher.documents():
#     #     print(doc)
#
#     # grouping_q = And([Term(cme.F_PRODUCT_GROUP, pdgp), Term(cme.F_CLEARED_AS, ca), Term(cme.F_MATCHED, False)])
#     grouping_q = And([Term(cme.F_PRODUCT_GROUP, pdgp), Term(cme.F_CLEARED_AS, ca)])
#     # grouping_q = qparser.QueryParser(pdgp_field, schema=ix.schema).parse(pdgp)
#     parser = qparser.QueryParser(cme.F_PRODUCT_NAME, schema=ix.schema)
#     query = parser.parse(prod)
#     # ts = query.iter_all_terms()
#     # fuzzy_terms = And([FuzzyTerm(f, t, maxdist=2, prefixlength=1) for f, t in ts])
#     results = searcher.search(query, filter=grouping_q, limit=None)
#     # results = searcher.search(Term(cme.F_PRODUCT_NAME, 'Eurodollar'), filter=grouping_q, limit=None)
#     if results:
#         rs = [r.fields() for r in results]
#         print(pd.DataFrame(rs))
#     # results = searcher.search(grouping_q, limit=None)
#     # if results:
#     #     # hit = results[0].fields()
#     #     rs = [r.fields() for r in results]
#     #     print(pd.DataFrame(rs))
#
# docs = list(ix.searcher().documents())
# print(len(docs))

# ana = StandardAnalyzer('\S+', stoplist=cme.STOP_LIST, minsize=1) | SplitFilter()
# # ana = RegexTokenizer('\S+') | SplitFilter()
# print([t.text for t in ana(' E-Mini S&P500')])


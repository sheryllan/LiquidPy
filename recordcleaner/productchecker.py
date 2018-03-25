import pandas as pd
import numpy as np
import configparser as cp
import os
import re
import inflect
import itertools



import datascraper as dtsp
from whooshext import *

# Parse the config files
# cp.parse_save()


reports_path = '/home/slan/Documents/exch_report/'
configs_path = '/home/slan/Documents/config_files/'
# checked_path = '/home/slan/Documents/checked_report/'
checked_path = os.getcwd()

exchanges = ['asx', 'bloomberg', 'cme', 'cbot', 'nymex_comex', 'eurex', 'hkfe', 'ice', 'ose', 'sgx']
report_fmtname = 'Web_ADV_Report_{}.xlsx'

report_files = {e: report_fmtname.format(e.upper()) for e in exchanges}


# config_files = {e: e + '.xlsx' for e in exchanges}
#
# test_input = [(reports_path + report_files['cme'], 'Summary'), (configs_path + config_files['cme'], 'Config')]
# test_output = test_input[0][0]




# parameter file2sheet is a tuple
def xl_consolidate(file2sheet, dest):
    wrt = cp.XlsxWriter.create_xlwriter(dest, False)
    for fl, sht in file2sheet:
        xl = pd.ExcelFile(fl)
        dt = xl.parse(sht)
        cp.XlsxWriter.to_xlsheet(dt, wrt, sht)
    wrt.save()



# xl_consolidate(test_input, test_output)
# xl = pd.ExcelFile(test_input[0][0])
# summary = xl.parse(test_input[0][1])
# products = xl.parse(test_input[1][1])['commodity_name']
# exp = lambda x: x in products.tolist()
# results = summary[list(filter(summary, 'Globex',  exp))]
# print((summary[list(filter(summary, 'Globex',  exp))].head()))

class WhooshExtension(object):
    STEM_ANA = StemmingAnalyzer('[^ /\.\(\)]+')
    CME_SPECIAL_MAPPING = {'midcurve': 'mc',
                           'mc': 'midcurve',
                           '$': 'USD'}

    @staticmethod
    def CMESpecialFilter(stream):
        for token in stream:
            if token.text in WhooshExtension.CME_SPECIAL_MAPPING:
                token.text = WhooshExtension.CME_SPECIAL_MAPPING[token.text]
            yield token





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


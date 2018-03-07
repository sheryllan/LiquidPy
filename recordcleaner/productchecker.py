import pandas as pd
import configparser as cp
import xlrd


# Parse the config files
#cp.parse_save()


reports_path = '/home/slan/Documents/exch_report/'
configs_path = '/home/slan/Documents/config_files/'

exchanges = ['asx', 'bloomberg', 'cme', 'eurex', 'hkfe', 'ice', 'ose', 'sgx']
report_fmtname = '_Volume_Report.xlsx'

report_files = {e: e.upper() + report_fmtname for e in exchanges}
config_files = {e: e + '.xlsx' for e in exchanges}

test_input = [(reports_path + report_files['cme'], 'Summary'), (configs_path + config_files['cme'], 'Config')]
test_output = test_input[0][0]

# parameter file2sheet is a tuple
def xl_consolidate(file2sheet, dest):
    wrt = cp.XlsxWriter.create_xlwriter(dest, False)
    for fl, sht in file2sheet:
        xl = pd.ExcelFile(fl)
        dt = xl.parse(sht)
        cp.XlsxWriter.to_xlsheet(dt, wrt, sht)
    wrt.save()

def filter(df, col, exp):
    return df[col].map(exp)

#xl_consolidate(test_input, test_output)
xl = pd.ExcelFile(test_input[0][0])
summary = xl.parse(test_input[0][1])
products = xl.parse(test_input[1][1])['commodity_name']
exp = lambda x: x in products.tolist()
results = summary[filter(summary, 'Globex',  exp)]
print(summary[filter(summary, 'Globex',  exp)].head())



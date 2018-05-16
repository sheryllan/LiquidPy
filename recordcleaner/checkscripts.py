from cmegcheck import *
from tempfile import TemporaryDirectory


def cmeg_check(outpath=None):
    scraper = CMEGScraper()
    prods_file, dfs_adv = scraper.run_scraper(years=[last_year()])
    df_prods = pd.read_excel(prods_file)

    matcher = CMEGMatcher()
    with TemporaryDirectory() as ixfolder_cme, TemporaryDirectory() as ixfolder_cbot:
        dfs_matched = matcher.run_pd_mtch(dfs_adv, df_prods, (ixfolder_cme, ixfolder_cbot), True)
        checker = CMEGChecker()
        checker.run_pd_check(dfs_matched, outpath=outpath)


outpath='CMEG_checked.xlsx'
cmeg_check(outpath)



from cmegcheck import *
from tempfile import TemporaryDirectory


def cmeg_check(outpath=None):
    scraper = CMEGScraper()
    df_prods, dfs_adv = scraper.run_scraper()
    matcher = CMEGMatcher(dfs_adv, df_prods)
    with TemporaryDirectory() as ixfolder_cme, TemporaryDirectory() as ixfolder_cbot:
        dfs_matched = matcher.run_pd_mtch((ixfolder_cme, ixfolder_cbot), True)
        checker = CMEGChecker(matcher)
        checker.run_pd_check(dfs_matched, outpath=outpath)


outpath='CMEG_checked.xlsx'
cmeg_check(outpath)



import urllib.error
import urllib.parse
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import re

from commonlib.commonfuncs import *


USER_AGENT = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7; X11; Linux x86_64) ' \
             'Gecko/2009021910 Firefox/3.0.7 Chrome/23.0.1271.64 Safari/537.11'
TABLE_TAB = 'table'
TR_TAB = 'tr'
TH_TAB = 'th'
TD_TAB = 'td'
A_TAB = 'a'
HREF_ATTR = 'href'


def download(url, fh):
    request = Request(url, headers={'User-Agent': USER_AGENT})
    try:
        response = urlopen(request)
        print(('\n[*] Downloading from: {}'.format(url)))
        fh.write(response.read())
        fh.flush()
        print('\n[*] Successfully downloaded to ' + fh.name)
        return fh
    except urllib.error.HTTPError as e:
        print(e.fp.read())


def make_soup_from_url(url):
    request = Request(url, headers={'User-Agent': USER_AGENT})
    html = urlopen(request)
    soup = BeautifulSoup(html, 'html.parser')
    return soup


def make_soup_from_file(path):
    with open(path, 'r') as input:
        raw_html = input.read()
        return BeautifulSoup(raw_html, 'html.parser')


def make_soup(input):
    if isinstance(input, str):
        try:
            return make_soup_from_url(input)
        except ValueError:
            return make_soup_from_file(input)
    else:
        return BeautifulSoup(input, 'html.parser')


def fltr_attrs(tags, attrs, mapping=None):
    mapping = {} if mapping is None else mapping
    return ({mapping.get(k, k): v for k, v in tag.attrs.items() if k in attrs} for tag in tags)


def find_link(soupobjs, pattern):
    links = [str(sobj.find(href=True)[HREF_ATTR]) for sobj in soupobjs]
    return find_first_n(links, lambda x: re.match(pattern, x))


class HtmlTableParser(object):
    def __init__(self, src, filterfunc=None):
        tables = make_soup(src).find_all(TABLE_TAB)
        if not tables:
            raise ValueError('No tables found in the source')
        self.table = tables[0] if filterfunc is None else filterfunc(tables)

    def get_tb_headers(self):
        return [th.text for th in self.table.find(TR_TAB).find_all(TH_TAB)]

    def get_td_rows(self, filterfunc=None):
        trs = self.table.find_all(TR_TAB)
        if filterfunc is not None:
            tds = [filterfunc(tr.find_all(TD_TAB)) for tr in trs if tr.find_all(TD_TAB)]
        else:
            tds = [tr.find_all(TD_TAB) for tr in trs if tr.find_all(TD_TAB)]
        return tds
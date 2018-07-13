import json
import re
import logging
import requests
from bs4 import BeautifulSoup
from requests.auth import HTTPBasicAuth

from commonlib.commonfuncs import *

USER_AGENT = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7; X11; Linux x86_64) ' \
             'Gecko/2009021910 Firefox/3.0.7 Chrome/23.0.1271.64 Safari/537.11'
TABLE_TAB = 'table'
TR_TAB = 'tr'
TH_TAB = 'th'
TD_TAB = 'td'
A_TAB = 'a'
HREF_ATTR = 'href'


def http_post(url, data, auth=None, cert=None):
    logger = logging.getLogger(__name__)
    json_data = data if isinstance(data, str) else json.dumps(data)
    cert = False if cert is None else cert
    if auth is not None:
        response = requests.post(url, json_data, verify=cert, auth=HTTPBasicAuth(*auth),
                                 headers={'Accept': 'application/json'})
    else:
        response = requests.post(url, json_data, verify=cert, headers={'Accept': 'application/json'})
    logger.info(response.content.decode())
    return response


def download(url, fh):
    logger = logging.getLogger(__name__)
    response = requests.get(url, stream=True, headers={'User-Agent': USER_AGENT})
    for chunk in response.iter_content(1024):
        if chunk:
            fh.write(chunk)
        fh.flush()
    logger.info('Successfully downloaded to ' + fh.name)
    return fh


def make_soup_from_url(url):
    response = requests.get(url, headers={'User-Agent': USER_AGENT})
    html = response.text
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
    # returns tables whose first tr has first th of which text = title
    @staticmethod
    def get_tables_by_th(url, title):
        tables = make_soup(url).find_all(TABLE_TAB)
        if not tables:
            raise ValueError('No tables found in the source')
        return [tbl for tbl in tables if tbl.find(TR_TAB).find(TH_TAB, text=title)]

    @staticmethod
    def get_tb_headers(table):
        return [th.text for th in table.find(TR_TAB).find_all(TH_TAB)]

    @staticmethod
    def get_td_rows(table, filterfunc=None):
        trs = table.find_all(TR_TAB)
        if filterfunc is not None:
            tds = [filterfunc(tr.find_all(TD_TAB)) for tr in trs if tr.find_all(TD_TAB)]
        else:
            tds = [tr.find_all(TD_TAB) for tr in trs if tr.find_all(TD_TAB)]
        return tds

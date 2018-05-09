import collections
import urllib.error
import urllib.parse
import urllib.request

from bs4 import BeautifulSoup

USER_AGENT = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7; X11; Linux x86_64) ' \
             'Gecko/2009021910 Firefox/3.0.7 Chrome/23.0.1271.64 Safari/537.11'


def nontypes_iterable(arg, excl_types=(str,)):
    return isinstance(arg, collections.Iterable) and not isinstance(arg, excl_types)


def flatten_iter(items, incl_level=False, types=(str,)):
    def flattern_iter_rcrs(items, flat_list, level):
        if not items:
            return flat_list

        if nontypes_iterable(items, types):
            level = None if level is None else level + 1
            for sublist in items:
                flat_list = flattern_iter_rcrs(sublist, flat_list, level)
        else:
            level_item = items if level is None else (level, items)
            flat_list.append(level_item)
        return flat_list

    level = -1 if incl_level else None
    return flattern_iter_rcrs(items, list(), level)


def to_list(x):
    return [x] if not nontypes_iterable(x) else list(x)


def find_first_n(arry, condition, n=1):
    result = list()
    for a in arry:
        if n == 0:
            break
        if condition(a):
            result.append(a)
            n -= 1
    return result if len(result) != 1 else result[0]


def download(url, fh):
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    try:
        response = urllib.request.urlopen(request)
        print(('\n[*] Downloading from: {}'.format(url)))
        fh.write(response.read())
        fh.flush()
        print('\n[*] Successfully downloaded to ' + fh.name)
        return fh
    except urllib.error.HTTPError as e:
        print(e.fp.read())


def make_soup(url):
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    html = urllib.request.urlopen(request)
    soup = BeautifulSoup(html, 'html.parser')
    return soup


def swap(a, b):
    tmp = a
    a = b
    b = tmp
    return a, b


def to_dict(items, tkey, tval):
    return {tkey(x): tval(x) for x in items}


def rreplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)

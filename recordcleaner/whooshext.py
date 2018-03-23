import re

from whoosh.analysis import *


class VowelFilter(Filter):
    VOWELS = ('a', 'e', 'i', 'o', 'u')

    def __init__(self, exclusions=list()):
        self.exclusions = exclusions

    def __remove_vowels(self, string):
        if len(string) < 4:
            return string
        result = string[0]
        for s in string[1:]:
            if s not in self.VOWELS:
                result = result + s
        return result

    def __call__(self, stream):
        for token in stream:
            if token.text not in self.exclusions:
                txt_changed = self.__remove_vowels(token.text)
                if txt_changed != token.text:
                    yield token
                    token.text = txt_changed
                    yield token
                else:
                    yield token
            else:
                yield token



class CurrencyConverter(Filter):
    CRRNCY_MAPPING = {'ad': 'australian dollar',
                      'bp': 'british pound',
                      'cd': 'canadian dollar',
                      'ec': 'euro',
                      'efx': 'euro',
                      'jy': 'japanese yen',
                      'jpy': 'japanese yen',
                      'ne': 'new zealand dollar',
                      'nok': 'norwegian krone',
                      'sek': 'swedish krona',
                      'sf': 'swiss franc',
                      'skr': 'swedish krona',
                      'zar': 'south african rand',
                      'aud': 'australian dollar',
                      'cad': 'canadian dollar',
                      'eur': 'euro',
                      'gbp': 'british pound',
                      'pln': 'polish zloty',
                      'nkr': 'norwegian krone',
                      'inr': 'indian rupee',
                      'rmb': 'chinese renminbi',
                      'usd': 'us dollar'}

    @classmethod
    def get_cnvtd_kws(cls):
        kws = set()
        for val in CurrencyConverter.CRRNCY_MAPPING.values():
            kws.update(val.split(' '))
        return list(kws)

    def __call__(self, stream):
        for token in stream:
            if token.text in self.CRRNCY_MAPPING:
                currency = self.CRRNCY_MAPPING[token.text].split(' ')
                yield token
                for c in currency:
                    token.text = c
                    yield token
            else:
                yield token


class SplitFilter(Filter):
    PTN_SPLT_WRD = '([A-Z]+[^A-Z]*|[^A-Z]+)((?=[A-Z])|$)'
    PTN_SPLT_NUM = '[0-9]+((?=[^0-9])|$)|[^0-9]+((?=[0-9])|$)'
    PTN_SPLT_WRDNUM = '[0-9]+((?=[^0-9])|$)|([A-Z]+[^A-Z0-9]*|[^A-Z0-9]+)((?=[A-Z])|(?=[0-9])|$)'

    PTN_MRG_WRD = '[A-Za-z]+'
    PTN_MRG_NUM = '[0-9]+'

    def __init__(self, delims='\W+', origin=True, splitwords=True, splitnums=True, mergewords=False, mergenums=False):
        self.delims = delims
        self.origin = origin
        self.splt_ptn = None
        self.mrg_ptn = None

        if mergewords and mergenums:
            self.mrg_ptn = '|'.join([self.PTN_MRG_WRD, self.PTN_MRG_NUM])
        elif mergewords:
            self.mrg_ptn = self.PTN_MRG_WRD
        elif mergenums:
            self.mrg_ptn = self.PTN_MRG_NUM

        if splitwords & splitnums:
            self.splt_ptn = self.PTN_SPLT_WRDNUM
        else:
            if splitwords:
                self.splt_ptn = self.PTN_SPLT_WRD
            elif splitnums:
                self.splt_ptn = self.PTN_SPLT_NUM

    def __call__(self, stream):
        for token in stream:
            text = token.text
            if self.origin:
                yield token
            words = re.split(self.delims, text) if re.search(self.delims, text) else [text]
            for token in self.__split_merge(words, token):
                if not (self.origin and token.text == text):
                    yield token

    def __split_merge(self, words, token):
        mrg_stream = ''
        yielded = False
        splits = []
        for word in words:
            # yield splits
            if self.splt_ptn is None:
                continue
            splits = list(self.__findall(self.splt_ptn, word))
            for split in splits:
                token.text = split
                yield token

            # yield merges
            if self.mrg_ptn is None:
                continue
            mtchobjs = re.finditer(self.mrg_ptn, word)
            for match in mtchobjs:
                if self.__can_merge(mrg_stream, match):
                    mrg_stream = mrg_stream + match.group()
                    yielded = False
                else:
                    if not mrg_stream in splits:
                        token.text = mrg_stream
                        yield token
                    mrg_stream = match.group()
                    yielded = True

        if (self.mrg_ptn is not None) and (not yielded) and (mrg_stream not in splits):
            token.text = mrg_stream
            yield token

    def __are_same_type(self, s1, s2):
        return (re.match(self.PTN_MRG_WRD, s1) and re.match(self.PTN_MRG_WRD, s2)) \
               or (re.match(self.PTN_MRG_NUM, s1) and re.match(self.PTN_MRG_NUM, s2))

    def __can_merge(self, string, mtchobj):
        return (not string) or (self.__are_same_type(string, mtchobj.group()) and mtchobj.start() == 0)

    def __findall(self, pattern, word):
        for mobj in re.finditer(pattern, word):
            yield mobj.group()


STOP_LIST = ['and', 'is', 'it', 'an', 'as', 'at', 'have', 'in', 'yet', 'if', 'from', 'for', 'when',
                 'by', 'to', 'you', 'be', 'we', 'that', 'may', 'not', 'with', 'tbd', 'a', 'on', 'your',
                 'this', 'of', 'will', 'can', 'the', 'or', 'are']

STD_ANA = StandardAnalyzer('[^\s/]+', stoplist=STOP_LIST, minsize=1)

ana = STD_ANA | SplitFilter() | VowelFilter(CurrencyConverter.get_cnvtd_kws()) | CurrencyConverter()
# ana = FancyAnalyzer()
# print([t.text for t in ana('Premium-Quoted European Style on Australian Dollar/US Dollar  CHINESE RENMINBI (CNH) E-MICRO CAD/USD aud')])
# print([t.text for t in ana(' E-MINI S&P500*30 ECapTotal5-3-city')])
print([t.text for t in ana(' NASDAQ BIOTECH')])


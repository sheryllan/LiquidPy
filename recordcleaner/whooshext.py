import re
import itertools
import copy
import collections
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

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
            yield token
            if token.text not in self.exclusions:
                txt_changed = self.__remove_vowels(token.text)
                if txt_changed != token.text:
                    token.text = txt_changed
                    yield token

    # def __and__(self, other):
    #     other
    #     self.__call__()


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
            yield token
            if token.text in self.CRRNCY_MAPPING:
                currency = self.CRRNCY_MAPPING[token.text].split(' ')
                for c in currency:
                    token.text = c
                    yield token


class SplitFilter(Filter):
    SRC_PTN_UPCS = 'A-Z'
    SRC_PTN_LWCS = 'a-z'
    SRC_PTN_NUM = '0-9'
    SRC_PTN_COLL = '([{}]+)'

    def __init__(self, delims='\W+', origin=True, splitwords=True, splitnums=True, mergewords=False, mergenums=False):
        self.delims = delims
        self.origin = origin
        # if splitwords == mergewords:
        #     wrd_ptn = [self.SRC_PTN_UPCS + self.SRC_PTN_LWCS]
        #     wrdnum_ptn = [w + self.SRC_PTN_NUM for w in wrd_ptn] if splitnums == mergenums \
        #         else wrd_ptn + [self.SRC_PTN_NUM]
        # else:
        #     wrd_ptn = [self.SRC_PTN_UPCS, self.SRC_PTN_LWCS] if splitwords else [self.SRC_PTN_UPCS + self.SRC_PTN_LWCS]
        #     wrdnum_ptn = wrd_ptn + [self.SRC_PTN_NUM] if splitnums else [w + self.SRC_PTN_NUM for w in wrd_ptn]
        #
        # self.wrdnum_ptn = '|'.join([self.SRC_PTN_COLL.format(p) for p in wrdnum_ptn])

        splt_wrd_ptn = [self.SRC_PTN_UPCS, self.SRC_PTN_LWCS] if splitwords else [self.SRC_PTN_UPCS + self.SRC_PTN_LWCS]
        splt_wrdnum_ptn = splt_wrd_ptn + [self.SRC_PTN_NUM] if splitnums else [w + self.SRC_PTN_NUM for w in splt_wrd_ptn]
        mrg_wrd_ptn = [self.SRC_PTN_UPCS + self.SRC_PTN_LWCS] if mergewords else [self.SRC_PTN_UPCS, self.SRC_PTN_LWCS]
        mrg_wrdnum_ptn = [w + self.SRC_PTN_NUM for w in mrg_wrd_ptn] if mergenums else mrg_wrd_ptn + [self.SRC_PTN_NUM]

        self.splt_ptn = '|'.join([self.SRC_PTN_COLL.format(p) for p in splt_wrdnum_ptn])
        self.mrg_ptn = '|'.join([self.SRC_PTN_COLL.format(p) for p in mrg_wrdnum_ptn])

    def __call__(self, stream):
        for token in stream:
            if self.origin:
                yield token
            words = re.split(self.delims, token.text) if re.search(self.delims, token.text) else [token.text]
            for t in self.__get_matched_tokens(token, self.splt_ptn, self.mrg_ptn, words):
                yield t

    def __get_matched_tokens(self, token, splt_ptn, mrg_ptn, words):
        for w in self.__get_matched_words(splt_ptn, mrg_ptn, words):
            if w != token.text:
                cpy = copy.deepcopy(token)
                cpy.text = w
                yield cpy

    def __get_matched_words(self, splt_ptn, mrg_ptn, words):
        for word in words:
            chains = self.__findall(splt_ptn, word) if splt_ptn == mrg_ptn \
                else set(itertools.chain(self.__findall(splt_ptn, word), self.__findall(mrg_ptn, word)))
            for w in chains:
                yield w

    def __findall(self, pattern, word):
        for w in re.findall(pattern, word):
            if isinstance(w, collections.Iterable):
                w = ''.join(w)
            yield w


class ChainedAnalyzer(Analyzer):
    def __init__(self, analyzer, *composables):
        if not isinstance(analyzer, Analyzer):
            raise TypeError("First argument must be of type Analyzer")
        self.analyzer = analyzer
        self.items = []
        for composable in composables:
            if isinstance(composable, Tokenizer):
                raise CompositionError("No tokenizer allowed appending the analyzer: %r" % self.items)
            if isinstance(composable, Filter):
                self.items.append(composable)

    def __and__(self, other):
        if not isinstance(other, Composable):
            raise TypeError("%r is not composable with %r" % (self, other))
        return ChainedAnalyzer(self, other)

    def __call__(self, value, no_morph=False, **kwargs):
        pool = ThreadPoolExecutor()
        future = pool.submit(self.analyzer, value, **kwargs)
        while not future.done():
            pass
        gen = future.result()

        items = self.items
        for item in items:
            if not (no_morph and hasattr(item, "is_morph") and item.is_morph):
                gen = item(gen)
        return gen


    def __analyzer_cb(self, gen, no_morph):
        items = self.items
        for item in items:
            if not (no_morph and hasattr(item, "is_morph") and item.is_morph):
                gen = item(gen)
        return gen


STOP_LIST = ['and', 'is', 'it', 'an', 'as', 'at', 'have', 'in', 'yet', 'if', 'from', 'for', 'when',
                 'by', 'to', 'you', 'be', 'we', 'that', 'may', 'not', 'with', 'tbd', 'a', 'on', 'your',
                 'this', 'of', 'will', 'can', 'the', 'or', 'are']

STD_ANA = StandardAnalyzer(stoplist=STOP_LIST, minsize=1)
# CME_PDNM_ANA = STD_ANA | SplitFilter() | CurrencyConverter() | VowelFilter()
CME_PDNM_ANA = ChainedAnalyzer(STD_ANA) & VowelFilter() & CurrencyConverter()

ana = CME_PDNM_ANA
# print([t.text for t in ana('Premium Quoted European Style on Australian Dollar/US Dollar  CHINESE RENMINBI (CNH) E-MICRO CAD/USD aud')])
print([t.text for t in ana('ec zar')])


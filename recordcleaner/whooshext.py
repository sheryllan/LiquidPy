import re
import itertools
import math

from whoosh.analysis import *


class CompositeFilter(Filter):
    def __init__(self, *filters):
        self.filters = []
        for filter in filters:
            if isinstance(filter, Filter):
                self.filters.append(filter)
            else:
                raise TypeError('The input type must be a Filter/CompositeFilter')

    def __and__(self, other):
        if not isinstance(other, Filter):
            raise TypeError('{} is not composable(and) with {}'.format(self, other))
        return CompositeFilter(self, other)

    def __call__(self, stream):
        flts = self.filters
        gen = stream
        for f in flts:
            gen = f(gen)
        return gen


class VowelFilter(CompositeFilter):
    VOWELS = ('a', 'e', 'i', 'o', 'u')

    def __init__(self, exclusions=list(), boost=0.8):
        self.exclusions = exclusions
        self.boost = boost
        super().__init__(self)

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
                    token.boost = self.boost
                    yield token
                else:
                    yield token
            else:
                yield token


class SplitFilter(CompositeFilter):
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
        super().__init__(self)

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


class SpecialWordFilter(CompositeFilter):
    def __init__(self, worddict, tokenizer=RegexTokenizer()):
        self.word_dict = worddict
        if not isinstance(tokenizer, Tokenizer):
            raise TypeError('Input is not a valid instance of Tokenizer')
        self.tokenizer = tokenizer
        super().__init__(self)

    def __call__(self, stream):
        for token in stream:
            if token.text in self.word_dict:
                values = self.word_dict[token.text]
                if not isinstance(values, list):
                    values = [values]
                for val in values:
                    tks = self.tokenizer(val[0])
                    for t in tks:
                        token.text = t.text
                        token.boost = val[1]
                        yield token
            else:
                yield token


def min_dist_rslt(results, qstring, fieldname, field, minboost=1):
    min_dist = math.inf
    q_tokens = sorted([(token.text, token.boost) for token in field.analyzer(qstring) if token.boost >= minboost], key=lambda x: x[0])
    qt_len = sum([token[1] for token in q_tokens])
    best_result = results[0]
    for r in results:
        field_value = r.fields()[fieldname]
        r_tokens = sorted([(token.text, token.boost) for token in field.analyzer(field_value) if token.boost >= minboost], key=lambda x: x[0])

        iter_qt = iter(r_tokens)
        iter_rt = iter(q_tokens)
        next_qt = next(iter_qt, None)
        next_rt = next(iter_rt, None)
        dist = sum([token[1] for token in r_tokens]) + qt_len
        while next_qt is not None:
            while next_rt is not None:
                if next_qt[0] == next_rt[0]:
                    dist -= (next_rt[1] + next_qt[1])
                    next_rt = next(iter_rt, None)
                    break
                else:
                    next_rt = next(iter_rt, None)
            next_qt = next(iter_qt, None)
        if dist < min_dist:
            min_dist = dist
            best_result = r
    return best_result









STOP_LIST = ['and', 'is', 'it', 'an', 'as', 'at', 'have', 'in', 'yet', 'if', 'from', 'for', 'when',
                 'by', 'to', 'you', 'be', 'we', 'that', 'may', 'not', 'with', 'tbd', 'a', 'on', 'your',
                 'this', 'of', 'will', 'can', 'the', 'or', 'are']

STD_ANA = StandardAnalyzer('[^\s/]+', stoplist=STOP_LIST, minsize=1)

# ana = STD_ANA | SplitFilter(origin=False, mergewords=True, mergenums=True) | VowelFilter(CurrencyConverter.get_cnvtd_kws()) | CurrencyConverter()
# ana = FancyAnalyzer()
# print([t.text for t in ana('Premium-Quoted European Style on Australian Dollar/US Dollar  CHINESE RENMINBI (CNH) E-MICRO CAD/USD aud')])
# print([t.text for t in ana(' E-MINI S&P500*30 ECapTotal5-3-city')])
# print([t.text for t in ana('  E-MICRO AUD/USD')])


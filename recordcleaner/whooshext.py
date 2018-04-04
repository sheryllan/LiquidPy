import os
import re
import pandas as pd
import itertools
from collections import OrderedDict

from whoosh.fields import *
from whoosh.analysis import *
from whoosh.index import create_in
from whoosh.index import open_dir
from whoosh import writing


def create_index(ix_path, fields, clean=False):
    if (not clean) and os.path.exists(ix_path):
        return open_dir(ix_path)
    if not os.path.exists(ix_path):
        os.mkdir(ix_path)
    schema = Schema(**fields)
    return create_in(ix_path, schema)


def index_from_df(ix, df, clean=False):
    wrt = ix.writer()
    fields = ix.schema.names()
    records = df[fields].to_dict('records')
    for record in records:
        record = {k: record[k] for k in record if not pd.isnull(record[k])}
        if clean:
            wrt.add_document(**record)
        else:
            wrt.update_document(**record)
    wrt.commit()


def setup_ix(fields, df, ix_path, clean=False):
    ix = create_index(ix_path, fields, clean)
    index_from_df(ix, df, clean)
    return ix


def clear_index(ix):
    wrt = ix.writer()
    wrt.commit(mergetype=writing.CLEAR)


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


class VowelFilter(Filter):
    VOWELS = ('a', 'e', 'i', 'o', 'u')

    def __init__(self, exclusions=list(), boost=0.8):
        self.exclusions = exclusions
        self.boost = boost

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
                    boost = token.boost
                    yield token
                    token.text = txt_changed
                    token.boost = self.boost * boost
                    yield token
                else:
                    yield token
            else:
                yield token


class SplitMergeFilter(Filter):
    PTN_SPLT_WRD = '[A-Z]*[^A-Z]*'
    PTN_SPLT_NUM = '[0-9]+|[^0-9]+'
    PTN_SPLT_WRDNUM = '[0-9]+|([A-Z]*[^A-Z0-9]*)'

    PTN_MRG_WRD = '[A-Za-z]+'
    PTN_MRG_NUM = '[0-9]+'

    def __init__(self, delims_splt='\W+', delims_mrg='\W+', min_spltlen=2, origin=False, splitcase=False, splitnums=False, mergewords=False, mergenums=False):
        self.delims_splt = delims_splt
        self.delims_mrg = delims_mrg
        self.min_spltlen = min_spltlen
        self.origin = origin
        self.splt_ptn = None
        self.mrg_ptn = None

        if mergewords and mergenums:
            self.mrg_ptn = '|'.join([self.PTN_MRG_WRD, self.PTN_MRG_NUM])
        elif mergewords:
            self.mrg_ptn = self.PTN_MRG_WRD
        elif mergenums:
            self.mrg_ptn = self.PTN_MRG_NUM

        if splitcase & splitnums:
            self.splt_ptn = self.PTN_SPLT_WRDNUM
        else:
            if splitcase:
                self.splt_ptn = self.PTN_SPLT_WRD
            elif splitnums:
                self.splt_ptn = self.PTN_SPLT_NUM

    def __call__(self, stream):
        for token in stream:
            tk_text = token.text
            tk_boost = token.boost
            if self.origin:
                yield token
            for text, boost in self.__split_merge(tk_text):
                if not (self.origin and text == tk_text):
                    token.text = text
                    token.boost = tk_boost * boost
                    yield token

    def __split(self, text):
        words = self.__join_short_words(re.split(self.delims_splt, text), self.min_spltlen)
        for word in words:
            splits = self.__findall(self.splt_ptn, word) \
                if self.splt_ptn is not None else [word]
            for split in splits:
                yield split

    def __merge(self, text):
        if self.mrg_ptn is not None:
            string = re.sub(self.delims_mrg, '', text)
            merges = self.__findall(self.mrg_ptn, string)
        else:
            merges = re.split(self.delims_mrg, text)
        for merge in merges:
            yield merge

    def __split_merge(self, text):
        splits = list(self.__split(text))
        merges = list(self.__merge(text))

        if len(splits) != 0 and len(merges) != 0:
            s_boost = 1 / (2 * len(splits))
            m_boost = 1 / (2 * len(merges))
            word_boost = OrderedDict({text: s_boost for text in splits})
            for text in merges:
                if text not in word_boost:
                    word_boost.update({text: m_boost})
                else:
                    word_boost.update({text: s_boost + m_boost})
            results = word_boost.items()
        elif len(splits) == 0:
            results = ((text, 1/len(merges)) for text in merges)
        elif len(merges) == 0:
            results = ((text, 1/len(splits)) for text in splits)
        else:
            results = [text]
        return results

    def __findall(self, pattern, word):
        for mobj in re.finditer(pattern, word):
            text = mobj.group()
            if text:
                yield text

    def __join_short_words(self, words, minlen=2):
        result = ''
        for w in words:
            if len(w) < minlen:
                result = result + w
            else:
                if result:
                    yield result
                    result = ''
                yield w
        if result:
            yield result


class SpecialWordFilter(Filter):
    def __init__(self, worddict, tokenizer=RegexTokenizer()):
        self.word_dict = worddict
        if not isinstance(tokenizer, Tokenizer):
            raise TypeError('Input is not a valid instance of Tokenizer')
        self.tokenizer = tokenizer

    def __call__(self, stream):
        for token in stream:
            if token.text in self.word_dict:
                boost = token.boost
                values = self.word_dict[token.text]
                if not isinstance(values, list):
                    values = [values]
                for val in values:
                    tks = self.tokenizer(val[0])
                    for t in tks:
                        token.text = t.text
                        token.boost = val[1] * boost
                        yield token
            else:
                yield token


def min_dist_rslt(results, qstring, fieldname, field, minboost=0):
    ana = field.analyzer
    q_tokens = sorted([(token.text, token.boost) for token in ana(qstring, mode='query') if token.boost >= minboost],
                      key=lambda x: x[0])
    results_srted = TreeMap()
    head = None
    for r in results:
        field_value = r.fields()[fieldname]
        r_tokens = sorted(
            [(token.text, token.boost) for token in ana(field_value, mode='index') if token.boost >= minboost],
            key=lambda x: x[0])

        qt_len = sum([token[1] for token in q_tokens])
        rt_len = sum([token[1] for token in r_tokens])

        midx = 0
        for qt in q_tokens:
            for i in range(midx, len(r_tokens)):
                if qt[0] == r_tokens[i][0]:
                    qt_len -= qt[1]
                    rt_len -= r_tokens[i][1]
                    midx = i + 1
                    break

        head = results_srted.add(((qt_len, rt_len), r), head)
    return [r for _, r in results_srted.get_items(head)]


class TreeMap(object):
    class Node(object):
        def __init__(self, data, left, right):
            self.data = data
            self.left = left
            self.right = right
            self.level = 1

    def skew(self, node):
        if node.left is None:
            return node
        if node.level == node.left.level:
            left = node.left
            node.left = left.right
            left.right = node
            return left
        return node

    def split(self, node):
        if node.right is None or node.right.right is None:
            return node
        if node.level == node.right.right.level:
            right = node.right
            node.right = right.left
            right.left = node
            right.level += 1
            return right
        return node

    def add(self, item, node=None):
        if node is None:
            return TreeMap.Node(item, None, None)
        key, value = item
        data = node.data
        nkey, nval = data

        if key < nkey:
            node.left = self.add(item, node.left)
        elif key == nkey:
            node.right = TreeMap.Node(item, None, node.right)
        else:
            node.right = self.add(item, node.right)

        node = self.skew(node)
        node = self.split(node)
        return node

    def get_items(self, head):

        def get_items_recursive(node, items):
            if node is None:
                return items
            items = get_items_recursive(node.left, items)
            items.append(node.data)
            items = get_items_recursive(node.right, items)
            return items

        return get_items_recursive(head, [])



STOP_LIST = ['and', 'is', 'it', 'an', 'as', 'at', 'have', 'in', 'yet', 'if', 'from', 'for', 'when',
             'by', 'to', 'you', 'be', 'we', 'that', 'may', 'not', 'with', 'tbd', 'a', 'on', 'your',
             'this', 'of', 'will', 'can', 'the', 'or', 'are']

# STD_ANA = StandardAnalyzer('[^\s/]+', stoplist=STOP_LIST, minsize=1)

# ana = STD_ANA | SplitFilter(origin=False, mergewords=True, mergenums=True) | VowelFilter(CurrencyConverter.get_cnvtd_kws()) | CurrencyConverter()
# ana = FancyAnalyzer()
# print([t.text for t in ana('Premium-Quoted European Style on Australian Dollar/US Dollar  CHINESE RENMINBI (CNH) E-MICRO CAD/USD aud')])
# print([t.text for t in ana(' E-MINI S&P500*30 ECapTotal5-3-city')])
# print([t.text for t in ana('  E-MICRO AUD/USD')])

import os
import re
import pandas as pd
import itertools
from collections import namedtuple
from collections import OrderedDict
from collections import Mapping

from whoosh.fields import *
from whoosh.analysis import *
from whoosh.index import create_in
from whoosh.index import open_dir
from whoosh import writing

import datascraper as dtsp


def namedtuple_with_defaults(typename, field_names, default_values=()):
    T = namedtuple(typename, field_names)
    T.__new__.__defaults__ = (None,) * len(T._fields)
    if isinstance(default_values, Mapping):
        prototype = T(**default_values)
    else:
        prototype = T(*default_values)
    T.__new__.__defaults__ = tuple(prototype)
    return T


TokenSub = namedtuple_with_defaults('TokenSub', ['text', 'boost', 'ignored', 'required'], [None, 1, False, False])


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


def join_words(words, minlen=2):
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


def combine_iters(*args):
    for items in zip(*args):
        yielded = set()
        for item in items:
            if item not in yielded:
                yielded.add(item)
                yield item


def set_token(token, **kwargs):
    for k, v in kwargs.items():
        setattr(token, k, v)
    return token


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

    def __init__(self, exclusions=list(), weight=0.2, ignore=True, lift_ignore=True, original=True):
        self.exclusions = exclusions
        self.weight = weight
        self.ignore = ignore
        self.lift_ignore = lift_ignore
        self.original = original

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
            tk_text, tk_boost, tk_ignored = token.text, token.boost, token.ignored
            ignored = False if self.lift_ignore else tk_ignored

            if tk_ignored or tk_text in self.exclusions:
                yield set_token(token, ignored=ignored)
                continue

            txt_changed = self.__remove_vowels(tk_text)
            if txt_changed == tk_text:
                yield set_token(token, ignored=ignored)
                continue

            if self.original:
                yield set_token(token,
                                boost=(1 - self.weight) * tk_boost,
                                ignored=ignored)

            yield set_token(token, text=txt_changed, boost=self.weight * tk_boost, ignored=ignored)


class SplitMergeFilter(Filter):
    PTN_SPLT_WRD = '[A-Z]*[^A-Z]*'
    PTN_SPLT_NUM = '[0-9]+|[^0-9]+'
    PTN_SPLT_WRDNUM = '[0-9]+|([A-Z]*[^A-Z0-9]*)'

    PTN_MRG_WRD = '[A-Za-z]+'
    PTN_MRG_NUM = '[0-9]+'

    def __init__(self, delims_splt='\W+', delims_mrg='\W+', min_spltlen=2, original=False,
                 splitcase=False, splitnums=False, mergewords=False, mergenums=False,
                 ignore_splt=False, ignore_mrg=True):
        self.delims_splt = delims_splt
        self.delims_mrg = delims_mrg
        self.min_spltlen = min_spltlen
        self.original = original
        self.ignore_splt = ignore_splt
        self.ignore_mrg = ignore_mrg
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
            tk_text, tk_boost, tk_ignored = token.text, token.boost, token.ignored
            if self.original:
                yield token
            for tksub in self.__split_merge(tk_text):
                if not (self.original and tksub.text == tk_text):
                    yield set_token(token, text=tksub.text,
                                    boost=tksub.boost * tk_boost,
                                    ignored=tk_ignored or tksub.ignored)

    def __split(self, text, ignore=False):
        words = [word for word in re.split(self.delims_splt, text) if len(word) >= self.min_spltlen]
        for word in words:
            splits = self.__findall(self.splt_ptn, word) \
                if self.splt_ptn is not None else [word]
            for split in splits:
                if len(split) >= self.min_spltlen:
                    yield TokenSub(split, 1, split != word and ignore)

    def __merge(self, text, ignore=True):
        if self.mrg_ptn is not None:
            string = re.sub(self.delims_mrg, '', text)
            merges = self.__findall(self.mrg_ptn, string)
        else:
            merges = re.split(self.delims_mrg, text)
        for merge in merges:
            yield TokenSub(merge, 1, merge != text and ignore)

    def __split_merge(self, text):
        splits = list(self.__split(text, self.ignore_splt))
        merges = list(self.__merge(text, self.ignore_mrg))

        if len(splits) != 0 and len(merges) != 0:
            s_boost = 1 / (2 * len(splits))
            m_boost = 1 / (2 * len(merges))
            word_boost = OrderedDict({split.text: TokenSub(
                split.text, split.boost * s_boost, split.ignored) for split in splits})
            for merge in merges:
                if merge.text not in word_boost:
                    word_boost.update({merge.text: TokenSub(merge.text, merge.boost * m_boost, merge.ignored)})
                else:
                    ignored = word_boost[merge.text].ignored
                    word_boost.update({merge.text: TokenSub(
                        merge.text, merge.boost * (s_boost + m_boost), ignored or merge.ignored)})
            results = word_boost.values()
        elif len(splits) == 0:
            results = (TokenSub(merge.text, merge.boost * 1 / len(merges), merge.ignored) for merge in merges)
        elif len(merges) == 0:
            results = (TokenSub(split.text, split.boost * 1 / len(splits), split.ignored) for split in splits)
        else:
            results = [TokenSub(text)]
        return results

    def __findall(self, pattern, word):
        for mobj in re.finditer(pattern, word):
            text = mobj.group()
            if text:
                yield text


class SpecialWordFilter(Filter):
    def __init__(self, word_dict, tokenizer=RegexTokenizer(), original=False):
        self.word_dict = word_dict
        if not isinstance(tokenizer, Tokenizer):
            raise TypeError('Input is not a valid instance of Tokenizer')
        self.tokenizer = tokenizer
        self.original = original
        self.mapped_kws = self.__groupby(word_dict.values())

    def __call__(self, stream):
        memory = set()
        prev_kws = list()
        next_group = self.mapped_kws
        token = None
        for token in stream:
            tk_cpy = TokenSub(token.text, token.boost, token.ignored, token.required)

            if prev_kws and next_group != self.mapped_kws and tk_cpy.text not in next_group:
                trans_tokens, prev_kws, next_group = self.__cleanup(prev_kws, next_group)
                for tk in self.__gen_tokens(token, trans_tokens, memory):
                    yield tk

            if tk_cpy.text not in self.word_dict:
                if tk_cpy.text in next_group:
                    prev_kws.append(tk_cpy)
                    next_group = next_group[tk_cpy.text]
                elif tk_cpy in next_group:
                    prev_kws.append(tk_cpy)
                else:
                    yield set_token(token, **tk_cpy._asdict())
            else:
                if self.original:
                    memory.add((tk_cpy.text, tk_cpy.boost))
                    yield token

                values = dtsp.to_list(self.word_dict[tk_cpy.text])
                tks = [TokenSub(tk.text, val.boost * tk_cpy.boost,
                                val.ignored or tk_cpy.ignored,
                                val.required or tk_cpy.required)
                       for val in values for tk in self.tokenizer(val.text)]

                if any((t.text, t.boost) not in memory for t in tks):
                    for t in self.__gen_tokens(token, tks, memory):
                        yield t

        if prev_kws and next_group != self.mapped_kws and token is not None:
            trans_tokens, prev_kws, next_group = self.__cleanup(prev_kws, next_group)
            for tk in self.__gen_tokens(token, trans_tokens, memory):
                yield tk

    def __cleanup(self, prev_kws, next_group):
        trans_tokens = prev_kws if None not in next_group \
            else (combine_iters(prev_kws, next_group[None]) if self.original else next_group[None])
        return trans_tokens, list(), self.mapped_kws

    def __gen_tokens(self, token, trans_tokens, memory=None):
        for ts_tk in dtsp.to_list(trans_tokens):
            if memory is not None:
                memory.add((ts_tk.text, ts_tk.boost))
            yield set_token(token, **ts_tk._asdict())

    def __groupby(self, items):

        def keyfunc(items, idx):
            return items[idx].text if idx < len(items) else None

        def item_todict(item, key_idx, results):
            key = keyfunc(item, key_idx)
            if key_idx >= len(item):
                return {key: item}

            val = results[key] if key in results else dict()
            val.update(item_todict(item, key_idx + 1, val))
            results.update({key: val})
            return results

        groups = dict()
        for item in items:
            groups = item_todict(item, 0, groups)
        return groups


class MultiFilterFixed(Filter):
    default_filter = PassFilter()

    def __init__(self, **kwargs):
        self.filters = kwargs

    def __eq__(self, other):
        return (other
                and self.__class__ is other.__class__
                and self.filters == other.filters)

    def __call__(self, tokens):
        # Only selects on the first token
        try:
            t = next(tokens)
            filter = self.filters.get(t.mode, self.default_filter)
            return filter(chain([t], tokens))
        except StopIteration:
            return iter('')


class RegexTokenizerExtra(RegexTokenizer):
    def __init__(self, expression=default_pattern, gaps=False, **attrs):
        super().__init__(expression, gaps)
        self.attrs = attrs

    def __call__(self, value, positions=False, chars=False, keeporiginal=False,
                 removestops=True, start_pos=0, start_char=0, tokenize=True,
                 mode='', **kwargs):
        stream = super().__call__(value, positions, chars, keeporiginal, removestops,
                                  start_pos, start_char, tokenize, mode, **kwargs)

        for token in stream:
            yield set_token(token, **self.attrs)


# class TokenAttrFilter(Filter):
#     def __init__(self, **attrs):
#         self.attrs = attrs
#
#     def __call__(self, stream):
#         try:
#             token = next(stream)
#             token.__dict__.update(self.attrs)
#             yield token
#             for token in stream:
#                 yield token
#         except StopIteration:
#             return iter('')


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


class Tree(object):
    class Node(object):
        def __init__(self, data=None, children=None):
            self.data = data
            self.children = children

    # def __init__(self, head=Node()):
    #     self.head = head


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
        elif key == nkey:  # preserving the input order
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

# STD_ANA = StandardAnalyzer('[^\s/]+', stoplist=STOP_LIST, minsize=1)

# ana = STD_ANA | SplitFilter(origin=False, mergewords=True, mergenums=True) | VowelFilter(CurrencyConverter.get_cnvtd_kws()) | CurrencyConverter()
# ana = FancyAnalyzer()
# print([t.text for t in ana('Premium-Quoted European Style on Australian Dollar/US Dollar  CHINESE RENMINBI (CNH) E-MICRO CAD/USD aud')])
# print([t.text for t in ana(' E-MINI S&P500*30 ECapTotal5-3-city')])
# print([t.text for t in ana('  E-MICRO AUD/USD')])

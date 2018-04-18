import os
import re
import pandas as pd
import itertools
import warnings
from collections import namedtuple
from collections import OrderedDict
from collections import Mapping
from collections import deque
import copy

from whoosh.fields import *
from whoosh.analysis import *
from whoosh.index import create_in
from whoosh.index import open_dir
from whoosh import writing

import datascraper as dtsp


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


def __update_doc(self, ix, doc):
    wrt = ix.writer()
    wrt.update_document(**doc)
    wrt.commit()
    print(len(list(ix.searcher().documents())))


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


def set_token(token, **kwargs):
    for k, v in kwargs.items():
        setattr(token, k, v)
    return token


def update_tksub(tksub_old, tksub_new, boostfunc=None):
    text = tksub_new.text
    if text != tksub_old.text:
        return tksub_new
    else:
        boost = boostfunc(tksub_old.boost, tksub_new.boost) if boostfunc else tksub_new.boost
        ignored = tksub_old.ignored or tksub_new.ignored
        required = tksub_old.required or tksub_new.required
        return TokenSub(text, boost, ignored, required)


def last_indexof(items, target):
    j = None
    for i in range(len(items) - 1, -1, -1):
        if items[i] == target:
            j = i
            break
    return j


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


class VowelFilter(Filter):
    VOWELS = ('a', 'e', 'i', 'o', 'u')

    def __init__(self, exclusions=list(), weight=0.2, lift_ignore=True, original=True):
        self.exclusions = exclusions
        self.weight = weight
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
        self.kws_treedict = self.__groupby(word_dict.values())

    def __call__(self, stream):
        memory = dict()
        prev_kws = list()
        next_group = self.kws_treedict
        token = None
        ismapped = False
        for token in stream:
            tk_cpy = TokenSub(token.text, token.boost, token.ignored, token.required)
            if self.original:
                memory.update({tk_cpy: (False, True)})
                yield token

            if tk_cpy.text in next_group:
                prev_kws, next_group = self.__move_down(prev_kws, next_group, tk_cpy)
            else:
                if prev_kws and next_group != self.kws_treedict:
                    trans_tokens = self.__prev_tokens(prev_kws)
                    for tk in self.__gen_tokens(trans_tokens, memory, ismapped):
                        yield set_token(token, **tk._asdict())
                    prev_kws, next_group = self.__clear()

                if tk_cpy.text in self.word_dict:
                    ismapped = True
                    mapped_tksubs = self.__mapped_kwtokens(tk_cpy)
                    last = mapped_tksubs[-1]
                    if last.text in next_group:
                        prev_kws.extend(mapped_tksubs)
                        next_group = self.__update_treedict(next_group, mapped_tksubs)
                    else:
                        for tk in self.__gen_tokens(mapped_tksubs, memory, ismapped):
                            yield set_token(token, **tk._asdict())
                elif tk_cpy.text in next_group:
                    ismapped = False
                    prev_kws, next_group = self.__move_down(prev_kws, next_group, tk_cpy)
                else:
                    ismapped = False
                    memory.update({tk_cpy: (ismapped, True)})
                    yield set_token(token, **tk_cpy._asdict())

        if prev_kws and next_group != self.kws_treedict and token is not None:
            trans_tokens = self.__prev_tokens(prev_kws)
            for tk in self.__gen_tokens(trans_tokens, memory, ismapped):
                yield set_token(token, **tk._asdict())

    def __update_treedict(self, treedict, to_append):
        child = treedict[to_append[-1].text]
        dict_toupdate = treedict
        for item in to_append:
            dict_toupdate = dict_toupdate[item.text]
        dict_toupdate.update(self.__get_extended_treedict(child, to_append))
        return dict_toupdate

    def __get_extended_treedict(self, treedict, tokens):

        def update_rcrs(value, items):
            if not isinstance(value, dict) and isinstance(value, list):
                item_list = dtsp.to_list(items)
                return item_list + [v for v in value if v not in item_list]

            for k, v in value.items():
                new_val = update_rcrs(v, items)
                if new_val:
                    value.update({k: new_val})

        new_trdict = copy.deepcopy(treedict)
        update_rcrs(new_trdict, tokens)
        return new_trdict

    def __mapped_kwtokens(self, tk_cpy):
        if not isinstance(tk_cpy, TokenSub):
            raise TypeError('The first argument must be of type TokenSub')

        values = dtsp.to_list(self.word_dict[tk_cpy.text])
        return [TokenSub(tk.text, val.boost * tk_cpy.boost,
                        val.ignored or tk_cpy.ignored,
                        val.required or tk_cpy.required)
               for val in values for tk in self.tokenizer(val.text)]

    def __move_down(self, prev_kws, next_group, token):
        prev_kws.append(token)
        next_group = next_group[token.text]
        if None in next_group:
            prev_kws.append(None)
        return prev_kws, next_group

    def __prev_tokens(self, prev_kws, ismapped=False):
        if not prev_kws:
            return prev_kws
        last_idx = last_indexof(prev_kws, None)
        trans_tokens = prev_kws
        if last_idx is not None:
            longest_kwtokens = self.__longest_kwtokens(prev_kws[0: last_idx], not ismapped)
            longest_kwtokens = longest_kwtokens if ismapped \
                else [update_tksub(to, tn, lambda x1, x2: x1 * x2) for to, tn in zip(*longest_kwtokens)]
            trans_tokens = longest_kwtokens + prev_kws[last_idx + 1:]
        return trans_tokens

    def __clear(self):
        return list(), self.kws_treedict

    def __longest_kwtokens(self, tokens, keeporigs=False):
        next_dict = self.kws_treedict
        orig_tokens = list()
        for token in tokens:
            if token is None:
                continue
            orig_tokens.append(token)
            next_dict = next_dict[token.text]
        if None not in next_dict:
            raise ValueError('None not in kws_treedict: the recorded items not consistent with the tree dict')
        return (orig_tokens, next_dict[None]) if keeporigs else next_dict[None]

    def __multi_all(self, trans_tokens, memory):
        all_in, all_mapped, all_notmapped = (True,) * 3
        all_yielded, all_notyielded = (True,) * 2
        for ts_tk in trans_tokens:
            record = TokenSub(ts_tk.text, ts_tk.boost, ts_tk.ignored, ts_tk.required)
            if record not in memory:
                all_in, all_mapped, all_notmapped = (False,) * 3
                break
            else:
                all_mapped = all_mapped and memory[record][0]
                all_notmapped = all_notmapped and not memory[record][0]
                all_yielded = all_yielded and memory[record][1]
                all_notyielded = all_notyielded and not memory[record][1]

        return all_in, all_mapped, all_notmapped, all_yielded, all_notyielded

    def __gen_tokens(self, trans_tokens, memory=None, ismapped=False):
        trans_tokens = dtsp.to_list(trans_tokens)
        if memory is not None:
            all_in, all_mapped, all_notmapped, all_yielded, all_notyielded = self.__multi_all(trans_tokens, memory)
            to_yield = not all_in or (all_in and
                                      ((all_mapped and all_notyielded) or
                                       (all_notmapped and (not ismapped or all_notyielded))))

            for ts_tk in trans_tokens:
                record = TokenSub(ts_tk.text, ts_tk.boost, ts_tk.ignored, ts_tk.required)
                if to_yield:
                    yield ts_tk

                memory.update({record: (ismapped, to_yield)})
        else:
            for ts_tk in trans_tokens:
                yield ts_tk

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


class AdvSearch(object):
    def __init__(self, searcher):
        self.searcher = searcher
        self.results = None

    def chain_search(self, query, init=False, callback=None, **kwargs):
        if init:
            self.results = None

        if not self.results:
            self.results = self.searcher.search(query, **kwargs)
            if callback:
                callback()

        return self

# region unused codes

# class CompositeFilter(Filter):
#     def __init__(self, *filters):
#         self.filters = []
#         for filter in filters:
#             if isinstance(filter, Filter):
#                 self.filters.append(filter)
#             else:
#                 raise TypeError('The input type must be a Filter/CompositeFilter')
#
#     def __and__(self, other):
#         if not isinstance(other, Filter):
#             raise TypeError('{} is not composable(and) with {}'.format(self, other))
#         return CompositeFilter(self, other)
#
#     def __call__(self, stream):
#         flts = self.filters
#         gen = stream
#         for f in flts:
#             gen = f(gen)
#         return gen

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

# class Tree(object):
#     class Node(object):
#         def __init__(self, data=None, children=None):
#             self.data = data
#             self.children = children

# def merge_lists(*args):
#     len_iter = len(args[0])
#     if any(len(arg) != len_iter for arg in args):
#         warnings.warn('Merging iterables of different length')
#     for items in zip(*args):
#         yielded = set()
#         for item in items:
#             if item not in yielded:
#                 yielded.add(item)
#                 yield item
#
#
# def slice_last(iterable):
#     it = iter(iterable)
#     try:
#         current = next(it)
#         for i in it:
#             yield current
#             current = i
#         return current
#     except StopIteration:
#         return iterable

# endregion

# STD_ANA = StandardAnalyzer('[^\s/]+', stoplist=STOP_LIST, minsize=1)

# ana = STD_ANA | SplitFilter(origin=False, mergewords=True, mergenums=True) | VowelFilter(CurrencyConverter.get_cnvtd_kws()) | CurrencyConverter()
# ana = FancyAnalyzer()
# print([t.text for t in ana('Premium-Quoted European Style on Australian Dollar/US Dollar  CHINESE RENMINBI (CNH) E-MICRO CAD/USD aud')])
# print([t.text for t in ana(' E-MINI S&P500*30 ECapTotal5-3-city')])
# print([t.text for t in ana('  E-MICRO AUD/USD')])

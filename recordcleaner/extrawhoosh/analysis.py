import copy
from collections import OrderedDict
from collections import namedtuple
from whoosh.analysis import *

from commonlib.commonfuncs import *
from commonlib.datastruct import *


TokenSub = namedtuple_with_defaults(namedtuple('TokenSub', ['text', 'boost', 'ignored', 'required']), [None, 1, False, False])


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

    PTN_MRG_WRD = '[^0-9]+'
    PTN_MRG_NUM = '[^A-Za-z]+'

    def __init__(self, delims_splt='\W+', delims_mrg='\W+', min_spltlen=1, original=False,
                 splitcase=False, splitnums=False, mergewords=False, mergenums=False,
                 ignore_splt=False, ignore_mrg=False):
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
            tk_cpy = TokenSub(token.text, token.boost, token.ignored)
            processed = list(self.__split_merge(tk_cpy.text))
            for tksub in self.__re_boost(tk_cpy, processed):
                updated_tksub = update_tksub(tk_cpy, tksub, lambda o, n: o * n)
                yield set_token(token, **updated_tksub._asdict())

    def __split(self, text, ignore=False):
        words = [word for word in re.split(self.delims_splt, text) if len(word) >= self.min_spltlen]
        for word in words:
            splits = self.__findall(self.splt_ptn, word) \
                if self.splt_ptn is not None else [word]
            for split in splits:
                if len(split) >= self.min_spltlen:
                    yield TokenSub(split, 1, split != word and ignore)

    def __merge(self, text, ignore=True):
        string = re.sub(self.delims_mrg, '', text)
        if self.mrg_ptn is not None:
            merges = self.__findall(self.mrg_ptn, string)
        else:
            merges = [string]
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

    def __re_boost(self, original, processed):
        if not self.original:
            return processed

        def boost_by(tokens, scale):
            return (update_tksub(t, TokenSub(t.text, scale), lambda o, n: o * n) for t in tokens)

        scale = 0.5
        original_text = original.text
        original = boost_by([original], scale)
        processed = (t for t in boost_by(processed, scale) if t.text != original_text)
        return chain(original, processed)

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
        kws_treedict = copy.deepcopy(self.kws_treedict)
        next_group = kws_treedict
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
                if prev_kws and next_group != kws_treedict:
                    trans_tokens = self.__prev_tokens(prev_kws, kws_treedict)
                    for tk in self.__gen_tokens(trans_tokens, memory, ismapped):
                        yield set_token(token, **tk._asdict())
                    prev_kws, next_group = self.__clear(kws_treedict)

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

        if prev_kws and next_group != kws_treedict and token is not None:
            trans_tokens = self.__prev_tokens(prev_kws, kws_treedict)
            for tk in self.__gen_tokens(trans_tokens, memory, ismapped):
                yield set_token(token, **tk._asdict())

    def __update_treedict(self, treedict, to_append):
        child = treedict[to_append[-1].text]
        if len(to_append) == 1:
            if None not in child:
                child.update({None: to_append[0]})
            return child
        dict_toupdate = treedict
        for item in to_append:
            dict_toupdate = dict_toupdate[item.text]
        dict_toupdate.update(self.__get_extended_treedict(child, to_append))
        return dict_toupdate

    def __get_extended_treedict(self, treedict, tokens):

        def update_rcrs(value, items):
            if not isinstance(value, dict) and isinstance(value, list):
                item_list = to_list(items)
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

        values = to_list(self.word_dict[tk_cpy.text])
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

    def __prev_tokens(self, prev_kws, treedict):
        if not prev_kws:
            return prev_kws
        last_idx = last_indexof(prev_kws, lambda x: x is None)
        trans_tokens = prev_kws
        if last_idx is not None:
            longest_kwtokens = self.__longest_kwtokens(prev_kws[0: last_idx], treedict, True)
            longest_kwtokens = [update_tksub(to, tn, lambda x1, x2: x1 * x2) for to, tn in zip(*longest_kwtokens)]
            trans_tokens = longest_kwtokens + prev_kws[last_idx + 1:]
        return trans_tokens

    def __clear(self, kws_treedict):
        return list(), kws_treedict

    def __longest_kwtokens(self, tokens, treedict, keeporigs=False):
        next_dict = treedict
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
        trans_tokens = to_list(trans_tokens)
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


# def join_words(words, minlen=2):
#     result = ''
#     for w in words:
#         if len(w) < minlen:
#             result = result + w
#         else:
#             if result:
#                 yield result
#                 result = ''
#             yield w
#     if result:
#         yield result
from whoosh.query import Every
from extrawhoosh.indexing import get_field
from commonlib.datastruct import TreeMap


def search_func(searcher, query, qparams, callback=lambda: None, **kwargs):
    searcher = searcher

    def srchfunc():
        results = searcher.search(query(**qparams), **kwargs)
        callback()
        return results

    return srchfunc


def chain_search(searches, chain_condition=lambda r: False if r else True):
    results = None
    for search in searches:
        if not chain_condition(results):
            return results
        results = search()
    return results


def get_idx_lexicon(searcher, *fds, **kwfds):
    grouped_docs = searcher.search(Every(), groupedby=fds)
    lexicons = dict()
    for fd in fds:
        gp_dict = {gp: [searcher.stored_fields(docid) for docid in ids]
                   for gp, ids in grouped_docs.groups(fd).items()}
        lexicons.update({fd: set(gp_dict.keys())})
        if fd in kwfds:
            subfd = kwfds[fd]
            subgp_dict = {gp: set([doc[subfd] for doc in docs]) for gp, docs in gp_dict.items()}
            lexicons.update({subfd: subgp_dict})
    return lexicons


def min_dist_rslt(results, qstring, fieldname, schema, minboost=0):
    field = get_field(schema, fieldname)
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
    return (r for _, r in results_srted.get_items(head))
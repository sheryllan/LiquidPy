from whoosh.query import *
from whoosh import qparser

from commonlib.commonfuncs import *
from extrawhoosh.indexing import get_field


# region queries
def fuzzy_terms(fieldname, schema, qstring, boost=1, maxdist=2, prefixlength=1):
    parser = qparser.QueryParser(fieldname, schema=schema)
    query = parser.parse(qstring)
    fuzzy_terms = [FuzzyTerm(f, t, boost=boost, maxdist=maxdist, prefixlength=prefixlength)
                   if len(t) > maxdist else Term(f, t) for f, t in query.iter_all_terms()]
    return fuzzy_terms


def and_query(fieldname, schema, qstring, boost=1, termclass=Term, **kwargs):
    terms = []
    if termclass == Term:
        parser = qparser.QueryParser(fieldname, schema=schema)
        terms = to_iter(parser.term_query(fieldname, qstring, termclass=termclass, boost=boost))
    elif termclass == FuzzyTerm:
        terms = fuzzy_terms(fieldname, schema, qstring, **kwargs)
    return And(terms, boost=boost)


def or_query(fieldname, schema, qstring, boost=0.9, termclass=Term, **kwargs):
    if termclass == Term:
        og = qparser.OrGroup.factory(boost)
        parser = qparser.QueryParser(fieldname, schema=schema, group=og)
        return parser.parse(qstring)
    elif termclass == FuzzyTerm:
        fz_terms = fuzzy_terms(fieldname, schema, qstring, **kwargs)
        return Or(fz_terms, boost=boost, minmatch=1)


def tokenize_split(fieldname, qstring, keyfunc, mode='query'):
    tokens = fieldname.analyzer(qstring, mode=mode)
    and_words, maybe_words = [], []
    for token in tokens:
        (and_words if keyfunc(token) else maybe_words).append(token.text)
    return and_words, maybe_words


def andmaybe_query(fieldname, schema, qstring, keyfunc=lambda x: x.required, and_extras=None, maybe_extras=None,
                   boost=1, termclass=Term, **kwargs):
    field = get_field(schema, fieldname)
    and_words, maybe_words = tokenize_split(field, qstring, keyfunc)
    and_words = and_words + list(field.process_text(and_extras)) if and_extras else and_words
    maybe_words = maybe_words + list(field.process_text(maybe_extras)) if maybe_extras else maybe_words
    and_terms = And([termclass(fieldname, w, boost=boost, **kwargs) for w in and_words])
    maybe_terms = Or([termclass(fieldname, w, boost=boost, **kwargs) for w in maybe_words])
    return AndMaybe(and_terms, maybe_terms) if and_terms else maybe_terms


def andnot_query(fieldname, schema, and_words, not_words, boost=1, termclass=Term, **kwargs):
    t_and = []
    parser = qparser.QueryParser(fieldname, schema=schema)
    if termclass == Term:
        t_and = parser.term_query(fieldname, and_words, termclass, boost)

    elif termclass == FuzzyTerm:
        t_and = fuzzy_terms(fieldname, schema, and_words, boost, **kwargs)

    q_and = And(t_and)
    q_not = parser.parse(not_words)
    return AndNot(q_and, q_not)


def or_of_and_query(fieldname, schema, and_list, boost=1, termclass=Term, **kwargs):
    q_and = []
    for and_words in and_list:
        q_and.append(and_query(fieldname, schema, and_words, boost, termclass, **kwargs))
    return Or(q_and)


def every_query(fieldname=None):
    if fieldname is not None:
        return Every(fieldname)
    return Every()


def filter_query(*args):
    qterms = list()
    for arg in args:
        if all(arg):
            qterms.append(Term(*arg))
    return And(qterms)

#endregion


QUERY, FIELDNAME, SCHEMA, QSTRING = 'query', 'fieldname', 'schema', 'qstring'
BOOST, TERMCLASS = 'boost', 'termclass'
ANDEXTRAS, MAYBEEXTRAS = 'and_extras', 'maybe_extras'
KEYFUNC, ANDWORDS, NOTWORDS = 'keyfunc', 'and_words', 'not_words'
ANDLIST = 'and_list'

query_dict = {'and': and_query,
              'or': or_query,
              'andmaybe': andmaybe_query,
              'andnot': andnot_query,
              'orofand': or_of_and_query,
              'every': every_query}

params_dict = {'and': {FIELDNAME, SCHEMA, QSTRING, BOOST, TERMCLASS},
               'or': {FIELDNAME, SCHEMA, QSTRING, BOOST, TERMCLASS},
               'andmaybe': {FIELDNAME, SCHEMA, QSTRING, KEYFUNC, ANDEXTRAS, MAYBEEXTRAS, BOOST, TERMCLASS},
               'andnot': {FIELDNAME, SCHEMA, ANDWORDS, NOTWORDS, BOOST, TERMCLASS},
               'orofand': {FIELDNAME, SCHEMA, ANDLIST, BOOST, TERMCLASS},
               'every': {FIELDNAME}}

reserved_params = set([QUERY] + list(flatten_iter(params_dict.values())))


def get_query_params(query, **kwargs):
    essentials = [FIELDNAME, SCHEMA, QSTRING]
    if any(k not in kwargs for k in essentials):
        raise KeyError('Input must have values for the keys: {}'.format(essentials))

    params = dict()
    queryobj = query_dict[query]
    if query == 'andnot':
        kwargs[ANDWORDS] = kwargs[QSTRING] if ANDWORDS not in kwargs or not kwargs[ANDWORDS] \
            else ' '.join([kwargs[ANDWORDS], kwargs[QSTRING]])

    params.update({k: v for k, v in kwargs.items() if k in params_dict[query] or k not in reserved_params})
    return queryobj, params

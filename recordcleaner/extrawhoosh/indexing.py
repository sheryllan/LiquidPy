import os
from pandas import isnull
from whoosh import writing
from whoosh.fields import *
from whoosh.index import create_in
from whoosh.index import open_dir

from commonlib.commonfuncs import find_first_n


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
        record = {k: record[k] for k in record if not isnull(record[k])}
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


def update_doc(ix, doc):
    wrt = ix.writer()
    wrt.update_document(**doc)
    wrt.commit()
    print(len(list(ix.searcher().documents())))


def get_field(schema, fieldname):
    return find_first_n(schema.items(), lambda x: x[0] == fieldname)[1]



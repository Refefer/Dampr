import os
import glob
import logging
import sqlite3
import codecs

from dampr import Dampr
from dampr.inputs import read_paths

class Indexer(object):
    def __init__(self, path, suffix='.index'):
        self.path = path
        self.suffix = suffix

    def get_idx(self, path):
        dirname, base = os.path.split(path)
        return os.path.join(dirname, "." + base + self.suffix)

    def exists(self, path):
        return os.path.isfile(self.get_idx(path))

    def create_db(self, path):
        db = self.open_db(path, True)
        c = db.cursor()
        c.execute('''CREATE TABLE key_index (key text, offset integer)''')
        return db

    def open_db(self, path, delete=False):
        path = self.get_idx(path)
        if delete and os.path.isfile(path):
            os.unlink(path)

        return sqlite3.connect(path)

    def build(self, key_f, force=False):
        paths = list(read_paths(self.path, False))
        paths.sort()

        def index_file(fname):
            logging.debug("Indexing %s", fname)
            db = self.create_db(fname)
            def it():
                offset = 0
                with codecs.open(fname, encoding='utf-8') as f:
                    while True:
                        line = f.readline()
                        if len(line) == 0:
                            break

                        for key in key_f(line):
                            yield key, offset

                        offset += len(line.encode('utf-8'))

            c = db.cursor()
            c.executemany("INSERT INTO key_index values (?, ?)", it())
            db.commit()
            c.execute("create index key_idx on key_index (key)")
            db.commit()
            c.execute("select count(*) from key_index")
            count = c.fetchone()[0]
            logging.debug("Keys indexed for %s: %s", fname, count)
            
            return count

        return Dampr.memory(paths) \
                .filter(lambda fname: force or not self.exists(fname)) \
                .map(index_file) \
                .fold_by(key=lambda x: 1, binop=lambda x,y: x + y) \
                .read(name="indexing")

    def union(self, keys):
        if not isinstance(keys, (list, tuple)):
            keys = [keys]

        paths = read_paths(self.path, self.suffix)

        query = """select distinct offset from key_index 
            where key in ({}) order by offset asc""".format(
                ','.join('"{}"'.format(key) for key in keys))

        def read_db(fname):
            db = self.open_db(fname)

            cur = db.cursor()
            cur.execute(query)
            with codecs.open(fname, encoding='utf-8') as f:
                for (offset,) in cur:
                    f.seek(offset)
                    yield f.readline()

        return Dampr.memory(paths).flat_map(read_db)

    def intersect(self, keys, min_match=None):
        if not isinstance(keys, (list, tuple)):
            keys = [keys]

        if min_match is None:
            min_match = len(keys)

        if isinstance(min_match, float):
            min_match = int(min_match * len(keys))

        paths = read_paths(self.path, self.suffix)

        str_keys = u','.join(u'"{}"'.format(key) for key in keys)
        query = u"""
            select offset from 
            (select offset, count(*) as c 
                from key_index 
                where key in ({}) 
                group by offset) where c >= {}
            order by offset asc""".format(str_keys, min_match)

        def read_db(fname):
            db = self.open_db(fname)

            cur = db.cursor()
            cur.execute(query)
            with codecs.open(fname, encoding='utf-8') as f:
                for (offset,) in cur:
                    f.seek(offset)
                    yield f.readline()

        return Dampr.memory(paths).flat_map(read_db)

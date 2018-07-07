import sys
import os
import itertools
import heapq
import gzip

PY_MAJOR = sys.version_info.major

try:
    import cPickle as pickle

    def dump_pickle(obj, f):
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

except ImportError:
    import pickle

    dump_pickle = pickle.dump

class Splitter(object):
    def partition(self, key, n_partitions):
        return hash(key) % n_partitions

class Dataset(object):
    def read(self):
        raise NotImplementedError()

    def delete(self):
        raise NotImplementedError()

    def __iter__(self):
        return self.read()

class Chunker(object):
    def chunks(self):
        raise NotImplementedError()

class TextInput(Chunker):
    def __init__(self, path, chunk_size=1*1024**2):
        self.path = path
        self.chunk_size = chunk_size

    def chunks(self):
        file_size = os.stat(self.path).st_size
        offset = 0
        while offset < file_size:
            yield TextLineDataset(self.path, offset, offset + self.chunk_size)
            offset += self.chunk_size

class MemoryInput(Chunker):
    def __init__(self, items, chunk_size=1000):
        self.items = items
        self.chunk_size = chunk_size

    def chunks(self):
        for start in range(0, len(self.items), self.chunk_size):
            yield MemoryDataset(self.items[start:start+self.chunk_size])

class TextLineDataset(Dataset):
    def __init__(self, path, start=0, end=None):
        self.path = path
        self.start = start
        self.end = end

    def read(self):
        with open(self.path) as f:
            f.seek(self.start)
            cur_pos = self.start
            if self.start > 0:
                cur_pos += len(f.readline())

            for i, line in enumerate(f):
                yield self.start + i, line
                cur_pos += len(line)
                if self.end is not None and cur_pos > self.end:
                    break

    def __str__(self):
        return "Text[path={},start={},end={}]".format(self.path,self.start, self.end)

    def delete(self):
        pass

class DMChunker(Chunker):
    def __init__(self, data_mapping):
        self.dm = data_mapping

    def chunks(self):
        for vs in self.dm.values():
            for v in vs:
                yield v

class DatasetWriter(object):
    def __init__(self, name):
        self.name       = name

    def start(self):
        raise NotImplementedError()

    def add_record(self, key, value):
        raise NotImplementedError()

    def finished(self):
        raise NotImplementedError()

class PickledDataset(Dataset):
    def __init__(self, path, compressed):
        self.path = path
        self.compressed = compressed

    def read(self):
        if self.compressed:
            f = gzip.GzipFile(self.path, 'rb', 1)
        else:
            f = open(self.path, 'rb')

        n_records = pickle.load(f)
        for _ in range(n_records):
            yield pickle.load(f)

        f.close()

    def delete(self):
        os.unlink(self.path)

    def __str__(self):
        return 'PickledDataset[path={}]'.format(self.path)
    __repr__ = __str__

class PickledDatasetWriter(DatasetWriter):
    def __init__(self, name, splitter, n_partitions, 
            buffer_size=10000, compressed=True):
        super(PickledDatasetWriter, self).__init__(name)
        self.splitter   = splitter
        self.n_partitions = n_partitions
        self.buffer_size = buffer_size
        self.compressed = compressed

    def start(self):
        if not os.path.isdir(self.name):
            os.makedirs(self.name)

        self.files = {}
        self.buffers = {}
        for i in range(self.n_partitions):
            self.files[i] = []
            self.buffers[i] = []

    def write_batch(self, i, buf):
        name = '{}/{}'.format(self.name, i)
        if self.compressed:
            f = gzip.GzipFile(name, 'wb', 1)
        else:
            f = open(name, 'wb') 

        dump_pickle(len(buf), f)
        for d in buf:
            dump_pickle(d, f)

        f.close()

        return name

    def flush(self, nidx):
        buf = self.buffers[nidx]
        buf.sort(key=lambda x: x[0])
        suffix = '{}.{}'.format(nidx, len(self.files[nidx]))
        dataset = PickledDataset(self.write_batch(suffix, buf), self.compressed)
        self.files[nidx].append(dataset)
        del buf[:]

    def add_record(self, key, value):
        nidx = self.splitter.partition(key, self.n_partitions)
        buf = self.buffers[nidx]
        buf.append((key, value))
        if len(buf) > self.buffer_size:
            self.flush(nidx)

    def finished(self):
        for nidx, buffer in self.buffers.items():
            if len(buffer) > 0:
                self.flush(nidx)

        return self.files

class CatDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def read(self):
        for ds in self.datasets:
            for x in ds.read():
                yield x

    def delete(self):
        for d in self.datasets:
            d.delete()

class MergeDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def read(self):
        its = [ds.read() for ds in self.datasets]
        if PY_MAJOR == 3:
            return heapq.merge(*its, key=lambda x: x[0])

        return heapq.merge(*its)

    def delete(self):
        for d in self.datasets:
            d.delete()

class MemoryDataset(Dataset):
    def __init__(self, kvs):
        self.kvs = kvs

    def read(self):
        for k, v in self.kvs:
            yield k, v

    def delete(self):
        pass

class Mapper(object):
    def map(self, *datasets):
        raise NotImplementedError()

class Map(Mapper):
    """
    Standard Mapper
    """
    def __init__(self, mapper):
        self.mapper = mapper

    def map(self, *datasets):
        assert len(datasets) == 1
        for key, value in datasets[0].read():
            for k2, v2 in self.mapper(key, value):
                yield k2, v2

class Reducer(object):
    def reduce(self, key, *datasets):
        raise NotImplementedError()

    def yield_groups(self, dataset):
        if len(dataset) > 0:
            if len(dataset) > 1:
                dataset = MergeDataset(dataset)
            else:
                dataset = dataset[0]

            for key, group in itertools.groupby(dataset.read(), key=lambda x: x[0]):
                it = (kv[1] for kv in group)
                yield key, it

class Reduce(Reducer):
    """
    Standard reduce
    """
    def __init__(self, reducer):
        self.reducer = reducer

    def reduce(self, *datasets):
        assert len(datasets) == 1
        for k, vs in self.yield_groups(datasets[0]):
            yield k, self.reducer(k, vs)

class Join(Reducer):
    def __init__(self, joiner_f):
        self.joiner_f = joiner_f

    def reduce(self, d1, d2):
        g1 = self.yield_groups(d1)
        g2 = self.yield_groups(d2)
        left, right = next(g1, None), next(g2, None)
        while left is not None and right is not None:
            if left[0] < right[0]:
                left = next(g1, None)
            elif left[0] > right[0]:
                right = next(g2, None)
            else:
                k = left[0]
                yield k, self.joiner_f(k, left[1], right[1])
                left, right = next(g1, None), next(g2, None)

def Filter(predicate):
    def _f(key, value):
        if predicate(key, value):
            yield key, value

    return Map(_f)


import os
import itertools
import cPickle
import heapq

class Splitter(object):
    def partition(self, key, n_partitions):
        return hash(key) % n_partitions

class Dataset(object):
    def read(self):
        raise NotImplementedError()

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
                yield i, line
                cur_pos += len(line)
                if self.end is not None and cur_pos > self.end:
                    break

    def __str__(self):
        return "Text[path={},start={},end={}]".format(self.path,self.start, self.end)

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
    def __init__(self, path):
        self.path = path

    def read(self):
        with open(self.path) as f:
            n_records = cPickle.load(f)
            for _ in range(n_records):
                yield cPickle.load(f)

class PickledDatasetWriter(DatasetWriter):
    def __init__(self, name, splitter, n_partitions, buffer_size=10000):
        super(PickledDatasetWriter, self).__init__(name)
        self.splitter   = splitter
        self.n_partitions = n_partitions
        self.buffer_size = buffer_size

    def start(self):
        self.files = {}
        self.buffers = {}
        for i in range(self.n_partitions):
            self.files[i] = []
            self.buffers[i] = []

    def write_batch(self, i, buf):
        name = '{}.{}'.format(self.name, i)
        with open(name, 'w') as f:
            cPickle.dump(len(buf), f, cPickle.HIGHEST_PROTOCOL)
            for d in buf:
                cPickle.dump(d, f, cPickle.HIGHEST_PROTOCOL)

        return name

    def flush(self, nidx):
        buf = self.buffers[nidx]
        buf.sort(key=lambda x: x[0])
        suffix = '{}.{}'.format(nidx, len(self.files[nidx]))
        dataset = PickledDataset(self.write_batch(suffix, buf))
        self.files[nidx].append(dataset)
        del buf[:]

    def add_record(self, key, value):
        nidx = self.splitter.partition(key, self.n_partitions)
        buf = self.buffers[nidx]
        buf.append((key, value))
        if len(buf) > self.buffer_size:
            self.flush(nidx)

    def finished(self):
        for nidx in range(self.n_partitions):
            if len(self.buffers[nidx]) > 0:
                self.flush(nidx)

        return self.files

class ChunkedPickledDatasetWriter(DatasetWriter):
    def __init__(self, name, buffer_size=10000):
        super(ChunkedPickledDatasetWriter, self).__init__(name)
        self.buffer_size = buffer_size

    def start(self):
        self.files = []
        self.buffer = []

    def write_batch(self, i, buf):
        name = '{}.{}'.format(self.name, i)
        with open(name, 'w') as f:
            cPickle.dump(len(buf), f, cPickle.HIGHEST_PROTOCOL)
            for d in buf:
                cPickle.dump(d, f, cPickle.HIGHEST_PROTOCOL)

        return name

    def flush(self):
        name = self.write_batch(len(self.files), self.buffer)
        self.files.append(PickledDataset(name))
        del self.buffer[:]

    def add_record(self, key, value):
        self.buffer.append((key, value))
        if len(self.buffer) > self.buffer_size:
            self.flush()

    def finished(self):
        if len(self.buffer) > 0:
            self.flush()

        return {0: self.files}

class CatDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def read(self):
        for ds in self.datasets:
            for x in ds.read():
                yield x

class MergeDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def read(self):
        its = [ds.read() for ds in self.datasets]
        return heapq.merge(*its)

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

def EZMap(f):
    return Map(f)

def EZReduce(f):
    return Reduce(f)



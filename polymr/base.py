import uuid
import sys
import os
import itertools
import heapq
import gzip

PY_MAJOR = sys.version_info.major

try:
    import cPickle as pickle
    from  cStringIO import StringIO

    def dump_pickle(obj, f):
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

except ImportError:
    import pickle
    from io import StringIO

    def dump_pickle(o, f):
        pickle.dump(o, f,  pickle.HIGHEST_PROTOCOL)

class Splitter(object):
    def partition(self, key, n_partitions):
        return hash(key) % n_partitions

class Dataset(object):

    def read(self):
        raise NotImplementedError()

    def grouped_read(self):
        for key, group in itertools.groupby(self.read(), key=lambda x: x[0]):
            it = (kv[1] for kv in group)
            yield key, it

    def delete(self):
        raise NotImplementedError()

    def __iter__(self):
        return self.read()

class EmptyDataset(Dataset):

    def read(self):
        return iter([])

    def delete(self):
        pass

class Chunker(object):
    def chunks(self):
        raise NotImplementedError()

class TextInput(Chunker):
    def __init__(self, path, chunk_size=64*1024**2):
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
    def __init__(self, worker_fs):
        self.worker_fs = worker_fs

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
        with gzip.GzipFile(self.path, 'rb', 1) as f:
            n_records = pickle.load(f)
            for _ in range(n_records):
                yield pickle.load(f)

    def delete(self):
        os.unlink(self.path)

    def __str__(self):
        return 'PickledDataset[path={}]'.format(self.path)
    __repr__ = __str__

class PickledDatasetWriter(DatasetWriter):
    def __init__(self, worker_fs, splitter, n_partitions, 
            buffer_size=10*1024*1024):
        super(PickledDatasetWriter, self).__init__(worker_fs)
        self.splitter     = splitter
        self.n_partitions = n_partitions
        self.buffer_size  = buffer_size

    def start(self):
        self.files   = []
        self.buffers = []
        self.keyoffs = []
        for i in range(self.n_partitions):
            self.files.append([])
            self.buffers.append(StringIO())
            self.keyoffs.append([])

    def write_batch(self, suffix, buf, keyoffs):
        path = self.worker_fs.get_file(suffix)
        with gzip.GzipFile(path, 'wb', 1) as f:
            dump_pickle(len(keyoffs), f)
            for kv in self._sort_kvs(keyoffs, buf):
                f.write(kv)

        return path

    def _sort_kvs(self, keyoffs, buf):
        keyoffs.sort(key=lambda x: x[0])
        for _, start, stop in keyoffs:
            buf.seek(start)
            yield buf.read(stop - start)

    def flush(self, nidx):
        buf, keyoffs = self.buffers[nidx], self.keyoffs[nidx]
        suffix = '{}.{}'.format(nidx, len(self.files[nidx]))
        file_name = self.write_batch(suffix, buf, keyoffs)
        dataset = PickledDataset(file_name)
        self.files[nidx].append(dataset)
        buf.truncate(0)
        self.keyoffs[nidx] = []

    def _write_to_buffer(self, key, value, f):
        key_start = f.tell()
        dump_pickle((key, value), f)
        return key_start, f.tell()

    def add_record(self, key, value):
        nidx = self.splitter.partition(key, self.n_partitions)
        buf, keyoffs = self.buffers[nidx], self.keyoffs[nidx]
        kvs, kvst = self._write_to_buffer(key, value, buf)
        keyoffs.append((key, kvs, kvst))
        if buf.tell() > self.buffer_size:
            self.flush(nidx)

    def finished(self):
        for nidx, buffer in enumerate(self.buffers):
            if buffer.tell() > 0:
                self.flush(nidx)

        return {i: fs for i, fs in enumerate(self.files)}

class MemGZipDataset(Dataset):
    def __init__(self, sio):
        self.sio = sio

    def read(self):
        with gzip.GzipFile(fileobj=StringIO(self.sio)) as sio:
            n_records = pickle.load(sio)
            for _ in range(n_records):
                yield pickle.load(sio)

    def delete(self):
        pass

class UnorderedWriter(DatasetWriter):
    def __init__(self, worker_fs, chunk_size=64*1024*1024, always_to_disk=False):
        super(UnorderedWriter, self).__init__(worker_fs)
        self.chunk_size = chunk_size
        self.chunk_id = 0
        self.files = []
        self.always_to_disk = always_to_disk
    
    def _write_to_gzip(self, fobj):
        buf = self.buffer
        with gzip.GzipFile(fileobj=fobj, mode='wb', compresslevel=1) as f:
            dump_pickle(self.records, f)

            buf.seek(0)
            data = buf.read(4096)
            while data:
                f.write(data)
                data = buf.read(4096)


    def flush(self):
        buf = self.buffer
        chunk_name = self.worker_fs.get_file(str(self.chunk_id))
        self.chunk_id += 1
        with open(chunk_name, 'wb') as out:
            self._write_to_gzip(out)
        
        self.files.append(PickledDataset(chunk_name, True))
        self._new_buffer()

    def _new_buffer(self):
        self.records = 0
        self.buffer = StringIO()

    def start(self):
        self._new_buffer()

    def add_record(self, key, value):
        dump_pickle((key, value), self.buffer)
        self.records += 1

        if self.buffer.tell() > self.chunk_size:
            self.flush()

    def flush_to_memory(self):
        tmp = StringIO()
        self._write_to_gzip(tmp)
        # Compress the data 
        d = MemGZipDataset(tmp.getvalue())
        self.files.append(d)

    def finished(self):
        if self.records > 0:
            if self.always_to_disk:
                self.flush()
            else:
                self.flush_to_memory()

        return {0: self.files}
    
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
    def reduce(self, *datasets):
        raise NotImplementedError()

    def yield_groups(self, dataset):
        if len(dataset) > 1:
            dataset = MergeDataset(dataset)
        elif len(dataset) == 1:
            dataset = dataset[0]
        else:
            dataset = EmptyDataset()

        return dataset.grouped_read()

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

    def reduce(self, *datasets):
        assert len(datasets) == 2
        g1 = self.yield_groups(datasets[0])
        g2 = self.yield_groups(datasets[1])
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

class Combiner(object):
    """
    Interface for the combining ordered chunks from the Map stage
    """

    def combine(self, datasets):
        """
        Takes in a set of datasets and streams out key/values
        """
        raise NotImplementedError()

class NoopCombiner(Combiner):

    def combine(self, datasets):
       md = MergeDataset(datasets)
       return md.read()

class PartialReduceCombiner(Combiner):
    def __init__(self, reducer):
        self.reducer = reducer

    def combine(self, datasets):
        pass

class Shuffler(object):
    def __init__(self, fs, n_partitions, splitter):
        self.fs = fs
        self.n_partitions = n_partitions
        self.splitter = splitter

    def shuffle(self, fs, datasets):
        """
        Needs to return a {partition_id: [datasets]}
        """
        raise NotImplementedError()

class DefaultShuffler(Shuffler):
    def shuffle(self, fs, datasets):
        partitions = []
        for i in range(self.n_partitions):
            writer = UnorderedWriter(fs.get_substage('partition_{}'.format(i)))
            writer.start()
            partitions.append(writer)

        for k, v in MergeDataset(datasets).read():
            p_idx = self.splitter.partitions(key, self.n_partitions)
            partitions[p_idx].add_record(k, v)

        splits = {}
        for i, writer in enumerate(partitions):
            splits[i] = writer.finished()[0]

        return splits

class FileSystem(object):
    def __init__(self, path):
        self.path = path

    def get_stage(self, name):
        return StageFileSystem(os.path.join(self.path, 'stage_{}'.format(name)))

class StageFileSystem(object):
    def __init__(self, path):
        self.path = path

    def get_worker(self, w_id):
        return WorkerFileSystem(os.path.join(self.path, 'worker_{}'.format(w_id)))

class WorkingFileSystem(object):
    def __init__(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)

        self.path = path 

    def get_file(self, name=None):
        if name is None:
            name = str(uuid.uuid4())

        new_file = os.path.join(self.path, name)
        return new_file

class WorkerFileSystem(WorkingFileSystem):
            
    def get_substage(self, w_id):
        return SubStageFileSystem(os.path.join(self.path, 'sub_{}'.format(w_id)))

class SubStageFileSystem(WorkingFileSystem):
    pass
    

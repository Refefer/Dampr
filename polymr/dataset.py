import gzip
import os
import sys
import itertools
import heapq
from operator import itemgetter

PY_MAJOR = sys.version_info.major

try:
    import cPickle as pickle
    from  cStringIO import StringIO

    def dump_pickle(obj, f):
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

except ImportError:
    import pickle
    from io import BytesIO as StringIO

    def dump_pickle(o, f):
        pickle.dump(o, f,  pickle.HIGHEST_PROTOCOL)


class DatasetWriter(object):
    def __init__(self, worker_fs):
        self.worker_fs = worker_fs

    def start(self):
        raise NotImplementedError()

    def add_record(self, key, value):
        raise NotImplementedError()

    def finished(self):
        raise NotImplementedError()


class BufferedWriter(DatasetWriter):
    def __init__(self, f, max_size=1024**2):
        self.f = f
        self.max_size = max_size
        self.buffer = []
        self.size = 0
        self.empty = True

    def write(self, data):
        self.empty = False
        self.buffer.append(data)
        self.size += len(data)
        if self.size > self.max_size:
            self.flush()

    def flush(self):
        for p in self.buffer:
            self.f.write(p)

        self.f.flush()
        del self.buffer[:]
        self.size = 0

    def close(self):
        self.flush()
        self.f.close()
 
class BufferedSortedWriter(DatasetWriter):
    def __init__(self, fs, buffer_size=1*1024**2, always_to_disk=True):
        self.fs = fs
        self.buffer_size = buffer_size
        self.always_to_disk = always_to_disk

    def start(self):
        self.files = []
        self.buffer = StringIO()
        self.keyoffs = []

    def write_batch_to_disk(self):
        path = self.fs.get_file(str(len(self.files)))
        with open(path, 'wb') as out:
            self._write_to_gzip(out)

        return path

    def _write_to_gzip(self, f):
        with gzip.GzipFile(fileobj=f, mode='wb', compresslevel=1) as f:
            dump_pickle(len(self.keyoffs), f)
            bw = BufferedWriter(f)
            for kv in self._sort_kvs():
                bw.write(kv)

            bw.flush()

    def _sort_kvs(self):
        self.keyoffs.sort(key=itemgetter(0))
        v = self.buffer.getvalue()
        for _, start, stop in self.keyoffs:
            yield v[start:stop]

    def flush_to_disk(self):
        file_name = self.write_batch_to_disk()
        dataset = PickledDataset(file_name)
        self.files.append(dataset)
        self.buffer.truncate(0)
        self.keyoffs = []

    def flush_to_memory(self):
        sio = StringIO()
        self._write_to_gzip(sio)
        self.files.append(MemGZipDataset(sio.getvalue()))
        self.buffer.truncate(0)
        self.keyoffs = []

    def _write_to_buffer(self, key, value):
        key_start = self.buffer.tell()
        dump_pickle((key, value), self.buffer)
        return key_start, self.buffer.tell()

    def add_record(self, key, value):
        kvs, kvst = self._write_to_buffer(key, value)
        self.keyoffs.append((key, kvs, kvst))
        if self.buffer.tell() > self.buffer_size:
            self.flush_to_disk()

    def finished(self):
        if self.buffer.tell() > 0:
            if self.always_to_disk:
                self.flush_to_disk()
            else:
                self.flush_to_memory()

        return {0: self.files}

class CSDatasetWriter(DatasetWriter):
    def __init__(self, worker_fs, splitter, n_partitions, 
            buffer_size=10*1024*1024):
        super(CSDatasetWriter, self).__init__(worker_fs)
        self.splitter     = splitter
        self.n_partitions = n_partitions
        self.buffer_size  = buffer_size

    def start(self):
        self.partitions = []
        for i in range(self.n_partitions):
            sub_fs = self.worker_fs.get_substage('partition_{}'.format(i))
            self.partitions.append(BufferedSortedWriter(sub_fs, self.buffer_size))
            self.partitions[-1].start()

    def add_record(self, key, value):
        nidx = self.splitter.partition(key, self.n_partitions)
        self.partitions[nidx].add_record(key, value)

    def finished(self):
        return {i: p.finished()[0] for i, p in enumerate(self.partitions)}

class ContiguousWriter(DatasetWriter):
    def __init__(self, worker_fs):
        super(ContiguousWriter, self).__init__(worker_fs)
        self.worker_fs = worker_fs
    
    def start(self):
        self.fname = self.worker_fs.get_file()
        self.f = BufferedWriter(gzip.GzipFile(self.fname, 'wb', 1))
    
    def add_record(self, key, value):
        dump_pickle((key, value), self.f)

    def finished(self):
        if self.f.empty:
            self.f.close()
            os.unlink(self.fname)
            return {0: []}

        self.f.close()
        return {0: [PickledDataset(self.fname, False)]}

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
            data = buf.read(4096 * 4)
            while data:
                f.write(data)
                data = buf.read(4096 * 4)

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

class Chunker(object):
    def chunks(self):
        raise NotImplementedError()

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

class PickledDataset(Dataset):
    def __init__(self, path, header_field=True):
        self.path = path
        self.header_field = header_field

    def read(self):
        with gzip.GzipFile(self.path, 'rb', 1) as f:
            if self.header_field:
                n_records = pickle.load(f)
                for _ in range(n_records):
                    yield pickle.load(f)
            else:
                try:
                    while True:
                        yield pickle.load(f)
                except EOFError:
                    pass

    def delete(self):
        os.unlink(self.path)

    def __str__(self):
        return 'PickledDataset[path={}]'.format(self.path)
    __repr__ = __str__


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

    
class CatDataset(Dataset, Chunker):
    def __init__(self, datasets):
        self.datasets = datasets

    def read(self):
        for ds in self.datasets:
            for x in ds.read():
                yield x

    def delete(self):
        for d in self.datasets:
            d.delete()

    def chunks(self):
        for d in self.datasets:
            yield d

class MergeDataset(Dataset, Chunker):
    def __init__(self, datasets):
        self.datasets = datasets

    def read(self):
        if len(self.datasets) == 1:
            return self.datasets[0].read()

        its = [ds.read() for ds in self.datasets]
        if PY_MAJOR == 3:
            return heapq.merge(*its, key=lambda x: x[0])

        return heapq.merge(*its)

    def chunks(self):
        for d in self.datasets:
            yield d


    def delete(self):
        for d in self.datasets:
            d.delete()

class MemoryDataset(Dataset, Chunker):
    def __init__(self, kvs, partitions=13):
        self.kvs = kvs
        self.partitions = partitions

    def read(self):
        for k, v in self.kvs:
            yield k, v

    def delete(self):
        pass

    def chunks(self):
        chunk_size = len(self.kvs) / self.partitions
        start = 0
        while start < len(self.vs):
            yield MemoryDataset(self.kvs[start:start+chunk_size])
            start += chunk_size

class StreamDataset(Dataset):
    def __init__(self, it):
        self.it = it
    
    def read(self):
        return self.it

    def delete(self):
        pass

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
    def __init__(self, items, partitions=50):
        self.items = items
        self.partitions = min(len(items), partitions)

    def chunks(self):
        chunk_size = int(len(self.items) // float(self.partitions))
        for start in range(0, len(self.items), chunk_size):
            yield MemoryDataset(self.items[start:start+chunk_size])

class DMChunker(Chunker):
    def __init__(self, data_mapping):
        self.dm = data_mapping

    def chunks(self):
        for vs in self.dm.values():
            for v in vs:
                yield v


from __future__ import print_function
import gzip
from math import ceil
import gc
import logging
import os
import sys
import itertools
import heapq
import io
from operator import itemgetter

from .memory import MemoryChecker

if sys.version_info.major == 3:
    import dampr.settings as settings
else:
    import settings

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

def gc_collect():
    logging.debug("Collecting gc: {}".format(gc.collect()))

def gzip_reader(*args, **kwargs):
    fd = gzip.GzipFile(*args, **kwargs)
    return io.BufferedReader(fd, 1024**2)

class DatasetWriter(object):
    def __init__(self, worker_fs):
        self.worker_fs = worker_fs

    def start(self):
        raise NotImplementedError()

    def add_record(self, key, value):
        raise NotImplementedError()

    def flush(self):
        raise NotImplementedError()

    def finished(self):
        raise NotImplementedError()

class BufferedWriter(object):
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
        self.f.write(b''.join(self.buffer))
        self.f.flush()
        del self.buffer[:]
        self.size = 0

    def close(self):
        self.flush()
        self.f.close()
 
class ReducedWriter(DatasetWriter):
    """
    Used when a binary reducer function can partially join records on
    the Map side.  Writes out to an underlying DataWriter when the number
    of unique values in active reduction exceed a certain threshold
    """
    SENTINEL = object()
    def __init__(self, dw, binop):
        self.dw = dw
        self.binop = binop
        self.key_set = {}

    def start(self):
        self.dw.start()
        self.key_set.clear()

    def add_record(self, key, value):
        cached_value = self.key_set.get(key, self.SENTINEL)
        if cached_value is not self.SENTINEL:
            self.key_set[key] = self.binop(value, cached_value)
        else:
            self.key_set[key] = value

    def flush(self, finished=False):
        for k, v in self.key_set.items():
            self.dw.add_record(k, v)

        self.dw.flush()
        if not finished:
            self.key_set.clear()

    def finished(self):
        self.flush(True)
        return self.dw.finished()

class SimpleWriter(DatasetWriter):

    def __init__(self, fs, batch_size=settings.batch_size):
        self.fs = fs
        self.batch_size = batch_size

    def start(self):
        self.files = []
        self.kvs = []

    def _write_to_gzip(self, f):
        with gzip.GzipFile(fileobj=f, mode='wb', compresslevel=settings.compress_level) as f:
            bw = BufferedWriter(f)
            i = 0
            while i < len(self.kvs):
                dump_pickle(self.kvs[i:i+self.batch_size], bw)
                i += self.batch_size

            bw.flush()

    def flush_buffer(self):
        raise NotImplementedError()

    def add_record(self, key, value):
        self.kvs.append((key, value))

    def reset(self):
        self.kvs = []

    def flush(self, finished=False):
        if len(self.kvs) > 0:
            dataset = self.flush_buffer()
            self.files.append(dataset)
            if not finished:
                self.reset()

    def finished(self):
        if len(self.kvs) > 0:
            self.flush(True)

        return {0: self.files}

class SortedWriter(SimpleWriter):
    def _write_to_gzip(self, f):
        self.kvs.sort(key=itemgetter(0))
        super(SortedWriter, self)._write_to_gzip(f)

class SortedMemoryWriter(SortedWriter):
    """
    Writes out the Buffer to Memory
    """
    def flush_buffer(self):
        sio = StringIO()
        self._write_to_gzip(sio)
        return MemGZipDataset(sio.getvalue(), batched=True)

class SortedDiskWriter(SortedWriter):
    """
    Writes out Buffer to Disk
    """
    def write_batch_to_disk(self):
        path = self.fs.get_file(str(len(self.files)))
        with open(path, 'wb') as out:
            self._write_to_gzip(out)

        return path

    def flush_buffer(self):
        file_name = self.write_batch_to_disk()
        return PickledDataset(file_name, batched=True)

class MaxMemoryWriter(DatasetWriter):
    def __init__(self, writer):
        self.writer = writer
        self.mem_checker = MemoryChecker(settings.max_memory_per_worker)

    def start(self):
        self.writer.start()
        self.mem_checker.start()

    def add_record(self, key, value):
        if self.mem_checker.check_over_highwatermark():
            self.writer.flush()
            gc_collect()
            self.mem_checker.reset()

        self.writer.add_record(key, value)

    def finished(self):
        return self.writer.finished()

class CSDatasetWriter(DatasetWriter):
    """
    Writes out a stream of key values, splitting into separate partitions based on
    the provided Splitter.
    """
    def __init__(self, worker_fs, splitter, n_partitions, 
            writer_cls=SortedDiskWriter, writer_args=None):
        super(CSDatasetWriter, self).__init__(worker_fs)
        self.splitter     = splitter
        self.n_partitions = n_partitions
        self.mem_checker = MemoryChecker(settings.max_memory_per_worker)

        assert issubclass(writer_cls, SortedWriter)
        self.writer_cls = writer_cls
        self.writer_args = writer_args if writer_args is not None else {}

    def start(self):
        self.kvs = []
        self.partitions = []
        for i in range(self.n_partitions):
            sub_fs = self.worker_fs.get_substage('partition_{}'.format(i))
            self.partitions.append(self.writer_cls(sub_fs, **self.writer_args))
            self.partitions[-1].start()

        self.mem_checker.start()

    def flush(self, finished=False):
        if not self.kvs:
            return

        # Add to the underlying SortedWriters
        for k, v in self.kvs:
            nidx = self.splitter.partition(k, self.n_partitions)
            self.partitions[nidx].add_record(k, v)

        # Flush them out
        for p in self.partitions:
            p.flush()

        # Cleanup
        if not finished:
            self.kvs = []
            gc_collect()
            self.mem_checker.reset()

    def add_record(self, key, value):
        self.kvs.append((key, value))
        if self.mem_checker.check_over_highwatermark():
            self.flush()
            
    def finished(self):
        self.flush(True)
        return {i: p.finished()[0] for i, p in enumerate(self.partitions)}

class SinkWriter(DatasetWriter):
    """
    Writes out the value portion of the key/value pair in text format.
    """
    def __init__(self, path, idx):
        super(SinkWriter, self).__init__(None)
        self.path = path
        self.idx = idx
        self.fname = os.path.join(self.path, 'part-{}'.format(self.idx))

    def start(self):
        self.f = open(self.fname, 'w')

    def add_record(self, key, value):
        print(value, file=self.f)

    def finished(self):
        self.f.close()
        return {0: [TextLineDataset(self.fname)]}

class ContiguousWriter(DatasetWriter):
    """
    Writes out data unordered into a Gzipped file
    """
    def __init__(self, worker_fs, batch_size=None):
        super(ContiguousWriter, self).__init__(worker_fs)
        self.worker_fs = worker_fs
        if batch_size is None:
            batch_size = settings.batch_size

        self.batch_size = batch_size
    
    def get_fileobj(self):
        raise NotImplementedError()

    def get_dataset(self):
        raise NotImplementedError()
    
    def start(self):
        self.buffer = []
        self.f = BufferedWriter(self.get_fileobj())
    
    def add_record(self, key, value):
        self.buffer.append((key, value))
        if len(self.buffer) == self.batch_size:
            self.flush()

    def flush(self):
        dump_pickle(self.buffer, self.f)
        self.buffer = []

    def finished(self):
        self.flush()
        if self.f.empty:
            self.f.close()
            return {0: []}

        self.f.close()
        return {0: [self.get_dataset()]}

class ContiguousDiskWriter(ContiguousWriter):
    """
    Writes out data unordered into a Gzipped file
    """
    def get_fileobj(self):
        self.fname = self.worker_fs.get_file()
        return gzip.GzipFile(self.fname, 'wb', settings.compress_level)

    def get_dataset(self):
        return PickledDataset(self.fname, True)
    
class ContiguousMemoryWriter(ContiguousWriter):
    """
    Writes out data unordered into a Gzipped file
    """
    def get_fileobj(self):
        self.buffer = StringIO()
        return gzip.GzipFile(fileobj=self.buffer, 
                mode='wb', 
                compresslevel=settings.compress_level)

    def get_dataset(self):
        return MemGZipDataset(self.buffer.getvalue(), True)

class UnorderedWriter(SimpleWriter):
    """
    Writes unordered chunks to disk
    """
    def _write_to_gzip(self, fobj):
        buf = self.buffer
        with gzip.GzipFile(fileobj=fobj, 
                mode='wb', 
                compresslevel=settings.compress_level) as f:

            buf.seek(0)
            data = buf.read(4096 * 4)
            while data:
                f.write(data)
                data = buf.read(4096 * 4)

    def flush(self):
        raise NotImplementedError()

    def flush_batch(self):
        if len(self.batch) > 0:
            dump_pickle(self.batch, self.buffer)
            self.batch = []

    def reset(self):
        self.buffer = StringIO()

    def start(self):
        self.batch = []
        self.reset()

    def add_record(self, key, value):
        self.batch.append((key, value))
        if len(self.batch) == self.batch_size:
            self.flush_batch()
            if self.buffer.tell() > self.chunk_size:
                self._flush_buffer()

    def _flush_buffer(self):
        dataset = self.flush()
        self.files.append(dataset)
        self.reset()

    def finished(self):
        self.flush_batch()
        if self.buffer.tell() > 0:
            self._flush_buffer()

        return {0: self.files}

class UnorderedDiskWriter(SimpleWriter):
    """
    Writes unordered chunks to disk
    """
    def flush_buffer(self):
        chunk_name = self.worker_fs.get_file(str(self.chunk_id))
        self.chunk_id += 1
        with open(chunk_name, 'wb') as out:
            self._write_to_gzip(out)
        
        return PickledDataset(chunk_name, True)

class UnorderedMemoryWriter(SimpleWriter):
    """
    Writes unordered chunks to memory
    """
    def flush_buffer(self):
        tmp = StringIO()
        self._write_to_gzip(tmp)
        # Compress the data 
        return MemGZipDataset(tmp.getvalue(), True)

class Chunker(object):
    def chunks(self):
        raise NotImplementedError()

class Dataset(Chunker):

    def read(self):
        raise NotImplementedError()

    def grouped_read(self):
        for key, group in itertools.groupby(self.read(), key=lambda x: x[0]):
            group = list(group)
            it = (kv[1] for kv in group)
            yield key, it

    def delete(self):
        raise NotImplementedError()

    def __iter__(self):
        return self.read()

    def chunks(self):
        yield self

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

            for line in f:
                yield cur_pos, line.rstrip(os.linesep)
                cur_pos += len(line)
                if self.end is not None and cur_pos > self.end:
                    break

    def __str__(self):
        return "Text[path={},start={},end={}]".format(self.path,self.start, self.end)

    def delete(self):
        pass

class GzipLineDataset(Dataset):
    def __init__(self, path):
        self.path = path

    def read(self):
        with gzip_reader(self.path) as f:
            cur_pos = 0
            for line in f:
                yield cur_pos, line.rstrip(os.linesep)
                cur_pos += len(line)

    def __str__(self):
        return "GzipFile[path={}]".format(self.path)

    def delete(self):
        pass

class PickledDataset(Dataset):
    def __init__(self, path, batched=False):
        self.path = path
        self.batched = batched

    def read(self):
        with gzip_reader(self.path, 'rb') as f:
            try:
                if self.batched:
                    while True:
                        for d in pickle.load(f):
                            yield d
                else:
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
    def __init__(self, sio, batched=False):
        self.sio = sio
        self.batched = batched

    def read(self):
        with gzip_reader(fileobj=StringIO(self.sio)) as sio:
            try:
                if not self.batched:
                    while True:
                        yield pickle.load(sio)
                else:
                    while True:
                        for d in pickle.load(sio):
                            yield d
            except EOFError:
                pass

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

class MemoryDataset(Dataset):
    def __init__(self, kvs, partitions=13):
        self.kvs = kvs
        self.partitions = partitions

    def read(self):
        for k, v in self.kvs:
            yield k, v

    def delete(self):
        pass

    def chunks(self):
        if self.partitions == 1:
            yield self
        else:
            chunk_size = int(ceil(len(self.kvs) / float(self.partitions)))
            start = 0
            while start < len(self.kvs):
                yield MemoryDataset(self.kvs[start:start+chunk_size], 1)
                start += chunk_size

class StreamDataset(Dataset):
    def __init__(self, it):
        self.it = it
    
    def read(self):
        return self.it

    def delete(self):
        pass

class DMChunker(Chunker):
    def __init__(self, data_mapping):
        self.dm = data_mapping

    def chunks(self):
        for vs in self.dm.values():
            for v in vs:
                yield v


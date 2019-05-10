import uuid
import os

from .dataset import *

class Splitter(object):
    def partition(self, key, n_partitions):
        return hash(key) % n_partitions

class Mapper(object):
    def map(self, *datasets):
        raise NotImplementedError()

class Streamable(object):
    def stream(self, kvs):
        raise NotImplementedError()

class Map(Mapper, Streamable):
    """
    Standard Mapper
    """
    def __init__(self, mapper):
        assert not isinstance(mapper, Mapper)
        self.mapper = mapper

    def map(self, *datasets):
        assert len(datasets) == 1
        return self.stream(datasets[0].read())

    def stream(self, kvs):
        for key, value in kvs:
            for nkv in self.mapper(key, value):
                yield nkv

    def __unicode__(self):
        name = getattr(self.mapper, '__name__', str(type(self.mapper)))
        return u'Map[{}]'.format(name)

    __str__ = __unicode__
    __repr__ = __unicode__

class ComposedStreamable(Streamable):
    def __init__(self, left, right):
        assert isinstance(left, Streamable)
        assert isinstance(right, Streamable)
        self.left = left
        self.right = right

    def stream(self, kvs):
        return self.right.stream(self.left.stream(kvs))

class ComposedMapper(Mapper):
    def __init__(self, left, right):
        assert isinstance(left, Mapper)
        assert isinstance(right, Streamable)
        self.left = left
        self.right = right

    def map(self, *datasets):
        return self.right.stream(self.left.map(*datasets))

class BlockMapper(Mapper, Streamable):
    """
    Custom BlockMapper.  User's can specify how a Mapper instance can
    consume a stream of key-values, allowing for custom logic.
    """
    
    def start(self):
        """
        Sets up instance variables for when a Mapper begins consumption of 
        a Map Block.
        """
        pass

    def add(self, key, value):
        """
        Logic for how to handle new key-value pairs.  This function is required
        to return an iterator regardless of whether it yields data: this gives
        a more flexible definition.
        """
        raise NotImplementedError()

    def finish(self):
        """
        Mapping is finished.  In the case of aggregations, this should yield out
        all remaining key-values to consume.
        """
        return ()

    def map(self, *datasets):
        assert len(datasets) == 1
        return self.stream(datasets[0].read())

    def stream(self, kvs):
        self.start()
        for key, value in kvs:
            for out in self.add(key, value):
                yield out

        for out in self.finish():
            yield out

class StreamMapper(Mapper, Streamable):
    """
    Simple generator based block mapper.
    """

    def __init__(self, streamer_f):
        self.streamer_f = streamer_f
    
    def map(self, *datasets):
        assert len(datasets) == 1
        return self.stream(datasets[0].read())

    def stream(self, kvs):
        it = (v for _k, v in kvs)
        return self.streamer_f(it)

    def __unicode__(self):
        name = getattr(self.streamer_f, '__name__', str(type(self.streamer_f)))
        return u'StreamMapper[{}]'.format(self.streamer_f.__name__)

    __str__ = __unicode__
    __repr__ = __unicode__

def group_datasets(dataset):
    if isinstance(dataset, Chunker):
        dataset = list(dataset.chunks())
    
    if len(dataset) > 1:
        dataset = CatDataset(dataset)
    elif len(dataset) == 1:
        dataset = dataset[0]
    else:
        dataset = EmptyDataset()
    
    return dataset

class MapCrossJoin(Mapper):
    """
    Cross products two datasets.  If `cache` is True, will load up the
    right dataset into memory instead of streaming multiple times from
    disk.
    """
    def __init__(self, crosser, cache):
        self.crosser = crosser 
        self.cache = cache

    def map(self, *datasets):
        assert len(datasets) == 2
        left, right = [group_datasets(d) for d in datasets]

        # Cache the results
        if self.cache:
            cached = list(right.read())
            read_right = lambda: iter(cached)
        else:
            read_right = right.read

        for key, value in left.read():
            for key2, value2 in read_right():
                for k3, v3 in self.crosser(key, value, key2, value2):
                    yield k3, v3

class MapAllJoin(Mapper):
    def __init__(self, crosser, load_f=lambda d: [v for k, v in d]):
        self.crosser = crosser 
        self.load_f = load_f

    def map(self, *datasets):
        assert len(datasets) == 2
        left, right = [group_datasets(d) for d in datasets]

        # Cache the results
        right = self.load_f(right.read())
        for key, value in left.read():
            for k, v in self.crosser(key, value, right):
                yield k, v

class Reducer(object):
    def reduce(self, *datasets):
        raise NotImplementedError()

    def yield_groups(self, dataset):
        return self.group_datasets(dataset).grouped_read()

    def group_datasets(self, dataset):
        if len(dataset) > 1:
            dataset = MergeDataset(dataset)
        elif len(dataset) == 1:
            dataset = dataset[0]
        else:
            dataset = EmptyDataset()
        
        return dataset

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

class BlockReducer(Reducer):
    """
    Custom BlockReducer.  User's can specify how a Reducer instance can
    consume a stream of key-valueiters, allowing for custom logic.
    """
    def start(self):
        pass 

    def add(self, k, it):
        raise NotImplementedError()

    def finish(self):
        return ()

    def reduce(self, *datasets):
        assert len(datasets) == 1
        self.start()
        for k, vs in self.yield_groups(datasets[0]):
            for nk, nv in self.add(k, vs):
                yield nk, nv

        for nk, nv in self.finish():
            yield nk, nv

class StreamReducer(Reducer):
    """
    Custom BlockReducer.  User's can specify how a Reducer instance can
    consume a stream of key-valueiters, allowing for custom logic.
    """
    def __init__(self, stream_f):
        self.stream_f = stream_f

    def reduce(self, *datasets):
        assert len(datasets) == 1
        for nk, nv in self.stream_f(self.yield_groups(datasets[0])):
            yield nk, (nk, nv)

    def __unicode__(self):
        name = getattr(self.stream_f, '__name__', str(type(self.stream_f)))
        return u'StreamReducer[{}]'.format(self.stream_f.__name__)

    __str__ = __unicode__
    __repr__ = __unicode__


class KeyedReduce(Reduce):
    def reduce(self, *datasets):
        for k, v in super(KeyedReduce, self).reduce(*datasets):
            yield k, (k, v)

class InnerJoin(Reducer):
    def __init__(self, joiner_f, many=False):
        self.joiner_f = joiner_f
        self.many = many

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
                it = self.joiner_f(k, left[1], right[1])
                if not self.many:
                    it = [it]

                for nv in it:
                    yield k, nv

                left, right = next(g1, None), next(g2, None)

class KeyedInnerJoin(InnerJoin):
    def reduce(self, *datasets):
        for k, v in super(KeyedInnerJoin, self).reduce(*datasets):
            yield k, (k, v)

class LeftJoin(Reducer):
    def __init__(self, joiner_f, default=lambda: iter([])):
        self.joiner_f = joiner_f
        self.default = default

    def reduce(self, *datasets):
        assert len(datasets) == 2
        g1 = self.yield_groups(datasets[0])
        g2 = self.yield_groups(datasets[1])
        left, right = next(g1, None), next(g2, None)
        while left is not None and right is not None:
            k = left[0]
            if left[0] < right[0]:
                yield k, self.joiner_f(k, left[1], self.default())
                left = next(g1, None)
            elif left[0] > right[0]:
                right = next(g2, None)
            else:
                yield k, self.joiner_f(k, left[1], right[1])
                left, right = next(g1, None), next(g2, None)

        # Finish off left
        while left is not None:
            k = left[0]
            yield k, self.joiner_f(k, left[1], self.default())
            left = next(g1, None)

class KeyedLeftJoin(LeftJoin):
    def reduce(self, *datasets):
        for k, v in super(KeyedLeftJoin, self).reduce(*datasets):
            yield k, (k, v)

class CrossJoin(Reduce):
    def __init__(self, joiner_f):
        self.joiner_f = joiner_f

    def reduce(self, *datasets):
        assert len(datasets) == 2
        for left in self.group_datasets(datasets[0]).read():
            for right in self.group_datasets(datasets[1]).read():
                yield self.joiner_f(left[0], left[1], right[0], right[1])

class KeyedCrossJoin(CrossJoin):
    def reduce(self, *datasets):
        for k, v in super(KeyedCrossJoin, self).reduce(*datasets):
            yield k, (k, v)

class OuterJoin(Reducer):
    def __init__(self, joiner_f, default=lambda: iter([])):
        self.joiner_f = joiner_f
        self.default = default

    def reduce(self, *datasets):
        assert len(datasets) == 2
        g1 = self.yield_groups(datasets[0])
        g2 = self.yield_groups(datasets[1])
        left, right = next(g1, None), next(g2, None)
        while left is not None and right is not None:
            if left[0] < right[0]:
                yield left[0], self.joiner_f(left[0], left[1], self.default())
                left = next(g1, None)
            elif left[0] > right[0]:
                yield right[0], self.joiner_f(right[0], iter([]), right[1])
                right = next(g2, None)
            else:
                yield k, self.joiner_f(k, left[1], right[1])
                left, right = next(g1, None), next(g2, None)

        # Finish off left
        while left is not None:
            yield left[0], self.joiner_f(left[0], left[1], self.default())
            left = next(g1, None)

        # Finish off right
        while right is not None:
            yield right[0], self.joiner_f(right[0], self.default(), right[1])
            right = next(g1, None)

class KeyedOuterJoin(OuterJoin):
    def reduce(self, *datasets):
        for k, v in super(KeyedOuterJoin, self).reduce(*datasets):
            yield k, (k, v)

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
       return MergeDataset(datasets)

class UnorderedCombiner(Combiner):
    def combine(self, datasets):
        return CatDataset(datasets)

class PartialReduceCombiner(Combiner):
    def __init__(self, reducer):
        self.reducer = reducer
    
    def _combine(self, datasets):
        for k, vs in MergeDataset(datasets).grouped_read():
            yield k, self.reducer.reducer(k, vs)

    def combine(self, datasets):
        return StreamDataset(self._combine(datasets))

class Shuffler(object):
    def __init__(self, n_partitions, splitter, writer_cls):
        self.n_partitions = n_partitions
        self.splitter = splitter
        self.writer_cls = writer_cls

    def shuffle(self, fs, datasets):
        """
        Needs to return a {partition_id: [datasets]}
        """
        raise NotImplementedError()

class DefaultShuffler(Shuffler):

    def shuffle(self, fs, datasets):
        partitions = []
        for i in range(self.n_partitions):
            writer = self.writer_cls(fs.get_substage('partition_{}'.format(i)))
            writer.start()
            partitions.append(writer)

        for k, v in MergeDataset(datasets).read():
            p_idx = self.splitter.partition(k, self.n_partitions)
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
        self.path = path 

    def get_file(self, name=None):
        if name is None:
            name = str(uuid.uuid4())

        if not os.path.isdir(self.path):
            os.makedirs(self.path)

        new_file = os.path.join(self.path, name)
        return new_file

class WorkerFileSystem(WorkingFileSystem):
            
    def get_substage(self, s):
        return SubStageFileSystem(os.path.join(self.path, 'sub_{}'.format(s)))

class SubStageFileSystem(WorkingFileSystem):
    pass
    

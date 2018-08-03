import uuid
import os

from .dataset import UnorderedWriter, StreamDataset, MergeDataset, EmptyDataset, CatDataset

class Splitter(object):
    def partition(self, key, n_partitions):
        return hash(key) % n_partitions

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

class KeyedReduce(Reduce):
    def reduce(self, *datasets):
        for k, v in super(KeyedReduce, self).reduce(*datasets):
            yield k, (k, v)

class InnerJoin(Reducer):
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
    

import multiprocessing
from multiprocessing.queues import SimpleQueue
import tempfile

from base import *

CPUS = multiprocessing.cpu_count()

def mr_map(in_q, out_q, mapper, dw, sentinel):
    dw.start()
    for datasets in iter(in_q, sentinel):
        for k, v in mapper.map(*datasets):
            dw.add_record(k, v)

    out_q.put(dw.finished())

def mr_reduce(out_q, datasets, reducer, dw):
    dw.start()
    for k, v in reducer.reduce(*datasets):
        dw.add_record(k, v)

    out_q.put(dw.finished())

class Input(object):
    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

class Graph(object):
    def __init__(self):
        self.inputs = {}
        self.outputs = []
        self.stages = []

    def add_input(self, dataset):
        inp = Input('Input:%s'. format(len(self.inputs)))
        self.inputs[inp] = dataset
        return inp

    def add_mapper(self, inputs, mapper):
        inp = Input('Map: {}'.format(len(self.stages)))
        self.stages.append((inp, inputs, mapper))
        return inp

    def add_reducer(self, inputs, reducer):
        inp = Input('Reduce: {}'.format(len(self.stages)))
        self.stages.append((inp, inputs, reducer))
        return inp

    def add_output(self, name):
        self.outputs.append(name)

class RunnerBase(object):
    def __init__(self, graph, 
            n_maps=CPUS,
            n_reducers=CPUS):

        self.graph = graph
        self.n_maps = n_maps
        self.n_reducers = n_reducers

    def run_map(self, stage_id, data, mapper):
        raise NotImplementedError()

    def run_reduce(self, stage_id, data, reducer):
        raise NotImplementedError()
        
    def collapse_datamappings(self, data_mappings):
        new_data = {}
        for dm in data_mappings:
            for partition, datasets in data_mappings:
                if partition not in new_data:
                    new_data[partition] = []

                new_data[partition].extend(datasets)

        return new_data

    def run(self):
        data = self.graph.inputs.copy()
        splitter = Splitter()
        for stage_id, (output, inputs, func) in enumerate(self.graph.stages):
            input_data = [data[i] for i in inputs]
            if isinstance(func, Mapper):
                data_mapping = self.run_map(stage_id, input_data, func)

            elif isinstance(func, Reducer):
                data_mapping = self.run_reducer(stage_id, input_data, func)

            assert isinstance(data_mapping, dict)
            data[output] = data_mapping

        ret = []
        for output in self.graph.outputs:
            ret.append(CatDataset([v for vs in data[output].values() for v in vs]))

        return ret

class SimpleRunner(RunnerBase):

    def run_map(self, stage_id, data_mappings, mapper):
        # if we get more than two input mappings, we only iterate over the first one
        iter_dm = data_mappings[0]
        if not isinstance(iter_dm, Chunker):
            iter_dm = DMChunker(iter_dm)

        dw = PickledDatasetWriter('/tmp/stage_{}_map'.format(stage_id), Splitter(), 1)
        dw.start()
        for chunk in iter_dm.chunks():
            for k, v in mapper.map(chunk, *data_mappings[1:]):
                dw.add_record(k, v)

        return dw.finished()

    def run_reducer(self, stage_id, data_mappings, reducer):
        dw = ChunkedPickledDatasetWriter('/tmp/stage_{}_reduce'.format(stage_id))
        dw.start()
        for r_idx in data_mappings[0].keys():
            datasets = [dm[r_idx] for dm in data_mappings]
            for k, v in reducer.reduce(*datasets):
                dw.add_record(k, v)

        return dw.finished()

#class MTRunner(RunnerBase):
#    def run_map(self, data_mappings, mapper):
#        dw = PickledDatasetWriter('/tmp/stage_{}'.format(i), splitter, self.n_reducers)
#        input_q = SimpleQueue()
#        output_q = SimpleQueue()
#        sentinel = object()
#        mappers = [multiprocessing.Process(target=map, 
#            args=(input_q, output_q, mapper, dw, sentinel)) for _ in range(self.n_maps)]
#
#        for m in mappers:
#            m.start()
#

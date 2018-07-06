import multiprocessing
from multiprocessing.queues import SimpleQueue
import tempfile

from base import *

CPUS = multiprocessing.cpu_count()


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
            for partition, datasets in dm.items():
                if partition not in new_data:
                    new_data[partition] = []

                new_data[partition].extend(datasets)

        return new_data

    def run(self):
        data = self.graph.inputs.copy()
        to_delete = set()
        splitter = Splitter()
        for stage_id, (output, inputs, func) in enumerate(self.graph.stages):
            input_data = [data[i] for i in inputs]
            if isinstance(func, Mapper):
                data_mapping = self.run_map(stage_id, input_data, func)

            elif isinstance(func, Reducer):
                data_mapping = self.run_reducer(stage_id, input_data, func)

            assert isinstance(data_mapping, dict)
            data[output] = data_mapping
            to_delete.add(output)

        ret = []
        for output in self.graph.outputs:
            cd = CatDataset([v for vs in data[output].values() for v in vs])
            ret.append(cd)
            to_delete.remove(output)

        for sd in to_delete:
            for ds in data[sd].values():
                for d in ds:
                    d.delete()

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

def mr_map(in_q, out_q, mapper, dw):
    dw.start()
    while True:
        datasets = in_q.get()
        if datasets is None:
            break

        main, supplemental = datasets
        for k, v in mapper.map(main, *supplemental):
            dw.add_record(k, v)

    out_q.put(dw.finished())

def mr_reduce(in_q, out_q, reducer, dw):
    dw.start()
    while True:
        datasets = in_q.get()
        if datasets is None:
            break

        for k, v in reducer.reduce(*datasets):
            dw.add_record(k, v)

    out_q.put(dw.finished())

class MTRunner(SimpleRunner):
    def run_map(self, stage_id, data_mappings, mapper):
        input_q = SimpleQueue()
        output_q = SimpleQueue()
        mappers = []
        for m_id in range(self.n_maps):
            dw = PickledDatasetWriter('/tmp/stage_{}_map.{}'.format(stage_id, m_id), 
                    Splitter(), self.n_reducers)

            mappers.append(multiprocessing.Process(target=mr_map,
                args=(input_q, output_q, mapper, dw)))

            mappers[-1].start()

        # if we get more than two input mappings, we only iterate over the first one
        iter_dm = data_mappings[0]
        if not isinstance(iter_dm, Chunker):
            iter_dm = DMChunker(iter_dm)
        
        for chunk in iter_dm.chunks():
            input_q.put((chunk, data_mappings[1:]))
        
        for _ in range(self.n_maps):
            input_q.put(None)

        res = []
        for m in mappers:
            m.join()
            res.append(output_q.get())

        return self.collapse_datamappings(res)

    def run_reducer(self, stage_id, data_mappings, reducer):
        input_q = SimpleQueue()
        output_q = SimpleQueue()
        reducers = []
        for r_id in range(self.n_reducers):
            dw = ChunkedPickledDatasetWriter(
                    '/tmp/stage_{}_map.{}'.format(stage_id, r_id)
            ) 
            reducers.append(multiprocessing.Process(target=mr_reduce,
                args=(input_q, output_q, reducer, dw)))

            reducers[-1].start()

        # Collect across inputs
        transpose = {}
        for dm in data_mappings:
            for key_id, datasets in dm.items():
                if key_id not in transpose:
                    transpose[key_id] = []

                transpose[key_id].append(datasets)
        
        for key_id, dms in transpose.items():
            input_q.put(dms)

        for _ in range(self.n_reducers):
            input_q.put(None)

        res = []
        for r in reducers:
            r.join()
            res.append(output_q.get())

        return self.collapse_datamappings(res)

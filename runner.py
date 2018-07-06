import logging
import multiprocessing
from multiprocessing.queues import SimpleQueue
import tempfile

from base import *

CPUS = multiprocessing.cpu_count()


class Source(object):
    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

    def __str__(self):
        return "Source[name={}]".format(self.name)

    __repr__ = __str__

class Graph(object):
    def __init__(self):
        self.inputs = {}
        self.outputs = []
        self.stages = []

    def add_input(self, dataset):
        inp = Source('Input:{}'. format(len(self.inputs)))
        self.inputs[inp] = dataset
        return inp

    def add_mapper(self, inputs, mapper):
        inp = Source('Map: {}'.format(len(self.stages)))
        self.stages.append((inp, inputs, mapper))
        return inp

    def add_reducer(self, inputs, reducer):
        inp = Source('Reduce: {}'.format(len(self.stages)))
        self.stages.append((inp, inputs, reducer))
        return inp

    def add_output(self, name):
        self.outputs.append(name)

class RunnerBase(object):
    def __init__(self, name, graph):

        self.name = name
        self.graph = graph

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
            logging.info("Starting stage %s/%s", stage_id, len(self.graph.stages))
            logging.info("Function - %s", type(func))
            input_data = [data[i] for i in inputs]
            for i, id in enumerate(input_data):
                logging.info("Source: %s", inputs[i])
                logging.debug("Source Datasets: %s", id)

            logging.info("Output: %s", output)

            if isinstance(func, Mapper):
                data_mapping = self.run_map(stage_id, input_data, func)

            elif isinstance(func, Reducer):
                data_mapping = self.run_reducer(stage_id, input_data, func)
            else:
                raise TypeError("Unknown type")

            assert isinstance(data_mapping, dict)
            data[output] = data_mapping
            to_delete.add(output)

        ret = []
        for output in self.graph.outputs:
            cd = MergeDataset([v for vs in data[output].values() for v in vs])
            ret.append(cd)
            to_delete.remove(output)

        for sd in to_delete:
            for ds in data[sd].values():
                for d in ds:
                    d.delete()

        return ret

class SimpleRunner(RunnerBase):

    def _gen_dw_name(self, stage_id, suffix):
        return '/tmp/{}_stage_{}_{}'.format(self.name, stage_id, suffix) 

    # Collect across inputs
    def transpose_dms(self, data_mappings):
        transpose = {}
        for dm in data_mappings:
            for key_id, datasets in dm.items():
                if key_id not in transpose:
                    transpose[key_id] = []

                transpose[key_id].append(datasets)

        return transpose

    def run_map(self, stage_id, data_mappings, mapper):
        # if we get more than two input mappings, we only iterate over the first one
        iter_dm = data_mappings[0]
        if not isinstance(iter_dm, Chunker):
            iter_dm = DMChunker(iter_dm)

        dw = PickledDatasetWriter(self._gen_dw_name(stage_id, 'map'), Splitter(), 1)
        dw.start()
        for chunk in iter_dm.chunks():
            for k, v in mapper.map(chunk, *data_mappings[1:]):
                dw.add_record(k, v)

        return dw.finished()

    def run_reducer(self, stage_id, data_mappings, reducer):
        dw = ChunkedPickledDatasetWriter(self._gen_dw_name(stage_id, 'red'))
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
    def __init__(self, name, graph, 
            n_maps=CPUS, 
            n_reducers=CPUS,
            n_partitions=200):
        super(MTRunner, self).__init__(name, graph)
        self.n_maps = n_maps
        self.n_reducers = n_reducers
        self.n_partitions = n_partitions

    def _collapse_output(self, input_q, output_q, procs):
        for _ in range(len(procs)):
            input_q.put(None)

        res = []
        for p in procs:
            p.join()
            res.append(output_q.get())

        return self.collapse_datamappings(res)

    def run_map(self, stage_id, data_mappings, mapper):
        input_q = SimpleQueue()
        output_q = SimpleQueue()
        mappers = []
        for m_id in range(self.n_maps):
            dw = PickledDatasetWriter(self._gen_dw_name(stage_id, 'map.{}'.format(m_id)),
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
        
        return self._collapse_output(input_q, output_q, mappers)

    def run_reducer(self, stage_id, data_mappings, reducer):
        input_q = SimpleQueue()
        output_q = SimpleQueue()
        reducers = []
        for r_id in range(self.n_reducers):
            dw = ChunkedPickledDatasetWriter(
                self._gen_dw_name(stage_id, 'red.{}'.format(r_id))
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

        return self._collapse_output(input_q, output_q, reducers)

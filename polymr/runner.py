import os
import logging
import multiprocessing
from multiprocessing import Queue
from multiprocessing.queues import Empty
import tempfile

from .base import *

CPUS = multiprocessing.cpu_count()

class Source(object):
    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

    def __str__(self):
        return "Source[`{}`]".format(self.name)

    __repr__ = __str__

class Graph(object):
    def __init__(self):
        self.inputs = {}
        self.stages = []

    def add_input(self, dataset):
        inp = Source('Input:{}'. format(len(self.inputs)))
        self.inputs[inp] = dataset
        return inp

    def add_mapper(self, inputs, mapper, name=None):
        assert isinstance(mapper, Mapper)
        assert all(isinstance(inp, Source) for inp in inputs)
        if name is None:
            name = 'Map: {}'

        inp = Source(name.format(len(self.stages)))
        self.stages.append((inp, inputs, mapper))
        return inp

    def add_reducer(self, inputs, reducer, name=None):
        assert isinstance(reducer, Reducer)
        assert all(isinstance(inp, Source) for inp in inputs)
        if name is None:
            name = 'Reduce: {}'

        inp = Source(name.format(len(self.stages)))
        self.stages.append((inp, inputs, reducer))
        return inp

class RunnerBase(object):
    def __init__(self, name, graph, working_dir='/tmp'):

        self.name = name
        self.working_dir = working_dir
        self.graph = graph

    @property
    def base_path(self):
        return os.path.join(self.working_dir, self.name)

    def _gen_dw_name(self, stage_id, suffix):
        return os.path.join(self.base_path, 
                'stage_{}_{}'.format(stage_id, suffix))

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

    def run(self, outputs):
        # Create the temp directory if it doesn't exist
        if not os.path.isdir(self.base_path):
            os.makedirs(self.base_path)

        data = self.graph.inputs.copy()
        to_delete = set()
        splitter = Splitter()
        for stage_id, (source, inputs, func) in enumerate(self.graph.stages):
            logging.info("Starting stage %s/%s", stage_id, len(self.graph.stages))
            logging.info("Function - %s", type(func))
            input_data = [data[i] for i in inputs]
            for i, id in enumerate(input_data):
                logging.info("Input: %s", inputs[i])
                logging.debug("Source Datasets: %s", id)

            logging.info("Output: %s", source)

            if isinstance(func, Mapper):
                data_mapping = self.run_map(stage_id, input_data, func)

            elif isinstance(func, Reducer):
                data_mapping = self.run_reducer(stage_id, input_data, func)
            else:
                raise TypeError("Unknown type")

            assert isinstance(data_mapping, dict)
            data[source] = data_mapping
            to_delete.add(source)

        logging.info("Finished...")
        # Collect the outputs and determine what to delete
        ret = []
        for source in outputs:
            dataset = data[source]
            if isinstance(dataset, Dataset):
                cd = source
            elif isinstance(dataset, Chunker):
                cd = MergeDataset(list(dataset.chunks()))
            else:
                cd = MergeDataset([v for vs in dataset.values() for v in vs])

            ret.append(cd)
            if source in to_delete:
                to_delete.remove(source)

        # Cleanup
        for sd in to_delete:
            for ds in data[sd].values():
                for d in ds:
                    d.delete()

        return ret

class SimpleRunner(RunnerBase):

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
        dw = PickledDatasetWriter(self._gen_dw_name(stage_id, 'red'), Splitter(), 1)
        dw.start()
        for r_idx in data_mappings[0].keys():
            datasets = [dm[r_idx] for dm in data_mappings]
            for k, v in reducer.reduce(*datasets):
                dw.add_record(k, v)

        return dw.finished()

def mr_map(w_id, in_q, out_q, mapper, dw):
    dw.start()
    while True:
        datasets = in_q.get()
        if datasets is None:
            break

        m_id, main, supplemental = datasets
        logging.debug("Mapper %i: Computing map: %i", w_id, m_id)
        for k, v in mapper.map(main, *supplemental):
            dw.add_record(k, v)

    logging.debug("Mapper %i: No more chunks", w_id)
    out_q.put(dw.finished())
    logging.debug("Mapper: %i: Finished", w_id)

def mr_reduce(w_id, in_q, out_q, reducer, dw):
    dw.start()
    while True:
        payload = in_q.get()
        if payload is None:
            break

        r_id, datasets = payload
        logging.debug("Reducer %i: Reduce received: %s", w_id, r_id)
        for k, v in reducer.reduce(*datasets):
            dw.add_record(k, v)

    logging.debug("Reducer %i: No more chunks", w_id)
    output = dw.finished()
    out_q.put(output)
    logging.debug("Reducer %i: Finished", w_id)

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
        # Add end sentinel
        for _ in range(len(procs)):
            input_q.put(None)

        res = []
        for i in range(len(procs)):
            while True:
                try:
                    res.append(output_q.get(timeout=0.01))
                    logging.debug("Worker %s/%s completed", i + 1, len(procs))
                    break
                except Empty:
                    pass

        for p in procs:
            p.join()

        return self.collapse_datamappings(res)

    def run_map(self, stage_id, data_mappings, mapper):
        input_q = Queue()
        output_q = Queue()
        mappers = []
        for m_id in range(self.n_maps):
            dw = PickledDatasetWriter(self._gen_dw_name(stage_id, 'map/{}'.format(m_id)),
                    Splitter(), self.n_partitions)

            mappers.append(multiprocessing.Process(target=mr_map,
                args=(m_id, input_q, output_q, mapper, dw)))

            mappers[-1].start()

        # if we get more than two input mappings, we only iterate over the first one
        iter_dm = data_mappings[0]
        if not isinstance(iter_dm, Chunker):
            iter_dm = DMChunker(iter_dm)
        
        for i, chunk in enumerate(iter_dm.chunks()):
            input_q.put((i, chunk, data_mappings[1:]))

        logging.debug("Processing %s maps", i + 1)
        
        return self._collapse_output(input_q, output_q, mappers)

    def run_reducer(self, stage_id, data_mappings, reducer):
        input_q = Queue()
        output_q = Queue()
        reducers = []
        for r_id in range(self.n_reducers):
            dw = PickledDatasetWriter(self._gen_dw_name(stage_id, 'red/{}'.format(r_id)),
                    Splitter(), self.n_partitions)

            reducers.append(multiprocessing.Process(target=mr_reduce,
                args=(r_id, input_q, output_q, reducer, dw)))

            reducers[-1].start()

        # Collect across inputs
        transpose = {}
        for dm in data_mappings:
            for key_id, datasets in dm.items():
                if key_id not in transpose:
                    transpose[key_id] = []

                transpose[key_id].append(datasets)
        
        for key_id, dms in transpose.items():
            input_q.put((key_id, dms))

        return self._collapse_output(input_q, output_q, reducers)

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

def mr_map(job, out_q, mapper, dw):
    w_id = os.getpid()
    dw.start()
    m_id, main, supplemental = job
    logging.debug("Mapper %i: Computing map: %i", w_id, m_id)
    for k, v in mapper.map(main, *supplemental):
        dw.add_record(k, v)

    out_q.put((w_id, m_id, dw.finished()))
    logging.debug("Mapper: %i: Finished", w_id)

def mr_reduce(job, out_q, reducer, dw):
    w_id = os.getpid()
    dw.start()
    r_id, datasets= job
    logging.debug("Reducer %i: Computing map: %i", w_id, r_id)
    for k, v in reducer.reduce(*datasets):
        dw.add_record(k, v)

    out_q.put((w_id, r_id, dw.finished()))
    logging.debug("Reducer: %i: Finished", w_id)

class StageRunner(object):
    def __init__(self, max_procs):
        self.max_procs = max_procs

    def execute_stage(self, t_id, payload):
        raise NotImplementedError()

    def run(self, job_queue):
        output_q = Queue()
        jobs = {}
        finished = []
        def get_output():
            child_pid, task_id, payload = output_q.get()
            finished.append(payload)
            
            # Cleanup pid
            jobs[child_pid].join()
            del jobs[child_pid]

        # Assign tasks via forking
        jobs_left = True
        while jobs_left:
            next_job = next(job_queue, None)
            jobs_left = next_job is not None
            if next_job is not None:
                # Are we full?  If so, wait for one to exit
                if len(jobs) == self.max_procs:
                    get_output()

                proc = self.execute_stage(next_job[0], next_job, output_q)
                proc.start()
                jobs[proc.pid] = proc

        while len(jobs) != 0:
            get_output()

        return finished

class MapStageRunner(StageRunner):
    def __init__(self, max_procs, stage_id, name_gen, n_partitions, mapper):
        super(MapStageRunner, self).__init__(max_procs)
        self.name_gen = name_gen
        self.n_partitions = n_partitions
        self.stage_id = stage_id
        self.mapper = mapper

    def execute_stage(self, t_id, payload, output_q):
        # Start next job
        dw = PickledDatasetWriter(self.name_gen(
            self.stage_id, 'map/{}'.format(t_id)),
                Splitter(), self.n_partitions)

        m = multiprocessing.Process(target=mr_map,
            args=(payload, output_q, self.mapper, dw))

        return m

class ReduceStageRunner(StageRunner):
    def __init__(self, max_procs, stage_id, name_gen, reducer):
        super(ReduceStageRunner, self).__init__(max_procs)
        self.name_gen = name_gen
        self.stage_id = stage_id
        self.reducer = reducer

    def execute_stage(self, t_id, payload, output_q):
        dw = UnorderedWriter(self.name_gen(self.stage_id, 'red/{}'.format(t_id)))

        m = multiprocessing.Process(target=mr_reduce,
            args=(payload, output_q, self.reducer, dw))

        return m

class MTRunner(RunnerBase):
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
        # if we get more than two input mappings, we only iterate over the first one
        iter_dm = data_mappings[0]
        if not isinstance(iter_dm, Chunker):
            iter_dm = DMChunker(iter_dm)

        jobs_queue = ((i, chunk, data_mappings[1:]) for i, chunk in enumerate(iter_dm.chunks()))
        msr = MapStageRunner(self.n_maps, stage_id,
                self._gen_dw_name, self.n_partitions, mapper)

        finished = msr.run(jobs_queue)

        return self.collapse_datamappings(finished)

    def run_reducer(self, stage_id, data_mappings, reducer):
        # Collect across inputs
        transpose = {}
        for dm in data_mappings:
            for key_id, datasets in dm.items():
                if key_id not in transpose:
                    transpose[key_id] = []

                transpose[key_id].append(datasets)
        
        rds = ReduceStageRunner(self.n_reducers, stage_id, self._gen_dw_name, reducer)
        finished = rds.run(iter(transpose.items()))

        return self.collapse_datamappings(finished)

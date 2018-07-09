import math
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

class GMap(object):
    def __init__(self, output, inputs, mapper, combiner, shuffler):
        self.output = output
        self.inputs = inputs
        self.mapper = mapper
        self.combiner = combiner
        self.shuffler = shuffler

class GReduce(object):
    def __init__(self, output, inputs, reducer):
        self.output = output
        self.inputs = inputs
        self.reducer = reducer

class Graph(object):
    def __init__(self):
        self.inputs = {}
        self.stages = []

    def add_input(self, dataset):
        inp = Source('Input:{}'. format(len(self.inputs)))
        self.inputs[inp] = dataset
        return inp

    def add_mapper(self, inputs, mapper, combiner=None, shuffler=None, name=None):
        assert isinstance(mapper, Mapper)
        assert isinstance(combiner, (type(None), Combiner))
        assert isinstance(shuffler, (type(None), Shuffler))
        assert all(isinstance(inp, Source) for inp in inputs)
        if name is None:
            name = 'Map: {}'

        inp = Source(name.format(len(self.stages)))
        self.stages.append(GMap(inp, inputs, mapper, combiner, shuffler))
        return inp

    def add_reducer(self, inputs, reducer, name=None):
        assert isinstance(reducer, Reducer)
        assert all(isinstance(inp, Source) for inp in inputs)
        if name is None:
            name = 'Reduce: {}'

        inp = Source(name.format(len(self.stages)))
        self.stages.append(GReduce(inp, inputs, reducer))
        return inp

class RunnerBase(object):
    def __init__(self, name, graph, working_dir='/tmp'):

        self.file_system = FileSystem(os.path.join(working_dir, name))
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

    def run(self, outputs):
        data = self.graph.inputs.copy()
        to_delete = set()
        splitter = Splitter()
        for stage_id, stage in enumerate(self.graph.stages):
        #for stage_id, (source, inputs, func) in enumerate(self.graph.stages):
            logging.info("Starting stage %s/%s", stage_id, len(self.graph.stages))
            logging.info("Function - %s", type(stage))
            input_data = [data[i] for i in stage.inputs]
            for i, id in enumerate(input_data):
                logging.info("Input: %s", stage.inputs[i])

            logging.info("Output: %s", stage.output)

            if isinstance(stage, GMap):
                data_mapping = self.run_map(stage_id, input_data, stage)

            elif isinstance(stage, GReduce):
                data_mapping = self.run_reducer(stage_id, input_data, stage)
            else:
                raise TypeError("Unknown type")

            assert isinstance(data_mapping, dict)
            data[stage.output] = data_mapping
            to_delete.add(stage.output)

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

def mr_map(job, out_q, stage, fs, n_partitions):
    w_id = os.getpid()

    # Default job, nothing special
    dw = CSDatasetWriter(fs, Splitter(), n_partitions)

    dw.start()
    m_id, main, supplemental = job
    logging.debug("Mapper %i: Computing map: %i", w_id, m_id)
    for k, v in stage.mapper.map(main, *supplemental):
        dw.add_record(k, v)

    out_q.put((w_id, m_id, dw.finished()))
    logging.debug("Mapper: %i: Finished", w_id)

def mrcs_map(job, out_q, stage, combiner, shuffler, fs):
    """
    Runs a more fine grained map/combine/shuffler
    """
    w_id = os.getpid()
    dw = BufferedSortedWriter(fs, always_to_disk=False)
    dw.start()
    m_id, main, supplemental = job
    logging.debug("Mapper %i: Computing map: %i", w_id, m_id)
    for k, v in stage.mapper.map(main, *supplemental):
        dw.add_record(k, v)

    sources = dw.finished()[0]

    logging.debug("Combining outputs...")
    combined_stream = combiner.combine(sources)
    results = shuffler.shuffle(fs, [combined_stream])

    out_q.put((w_id, m_id, results))
    logging.debug("Mapper: %i: Finished", w_id)

def mr_reduce(job, out_q, stage, dw):
    w_id = os.getpid()
    dw.start()
    r_id, datasets= job
    logging.debug("Reducer %i: Computing map: %i", w_id, r_id)
    for k, v in stage.reducer.reduce(*datasets):
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
    def __init__(self, max_procs, fs, n_partitions, mapper):
        super(MapStageRunner, self).__init__(max_procs)
        self.fs = fs
        self.n_partitions = n_partitions
        self.mapper = mapper

    def execute_stage(self, t_id, payload, output_q):
        fs = self.fs.get_worker('map/{}'.format(t_id))

        if self.mapper.combiner is None and self.mapper.shuffler is None:

            p = multiprocessing.Process(target=mr_map,
                args=(payload, output_q, self.mapper, fs, self.n_partitions))
        else:
            c = NoopCombiner() if self.mapper.combiner is None else self.mapper.combiner
            s = DefaultShuffler(self.n_partitions, Splitter())
            p = multiprocessing.Process(target=mrcs_map,
                args=(payload, output_q, self.mapper, c, s, fs))

        return p

class CombinerStageRunner(StageRunner):
    def __init__(self, max_procs, fs, combiner):
        super(CombinerStageRunner, self).__init__(max_procs)
        self.fs = fs
        self.combiner = combiner

    def combine(self, payload, output_q, fs):
        w_id = os.getpid()
        t_id, datasets = payload

        dw = UnorderedWriter(fs)
        dw.start()
        for k,v in self.combiner.combine(datasets):
            dw.add_record(k,v)

        for d in datasets:
            d.delete()

        output_q.put((w_id, t_id, dw.finished()[0]))

    def execute_stage(self, t_id, payload, output_q):
        fs = self.fs.get_worker('merge/{}'.format(t_id))

        p = multiprocessing.Process(target=self.combine,
            args=(payload, output_q, fs))

        return p

class ReduceStageRunner(StageRunner):
    def __init__(self, max_procs, fs, reducer):
        super(ReduceStageRunner, self).__init__(max_procs)
        self.fs = fs
        self.reducer = reducer

    def execute_stage(self, t_id, payload, output_q):
        fs = self.fs.get_worker('red/{}'.format(t_id))
        dw = ContiguousWriter(fs)

        m = multiprocessing.Process(target=mr_reduce,
            args=(payload, output_q, self.reducer, dw))

        return m

class MTRunner(RunnerBase):
    def __init__(self, name, graph, 
            n_maps=CPUS, 
            n_reducers=CPUS,
            n_partitions=None):
        super(MTRunner, self).__init__(name, graph)
        self.n_maps = n_maps
        self.n_reducers = n_reducers

        if n_partitions is None:
            n_partitions = n_reducers * 4

        self.n_partitions = n_partitions

    def run_map(self, stage_id, data_mappings, mapper):
        # if we get more than two input mappings, we only iterate over the first one
        iter_dm = data_mappings[0]
        if not isinstance(iter_dm, Chunker):
            iter_dm = DMChunker(iter_dm)

        jobs_queue = ((i, chunk, data_mappings[1:]) for i, chunk in enumerate(iter_dm.chunks()))
        stage_fs = self.file_system.get_stage(stage_id)
        msr = MapStageRunner(self.n_maps, stage_fs, self.n_partitions, mapper)

        finished = msr.run(jobs_queue)

        collapsed = self.collapse_datamappings(finished)
        # Check for number of files
        for k, v in collapsed.items():
            while len(v) > 50:
                logging.debug("Partition %s needs to be merged: found %s files", k, len(v))
                num_files = int(math.ceil(len(v) / 50.))
                chunks = ((i, v[s:s+num_files])
                        for i, s in enumerate(range(0, len(v), num_files)))
                c = NoopCombiner() if mapper.combiner is None else mapper.combiner
                csr = CombinerStageRunner(self.n_maps, stage_fs, c)
                v = [f for fs in csr.run(chunks) for f in fs]
                collapsed[k] = v

        return collapsed

    def run_reducer(self, stage_id, data_mappings, reducer):
        # Collect across inputs
        keys = sorted({k for dm in data_mappings for k in dm})
        transpose = {k: [] for k in keys}
        for dm in data_mappings:
            for k in keys:
                transpose[k].append(dm.get(k, []))
        
        stage_fs = self.file_system.get_stage(stage_id)
        rds = ReduceStageRunner(self.n_reducers, stage_fs, reducer)
        finished = rds.run(iter(transpose.items()))

        return self.collapse_datamappings(finished)

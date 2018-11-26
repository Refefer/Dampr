import math
import os
import shutil
import logging
import multiprocessing
from multiprocessing import Queue
from multiprocessing.queues import Empty
import tempfile

from .base import *
from .dataset import *

CPUS = multiprocessing.cpu_count()

class Source(object):
    CNT = 0
    def __init__(self, name):
        self.name = name
        self.cnt = self.CNT
        type(self).CNT += 1

    def __hash__(self):
        return hash(self.cnt)

    def __eq__(self, other):
        return self.cnt == other.cnt

    def __str__(self):
        return "Source[`{}`]".format(self.name)

    __repr__ = __str__

class GMap(object):
    def __init__(self, output, inputs, mapper, combiner, shuffler, options=None):
        self.output = output
        self.inputs = inputs
        self.mapper = mapper
        self.combiner = combiner
        self.shuffler = shuffler
        self.options = options if options is not None else  {}

    def __unicode__(self):
        return u"Map".format()

    __repr__ = __unicode__

class GReduce(object):
    def __init__(self, output, inputs, reducer, options):
        self.output = output
        self.inputs = inputs
        self.reducer = reducer
        self.options = options if options is not None else  {}

    def __unicode__(self):
        return u"Reducer".format()

    __repr__ = __unicode__

class GSink(object):
    def __init__(self, output, inputs, mapper, path, options=None):
        self.output = output
        self.inputs = inputs
        self.mapper = mapper
        self.path = path
        self.options = options if options is not None else  {}

    def __unicode__(self):
        return u"Sink[path={}]".format(self.path)

    __repr__ = __unicode__

class Graph(object):
    def __init__(self):
        self.inputs = {}
        self.stages = []

    def _copy_graph(self):
        graph = Graph()
        graph.inputs.update(self.inputs)
        graph.stages.extend(self.stages)
        return graph

    def add_input(self, dataset):
        ng = self._copy_graph()
        inp = Source('Input:{}'. format(len(self.inputs)))
        ng.inputs[inp] = dataset
        return inp, ng

    def add_mapper(self, inputs, mapper, combiner=None, 
            shuffler=None, name=None, options=None):
        assert isinstance(mapper, Mapper)
        assert isinstance(combiner, (type(None), Combiner))
        assert isinstance(shuffler, (type(None), Shuffler))
        assert all(isinstance(inp, Source) for inp in inputs)
        if name is None:
            name = 'Map: {}'

        inp = Source(name.format(len(self.stages)))
        ng = self._copy_graph()
        ng.stages.append(GMap(inp, inputs, mapper, combiner, shuffler, options))
        return inp, ng

    def add_reducer(self, inputs, reducer, name=None, options=None):
        assert isinstance(reducer, Reducer)
        assert all(isinstance(inp, Source) for inp in inputs)
        if name is None:
            name = 'Reduce: {}'

        inp = Source(name.format(len(self.stages)))
        ng = self._copy_graph()
        ng.stages.append(GReduce(inp, inputs, reducer, options))
        return inp, ng

    def add_sink(self, inputs, mapper, path, name=None, options=None):
        assert isinstance(mapper, Mapper)
        assert all(isinstance(inp, Source) for inp in inputs)
        if name is None:
            name = 'Sink: {}'

        inp = Source(name.format(path))
        ng = self._copy_graph()
        ng.stages.append(GSink(inp, inputs, mapper, path, options))
        return inp, ng

    def union(self, other_graph):
        ng = self._copy_graph()
        ng.inputs.update(other_graph.inputs)
        seen_stages = set(ng.stages)
        for s in other_graph.stages:
            if s not in seen_stages:
                ng.stages.append(s)

        return ng
        
class RunnerBase(object):
    def __init__(self, name, graph, working_dir='/tmp'):
        self.file_system = FileSystem(os.path.join(working_dir, name))
        self.graph = graph

    def run_map(self, stage_id, data, mapper):
        raise NotImplementedError()

    def run_reduce(self, stage_id, data, reducer):
        raise NotImplementedError()

    def run_sink(self, stage_id, data, reducer):
        raise NotImplementedError()

    def format_outputs(self, outputs):
        new_ret = []
        for output in outputs:
            if len(output) == 1:
                output = output[0]
            else:
                output = MergeDataset(output)

            new_ret.append(output)
        
        return new_ret
        
    def collapse_datamappings(self, data_mappings):
        new_data = {}
        for dm in data_mappings:
            for partition, datasets in dm.items():
                if partition not in new_data:
                    new_data[partition] = []

                new_data[partition].extend(datasets)

        return new_data

    def run(self, outputs, cleanup=True):
        data = self.graph.inputs.copy()
        to_delete = set()
        splitter = Splitter()
        for stage_id, stage in enumerate(self.graph.stages):
            logging.info("Stage %s/%s", stage_id + 1, len(self.graph.stages))
            logging.info("Function - %s", stage)
            input_data = [data[i] for i in stage.inputs]
            for i, id in enumerate(input_data):
                logging.info("Input: %s", stage.inputs[i])

            logging.info("Output: %s", stage.output)

            cleanup_stage = True
            if isinstance(stage, GMap):
                data_mapping = self.run_map(stage_id, input_data, stage)

            elif isinstance(stage, GReduce):
                data_mapping = self.run_reducer(stage_id, input_data, stage)

            elif isinstance(stage, GSink):
                data_mapping = self.run_sink(stage_id, input_data, stage)
                # We're sinking to disk, don't delete it
                cleanup_stage = False

            else:
                raise TypeError("Unknown type")

            assert isinstance(data_mapping, dict)
            data[stage.output] = data_mapping
            if cleanup_stage:
                to_delete.add(stage.output)

        # Collect the outputs and determine what to delete
        ret = []
        for source in outputs:
            dataset = data[source]
            if isinstance(dataset, Dataset):
                cd = [dataset]
            elif isinstance(dataset, Chunker):
                cd = list(dataset.chunks())
            else:
                cd = [v for vs in dataset.values() for v in vs]

            ret.append(cd)
            if source in to_delete:
                to_delete.remove(source)

        ret = self.format_outputs(ret)
        # Cleanup
        if cleanup:
            for sd in to_delete:
                for ds in data[sd].values():
                    for d in ds:
                        d.delete()

        logging.info("Finished...")

        return ret


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
    def __init__(self, max_procs, fs, n_partitions, mapper, options):
        super(MapStageRunner, self).__init__(max_procs)
        self.fs = fs
        self.n_partitions = n_partitions
        self.mapper = mapper
        self.options = options

    def simple_map(self, job, out_q, fs):
        w_id = os.getpid()

        # Default job, nothing special
        if self.options.get('memory', False):
            writer_cls = BufferedSortedMemoryWriter
        else:
            writer_cls = BufferedSortedDiskWriter

        dw = CSDatasetWriter(fs, Splitter(), self.n_partitions, writer_cls=writer_cls)

        dw.start()
        m_id, main, supplemental = job
        logging.debug("Mapper %i: Computing map: %i", w_id, m_id)
        for k, v in self.mapper.mapper.map(main, *supplemental):
            dw.add_record(k, v)

        out_q.put((w_id, m_id, dw.finished()))
        logging.debug("Mapper: %i: Finished", w_id)

    def medium_map(self, job, out_q, combiner, shuffler, fs):
        """
        Runs a more fine grained map/combine/shuffler
        """
        w_id = os.getpid()
        if self.options.get('memory', False):
            dw = BufferedSortedMemoryWriter(fs)
        else:
            dw = BufferedSortedDiskWriter(fs)

        # Do we have a map side partial reducer?
        binop = self.options.get('binop')
        if callable(binop):
            reduce_buffer = self.options.get('reduce_buffer', 1000)
            if reduce_buffer > 0:
                # Zero buffer means all reductions will happen reduce side
                dw = ReducedWriter(dw, binop, max_values=reduce_buffer)

        dw.start()
        m_id, main, supplemental = job
        logging.debug("Mapper %i: Computing map: %i", w_id, m_id)
        for k, v in self.mapper.mapper.map(main, *supplemental):
            dw.add_record(k, v)

        sources = dw.finished()[0]

        if len(sources) > 1:
            logging.debug("Combining outputs: found %i files", len(sources))
            logging.debug("Combining: %s", sources)
            combined_stream = combiner.combine(sources)
        elif len(sources) == 1:
            combined_stream = sources[0]
        else:
            combined_stream = EmptyDataset()

        results = shuffler.shuffle(fs, [combined_stream])

        out_q.put((w_id, m_id, results))
        logging.debug("Mapper: %i: Finished", w_id)


    def execute_stage(self, t_id, payload, output_q):
        fs = self.fs.get_worker('map/{}'.format(t_id))

        if self.mapper.combiner is None and self.mapper.shuffler is None:

            p = multiprocessing.Process(target=self.simple_map,
                args=(payload, output_q, fs))
        else:
            c = NoopCombiner() if self.mapper.combiner is None else self.mapper.combiner
            if self.options.get('memory', False):
                writer_cls = UnorderedDiskWriter
            else:
                writer_cls = UnorderedMemoryWriter

            s = DefaultShuffler(self.n_partitions, Splitter(), writer_cls)
            o = self.mapper.options
            p = multiprocessing.Process(target=self.medium_map,
                args=(payload, output_q, c, s, fs))

        return p

class SinkStageRunner(StageRunner):
    def __init__(self, max_procs, mapper, path):
        super(SinkStageRunner, self).__init__(max_procs)
        if not os.path.exists(path):
            try:
                os.makedirs(path)
            except OSError as e:
                if e.errno != 17:
                    raise

        self.path = path
        self.mapper = mapper

    def sink(self, job, out_q):
        """
        Writes line delimited items as a sink
        """
        w_id = os.getpid()

        m_id, main, supplemental = job
        dw = SinkWriter(self.mapper.path, m_id)

        dw.start()
        logging.debug("Sink %i: Computing map: %i", w_id, m_id)
        for k, v in self.mapper.mapper.map(main, *supplemental):
            dw.add_record(k, v)

        out_q.put((w_id, m_id, dw.finished()))
        logging.debug("Sink: %i: Finished", w_id)

    def execute_stage(self, t_id, payload, output_q):
        p = multiprocessing.Process(target=self.sink,
            args=(payload, output_q))

        return p

class CombinerStageRunner(StageRunner):
    """
    Merges files together to reduce their on disk presence
    """
    def __init__(self, max_procs, fs, combiner, options):
        super(CombinerStageRunner, self).__init__(max_procs)
        self.fs = fs
        self.combiner = combiner
        self.options = options

    def combine(self, payload, output_q, fs):
        w_id = os.getpid()
        t_id, datasets = payload

        if self.options.get('memory', False):
            dw = ContiguousMemoryWriter(fs)
        else:
            dw = ContiguousDiskWriter(fs)

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
    def __init__(self, max_procs, fs, reducer, options):
        super(ReduceStageRunner, self).__init__(max_procs)
        self.fs = fs
        self.reducer = reducer
        self.options = options

    def reduce(self, job, out_q, dw):
        w_id = os.getpid()
        dw.start()
        r_id, datasets = job
        logging.debug("Reducer %i: Computing reduce: %i", w_id, r_id)
        for k, v in self.reducer.reducer.reduce(*datasets):
            dw.add_record(k, v)

        out_q.put((w_id, r_id, dw.finished()))
        logging.debug("Reducer: %i: Finished", w_id)

    def execute_stage(self, t_id, payload, output_q):
        fs = self.fs.get_worker('red/{}'.format(t_id))

        if self.options.get('memory', False):
            dw = ContiguousMemoryWriter(fs)
        else:
            dw = ContiguousDiskWriter(fs)

        m = multiprocessing.Process(target=self.reduce,
            args=(payload, output_q, dw))

        return m

class MTRunner(RunnerBase):
    def __init__(self, name, graph, 
            n_maps=CPUS, 
            n_reducers=CPUS,
            n_partitions=91,
            max_files_per_stage=50):
        super(MTRunner, self).__init__(name, graph)
        self.n_maps = n_maps
        self.n_reducers = n_reducers
        self.n_partitions = n_partitions
        self.max_files_per_stage = max_files_per_stage

    def chunk_list(self, v):
        """
        Given a list of files, creates subsets of those files.
        """
        chunks = min(self.max_files_per_stage, self.n_maps)
        num_files = min(int(math.ceil(len(v) / float(chunks))), self.max_files_per_stage)
        return ((i, v[s:s+num_files])
                for i, s in enumerate(range(0, len(v), num_files)))

    def run_map(self, stage_id, data_mappings, mapper):
        # if we get more than two input mappings, we only iterate over the first one
        iter_dm = data_mappings[0]
        if not isinstance(iter_dm, Chunker):
            iter_dm = DMChunker(iter_dm)

        supplementary = []
        for dm in data_mappings[1:]:
            if not isinstance(dm, Chunker):
                dm = DMChunker(dm)

            supplementary.append(list(dm.chunks()))

        jobs_queue = ((i, chunk, supplementary) for 
                i, chunk in enumerate(iter_dm.chunks()))

        stage_fs = self.file_system.get_stage(stage_id)
        n_maps = mapper.options.get('n_maps', self.n_maps)
        msr = MapStageRunner(n_maps, stage_fs, self.n_partitions, mapper, mapper.options)

        finished = msr.run(jobs_queue)

        collapsed = self.collapse_datamappings(finished)
        # Check for number of files
        for k, v in collapsed.items():
            while len(v) > self.max_files_per_stage:
                logging.debug("Partition %s needs to be merged: found %s files", k, len(v))
                chunks = self.chunk_list(v)
                c = NoopCombiner() if mapper.combiner is None else mapper.combiner
                csr = CombinerStageRunner(n_maps, stage_fs, c, mapper.options)
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
        n_reducers = reducer.options.get('n_reducers', self.n_reducers)
        rds = ReduceStageRunner(n_reducers, stage_fs, reducer, reducer.options)
        finished = rds.run(iter(transpose.items()))

        return self.collapse_datamappings(finished)

    def run_sink(self, stage_id, data_mappings, sink):
        # if we get more than two input mappings, we only iterate over the first one
        iter_dm = data_mappings[0]
        if not isinstance(iter_dm, Chunker):
            iter_dm = DMChunker(iter_dm)

        jobs_queue = ((i, chunk, data_mappings[1:]) 
                for i, chunk in enumerate(iter_dm.chunks()))
        n_maps = sink.options.get('n_maps', self.n_maps)
        ssr = SinkStageRunner(self.n_maps, sink, sink.path)

        finished = ssr.run(jobs_queue)

        return self.collapse_datamappings(finished)

    def format_outputs(self, outputs):
        
        # if mapped, we ordered the output so we use a merged dataset
        ret = []
        for output in outputs:
            while len(output) > self.max_files_per_stage:
                stage_fs = self.file_system.get_stage('final_combine')

                logging.debug("Combining final files: found %i", len(output))
                c = NoopCombiner() 
                csr = CombinerStageRunner(self.n_maps, stage_fs, c, {})
                jobs = self.chunk_list(output)
                output = [p for ps in csr.run(jobs) for p in ps]

            if len(output) == 1:
                output = output[0]

            else:
                output = MergeDataset(output)

            ret.append(output)

        return ret

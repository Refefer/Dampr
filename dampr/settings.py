import multiprocessing


# Total number of processes to process data with
max_processes = multiprocessing.cpu_count()

# Compression level for gzip
compress_level = 1

# Number of reduce partitions to emit.  Increase if using larger machines
partitions = 91

# Total number of files per stage.  If the number of created files is larger, will
# merge them into a smaller number of files.  This prevents issues with file descriptor
# limits
max_files_per_stage = 50

# To improve pickling speed to disk, dampr batches tuples to disk.  Increase or decrease
# this size to trade off between serialization versus memory
batch_size = 1000

memory_checker_type = "interpolative"

# Highwater mark for memory in a process.  If it goes over this amount, it will
# flush the buffers to disk.  Max memory will be around 
# max_processes * max_memory_per_worker * scaler
max_memory_per_worker = 512

# For the exponential memory, it every max(min_count, log(items_seen) / log(memory_check_base))
# Increase the base to check less frequently, decrease to check more frequently
memory_check_base = 1.2

# Minimum number of tuples to see before checking memory
memory_min_count = 10000

# Go a max of this many before force checking
memory_max_count_before_check = 100000

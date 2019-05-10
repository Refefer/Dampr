import logging
import math
import platform
import resource

def get_cur_memory():
    """ Return the used memory in MB """ 
    if platform.system() == 'Linux':
        # Slightly more accurate than below
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split(None, 2)[1]) >> 10

    elif platform.system() == "Darwin":
        rusage_denom = 1024. ** 2
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rusage_denom
        return mem

    # Sorry windows, not sure what to do
    raise Exception("This method doesn't support Windows currently")

class MemoryChecker(object):
    """
    Checks memory on an exponential scale of new items; when it reaches the high water mark
    """
    def __init__(self, max_memory_in_mbs, min_count=10000, base=1.3):
        self.max_memory_in_mbs = max_memory_in_mbs
        self.base = base
        self.min_count = min_count

    def start(self):
        self.start_memory = get_cur_memory()
        self.count = 0
        self.last_check = 0 

    def check_over_highwatermark(self):
        self.count += 1
        if self.count > self.min_count:
            next_check = math.log(self.count) / math.log(self.base)
            # Check memory
            if (self.last_check + 1) < next_check:
                cur_memory = get_cur_memory()
                if cur_memory >= self.start_memory + self.max_memory_in_mbs:
                    logging.debug("cur_memory:{},start_memory:{}".format(
                        cur_memory, self.start_memory))

                    # ok, time to dump
                    return True

                # Only update if we aren't over the high watermark
                self.last_check += 1

        return False



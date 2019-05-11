import logging
import math
import platform
import resource
import sys

if sys.version_info.major == 3:
    import dampr.settings as settings
else:
    import settings

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

class ExponentialMemoryChecker(object):
    """
    Checks memory on an exponential scale of new items; when it reaches the high water mark
    """
    def __init__(self, max_memory_in_mbs, min_count=None, base=None):
        self.max_memory_in_mbs = max_memory_in_mbs
        if min_count is None:
            min_count = settings.memory_min_count

        self.min_count = min_count

        if base is None:
            base = settings.memory_min_count

        self.min_count = min_count

    def start(self):
        self.start_memory = get_cur_memory()
        self.reset()

    def reset(self):
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

class InterpolativeMemoryChecker(object):
    """
    Attempts to estimate the per/item cost and estimate when to check next.
    """
    def __init__(self, max_memory_in_mbs):
        self.max_memory_in_mbs = max_memory_in_mbs

    def start(self):
        self.mem_per_item = 1e-7
        self.start_memory = get_cur_memory()
        self.count = 0.
        self.next_check = settings.memory_min_count

    def reset(self):
        self.count = 0
        self.next_check = self.estimate_next_check(get_cur_memory()) 

    def update_mem_per_item(self, cur_memory):
        self.mem_per_item = max(
                self.mem_per_item, 
                (cur_memory - self.start_memory) / float(self.count))

        self.next_check = self.count + self.estimate_next_check(cur_memory)

    def estimate_next_check(self, cur_memory):
        item_estimate = (cur_memory - self.start_memory) / self.mem_per_item
        return min(settings.memory_max_count_before_check, item_estimate)

    def check_over_highwatermark(self):
        self.count += 1
        if self.count >= self.next_check:
            # Check memory
            cur_memory = get_cur_memory()
            self.update_mem_per_item(cur_memory)
            if cur_memory >= self.start_memory + self.max_memory_in_mbs:
                logging.debug("cur_memory:{},start_memory:{}".format(
                    cur_memory, self.start_memory))

                # ok, time to dump
                return True

        return False


def MemoryChecker(*args, **kwargs):
    if settings.memory_checker_type == "exponential":
        return ExponentialMemoryChecker(*args, **kwargs)
    elif settings.memory_checker_type == "interpolative":
        return InterpolativeMemoryChecker(*args, **kwargs)
    else:
        raise TypeError("Unknown mem_checker {}".format(settings.memory_checker_type))

import time


def to_hms_str(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


class Stopwatch:
    def __init__(self): self.reset()

    def reset(self): self.start_time = time.time()

    def elapsed_time(self): return time.time() - self.start_time

    def to_str(self): return to_hms_str(self.elapsed_time())

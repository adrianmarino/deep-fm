class MetricLogger:
    def __init__(self): self.logs = {}

    def append(self, name, value):
        if name not in self.logs:
            self.logs[name] = []
        self.logs[name].append(value)

    def is_empty(self): return len(self.logs.keys()) == 0


class Metric:
    def __init__(self, name = "metric"):
        self.sum = 0
        self.history = []
        self.name = name

    def add_append(self, param):
        self.add(param)
        self.append(self.sum)

    def add(self, param):
        self.sum += param

    def append(self, param):
        self.history.append(param)

    def dump(self, file = None):
        import json
        to_dump = {'sum':self.sum, 'history':self.history}
        with open(file, 'w') as f:
            json.dump(to_dump, f)

    def reset(self):
        self.sum = 0
        self.history = []


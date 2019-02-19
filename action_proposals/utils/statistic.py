import numpy as np
class Statistic():
    def __init__(self):
        self.datas = {}

    def update(self, name, value):
        if name not in self.datas:
            self.datas[name] = []
        self.datas[name].append(value)

    def get_mean(self, name):
        pass

    def format(self):
        format_str = ''
        for name in self.datas:
            data = self.datas[name]
            mean = np.mean(data)
            format_str += '{} mean:{:.5f} '.format(name, mean)
        return format_str

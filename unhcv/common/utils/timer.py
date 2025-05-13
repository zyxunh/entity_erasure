from time import time, sleep

class Timer:
    def __init__(self, warm_up_iter=3):
        self.warm_up_iter = warm_up_iter
        self.iter_num = 0
        self.time_accumulate = 0
        self.begin_time = None
    
    def tic(self):
        self.begin_time = time()

    def toc(self):
        if self.begin_time is None:
            self.begin_time = time()
        if self.iter_num >= self.warm_up_iter:
            self.time_accumulate += (time() - self.begin_time)
        self.iter_num += 1
    
    def mean(self):
        if self.iter_num > self.warm_up_iter:
            return self.time_accumulate / (self.iter_num - self.warm_up_iter)
        else:
            return 0

class TimerDict:
    def __init__(self, warm_up_iter=3) -> None:
        self.time_dict = {}
        self.warm_up_iter = warm_up_iter

    def tic(self, key):
        if key not in self.time_dict:
            self.time_dict[key] = Timer(self.warm_up_iter)
        self.time_dict[key].tic()

    def toc(self, key):
        if key not in self.time_dict:
            self.time_dict[key] = Timer(self.warm_up_iter)
        self.time_dict[key].toc()
    
    def get_mean_time(self):
        mean_time_dict = {}
        for key, timer in self.time_dict.items():
            mean_time_dict[key] = timer.mean()
        return mean_time_dict
    
if __name__ == '__main__':
    timer_dict = TimerDict()
    for _ in range(5):
        timer_dict.tic('0.1')
        sleep(0.1)
        timer_dict.toc('0.1')

        timer_dict.tic('0.2')
        sleep(0.2)
        timer_dict.toc('0.2')

        timer_dict.tic('0.3')
        sleep(0.3)
        timer_dict.toc('0.3')
    print(timer_dict.get_mean_time())
        
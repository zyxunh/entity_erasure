import sys
from .timer import Timer
import time
from tqdm.auto import tqdm


class MeanCache:
    def __init__(self):
        self.cache = {}
        self.cache_num = {}

    def update(self, data_dict):
        for key, value in data_dict.items():
            mem = self.cache.get(key, 0)
            num = self.cache_num.get(key, 0)
            self.cache[key] = mem + value
            self.cache_num[key] = num + 1

    def clear(self):
        self.cache.clear()
        self.cache_num.clear()

    def mean(self, clear=True):
        mean_dict = {}
        for key in self.cache.keys():
            mean_dict[key] = self.cache[key] / self.cache_num[key]
        if clear:
            self.clear()
        return mean_dict


class ProgressBarTqdm:
    """A progress bar which can print the progress."""
    def __init__(self, num_step, disable=False, smoothing=0.3, **kwargs):
        self.progress_bar = tqdm(range(0, num_step), disable=disable, smoothing=smoothing, **kwargs)
        self.progress_bar.set_description("Steps")

    def update(self, num=1):
        self.progress_bar.update(num)
    
    def log(self, items: dict):
        self.progress_bar.set_postfix(**items)

class ProgressBar:
    """A progress bar which can print the progress."""

    def __init__(self, task_num, *args, pre_str=None, bar_width=50, unit=60, start=True, display_gap=0.1, file=sys.stdout, **kwargs):
        self.display_gap = display_gap * unit
        self.unit = unit
        self.task_num = task_num
        self.bar_width = bar_width
        self.completed = 0
        self.file = file
        self.pre_str = pre_str
        if start:
            self.start()

    def start(self):
        # if self.task_num > 0:
        #     self.file.write(f'0/{self.task_num}, '
        #                     'elapsed: 0s, ETA:')
        # else:
        #     self.file.write('completed: 0, elapsed: 0s')
        self.file.flush()
        self.timer = Timer()
        self.display_time = time.time()

    def update(self, num_tasks=1):
        assert num_tasks > 0
        self.completed += num_tasks
        if time.time() - self.display_time > self.display_gap or self.completed == self.task_num:
            self.display_time = time.time()
            elapsed = self.timer.since_start()
            if elapsed > 0:
                fps = self.completed / elapsed
            else:
                fps = float('inf')
            if self.task_num > 0:
                percentage = self.completed / float(self.task_num)
                eta = int(elapsed * (1 - percentage) / percentage + 0.5)
                msg = f'{self.completed}/{self.task_num}, ' \
                    f'{fps:.1f} task/s, elapsed: {int(elapsed + 0.5)}s, ' \
                    f'ETA: {eta:5}s\n'
                if self.pre_str is not None:
                    msg = f'{self.pre_str}: ' + msg

                # bar_width = min(self.bar_width,
                #                 int(self.terminal_width - len(msg)) + 2,
                #                 int(self.terminal_width * 0.6))
                # bar_width = max(2, bar_width)
                # mark_width = int(bar_width * percentage)
                # bar_chars = '>' * mark_width + ' ' * (bar_width - mark_width)
                self.file.write(msg)
            else:
                self.file.write(
                    f'completed: {self.completed}, elapsed: {int(elapsed + 0.5)}s,'
                    f' {fps:.1f} tasks/s')
            self.file.flush()


# class ProgressBar_(MMCVProgressBar):
#     """A progress bar which can print the progress."""

#     def __init__(self, task_num, *args, bar_width=50, start=True, display_gap=30, **kwargs):
#         super().__init__(*args, task_num=task_num, bar_width=bar_width, start=start, **kwargs)
#         self.display_gap = display_gap

#     def start(self):
#         # if self.task_num > 0:
#         #     self.file.write(f'0/{self.task_num}, '
#         #                     'elapsed: 0s, ETA:')
#         # else:
#         #     self.file.write('completed: 0, elapsed: 0s')
#         self.file.flush()
#         self.timer = Timer()
#         self.display_time = time.time()

#     def update(self, num_tasks=1):
#         assert num_tasks > 0
#         self.completed += num_tasks
#         if time.time() - self.display_time > self.display_gap or self.completed == self.task_num:
#             self.display_time = time.time()
#             elapsed = self.timer.since_start()
#             if elapsed > 0:
#                 fps = self.completed / elapsed
#             else:
#                 fps = float('inf')
#             if self.task_num > 0:
#                 percentage = self.completed / float(self.task_num)
#                 eta = int(elapsed * (1 - percentage) / percentage + 0.5)
#                 msg = f'{self.completed}/{self.task_num}, ' \
#                     f'{fps:.1f} task/s, elapsed: {int(elapsed + 0.5)}s, ' \
#                     f'ETA: {eta:5}s\n'

#                 # bar_width = min(self.bar_width,
#                 #                 int(self.terminal_width - len(msg)) + 2,
#                 #                 int(self.terminal_width * 0.6))
#                 # bar_width = max(2, bar_width)
#                 # mark_width = int(bar_width * percentage)
#                 # bar_chars = '>' * mark_width + ' ' * (bar_width - mark_width)
#                 self.file.write(msg)
#             else:
#                 self.file.write(
#                     f'completed: {self.completed}, elapsed: {int(elapsed + 0.5)}s,'
#                     f' {fps:.1f} tasks/s')
#             self.file.flush()


if __name__ == '__main__':
    import time
    import torch
    if 1:
        from time import sleep
        from tqdm import tqdm
        import random

        # Default smoothing of 0.3 - irregular updates and medium-useful ETA
        for i in tqdm(range(100), smoothing=0.0):
            sleep(random.randint(0,5)/10)

        # Immediate updates - not useful for irregular updates
        for i in tqdm(range(100), smoothing=1):
            sleep(random.randint(0,5)/10)

        # Global smoothing - most useful ETA in this scenario
        for i in tqdm(range(100), smoothing=0):
            sleep(random.randint(0,5)/10)
    if 0:
        progress_bar = ProgressBarTqdm(100)
        for i in range(10):
            time.sleep(2)
            progress_bar.update()
            progress_bar.log(dict(loss=torch.tensor(0.2152112).item(), grad=i))
    if 0:
        progress_bar = ProgressBar(10, display_gap=2, pre_str='test')
        for _ in range(10):
            time.sleep(2)
            progress_bar.update()
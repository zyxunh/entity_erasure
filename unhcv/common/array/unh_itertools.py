from itertools import islice
import math

def chunk(it, size, return_list=True):
    it = iter(it)
    it_chunk = iter(lambda: tuple(islice(it, size)), ())
    if return_list:
        return tuple(it_chunk)
    return it_chunk

def split(it, num, return_list=True):
    it_num = len(it)
    assert it_num >= num
    it = iter(it)
    size_floor = int(math.floor(it_num / num))
    size = it_num / num
    it_num_extra = round((size - size_floor) * num)
    assert it_num_extra + num * size_floor == it_num, "it_num, num: {}, {}".format(it_num, num)
    out = []
    for i in range(num):
        if it_num_extra > 0:
            out.append(tuple(islice(it, size_floor + 1)))
            it_num_extra -= 1
        else:
            out.append(tuple(islice(it, size_floor)))
    return out

if __name__ == '__main__':
    import numpy as np
    k = split(np.arange(173), 72)
    print(k)
    tuple(iter(lambda: tuple(islice(iter(np.arange(10)), 3)), ()))
    out = list(chunk([1,2,3,4,5], 2))
    k = list(split(np.arange(95), num=10))
from typing import Dict


def human_format_num(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])

def dict2strs(x_dict: Dict):
    strs = []
    for key, value in x_dict.items():
        strs.append("{}: {}".format(key, value))
    return strs

if __name__ == "__main__":
    human_format_num(100000)
    pass
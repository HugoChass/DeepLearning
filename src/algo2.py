import numpy

def algo2(ratio):
    lst = [x for x in ratio if x < 0]
    if len(lst) == 1:
        return numpy.where(ratio == lst[0])
    else:
        return 0

#coding:utf8
import numpy as np
import random
from functools import reduce
from scipy.signal import convolve
from scipy.special import comb
from datetime import datetime

def get_f(i,n):
    res = []
    b = 0.1*i+0.05
    for m in range(n+1):
        p = comb(n,m)*(b**m)*((1-b)**(n-m))
        res.append(p)
    return np.nan_to_num(res)

def app_convolve(nums):
    count = [0]*10
    for num in nums:
        i = int(10*num)
        if i>=10:
            count[9] +=1
            continue
        count[i] += 1
    fs = []
    for i,n in enumerate(count):
        fs += [get_f(i,n)]
    res = reduce(lambda x,y:np.nan_to_num(convolve(x,y)),fs)
    return res

# normalise the infos
def sum_res(res0, res1):
    s = res0 + res1
    if s != 0:
        res1 = res1 / s
        if res1 > (1-10 ** (-16)):
            res1 = 1 - 10 ** (-16)
        if res1 < 10 ** (-16):
            res1 = 10 ** (-16)
    else:
        res1 = 0.5
    return res1

# summarize all the situation of permutation and combination
def sum_m(b_l,m):
    #b_l = arg[0]
    #m = arg[1]
    # b_l: a list of values
    #   m: the number of values which is choose as positive part
    # if m is 0:
    if m==0:
        s = 1
        for e in b_l:
            s *= (1 - e)
        return s
    if m == len(b_l):
        s = 1
        for e in b_l:
            s *= e
        return s
    sub_b_l = b_l[1:]
    f0 = (1 - b_l[0]) * sum_m(sub_b_l, m)
    f1 = b_l[0] * sum_m(sub_b_l, m - 1)
    return f0 + f1



if __name__ == '__main__':
    nums = []
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "Random data......")
    for i in range(1000):
        nums.append(random.random())
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "Calcul convolve......")
    res = app_convolve(nums)
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "Present result......")
    print(len(res))
    print(res)
    print(sum(res))

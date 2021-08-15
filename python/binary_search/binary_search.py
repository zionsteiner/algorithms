import matplotlib.pyplot as plt
import numpy as np
from time import perf_counter as time
import math

# O(logn)

def binary_search_rec(a: list, item):
    return binary_search_rec_util(a, 0, len(a), item)
        
def binary_search_rec_util(a: list, start: int, end: int, item):
    if start == end:
        return -start
    
    mid_idx = (end - start - 1) // 2 + start
    mid_item = a[mid_idx]
    
    if item == mid_item:
        return mid_idx
    elif item < mid_item:
        return binary_search_rec_util(a, start, mid_idx, item)
    else:
        return binary_search_rec_util(a, mid_idx + 1, end, item)
            
def binary_search_iter(a: list, item):
    start = 0
    end = len(a)
    while True:
        if start == end:
            return -start
            
        mid_idx = (end - start - 1) // 2 + start
        mid_item = a[mid_idx]
        if item == mid_item:
            return mid_idx
        elif item < mid_item:
            end = mid_idx
        else:
            start = mid_idx + 1

# Test        
searches = [binary_search_rec, binary_search_iter]
for search in searches:
    print(search.__name__, '- Test 1')
    a = [x for x in range(5)]
    for x in range(len(a)):
        assert a[search(a, x)] == a[x]
    print('PASSED')
        
    print(search.__name__, '- Test 2')
    a = [-9, 0, 10, 23, 55]
    assert search(a, 24) == -4
    print('PASSED')

# Plot
n_trials = 5
n_arrays = 100
a_lens = sorted(np.logspace(0, 7, num=n_arrays).astype(int))

rec_timings = np.zeros(len(a_lens))
iter_timings = np.zeros(len(a_lens))

for i, a_len in enumerate(a_lens):    
    rec_avg_timing = 0
    iter_avg_timing = 0
    
    for trial in range(n_trials):
        a = np.random.random(size=a_len) * 1000000
        rand_idx = int(np.random.random() * len(a))
        
        # Rec
        start = time()
        binary_search_rec(a, a[rand_idx])
        end = time()
        rec_avg_timing += (end - start) / n_trials
        
        # Iter
        start = time()
        binary_search_iter(a, a[rand_idx])
        end = time()
        iter_avg_timing += (end - start) / n_trials
        
    rec_timings[i] = rec_avg_timing
    iter_timings[i] = iter_avg_timing
    
plt.plot(np.log10(a_lens), rec_timings * 10**6, label='recursive')
plt.plot(np.log10(a_lens), iter_timings * 10**6, label='iterative')
plt.title('Binary Search Exec. Time')
plt.xlabel('log(n)')
plt.ylabel('us')
plt.legend()
plt.savefig('python/binary_search/complexity.png')
plt.show()
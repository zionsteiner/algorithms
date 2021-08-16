import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter as time

# O(nlogn)
def mergesort(a: list, inplace=True):
    if 0 <= len(a) <= 1:
        return a
    
    if not inplace:
        arr = a.copy()
    else:
        arr = a
    
    mergesort_util(arr, 0, len(arr))
    
    return arr

def mergesort_util(a: list, i: int, j: int):
    if j - i == 1:
        return
    
    mid_idx = (j - i) // 2 + i
    mergesort_util(a, i, mid_idx)
    mergesort_util(a, mid_idx, j)
    merge(a, i, j)
    
def merge(a: list, i: int, j: int):
    mid_idx = (j - i) // 2 + i
    
    ii = i
    jj = mid_idx
    k = 0
    
    merged = [None for _ in range(j - i)]
    while ii < mid_idx and jj < j:
        if a[ii] < a[jj]:
            merged[k] = a[ii]
            ii += 1
        else:
            merged[k] = a[jj]
            jj += 1
        k += 1
        
    while ii < mid_idx:
        merged[k] = a[ii]
        ii += 1
        k += 1
        
    while jj < j:
        merged[k] = a[jj]
        jj += 1
        k += 1
        
    a[i:j] = merged
        
# Test
print('Mergesort Test 1')
n_tests = 10
for _ in range(n_tests):
    a = np.random.randint(0, 1000000, size=100)
    assert all(np.array(sorted(a)) == mergesort(a))
print('PASSED')

# Plot
n_trials = 5
n_arrays = 20
a_lens = np.logspace(0, 6, num=n_arrays).astype(int)

timings = np.zeros(len(a_lens))

for i, a_len in enumerate(a_lens):    
    avg_timing = 0
    
    for trial in range(n_trials):
        a = np.random.random(size=a_len) * 1000000
        
        start = time()
        mergesort(a)
        end = time()
        avg_timing += (end - start) / n_trials
        
    timings[i] = avg_timing
    
plt.plot(a_lens, timings * 10**6)
plt.title('Mergesort Exec. Time - O(nlogn)')
plt.xlabel('n')
plt.ylabel('us')
plt.savefig('python/mergesort/complexity.png')
plt.show()
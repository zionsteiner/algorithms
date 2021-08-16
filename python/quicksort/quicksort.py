import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter as time

# O(nlogn)
def quicksort(a: list, inplace=True):
    if not inplace:
        arr = a.copy()
    else:
        arr = a
    
    
    quicksort_util(a, 0, len(a))
    
    return arr

def quicksort_util(a, i, j):
    if j - i <= 1:
        return
    
    pivot_idx = partition_lomuto(a, i, j)
    
    quicksort_util(a, i, pivot_idx)
    quicksort_util(a, pivot_idx, j)
    
def partition_lomuto(a, i, j):
    pivot_idx = find_pivot(a, i, j)
    pivot = a[pivot_idx]
    
    a[j - 1], a[pivot_idx] =  a[pivot_idx], a[j - 1]
    
    low = i
    for high in range(i, j - 1):
        if a[high] <= pivot:
            a[low], a[high] = a[high], a[low]
            low += 1
            
    a[low], a[j - 1] = a[j - 1], a[low]
    
    return low
    
def find_pivot(a, i, j):
    rand_idxs = np.random.randint(i, j, size=3)
    rand_items = [a[x] for x in rand_idxs]
    median = mo3(*rand_items)
    return rand_idxs[rand_items.index(median)]

def mo3(a, b, c):
    return sorted([a, b, c])[1]

# Test
print('Quicksort Test 1')
n_tests = 10
for _ in range(n_tests):
    a = np.random.randint(0, 1000000, size=100)
    assert all(np.array(sorted(a)) == quicksort(a))
print('PASSED')

# Plot
n_trials = 5
n_arrays = 20
a_lens = np.logspace(0, 6, num=n_arrays).astype(int)

timings = np.zeros(len(a_lens))

for i, a_len in enumerate(a_lens):    
    avg_timing = 0
    
    for trial in range(n_trials):
        a = np.random.randint(0, 10000, size=a_len) * 1000000
        
        start = time()
        quicksort(a)
        end = time()
        avg_timing += (end - start) / n_trials
        
    timings[i] = avg_timing
    
plt.plot(a_lens, timings * 10**6)
plt.title('Quicksort Exec. Time - O(nlogn)')
plt.xlabel('n')
plt.ylabel('us')
plt.savefig('python/quicksort/complexity.png')
plt.show()
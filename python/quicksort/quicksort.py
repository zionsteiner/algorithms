import numpy as np
import random
import matplotlib.pyplot as plt
from time import perf_counter as time

# O(nlogn)
def quicksort(a: list, inplace=True, pivot_method='hoare'):
    if not inplace:
        arr = a.copy()
    else:
        arr = a
    
    
    quicksort_util(a, 0, len(a), pivot_method)
    
    return arr

def quicksort_util(a, i, j, pivot_method):
    if j - i <= 1:
        return
    
    if pivot_method == 'lomuto':
        pivot_idx = partition_lomuto(a, i, j)
    else: 
        pivot_idx = partition_hoare(a, i, j)
    
    quicksort_util(a, i, pivot_idx, pivot_method)
    quicksort_util(a, pivot_idx, j, pivot_method)
    
# Slower
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

# Faster
def partition_hoare(a, i, j):
    pivot_idx = find_pivot(a, i, j)
    pivot = a[pivot_idx]
    
    a[i], a[pivot_idx] =  a[pivot_idx], a[i]
    
    low = i - 1
    high = j
    
    while True:
        low +=1
        while a[low] < pivot:
            low += 1
        
        high -= 1
        while a[high] > pivot:
            high -= 1
            
        if low >= high:
            return low
        
        a[low], a[high] = a[high], a[low]
    
def find_pivot(a, i, j):
    rand_idxs = np.random.randint(i, j, size=3)
    rand_items = [a[x] for x in rand_idxs]
    median = mo3(*rand_items)
    return rand_idxs[rand_items.index(median)]

def mo3(a, b, c):
    return sorted([a, b, c])[1]

# Test
print('Quicksort Test 1 - Lomuto')
n_tests = 10
for _ in range(n_tests):
    a = np.random.randint(0, 1000000, size=100)
    assert all(np.array(sorted(a)) == quicksort(a, pivot_method='lomuto'))
print('PASSED')

print('Quicksort Test 2 - Hoare')
n_tests = 10
for _ in range(n_tests):
    a = np.random.randint(0, 1000000, size=100)
    assert all(np.array(sorted(a)) == quicksort(a))
print('PASSED')

# Plot
n_trials = 5
n_arrays = 20
a_lens = np.logspace(0, 6, num=n_arrays).astype(int)

lomuto_timings = np.zeros(len(a_lens))
hoare_timings = np.zeros(len(a_lens))

for i, a_len in enumerate(a_lens):    
    lomuto_avg_timing = 0
    hoare_avg_timing = 0
    
    for trial in range(n_trials):
        a = np.random.randint(0, 10000, size=a_len) * 1000000
        
        start = time()
        quicksort(a, pivot_method='lomuto')
        end = time()
        lomuto_avg_timing += (end - start) / n_trials
        
        start = time()
        quicksort(a)
        end = time()
        hoare_avg_timing += (end - start) / n_trials
        
    lomuto_timings[i] = lomuto_avg_timing
    hoare_timings[i] = hoare_avg_timing
    
plt.plot(a_lens, lomuto_timings * 10**6, label='lomuto')
plt.plot(a_lens, hoare_timings * 10**6, label='hoare')
plt.legend()
plt.title('Quicksort Exec. Time - O(nlogn)')
plt.xlabel('n')
plt.ylabel('us')
plt.savefig('python/quicksort/complexity.png')
plt.show()
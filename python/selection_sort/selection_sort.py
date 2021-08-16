import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter as time

# O(n^2)
def selection_sort(a: list, inplace=True):
    if not inplace:
        arr = a.copy()
    else:
        arr = a
        
    for i in range(len(a)):
        arr_sub = arr[i:]
        min_idx = i + min(list(range(len(arr_sub))), key=lambda x: arr_sub[x])
        
        temp = arr[i]
        arr[i] = arr[min_idx]
        arr[min_idx] = temp
        
    return arr

# Test
print('Selection sort Test 1')
n_tests = 10
for _ in range(n_tests):
    arr = np.random.randint(0, 10000, size=100)
    assert all(np.array(sorted(arr)) == selection_sort(arr))
print('PASSED')

# Plot
n_trials = 5
n_arrays = 20
a_lens = np.logspace(0, 4, num=n_arrays).astype(int)

timings = np.zeros(len(a_lens))

for i, a_len in enumerate(a_lens):    
    avg_timing = 0
    
    for trial in range(n_trials):
        a = np.random.random(size=a_len) * 1000000

        start = time()
        selection_sort(a)
        end = time()
        avg_timing += (end - start) / n_trials
        
    timings[i] = avg_timing
    
plt.plot(a_lens**2, timings * 10**6)
plt.title('Selection sort Exec. Time')
plt.xlabel('n^2')
plt.ylabel('us')
plt.savefig('python/selection_sort/complexity.png')
plt.show()
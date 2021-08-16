import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter as time

# O(n^2)
def bubblesort(a: list, inplace=True):
    if not inplace:
        arr = a.copy()
    else:
        arr = a
    
    for j in range(len(arr) - 1, -1, -1):
        for i in range(0, j):
            if arr[i] > arr[i + 1]:
                temp = arr[i]
                arr[i] = arr[i + 1]
                arr[i + 1] = temp
        
    return arr
    
# Test
print('Bubblesort Test 1')
n_tests = 10
for _ in range(n_tests):
    a = np.random.randint(0, 1000000, size=100)
    assert all(np.array(sorted(a)) == bubblesort(a))
print('PASSED')

# Plot
n_trials = 5
n_arrays = 20
a_lens = np.logspace(0, 3, num=n_arrays).astype(int)

timings = np.zeros(len(a_lens))

for i, a_len in enumerate(a_lens):    
    avg_timing = 0
    
    for trial in range(n_trials):
        a = np.random.random(size=a_len) * 1000000
        
        start = time()
        bubblesort(a)
        end = time()
        avg_timing += (end - start) / n_trials
        
    timings[i] = avg_timing
    
plt.plot(a_lens, timings * 10**6)
plt.title('Bubblesort Exec. Time - O(n^2)')
plt.xlabel('n')
plt.ylabel('us')
plt.savefig('python/bubblesort/complexity.png')
plt.show()
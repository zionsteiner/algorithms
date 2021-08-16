import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter as time

# O(n^2)
def insertion_sort(a: list, inplace=True):
    if not inplace:
        arr = a.copy()
    else:
        arr = a
        
    for i in range(1, len(arr)):
        j = i
        while j > 0 and arr[j] < arr[j - 1]:
            temp = arr[j - 1]
            arr[j - 1] = arr[j]
            arr[j] = temp
            j -= 1
    
    return arr

# Test
print('Insertion sort Test 1')
n_tests = 10
for _ in range(n_tests):
    a = np.random.randint(0, 1000000, size=100)
    assert all(np.array(sorted(a)) == insertion_sort(a))
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
        insertion_sort(a)
        end = time()
        avg_timing += (end - start) / n_trials
        
    timings[i] = avg_timing
    
plt.plot(a_lens**2, timings * 10**6)
plt.title('Insertion sort Exec. Time')
plt.xlabel('n^2')
plt.ylabel('us')
plt.savefig('python/insertion_sort/complexity.png')
plt.show()
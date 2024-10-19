import time;
import numpy as np;
import universal_linear_algebra

def benchmark_add():
    array1 = np.random.rand(100000000)
    array2 = np.random.rand(100000000)
    start = time.time()
    out1 = array1 + array2
    print(f"Numpy add: {time.time() - start}s")
    array1_list = array1.tolist()
    array2_list = array2.tolist()
    start = time.time()
    out2 = universal_linear_algebra.add_1d(array1_list, array2_list)
    print(f"Rust add: {time.time() - start}s")
    np.testing.assert_allclose(out1, out2, atol=1e-6)

benchmark_add()
    
import numpy as np
import numpy.random as npr
import time
 
# --- Test 1
N = 1
n = 1000
 
A = npr.randn(n,n)
B = npr.randn(n,n)
 
t = time.time()
for i in range(N):
    C = np.dot(A, B)
td = time.time() - t
print("dotted two (%d,%d) matrices in %0.1f ms" % (n, n, 1e3*td/N))
 
# --- Test 2
N = 100
n = 4000
 
A = npr.randn(n)
B = npr.randn(n)
 
t = time.time()
for i in range(N):
    C = np.dot(A, B)
td = time.time() - t
print("dotted two (%d) vectors in %0.2f us" % (n, 1e6*td/N))
 
# --- Test 3
m,n = (2000,1000)
 
A = npr.randn(m,n)
 
t = time.time()
[U,s,V] = np.linalg.svd(A, full_matrices=False)
td = time.time() - t
print("SVD of (%d,%d) matrix in %0.3f s" % (m, n, td))
 
# --- Test 4
n = 1500
A = npr.randn(n,n)
 
t = time.time()
w, v = np.linalg.eig(A)
td = time.time() - t
print("Eigendecomp of (%d,%d) matrix in %0.3f s" % (n, n, td))

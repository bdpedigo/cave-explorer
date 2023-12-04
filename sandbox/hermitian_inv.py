# %%
import numpy as np

n = 10
A = np.random.rand(n, n)
B = A.T @ A


# %%
B_inv = np.linalg.inv(B)

np.diag(B_inv)

# %%
np.linalg.inv(np.diag(np.diag(B)))

#%%
np.linalg.pinv(B)

#%%
np.allclose(np.linalg.pinv(B), np.linalg.inv(B))
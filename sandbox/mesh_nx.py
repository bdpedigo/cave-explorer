#%%
from scipy.sparse import csr_matrix

# create a dummy scipy sparse matrix
A = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])

#%%

import networkx as nx 
print(nx.__version__)

#%%
nx.from_scipy_sparse_array(A)
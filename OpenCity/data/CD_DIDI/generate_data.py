import numpy as np
import pandas as pd

file_path = 'dataset.npy'
data_read = np.load(file_path)[..., 0:1]
# test = data_read[:, :, 0]
# print(test[test==0].shape)

# np.savez('chengdu_didi.npz', data=data_read)
datas = np.load('chengdu_didi.npz')
print(datas.files)
print(datas['data'])
print(datas['data'].shape)

adj = np.load('matrix.npy')
print(adj.shape)
print(adj[:12, :12])


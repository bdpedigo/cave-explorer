# %%
import caveclient as cc
from cloudfiles import CloudFiles

bucket_path = "gs://allen-minnie-phase3/minniephase3-emily-pcg-skeletons/minnie_all/v661/meshworks"

cf = CloudFiles(bucket_path)

client = cc.CAVEclient("minnie65_phase3_v1")

cf.exists("864691134589823178_3269905.h5")

# %%
import pickle
import h5py    
import io
from meshparty.skeleton_io import read_skeleton_h5_by_part


files_generator = cf.list()
i = 0
while i < 2:
    file_name = next(files_generator)
    root_id = int(file_name.split("_")[0])
    binary = cf.get(file_name)
    with open('temp.binary', 'wb') as f:
        f.write(binary)
    sk = read_skeleton_h5_by_part("temp.binary")
    i += 1

#%%


#%%
h5.keys()
#%%
h5['skeleton'].keys()
# %%
h5['annotations'].keys()
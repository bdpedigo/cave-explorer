# %%

################

# %%
cv = client.info.segmentation_cloudvolume()
cv.get_chunk_layer(160807562097197861)

# %%


dir(cv)

# %%
chunk_mappings = cv.get_chunk_mappings(160807562097197861)
# %%
mesh = cv.mesh.get(160807562097197861)[160807562097197861]

# %%
l2_id = 160807562097197861
supervoxels = cg.get_children(l2_id)
cv.mesh.get(supervoxels)

# %%
dir(cv.image.download())
# %%
mesh = cv.mesh.get(l2_id)[l2_id]
verts = mesh.vertices
bounds = verts.min(axis=0), verts.max(axis=0)
# %%
cv.image.download()

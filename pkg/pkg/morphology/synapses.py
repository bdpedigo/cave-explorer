def get_pre_post_synapses(root_id, client, synapse_table=None, remove_self=True):
    if synapse_table is None:
        synapse_table = client.info.get_datastack_info()["synapse_table"]
    tables = []
    for side in ["pre", "post"]:
        # find all of the original objects that at some point were part of this neuron
        original_roots = client.chunkedgraph.get_original_roots(root_id)

        # now get all of the latest versions of those objects
        # this will likely be a larger set of objects than we started with since those
        # objects could have seen further editing, etc.
        latest_roots = client.chunkedgraph.get_latest_roots(original_roots)

        # get the pre/post-synapses that correspond to those objects
        syn_df = client.materialize.query_table(
            synapse_table,
            filter_in_dict={f"{side}_pt_root_id": latest_roots},
        )
        if remove_self:
            syn_df = syn_df.query("pre_pt_root_id != post_pt_root_id")
        tables.append(syn_df)
    return (tables[0], tables[1])


def map_synapses(pre_synapses, post_synapses, supervoxel_map, l2dict_mesh):
    outs = []
    for side, synapses in zip(["pre", "post"], [pre_synapses, post_synapses]):
        # map the supervoxels attached to the synapses to the level 2 ids as defined by the
        # supervoxel map. this supervoxel map comes from looking up all of the level2 ids
        # at any point in time
        # TODO I think that implies there *could* be collisions here but I don't think it
        # will be a big problem in practice, still might break the code someday, though
        synapses[f"{side}_pt_level2_id"] = synapses[f"{side}_pt_supervoxel_id"].map(
            supervoxel_map
        )

        # now we can map each of the synapses to the mesh index, via the level 2 id
        synapses = synapses.query(
            f"{side}_pt_level2_id.isin(@l2dict_mesh.keys())"
        ).copy()
        synapses[f"{side}_pt_mesh_ind"] = synapses[f"{side}_pt_level2_id"].map(
            l2dict_mesh
        )
        outs.append(synapses)
    return tuple(outs)


def apply_synapses_to_meshwork(meshwork, pre_synapses, post_synapses, overwrite=True):
    # apply these synapse -> mesh index mappings to the meshwork
    for side, synapses in zip(["pre", "post"], [pre_synapses, post_synapses]):
        meshwork.anno.add_annotations(
            f"{side}_syn",
            synapses,
            index_column=f"{side}_pt_mesh_ind",
            point_column="ctr_pt_position",
            voxel_resolution=synapses.attrs.get("dataframe_resolution"),
            overwrite=overwrite,
        )
    return meshwork

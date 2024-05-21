def extract_meshwork_node_mappings(meshwork):
    level2_nodes = meshwork.anno.lvl2_ids.df.copy()
    level2_nodes.set_index("mesh_ind_filt", inplace=True)
    level2_nodes[
        "skeleton_index"
    ] = meshwork.anno.lvl2_ids.mesh_index.to_skel_index_padded
    level2_nodes = level2_nodes.rename(columns={"lvl2_id": "level2_id"}).drop(
        columns="mesh_ind"
    )
    skeleton_to_level2 = level2_nodes.groupby("skeleton_index")["level2_id"].unique()

    level2_nodes = level2_nodes.reset_index(drop=True).set_index("level2_id")[
        "skeleton_index"
    ]
    return skeleton_to_level2, level2_nodes

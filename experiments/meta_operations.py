# %%

if False:
    edit_lineage_graph = nx.DiGraph()

    for operation_id, row in tqdm(
        merges.iterrows(), total=len(merges), desc="Finding merge lineage relationships"
    ):
        before1_root_id, before2_root_id = row["before_root_ids"]
        after_root_id = row["after_root_ids"][0]

        before1_nodes = cg.get_leaves(before1_root_id, stop_layer=2)
        before2_nodes = cg.get_leaves(before2_root_id, stop_layer=2)
        after_nodes = cg.get_leaves(after_root_id, stop_layer=2)

        before_nodes = np.concatenate((before1_nodes, before2_nodes))
        removed_nodes = np.setdiff1d(before_nodes, after_nodes)
        added_nodes = np.setdiff1d(after_nodes, before_nodes)
        for node1 in removed_nodes:
            for node2 in added_nodes:
                edit_lineage_graph.add_edge(
                    node1, node2, operation_id=operation_id, operation_type="merge"
                )

    for operation_id, row in tqdm(
        splits.iterrows(), total=len(splits), desc="Finding split lineage relationships"
    ):
        before_root_id = row["before_root_ids"][0]

        # TODO: this is a hack to get around the fact that some splits have only one after
        # root ID. This is because sometimes a split is performed but the two objects are
        # still connected in another place, so they don't become two new roots.
        # Unsure how to handle this case in terms of tracking edits to replay laters

        # after1_root_id, after2_root_id = row["roots"]
        after_root_ids = row["roots"]

        before_nodes = cg.get_leaves(before_root_id, stop_layer=2)

        after_nodes = []
        for after_root_id in after_root_ids:
            after_nodes.append(cg.get_leaves(after_root_id, stop_layer=2))
        after_nodes = np.concatenate(after_nodes)

        removed_nodes = np.setdiff1d(before_nodes, after_nodes)
        added_nodes = np.setdiff1d(after_nodes, before_nodes)

        for node1 in removed_nodes:
            for node2 in added_nodes:
                edit_lineage_graph.add_edge(
                    node1, node2, operation_id=operation_id, operation_type="split"
                )

    meta_operation_map = {}
    for i, component in enumerate(nx.weakly_connected_components(edit_lineage_graph)):
        subgraph = edit_lineage_graph.subgraph(component)
        subgraph_operations = set()
        for source, target, data in subgraph.edges(data=True):
            subgraph_operations.add(data["operation_id"])
        meta_operation_map[i] = subgraph_operations

    print("Total operations: ", len(merges) + len(splits))
    print("Number of meta-operations: ", len(meta_operation_map))
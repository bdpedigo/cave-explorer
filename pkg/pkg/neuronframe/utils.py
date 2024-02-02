from .neuronframe import NeuronFrame

def verify_neuron_matches_final(full_neuron: NeuronFrame, current_neuron: NeuronFrame):
    final_neuron = full_neuron.set_edits(full_neuron.edits.index, inplace=False)
    final_neuron.select_nucleus_component(inplace=True)
    final_neuron.remove_unused_synapses(inplace=True)

    assert final_neuron.nodes.index.sort_values().equals(
        current_neuron.nodes.index.sort_values()
    )

    assert final_neuron.edges.index.sort_values().equals(
        current_neuron.edges.index.sort_values()
    )

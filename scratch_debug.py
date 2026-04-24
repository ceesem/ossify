import caveclient
import ossify as osy

root_id = 864691136010651052
client = caveclient.CAVEclient("minnie65_phase3_v1")
client.version = 1718

cell = osy.load_cell_from_client(
    root_id,
    client=client,
    synapses=True,
    restore_graph=True,
    restore_properties=True,
)


# works
osy.algorithms.label_axon_from_spectral_split(
    cell, max_times=1,
)


# fails
osy.algorithms.label_axon_from_spectral_split(
    cell, max_times=1,
)


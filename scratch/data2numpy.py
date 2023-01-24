import numpy as np
import geopandas as gpd


#print("Reading Nodes")
#nodes = gpd.read_file("../data/sfo_roads.shp")
#print(nodes.head())
#graph_data = np.hstack([
#    nodes[["STARTID"]].to_numpy(),
#    nodes[["ENDID"]].to_numpy(),
#    nodes[["LENGTH"]].to_numpy()
#    ])
#print(graph_data)
#print("Writing Graph Info")
#np.save("graph.npy", graph_data)
#### Second: geometry of nodes (sufficient, geometry of lines inbetween can be added later)
vertices = gpd.read_file("../data/sfo_nodes.shp")
print(vertices.head())
I=vertices[["ID"]].to_numpy().astype(int)
locs=np.vstack([x[0].coords[:][0] for x in vertices[["geometry"]].to_numpy()])
np.save("i.npy",I)
np.save("locs.npy", locs)
#

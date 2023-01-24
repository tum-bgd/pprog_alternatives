import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
def load_graph():
    G = np.load("../data/graph.npy")
    I = np.load("../data/i.npy")
    C = np.load("../data/locs.npy")
    return G,I,C


def out_degree(G):
    out_degree = np.max(np.unique(G[:,0], return_counts=True)[1])
    return out_degree


def out_edges(G, v):
    " Returns the out-edges of vertex v in G, that is all edges that start at v"
    return G[G[:,0] == v, :]
    


# This class implements a priority queue, but not in an efficient way
import sys
class MinPriorityQueue:
    def __init__(self):
        self.queue = []
 
    def __str__(self):
        return ' '.join([str(i) for i in self.queue])
 
    # for checking if the queue is empty
    def empty(self):
        return len(self.queue) == 0
 
    # for inserting an element in the queue
    def insert(self, data):
        self.queue.append(data)
 
    # for popping an element based on Priority
    def priority(self,x):
        return -x # for now, priority is -x, later it will be -x[0]
    def pop(self):
        try:
            max_val = 0
            for i in range(len(self.queue)):
                if self.priority(self.queue[i]) > self.priority(self.queue[max_val]):
                    max_val = i
            item = self.queue[max_val]
            del self.queue[max_val]
            return item
        except IndexError:
            return None
    
    

def dijkstra(G,I, s):
    # Step 1: Initialize Working Data
#  for each vertex u in V (This loop is not run in dijkstra_shortest_paths_no_init)
#    d[u] := infinity 
#    p[u] := u 
#    color[u] := WHITE
#  end for
    d=np.ones(I.shape[0]) * np.inf
    p=np.array(range(I.shape[0]))
    color = np.zeros(I.shape[0]) # 0 is white, 1 is gray, 2 is black


    # create a min-queue and add (0,start) (first is distance, second, is where you start)

    # while not q.empty():
    # take min-element
    # find neighbors
    # add neighbors to the queue
    # mark black
    
    
    # Step N: Retunr distance and predecessors
    return d,p






#DIJKSTRA(G, s, w)
#  color[s] := GRAY 
#  d[s] := 0 
#  INSERT(Q, s)
#  while (Q != Ø)
#    u := EXTRACT-MIN(Q)
#    S := S U { u }
#    for each vertex v in Adj[u]
#      if (w(u,v) + d[u] < d[v])
#        d[v] := w(u,v) + d[u]
#        p[v] := u 
#        if (color[v] = WHITE) 
#          color[v] := GRAY
#          INSERT(Q, v) 
#        else if (color[v] = GRAY)
#          DECREASE-KEY(Q, v)
#      else
#        ...
#    end for
#    color[u] := BLACK
#  end while
#  return (d, p)
        
    


if __name__=="__main__":
    G,I,C = load_graph()
    print("We found %d edges" % (G.shape[0]))
    
    start = G[0,0] # this is surely a vertex id, it is the start of the first edge
    print("Out-edges of vertex %s" % (str(start)))
    print(out_edges(G,start).astype(int))
    dijkstra(G,I,start)
    
#    plt.scatter(C[:,0],C[:,1])
#    plt.savefig("/var/www/html/graph.png")

    # Let us now check our priority queue is working as we want
#    q = MinPriorityQueue()
#    q.insert(10)
#    q.insert(20)
#    q.insert(15)
#    print(q)
#    print(q.pop())
#    print(q)
#    while q.pop() is not None:
#        print("Found something")
#

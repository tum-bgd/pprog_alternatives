import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

def load_lecturemockdata():
    G = np.array(
        [
            [0,1,55],# M<->A
            [1,0,55], 
            [1,2,45],# A<->Gz
            [2,1,45],
            [0,3,70],# M<->In
            [3,0,70],
            [2,3,90], #Gz<->In
            [3,2,90],
            [3,4,80],#In<->N
            [4,3,80],
            [2,4,1234], #Gz<->N
            [4,2,1234],
            [4,5,90], # N<->Wue
            [5,4,90],
            [2,5,150], # Gz Wue
            [5,2,150]  
        ],dtype=float)
    L=["M","A","Gz", "In","N","Wue"]
    I=np.array(range(len(L))).reshape(-1,1)
    C = np.array([
        [100,0],
        [75,25],
        [50,50],
        [90,70],
        [80,90],
        [30,110]
    ])
    return G,I,C,L

        

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
        return -x[0] # for now, priority is -x, later it will be -x[0]
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
    
    

def dijkstra(G,I,L, s, e = None):
    # Step 1: Initialize Working Data
#  for each vertex u in V (This loop is not run in dijkstra_shortest_paths_no_init)
#    d[u] := infinity 
#    p[u] := u 
#    color[u] := WHITE
#  end for
    d=np.ones(I.shape[0]) * np.inf
    p=np.array(range(I.shape[0]))
    color = np.zeros(I.shape[0]) # 0 is white, 1 is gray, 2 is black
    c={"white":0,"gray":1,"black":2}
    
    # create a min-queue and add (0,start) (first is distance, second, is where you start)
    q = MinPriorityQueue()
    q.insert((0,s))
    d[s] = 0
    color[s] = c["gray"]

    while not q.empty():
        u = q.pop()
        if u[0]  != d[u[1]]:
          #  print("skip superseeded")
            continue
        if u[1] == e:
         #   print("Shortest Path encoded")
            dist = d[e]
            shortest_path=[e]
            while p[e] != e:
                shortest_path.insert(0,p[e])
                e = p[e]
            return dist,shortest_path
        
        #print(u)
        for _,vf,ed in out_edges(G,u[1]):
            v = int(vf)
            dst = d[u[1]]+ed
            #print("edge: %s @%f " %(str([u[1],v,ed]),dst))
            if dst < d[v]:
                d[v] = dst # a new path of this distance found
                p[v] = u[1] # this where we come from
                if color[v] == c["white"]:
                    color[v] = c["gray"]
                    q.insert((d[v],v))
                    #print("Queue after discover: %s " %(str(q)))
                elif color[v] == c["gray"]: # we already had a path, but this is shorter
                    q.insert((d[v],v)) # this is redundant with the previous block

        color[u[1]] = c["black"]
    return inf,[]








if __name__=="__main__":
    G,I,C,L = load_lecturemockdata()#load_graph()
    print("G.shape=%s" % (str(G.shape)))
    print("I.shape=%s" % (str(I.shape)))
    print("C.shape=%s" % (str(C.shape)))
    print("We found %d edges" % (G.shape[0]))
    
    start = int(G[0,0]) # this is surely a vertex id, it is the start of the first edge
    print("Out-edges of vertex %s" % (str(L[start])))
    print(out_edges(G,start).astype(int))

    # Now, we find alternatives through the Penalty algorithm:
    # increase the edge weights of the winning path each time and wait for a new path
    last = None
    for step in range(100):
        d,p = dijkstra(G,I,L,start,5)
        Lp = "->".join([L[x] for x in p])
        print("Distance: %f for %s" %(d,Lp))
        if last is not None:
            if not np.all(last == p):
                print("First alternative uncovered in step %d" % (step))
                break # end the search
            
        # enlarge all edges of the path
        for y,x in zip(p[1:], p[:-1]):
            mask = (G[:,0] == x) & (G[:,1] == y)
            G[mask,2] = int( G[mask,2]*1.02)
            
        last = p
    

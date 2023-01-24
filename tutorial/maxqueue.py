import numpy as np


class MaxQueue:
    def __init__(self):
        self.L = list() # our storage

    def __str__(self):
        return ",".join([str(x) for x in self.L])

    def insert(self, x):
        self.L = self.L + [x]
        print("Insert updated to %s" % (str(self.L)))
    def empty(self):
        return len(self.L) == 0
    def pop(self):
        # shall return the largest entry of L
        if self.empty():
            return None
        i = np.where(self.L == np.max(self.L))[0][0]
        x = self.L[i]
        del self.L[i]
        return x


if __name__=="__main__":
    print("hello")
    q = MaxQueue()
    print(q.empty())
    q.insert(2)
    q.insert(4)
    q.insert(1)
    print(q.empty())
    print(q.pop())
    print(q.pop())
    print(q.pop())
    print(q.pop())
    print(q)
    

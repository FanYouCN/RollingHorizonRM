import numpy as np
import math
from joblib import Parallel, delayed
from RHdata import RHinstance
from FHmodelsim import FhModelSim
from FHmodel import FhModel
from birkhoff import birkhoff_von_neumann_decomposition
import time

def prob_alloc(aRH, astar, aalphab, aalphad):
    astar[1,0,0] = 0
    for m in aRH.mcM:
        astar[1,m,0] = aRH.C - aalphab[m] - sum([astar[1,m,n] for n in aRH.mcN])
    for n in aRH.mcN:
        astar[1,0,n] = aalphad[n] - sum([astar[1,m,n] for m in aRH.mcM])
    Cstar, Dstar = {}, {}
    for m in range(aRH.M+1):
        Cstar[m] = sum(astar[1,m,n] for n in range(aRH.N+1))
    for n in range(aRH.N+1):
        Dstar[n] = sum(astar[1,m,n] for m in range(aRH.M+1))
    ahat = {}
    for m in range(aRH.M+1):
        for n in range(aRH.N+1):
            ahat[m,n] = astar[1,m,n] - math.floor(astar[1,m,n])
    Chat, Dhat = {}, {}
    for m in range(aRH.M+1):
        Chat[m] = round(Cstar[m] - sum([math.floor(astar[1,m,n]) for n in range(aRH.N+1)]))
    for n in range(aRH.N+1):
        Dhat[n] = round(Dstar[n] - sum([math.floor(astar[1,m,n]) for m in range(aRH.M+1)]))
    Mtilde, Ntilde = [], []
    for m in range(aRH.M+1):
        for k in range(1, int(Chat[m])+1):
            Mtilde.append((m,k))
    for n in range(aRH.N+1):
        for l in range(1, int(Dhat[n])+1):
            Ntilde.append((n,l))
    atilde = np.zeros((len(Mtilde), len(Ntilde)))
    for i in range(len(Mtilde)):
        for j in range(len(Ntilde)):
            m, n = Mtilde[i][0], Ntilde[j][0]
            atilde[i,j] = 1.0 / (Chat[m] * Dhat[n]) * ahat[m,n]
    res = birkhoff_von_neumann_decomposition(atilde)
    Astar = {}
    for m in aRH.mcM:
        for n in aRH.mcN:
            Astar[m,n] = math.floor(astar[1,m,n])
    print aalphad
    for k in Astar.keys():
        if Astar[k] >0:
            print k, Astar[k]
    print "------------------------------------------------------"
    if len(res) == 0:
        return Astar
    coefs, permus = zip(*res)
    permu = permus[np.random.choice(range(len(coefs)), p=coefs)]
    for i in range(len(Mtilde)):
        for j in range(len(Ntilde)):
            if permu[i,j] == 1:
                m, n = Mtilde[i][0], Ntilde[j][0]
                if m*n !=0:
                    Astar[m,n] += Chat[m] * Dhat[n]
    for k in Astar.keys():
        if Astar[k] >0:
            print k, Astar[k]
    return 0

if __name__ == '__main__':
    np.random.seed(134)
    aRH = RHinstance(0.95, 3.234)
    init_b = aRH.getbookingsample()
    init_d = aRH.getdemandsample()
    aFH = FhModel(aRH, 50, init_b, init_d)
    aFH.mod.optimize()
    A = {}
    for t in aFH.mcT:
        for m in aFH.RH.mcM:
            for n in aFH.RH.mcN:
                A[t,m,n] = aFH.a[t,m,n].getAttr("Pi")
    prob_alloc(aRH, A, init_b, init_d)

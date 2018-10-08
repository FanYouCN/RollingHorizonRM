from gurobipy import *
import numpy as np
from RHdata import RHinstance
from FHmodel import FhModel

class MFhModel:
    def __init__(self, aRH, aT, aAlphab, aAlphad):
        self.RH = aRH
        self.Ed = aRH.lam
        self.T = aT
        self.mcT = range(1, aT + 1)
        self.mcTmin = range(1, aT)

        self.mafd = Model()
        self.mafd.setParam('OutputFlag', 0)

        self.a = self.mafd.addVars(self.mcT, self.RH.mcM, self.RH.mcN, lb=0)
        self.b = self.mafd.addVars(self.mcT, self.RH.mcM, lb=0)

        obj = 0
        for t in self.mcT:
            for m in self.RH.mcM:
                for n in self.RH.mcN:
                    obj += self.RH.v[m,n] * self.a[t,m,n]

        self.mafd.setObjective(obj, GRB.MAXIMIZE)

        for t in self.mcTmin:
            for m in self.RH.mcMmin:
                if t == 1:
                    self.mafd.addConstr(self.b[t,m] == aAlphab[m])
                else:
                    self.mafd.addConstr(self.b[t,m] == self.RH.beta * (self.b[t-1,m+1] + self.a.sum(t-1,m+1,'*')))

        for m in self.RH.mcMmin:
            self.mafd.addConstr((1-self.RH.beta) * self.b[self.T,m] == self.RH.beta * (self.b[self.T-1,m+1] + self.a.sum(self.T-1,m+1,'*')))

        for m in self.RH.mcMmin:
            self.mafd.addConstr(self.RH.beta * self.b[self.T,m] == self.RH.beta * (self.b[self.T,m+1] + self.a.sum(self.T,m+1,'*')))

        for t in self.mcT:
            for m in self.RH.mcM:
                self.mafd.addConstr(self.a.sum(t,m,'*') + self.b[t,m] <= pow(self.RH.beta, t-1) * self.RH.C  / (1-self.RH.beta if t==self.T else 1) )

        for t in self.mcT:
            for n in self.RH.mcN:
                if t == 1:
                    self.mafd.addConstr(self.a.sum(t,'*',n) <= aAlphad[n])
                else:
                    self.mafd.addConstr(self.a.sum(t,'*',n) <= pow(self.RH.beta, t-1) * self.Ed[n] / (1-self.RH.beta if t==self.T else 1))


    def solve(self):
        self.mafd.optimize()
        if self.mafd.status == 2:
            return self.mafd.objVal
        else:
            return "Infeasible"


if __name__ == '__main__':
    aRH = RHinstance(0.95, 2.0)
    # aT = 6
    # np.random.seed(1234)
    init_b = aRH.getbookingsample()
    init_d = aRH.getdemandsample()
    # aFh = FhModel(aRH, aT, init_b, init_d)
    # aMFh = MFhModel(aRH, aT, init_b, init_d)
    # aFh.solve()
    # aMFh.solve()

    def get_minT(aRH, aalphab, aalphad):
        T = 2
        while True:
            aMFh = MFhModel(aRH, T, init_b, init_d)
            if aMFh.solve() != "Infeasible":
                return T
            T += 1
    print(get_minT(aRH, init_b, init_d))

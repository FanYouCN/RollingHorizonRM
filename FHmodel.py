from gurobipy import *
import numpy as np
import math
from joblib import Parallel, delayed
from RHdata import RHinstance
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
    if len(res) == 0:
        return Astar
    coefs, permus = zip(*res)
    permu = permus[np.random.choice(range(len(coefs)), p=coefs)]
    for i in range(len(Mtilde)):
        for j in range(len(Ntilde)):
            if permu[i,j] == 1:
                m, n = Mtilde[i][0], Ntilde[j][0]
                if m*n !=0:
                    Astar[m,n] += 1
    return Astar


class FhModel:
    def __init__(self, aRH, aT, aAlphab, aAlphad):
        self.RH = aRH
        self.Ed = aRH.lam
        self.T = aT
        self.mcT = range(1, aT + 1)

        self.mod = Model()
        self.mod.setParam('OutputFlag', 0)
        self.V = self.mod.addVars(self.mcT, aRH.mcM, lb=-GRB.INFINITY, name="V")
        self.U = self.mod.addVars(self.mcT, aRH.mcM, name="U")
        self.W = self.mod.addVars(self.mcT, aRH.mcN, name="W")

        expr = quicksum(aAlphab[m] * self.V[1, m] for m in aRH.mcMmin)
        expr += quicksum(aRH.C * (pow(aRH.beta, t - 1) / (1 - (t == self.T) * aRH.beta)) * self.U[t, m]
                         for t in self.mcT for m in aRH.mcM)
        expr += quicksum(
            (aAlphad[n] + (self.T == 1) * (aRH.beta / (1 - aRH.beta)) * self.Ed[n]) * self.W[1, n] for n in aRH.mcN)
        expr += quicksum(((pow(aRH.beta, t - 1) / (1 - (t == self.T) * aRH.beta)) * self.Ed[n]) * self.W[t, n]
                         for t in range(2, self.T + 1) for n in aRH.mcN)
        self.mod.setObjective(expr, GRB.MINIMIZE)

        self.b = {}
        for t in self.mcT:
            for m in aRH.mcMmin:
                if m > 1:
                    self.b[t, m] = self.mod.addConstr(
                        self.V[t, m] + self.U[t, m] == aRH.beta * self.V[min(t + 1, self.T), m - 1], name="b")
                else:
                    self.b[t, m] = self.mod.addConstr(self.V[t, m] + self.U[t, m] == 0, name="b")
        self.a = {}
        for t in self.mcT:
            for m in aRH.mcM:
                for n in aRH.mcN:
                    if m > 1:
                        self.a[t, m, n] = self.mod.addConstr(
                            self.U[t, m] + self.W[t, n] >= aRH.v[m, n] + aRH.beta * self.V[min(t + 1, self.T), m - 1],
                            name="a")
                    else:
                        self.a[t, m, n] = self.mod.addConstr(self.U[t, m] + self.W[t, n] >= aRH.v[m, n], name="a")

    def solve(self):
        # self.mod.write("test.lp")
        # self.mod.Params.OutputFlag = 0
        self.mod.optimize()
        lb = 0
        Vstar = {}
        for m in self.RH.mcMmin:
            if self.T == 1:
                Vstar[m] = self.V[1, m].getAttr("x")
            else:
                Vstar[m] = self.V[2, m].getAttr("x")
        # print self.mod.objVal
        return self.mod.objVal, Vstar

    def solve_lower(self):
        # if self.T == 1:
        #     return 0
        self.mod.optimize()
        astar = {}
        for t in self.mcT:
            for m in self.RH.mcM:
                for n in self.RH.mcN:
                    astar[t,m,n] = self.a[t,m,n].getAttr("Pi")
        astar_plus = {}
        for t in range(1,self.T):
            for m in self.RH.mcM:
                for n in self.RH.mcN:
                    astar_plus[t,m,n] = astar[t,m,n]
        for m in self.RH.mcM:
            for n in self.RH.mcN:
                astar_plus[self.T,m,n] = 0
        for m in self.RH.mcMmin:
            for n in self.RH.mcN:
                astar_plus[self.T+1,m,n] = self.RH.beta * astar[self.T,m+1,n]
        for n in self.RH.mcN:
            astar_plus[self.T+1,self.RH.M,n] = 0
        z = 0
        for t in range(1,self.T+2):
            for m in self.RH.mcM:
                for n in self.RH.mcN:
                    z += self.RH.v[m,n] * astar_plus[t,m,n]
        return z

    def solve_lower_2(self):
        # if self.T == 1:
        #     return 0
        self.mod.optimize()
        astar = {}
        for t in self.mcT:
            for m in self.RH.mcM:
                for n in self.RH.mcN:
                    astar[t,m,n] = self.a[t,m,n].getAttr("Pi")
        astar_plus = {}
        for t in range(1,self.T):
            for m in self.RH.mcM:
                for n in self.RH.mcN:
                    astar_plus[t,m,n] = astar[t,m,n]
        for m in self.RH.mcM:
            for n in self.RH.mcN:
                astar_plus[self.T,m,n] = 0
        for m in self.RH.mcMmin:
            for n in self.RH.mcN:
                astar_plus[self.T+1,m,n] = 0
        for n in self.RH.mcN:
            astar_plus[self.T+1,self.RH.M,n] = 0
        z = 0
        for t in range(1,self.T+2):
            for m in self.RH.mcM:
                for n in self.RH.mcN:
                    z += self.RH.v[m,n] * astar_plus[t,m,n]
        return z



    def getdual(self):
        self.mod.optimize()
        primal_astar = {}
        vstar = 0
        for m in self.RH.mcM:
            for n in self.RH.mcN:
                astar = self.a[1,m,n].getAttr("Pi")
                if astar > 1e-10:
                    if abs(astar - round(astar)) > 1e-10:
                        astar = math.floor(astar)
                    primal_astar[m,n] = round(astar)
                    vstar += self.RH.v[m,n] * astar
                else:
                    primal_astar[m,n] = 0
        return vstar, primal_astar

    def getdual_prob(self, currb, currd):
        self.mod.optimize()
        A = {}
        vstar = 0
        for t in self.mcT:
            for m in self.RH.mcM:
                for n in self.RH.mcN:
                    astar = self.a[t,m,n].getAttr("Pi")
                    if astar > 1e-10:
                        if abs(astar - round(astar)) > 1e-10:
                            A[t,m,n] = astar
                        A[t,m,n] = round(astar)
                    else:
                        A[t,m,n] = 0
        Astar = prob_alloc(self.RH, A, currb, currd)
        for m in self.RH.mcM:
            for n in self.RH.mcN:
                vstar += self.RH.v[m,n] * Astar[m,n]
        return vstar, Astar



    def setobjective(self, aAlphab, aAlphad):
        for m in self.RH.mcMmin:
            self.V[1, m].obj = aAlphab[m]
        for m in self.RH.mcM:
            for t in self.mcT:
                self.U[t, m].obj = self.RH.C * (pow(self.RH.beta, t - 1) / (1 - (t == self.T) * self.RH.beta))
        for n in self.RH.mcN:
            self.W[1, n].obj = aAlphad[n] + (self.T == 1) * (self.RH.beta / (1 - self.RH.beta)) * self.Ed[n]
        for t in range(2, self.T + 1):
            for n in self.RH.mcN:
                self.W[t, n].obj = (pow(self.RH.beta, t - 1) / (1 - (t == self.T) * self.RH.beta)) * self.Ed[n]

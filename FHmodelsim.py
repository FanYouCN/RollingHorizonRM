from gurobipy import *
from FHmodel import FhModel
from RHdata import RHinstance

class FhModelSim:
    def __init__(self, aRH, aT, aNsamplesteps):
        self.RH = aRH
        self.T = aT
        self.Nss = aNsamplesteps

        self.samod = Model()
        self.samod.setParam('OutputFlag', 0)
        self.saa = self.samod.addVars(self.RH.mcM, self.RH.mcN, name="a")

        self.samod.setObjective(quicksum(self.RH.v[m, n] * self.saa[m, n] for m in self.RH.mcM for n in self.RH.mcN),
                                GRB.MAXIMIZE)
        self.saV = {}
        for m in self.RH.mcM:
            self.saV[m] = self.samod.addConstr(quicksum(self.saa[m, n] for n in self.RH.mcN) <= self.RH.C)
        self.saW = {}
        for n in self.RH.mcN:
            self.saW[n] = self.samod.addConstr(quicksum(self.saa[m, n] for m in self.RH.mcM) <= self.RH.lam[n])

    def update_samodel(self, aVstar, aalphab, aalphad):
        for m in self.RH.mcM:
            for n in self.RH.mcN:
                if m == 1:
                    self.saa[m, n].obj = self.RH.v[m, n]
                else:
                    self.saa[m, n].obj = self.RH.v[m, n] + self.RH.beta * aVstar[m - 1]

        # print "Objective Function:"
        # for n in self.RH.mcN:
        #     for m in self.RH.mcM:
        #         if m == 1:
        #             print self.RH.v[m, n], ',',
        #         else:
        #             print self.RH.v[m, n] + self.RH.beta * aVstar[m - 1], ',',
        #     print ''

        for m in self.RH.mcM:
            self.saV[m].RHS = self.RH.C - aalphab[m]

        for n in self.RH.mcN:
            self.saW[n].RHS = aalphad[n]

    def solve_samodel(self):
        # self.samod.write("samod.lp")
        self.samod.optimize()

        vstar = 0
        astar = self.samod.getAttr('x', self.saa)
        for n in self.RH.mcN:
            for m in self.RH.mcM:
                vstar += self.RH.v[m, n] * astar[m, n]

        # print "Allocation:"
        # for n in self.RH.mcN:
        #     for m in self.RH.mcM:
        #         print astar[m,n], ",",
        #     print ''

        return vstar, astar

    def get_samplepath_primal(self, aAlphab, aAlphad):
        currb = aAlphab
        currd = aAlphad

        fhmod = FhModel(self.RH, self.T, currb, currd)
        sp_ub, _ = fhmod.solve()
        sp_lb = 0
        if self.T == 1:
            return [sp_lb, sp_ub]
        discount_factor = 1
        for ss in range(self.Nss):
            vstar, primal_astar = fhmod.getdual()
            sp_lb += discount_factor * vstar
            discount_factor *= self.RH.beta
            currd = self.RH.getdemandsample()
            currb = self.RH.gettransition(currb, primal_astar)
            fhmod.setobjective(currb, currd)
        return [sp_lb, sp_ub]


    def get_samplepath(self, aResolve, aAlphab, aAlphad):
        currb = aAlphab
        currd = aAlphad

        fhmod = FhModel(self.RH, self.T, currb, currd)
        sp_ub, currVstar = fhmod.solve()
        sp_lb = 0

        discount_factor = 1
        for ss in range(self.Nss):
            self.update_samodel(currVstar, currb, currd)
            vstar, astar = self.solve_samodel()
            sp_lb += discount_factor * vstar
            discount_factor *= self.RH.beta

            # update state
            currd = self.RH.getdemandsample()
            currb = self.RH.gettransition(currb, astar)

            # update V
            if aResolve > 0:
                fhmod.setobjective(currb, currd)
                currVstar = fhmod.solve()[1]

        return [sp_lb, sp_ub]

    def get_samplepath_greedy(self, aAlphab, aAlphad):
        currb = aAlphab
        currd = aAlphad
        fhmod = FhModel(self.RH, self.T, currb, currd)
        sp_lb = 0
        Vstar = {}
        for m in self.RH.mcMmin:
            Vstar[m] = 0
        discount_factor = 1
        for ss in range(self.Nss):
            self.update_samodel(Vstar, currb, currd)
            vstar, astar = self.solve_samodel()
            sp_lb += discount_factor * vstar
            discount_factor *= self.RH.beta

            # update state
            currd = self.RH.getdemandsample()
            currb = self.RH.gettransition(currb, astar)
        return sp_lb

    def get_samplepath_primal_prob(self, aAlphab, aAlphad):
        currb = aAlphab
        currd = aAlphad

        fhmod = FhModel(self.RH, self.T, currb, currd)
        sp_lb = 0
        if self.T == 1:
            return sp_lb
        discount_factor = 1
        for ss in range(self.Nss):
            vstar, primal_astar = fhmod.getdual_prob(currb, currd)
            sp_lb += discount_factor * vstar
            discount_factor *= self.RH.beta
            currd = self.RH.getdemandsample()
            currb = self.RH.gettransition(currb, primal_astar)
            fhmod.setobjective(currb, currd)
        return sp_lb



if __name__ == "__main__":
    import RHdata, math
    import numpy as np
    aRH = RHdata.RHinstance(0.95, 4)
    init_b = aRH.getbookingsample()
    init_d = aRH.getdemandsample()
    n_samplesteps = int(math.ceil(-6.0 / np.log10(aRH.beta)))
    afhsp = FhModelSim(aRH, 50, n_samplesteps)
    # print afhsp.get_samplepath_primal(init_b, init_d)
    # print afhsp.get_samplepath_primal_prob(init_b, init_d)
    # print "-------------------"

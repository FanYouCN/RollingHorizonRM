import numpy as np


class RHinstance:
    def __init__(self, abeta, aload):
        self.M = 30
        self.N = 3
        self.beta = abeta
        self.C = 60
        self.load = aload
        self.lam = {}
        self.lam[1] = self.load * 8
        self.lam[2] = self.load * 36
        self.lam[3] = self.load * 16
        # self.lam[4] = self.load * 16
        # self.lam[5] = self.load * 24
        # self.lam[6] = self.load * 25
        # self.lam[7] = self.load * 9
        # self.lam[8] = self.load * 16
        # self.lam[9] = self.load * 8
        # self.lam[10] = self.load * 20

        self.mcM = range(1, self.M + 1)
        self.mcMmin = range(1, self.M)
        self.mcN = range(1, self.N + 1)

        target = {}
        target[1] = 7
        target[2] = 14
        target[3] = 21

        f = {}
        f[1] = 200
        f[2] = 100
        f[3] = 50
        # f[4] = 180
        # f[5] = 240
        # f[6] = 100
        # f[7] = 50
        # f[8] = 360
        # f[9] = 400
        # f[10] = 120


        self.v = {}
        # np.random.seed(2017)
        # for m in self.mcM:
        #     for n in self.mcN:
        #         self.v[m, n] = (pow(self.beta, m-1) * f[n]) * (0.9 + np.random.rand() / 5)


        for m in self.mcM:
            for n in self.mcN:
                p = max(m - target[n], 0)
                self.v[m,n] = pow(self.beta, p) * f[n]


    def display(self):
        output_string = "Instance: " + "\n"
        output_string += "Horizon: " + str(self.M) + "\n"
        output_string += "Classes: " + str(self.N) + "\n"
        output_string += "Discount Factor: " + str(self.beta) + "\n"
        output_string += "Capacity: " + str(self.C) + "\n"
        output_string += "Demand Distribution Means: " + str(self.lam) + "\n"
        output_string += "Load: " + str(self.load) + "\n"
        output_string += "Revenue Coeff: " + "\n"
        for n in self.mcN:
            for m in self.mcM:
                output_string += str(self.v[m, n]) + ","
            print("\n")
        print(output_string)

    def gettransition(self, ab, aa):
        nb = {}
        for m in self.mcMmin:
            nb[m] = ab[m + 1] + sum(aa[m + 1, n] for n in self.mcN)
        nb[self.M] = 0
        return nb

    def getdemandsample(self):
        tmp = {}
        for n in self.mcN:
            tmp[n] = np.random.poisson(self.lam[n])
        return tmp

    def getbookingsample(self):
        tmp = {}
        for m in self.mcMmin:
            tmp[m] = np.random.randint(self.C + 1)
        tmp[self.M] = 0
        return tmp

    def getsamples(self, n_samples):
        np.random.seed(1234)
        alpha_b, alpha_d = {}, {}
        for sp in range(n_samples):
            alpha_b[sp] = self.getbookingsample()
            alpha_d[sp] = self.getdemandsample()
        return alpha_b, alpha_d

    def getEd(self):
        tmp = {}
        for n in self.mcN:
            tmp[n] = self.lam[n]
        return tmp

    def getEb(self):
        tmp = {}
        for m in self.mcMmin:
            tmp[m] = (self.C + 0.0) / 2.0
        return tmp

if __name__ == "__main__":
    import FHmodelsim
    testRH = RHinstance(0.95, 0.6)
    testRH.display()

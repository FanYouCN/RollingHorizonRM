import numpy as np
import math
from joblib import Parallel, delayed
from RHdata import RHinstance
from FHmodelsim import FhModelSim
from FHmodel import FhModel
from MFHmodel import MFhModel
import time


def get_greedy_bounds(aRH, asamplesteps, aalphab, aalphad):
    afhsp = FhModelSim(aRH, 1, asamplesteps)
    return afhsp.get_samplepath_greedy(aalphab, aalphad)

def get_primal_prob_bounds(aRH, aT, asamplesteps, aalphab, aalphad):
    afhsp = FhModelSim(aRH, aT, asamplesteps)
    return afhsp.get_samplepath_primal_prob(aalphab, aalphad)

def get_samplepathbounds(aRH, aT, asamplesteps, aresolve, primal_yn, aalphab, aalphad):
    afhsp = FhModelSim(aRH, aT, asamplesteps)
    if primal_yn == 0:
        return afhsp.get_samplepath(aresolve, aalphab, aalphad)
    else:
        return afhsp.get_samplepath_primal(aalphab, aalphad)

def get_minT(aRH, aalphab, aalphad):
    T = 2
    while True:
        aMFh = MFhModel(aRH, T, aalphab, aalphad)
        if aMFh.solve() != "Infeasible":
            return T
        T += 1


n_samples = 100
beta = 0.95
n_samplesteps = int(math.ceil(-6.0 / np.log10(beta)))
load_samples = [0.6, 1.0, 1.5, 2.0, 3.0, 6.0]
# load_samples = [1.8, 2.4, 3.6]
T_max = 50
mcT_max = range(1, T_max + 1)


def run_greedy():
    grdfile = open('results/GRD.txt', 'w')
    for i in range(len(load_samples)):
        rhinstance = RHinstance(beta, load_samples[i])
        alpha_b, alpha_d = rhinstance.getsamples(n_samples)
        lb = []
        lb = Parallel(n_jobs=-1, verbose=0)(delayed(get_greedy_bounds)(rhinstance, n_samplesteps, alpha_b[sp], alpha_d[sp]) for sp in range(n_samples))
        lb_mean = np.mean(lb, axis=0)
        print 'GRD.txt', ',', beta, ',', load_samples[i],  ',', lb_mean
        print >> grdfile, beta, ',', load_samples[i], ',', lb_mean
    grdfile.close()

def run_primal_prob():
    outfile = open('results/FIN_PP.txt', 'w')
    for i in range(len(load_samples)):
        rhinstance = RHinstance(beta, load_samples[i])
        alpha_b, alpha_d = rhinstance.getsamples(n_samples)
        for t in mcT_max:
            lb = []
            lb = Parallel(n_jobs=-1, verbose=0)(delayed(get_primal_prob_bounds)(rhinstance, t, n_samplesteps, alpha_b[sp], alpha_d[sp]) for sp in range(n_samples))
            lb_mean = np.mean(lb, axis=0)
            print 'FIN_PP.txt', ',', beta, ',', load_samples[i], ',', t, ',', lb_mean
            print >> outfile, beta, ',', load_samples[i], ',', t, ',', lb_mean
    outfile.close()

def run_simu(primal_yn, resolve_yn):
    if primal_yn != 0:
        filepath = 'results/FIN_P.txt'
    elif resolve_yn != 0:
        filepath = 'results/FIN_RE.txt'
    else:
        filepath = 'results/FIN.txt'

    outfile = open(filepath, 'w')
    for i in range(len(load_samples)):
        rhinstance = RHinstance(beta, load_samples[i])
        alpha_b, alpha_d = rhinstance.getsamples(n_samples)
        for t in mcT_max:
            lb = []
            lb = Parallel(n_jobs=-1, verbose=0)(delayed(get_samplepathbounds)(rhinstance, t, n_samplesteps, resolve_yn, primal_yn, alpha_b[sp], alpha_d[sp]) for sp in range(n_samples))
            lb_mean = np.mean(lb, axis=0)
            print filepath, ',', beta, ',', load_samples[i], ',', t, ',', lb_mean[0]
            print >> outfile, beta, ',', load_samples[i], ',', t, ',', lb_mean[0]
    outfile.close()

def run_minT():
    outfile = open('results/minT.txt', 'w')
    for i in range(len(load_samples)):
        rhinstance = RHinstance(beta, load_samples[i])
        alpha_b, alpha_d = rhinstance.getsamples(n_samples)
        minT = []
        minT = Parallel(n_jobs=-1, verbose=0)(delayed(get_minT)(rhinstance, alpha_b[sp], alpha_d[sp]) for sp in range(n_samples))
        for t in minT:
            print >> outfile, beta, ',', load_samples[i], ',', t
    outfile.close()

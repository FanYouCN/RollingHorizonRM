import simuT, bounds, time

if __name__ == "__main__":
    t = time.time()
    bounds.run_bounds()
    simuT.run_greedy()
    simuT.run_simu(0, 0)
    simuT.run_simu(0, 1)
    simuT.run_simu(1, 1)
    simuT.run_primal_prob()
    # simuT.run_minT()
    print(time.time() - t)

import matplotlib.pyplot as plt
import random
import numpy as np
from scipy.stats import norm as nr
import math




class Finance():

    def __init__(self ,S ,  K ,  T , t ,  r ,  q ,  sigma, nsteps , ntrajs):

        self.ntrajs = ntrajs == 5000
        self.nsteps = nsteps == 200
        self.S = S
        self.K = K
        self.T = T
        self.t = t
        self.r = r
        self.q = q
        self.sigma = sigma

    def BlackandSHoles(self , T, t , S , sigma , r, q , K):

        d1 = (np.log(S / K) + (r - q + 0.5 * pow(sigma, 2)) * (T-t)) / (sigma * np.sqrt(T-t))
        d2 = d1 - sigma * math.sqrt(T-t)
        Nd1 = nr.cdf(d1, 0, 1)
        Nd2 = nr.cdf(d2, 0, 1)
        Vcall = S * np.exp(-q * T) * Nd1 - K * np.exp(-r * (T-t)) * Nd2
        Payoff = max((S - K, 0))
        return Vcall


    def Monte_carlo(self, T, t , S , nsteps , ntrajs ,sigma , r, q , K):

        dt = (T - t) / nsteps
        t = []
        sumpayoff = 0
        random.seed(20000)
        t.extend([0])
        nc = 0.05
        alfa = 1 - nc
        z = nr.ppf(0.5 * alfa + 0.5)
        sumpayoff2 = 0
        for isteps in range(0, nsteps + 1):
            time = isteps * dt
            t.append(time)
        for itraj in range(0, ntrajs + 1):
            St = [S]
            for isteps in range(0, nsteps + 1):
                et = random.gauss(0, 1)
                St.append(St[isteps] * (1 + (r - q) * dt + et * sigma * math.sqrt(dt)))
            sumpayoff += np.max([St[isteps] - K, 0])
            sumpayoff2 += pow(np.max([St[isteps] - K, 0]), 2)

        pv2 = sumpayoff2 * np.exp(-r * T)
        pv = sumpayoff * np.exp(-r * T)
        desv = np.sqrt((ntrajs * pv2 - pow(pv, 2)) / (pow(ntrajs, 2)))
        err = math.sqrt(z * desv / ntrajs)
        call = pv / ntrajs
        return call, err
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List
import numpy as np
from scipy.stats import beta, gamma
from scipy.special import betaln

__author1__ = "Morten Arngren"
__author2__ = "Jordan Allen"

class Bayesian_AB_Test:
    """ class for pre-processing data and calculating Bayesian test statistics
    """

    def __init__(self):
        """ Inits class
        """
        # init
        self.rv = {}
    
    def p_ab(self, ints: List, imps: List, best: str = 'max', thr: float = 1, n_samples: int = 10000, dist: str = 'beta'):
        """ Calc. probability that all variant are better than the rest
            one by one.

        Args:
            ints (List): list of interactions for variants e.g., [n_ints_a, n_ints_b, n_ints_c].
            imps (List): list of impressions for variants e.g., [N_IMPR_A, N_IMPR_B, N_IMPR_C].
            thr (float, optional): threshold. Defaults to 1.
            n_samples (int, optional): number of samples. Defaults to 10000.
            dist (string, optional): distribution of variable. Either beta or gamma.

        Returns:
            P_ab_thr (List): list of probabilities that each variant is better than the rest
        """
        # Create list of scipy.stats objects
        if dist == 'beta':
            rvs = [beta(a_i, x_i-a_i) for a_i, x_i in zip(ints, imps)] # only add +1 to each variable if the used dataset is small, to prevent risk of 0 counts
        elif dist == 'gamma':
            rvs = [gamma(a_i*x_i, 1/x_i) for a_i, x_i in zip(ints, imps)] # only add +1 to first variable if the used dataset is small, to prevent risk of 0 counts

        # Generate samples from all variants
        samples = np.array( [rv.rvs(size=n_samples) for rv in rvs] )

        # Calc. probability that a variant is better than the rest
        P_ab_thr = []
        for id_ref in range(len(rvs)):
            # Identify the rest of the variants
            id_rest = [i for i in range(len(rvs)) if i != id_ref]
            # Calc. probability ratio that ref is better than the rest
            P_ratio = samples[id_ref] / np.max(samples[id_rest], axis=0)
            # Calc. prob. mass above threshold and save for each variant
            if best == 'max':
                P_ab_thr += [ (P_ratio>thr).sum() / n_samples ]
            if best == 'min':
                P_ab_thr += [ (P_ratio<thr).sum() / n_samples ]
        return P_ab_thr

    def p_ba_beta(self, alpha_a: float, beta_a: float, alpha_b: float, beta_b: float) -> float:
        """ probability of B having higher performance than A assuming the Beta distribution.
            ref: https://www.evanmiller.org/bayesian-ab-testing.html#implementation

        Args:
            alpha_a (float): alpha parameter for rv A
            beta_a (float): beta parameter for rv A
            alpha_b (float): alpha parameter for rv B
            beta_b (float): beta parameter for rv B
        """
        P = np.sum( [ np.exp( betaln(alpha_a+i, beta_b+beta_a) \
                    - np.log(beta_b+i) \
                    - betaln(1+i, beta_b) \
                    - betaln(alpha_a, beta_a) ) for i in range(alpha_b) ] )

        return P


    def loss(self, rv_a, rv_b, f_max: float=1, N: int=100) -> List:
        """ calc. the loss - ie. amount of performance lost if wrong variant is chosen
        
        Args:
            rv_a (scipy.stats): random variable function for variant A
            rv_b (scipy.stats): random variable function for variant B
            f_max (float): max. value for the pdf
            N (int): number of pdf divisions for integration
        """
        # util function to calc. loss
        def __loss(i, j):
            return max(j/N - i/N, 0)

        # util function
        def __joint_posterior_array(rv_a, rv_b, f_max, N=100):
            joint = np.array( [rv_a.pdf(ii) * rv_b.pdf(np.linspace(0,f_max,N)) for i,ii in enumerate(np.linspace(0,f_max,N))] ) + 1e-16
            return joint/joint.sum()

        loss_a, loss_b = 0, 0
        # calc. f_max based in std of gamma distributions
        # if isinstance(rv_a, gamma):
        #     f_max = 5 * rv_a.std()
        # if isinstance(rv_b, gamma):
        #     f_max = max(f_max, 5 * rv_b.std())
        if f_max == 0:
            f_max = max(5*rv_a.std(), 5*rv_b.std())

        # calc. loss
        joint = __joint_posterior_array(rv_a, rv_b, f_max=f_max, N=N)
        for i in range(N):
            loss_a += sum( [joint[i,j]*__loss(i,j) for j in range(N)] )
            loss_b += sum( [joint[i,j]*__loss(j,i) for j in range(N)] )

        return loss_a, loss_b
        
    def loss_beta(self, alpha_a: float, beta_a: float, alpha_b: float, beta_b: float, n_samples: int=1000) -> List:
        """ https://www.chrisstucchio.com/blog/2014/bayesian_ab_decision_rule.html

        Args:
            alpha_a (float): alpha parameter for rv A
            beta_a (float): beta parameter for rv A
            alpha_b (float): alpha parameter for rv B
            beta_b (float): beta parameter for rv B
        """
        # analytically
        from scipy.special import beta as B
        a, b, c, d = int(alpha_a), int(beta_a), int(alpha_b), int(beta_b)

        # normal domain - not numerically stable
        # loss_a = B(a+1, b) / B(a, b) * (1-self.p_ba_anal(a+1, b, c, d)) \
        #        - B(c+1, d) / B(c, d) * (1-self.p_ba_anal(a, b, c+1, d))
        # loss_b = B(c+1, d) / B(c, d) * (1-self.p_ba_anal(c+1, d, a, b) \
        #        - B(a+1, b) / B(a, b) * (1-self.p_ba_anal(c, d, a+1, b)

        # log domain calc. - TODO: p_ab has two outputs...[1] correct?
        loss_a = np.exp( betaln(c+1, d) - betaln(c, d) + np.log(1-self.p_ab([beta(c+1,d), beta(a,b)], n_samples=n_samples)[1]) ) \
               - np.exp( betaln(a+1, b) - betaln(a, b) + np.log(1-self.p_ab([beta(c,d), beta(a+1,b)], n_samples=n_samples)[1]) )
        loss_b = np.exp( betaln(a+1, b) - betaln(a, b) + np.log(1-self.p_ab([beta(a+1,b), beta(c,d)], n_samples=n_samples)[1]) ) \
               - np.exp( betaln(c+1, d) - betaln(c, d) + np.log(1-self.p_ab([beta(a,b), beta(c+1,d)], n_samples=n_samples)[1]) )

        return loss_a, loss_b

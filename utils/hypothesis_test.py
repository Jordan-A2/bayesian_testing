#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

__author1__ = "Morten Arngren"
__author2__ = "Jordan Allen"

class Hypothesis_AB_Test:
    """ class for pre-processing data and calculating hypothesis test statistics
    """

    def __init__(self):
        """
        """
        pass

    def calc_sample_size(self, p_list: List[float], Z_a: float=1.96, Z_b: float=0.842) -> int:
        """
        Calculate the sample size required for A/B/N testing with all pairwise comparisons.

        Args:
            p_list (List[float]): List of performance values for N variants, e.g., [0.35, 0.12, 0.30].
            Z_a (float): Z-value for confidence level (e.g., 1.96 for 95% confidence).
            Z_b (float): Z-value for statistical power (e.g., 0.842 for 80% power).

        Returns:
            int: The required sample size for the test.
        """
        n_variants = len(p_list)

        # Adjust the Z-value for multiple comparisons (Bonferroni correction)
        Z_a_corrected = Z_a / ((n_variants * (n_variants - 1)) / 2)  # Number of pairwise comparisons

        max_sample_size = 0  # Track the largest sample size needed

        # Loop through all pairs of variants
        for i in range(n_variants):
            for j in range(i + 1, n_variants):
                p1 = p_list[i]
                p2 = p_list[j]

                # Compute sample size for comparing p1 and p2
                n_samples = int( (Z_a_corrected + Z_b)**2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p2 - p1)**2 ) + 1
                max_sample_size = max(max_sample_size, n_samples)

        return max_sample_size


    def chi2_test(self, interactions: List[int], impressions: List[int], variants: List[str], print_opt: bool) -> None:
        """
        Perform pairwise chi-square tests for N variants and print the results for each pair.   

        Args:
            clicks (List[int]): List of number of clicks for N variants, e.g., [clicks_a, clicks_b, clicks_c, ...].
            impressions (List[int]): List of number of impressions for N variants, e.g., [impr_a, impr_b, impr_c, ...].

        Prints:
            Chi-square statistic and p-value for each pair of variants.
        """
        n_variants = len(interactions)
        vec = {}

        # Loop through all pairs of variants
        for i in range(n_variants):
            for j in range(i + 1, n_variants):
                # Get the clicks and impressions for variant i and variant j
                clicks_i, impr_i = interactions[i], impressions[i]
                clicks_j, impr_j = interactions[j], impressions[j]

                # Create the contingency table for the two variants
                contingency_table = np.array([[clicks_i, impr_i - clicks_i],
                                              [clicks_j, impr_j - clicks_j]]) # only add +1 to each variable if the used dataset is small, to prevent risk of 0 counts

                # Perform the chi-square test
                chi2, p, _, _ = chi2_contingency(contingency_table)

                if print_opt == True:
                    # Print the results for this pair
                    print(f"Chi-square test for Variant {variants[i]} vs Variant {variants[j]}:")
                    print(f"Chi-square statistic: {chi2}")
                    print(f"P-value: {p}\n")
                elif print_opt == False:
                    vec[f'{variants[i]}_{variants[j]}'] = [chi2, p]

        return vec



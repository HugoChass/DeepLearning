import numpy as np
import math
import random

from src.algo2 import algo2


def algo1(gradients):
    # Step 1
    M = 512
    K = 5  # For retinamnist
    G_H = gradients[-2].cpu().numpy()

    deterministic = []
    for m in range(M):
        sample = G_H[:, m]
        y = sum([1 if x < 0 else 0 for x in sample])
        if y != 1:
            y = 0
        deterministic.append(y)

    print("Starting first algorithm...")
    # Step 3
    r = []
    G_H1 = G_H[0, :]
    for c in range(K):
        r.append(G_H[c, :] / G_H1)

    # Make the disjoint index groupes for all the values occuring multiple times in r2
    r2 = r[1]
    r2_unique = np.unique(r2, return_counts=True)
    unique_vals = []
    I = []
    for i in range(r2_unique[0].shape[0]):
        if r2_unique[1][i] > 1:
            print(f"Value {r2_unique[0][i]} has {r2_unique[1][i]} occurences.")
            unique_vals.append(r2_unique[0][i])
    for val in unique_vals:
        I.append([])
        for i in range(r2.shape[0]):
            if not math.isnan(val):
                if r2[i] == val:
                    I[-1].append(i)
            else:
                if math.isnan(r2[i]):
                    I[-1].append(i)

    # exclude the group of disjoint indices which is made by the NaN values
    I = I[:-1]

    # M is now inferred batch size, no correlation with m used before (might be better to change the M before)
    M = len(I)
    # Make a new matrix with size K x M to store the g^m_c / g^m_1 values to parse to the next algorithm
    g_mc = np.zeros((K, M))

    for c in range(K):
        for m in range(M):
            # Get disjoint index group m
            Im = I[m]
            # select an arbitrary index from Im
            # Index does not really matter since any r[c][j] has the same value, picking the first might have been
            # easier
            j = random.choice(Im)
            # Get the value of r_cj
            rc = r[c][j]
            # make g^m_c / g^m_1 = r_cj, CAG: this needs to be ratio right?
            g_mc[c, m] = rc

    # The loop which runs from line 11 till the end
    for m in range(M):
        # Call algo 2 with the gradient things
        Y_m = algo2(g_mc[:, m])
        # Add steps 13-15 here
        delta_m = 1/g_mc[Y_m, m]  # CAG: ratio
        g_m1 = (2/3)*delta_m
        # Calc g_mc with ratio
        g_mc[:, m] = g_mc[:, m]*g_m1  # CAG: ratio

    return g_mc

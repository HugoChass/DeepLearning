import numpy as np
import math


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


    pass
    pass

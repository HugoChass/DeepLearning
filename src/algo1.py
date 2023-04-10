import numpy as np


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

    r_unique = np.unique(r[1], return_counts=True)
    r = np.array(r)

    pass
    pass

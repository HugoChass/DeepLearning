import numpy as np
import random


def algo3(gradients, I_mH, g_mc):
    H = 3  # amount of total layers
    M = len(I_mH)
    Dm = np.array([np.zeros((512, 512)), np.zeros((5, 5))], dtype=object)
    D = np.tile(Dm, (M, 1))
    # print(D[0][0].shape)
    G = [gradients[0].cpu().numpy(), gradients[2].cpu().numpy()]  # remove biases from gradients list

    I_cur = I_mH
    for i in reversed(range(H - 1)):
        for m in range(M):
            j = random.choice(I_cur[m])
            ind = 0
            # print(G[i][:,j])
            for val in G[i][:, j]:  # check for the zero's in the col, if so, not activated in diag
                if val != 0:
                    D[m][i][ind, ind] = 1
                ind += 1

        # If EXAN in layer, sum of diag == 1
        for m in range(M):
            # print(D[m][i].shape)
            if 0.99 <= np.sum(D[m][i]) <= 1.01:
                print("EXAN found!")
                print(np.sum(D[m][i]))
                raise SystemExit("Stop right there!")
            # else:
            #     # print("Nope...")
            #     # print(np.sum(D[m][i]))

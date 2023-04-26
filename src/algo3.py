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

        # Create temporary list for Exans sets
        I_sets_temp = []
        # Loop over the D maxtrices create for each sample at a certain layer
        for d in D[:, i]:
            I_set_temp = []  # Create temporary list for index set for a sample
            for n in range(d.shape[0]):
                if d[n, n] == 1:  # If the diagonal value is 1 then it is an exan and should be added to the set
                    I_set_temp.append(n)
            I_sets_temp.append(I_set_temp)  # Append smaple index set

        I_cur = I_sets_temp  # Update I_cur




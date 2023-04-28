import numpy as np
import random


def algo3(gradients, I_mH, g_mc):

    H = 1  # amount of total layers
    K = 5
    M = len(I_mH)
    Dm = np.array([np.zeros((512, 512)), np.zeros((5, 5))], dtype=object)
    D = np.tile(Dm, (M, 1))
    G = [gradients[0].cpu().numpy(), gradients[2].cpu().numpy()]  # remove biases from gradients list
    B = [gradients[1].cpu().numpy(), gradients[3].cpu().numpy()]
    I_cur = I_mH
    for i in reversed(range(1, H - 1)):
        for m in range(M):
            j = random.choice(I_cur[m])
            ind = 0
            for val in G[i][:, j]:  # check for the zero's in the col, if so, not activated in diag
                #print(val)
                if val != 0:
                    D[m][i][ind, ind] = 1
                ind += 1

        D_sum = D[:, i].sum()
        #D_sum = np.identity(5)
        exan_lst = []
        for k in range(D_sum.shape[0]):
            if D_sum[k, k] == 1:
                exan_lst.append(k)

        I_sets = []
        for d in D[:, i]:
            temp_lst = []
            for k in exan_lst:
                if d[k, k] == 1:
                    temp_lst.append(k)
            I_sets.append(temp_lst)
        I_cur = I_sets


    # left hand side: sum over m(1...M): sum over c(1...K): g_mc*W_Hc
    lhs = g_mc.T @ G[1] # now we dont have to sum over c
    lhs = np.sum(lhs, axis=0) # sum over m

    # right hand side: M*dLdb1
    rhs = M*B[0]

    # Create D diagonal matrix
    D_mH = np.zeros((512,512))

    # Check for EXAN positions if rhs equals the lhs
    for m in range(M):
        for ind in I_cur[m]:
            if lhs[ind] == rhs[ind]:
                D_mH[ind,ind] += 1 
                print("Found a match on index:  ", ind)

    return D_mH



        


        




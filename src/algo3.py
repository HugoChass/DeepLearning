import numpy as np
import random


def algo3(gradients, I_mH, g_mc):

    H = 3  # amount of total layers
    M = len(I_mH)
    Dm = np.array([np.zeros((512, 512)), np.zeros((5, 5))], dtype=object)
    D = np.tile(Dm, (M, 1))
    G = [gradients[0].cpu().numpy(), gradients[2].cpu().numpy()]  # remove biases from gradients list
    I_cur = I_mH
    for i in reversed(range(H - 1)):

        try:
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
                    print('exan found!')
                    exan_lst.append(k)

            I_sets = []
            for d in D[:, i]:
                temp_lst = []
                for k in exan_lst:
                    if d[k, k] == 1:
                        temp_lst.append(k)
                I_sets.append(temp_lst)
            I_cur = I_sets

        except(IndexError):
            print(' --- Not enough exans found, Failed --- ')
            break
        
    else:
        print(' --- Fucking worked mate --- ')


        


        




import numpy as np
import torchvision
import random


# net = torchvision.models.vgg13(weights='IMAGENET1K_V1')
# print(net)

def vector_loss(average_gradient, K, M):
    # ratio vector
    r = [average_gradient[c] / average_gradient[0] for c in range(K)]

    # index groups?
    I = []

    ratio = np.empty((M, K))
    for c in range(0, K):
        for m in range(0, M):
            j = random.choice(I[m])
            ratio[m, c] = r[c, j]
    
    loss_vector = []
    for m in range(0, M):
        m_loss_vector = ratio[m]
        Ym = label_reconstruction(m_loss_vector)
        delta_m = 1/m_loss_vector[Ym]
        g1 = 2 * delta_m / 3
        loss_vector.append(m_loss_vector * g1)

    return loss_vector, I

def label_reconstruction(loss_v):
    lst = [x for x in loss_v if x < 0]
    if len(lst) == 1:
        return loss_v.index(lst[0])
    else:
        return 0



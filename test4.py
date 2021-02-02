from statsmodels.stats import weightstats
import numpy as np

def one_sim(p, k, dprint=False):
    vecs = np.random.standard_normal(size=(4, p))
    vecs = np.array([np.array(v) / np.linalg.norm(v) for v in vecs])

    R = np.random.standard_normal(size=(p, k))
    V = np.matmul(vecs, R)
    V = np.sign(V)

    t1 = t2 = tcnt = ta = 0
    for (vi, vj, v1, v2) in np.transpose(V):
        if vi == vj:
            t1 += 1
        if v1 == v2:
            t2 += 1
        if vi == vj and v1 == v2:
            tcnt += 1
        ta += (vi == v1) * (vj == v1)
        # if vi == vj == v1:
        #     ta += 1

    angle_ij = np.arccos(np.dot(vecs[0], vecs[1]))
    angle_12 = np.arccos(np.dot(vecs[2], vecs[3]))

    angle_i1 = np.arccos(np.dot(vecs[0], vecs[2]))
    angle_j1 = np.arccos(np.dot(vecs[1], vecs[2]))

    # angle_i2 = np.arccos(np.dot(vecs[0], vecs[3]))
    # angle_j2 = np.arccos(np.dot(vecs[1], vecs[3]))

    act_ij = 1 - angle_ij / np.pi
    act_12 = 1 - angle_12 / np.pi

    est_ij = t1 / k
    est_12 = t2 / k

    err_ij = act_ij - est_ij
    err_12 = act_12 - est_12

    perr_ij = err_ij / act_ij * 100
    perr_12 = err_12 / act_12 * 100

    if dprint:
        # angle estimates match
        print("basic angles")
        print(act_ij, est_ij, perr_ij)
        print(act_12, est_12, perr_12)
        print()

        # show that E[i==j * i==1] is the three way angle similarity
        print("three way")
        act_three = 1 - (angle_i1 + angle_ij + angle_j1) / (2 * np.pi)
        est_three = ta / k
        perr_three = (act_three - est_three) / act_three * 100
        print(act_three, est_three, perr_three)
        print()

        # covariances
        print("covariance")
        # E[i==j * 1==2]
        print(act_ij * act_12)
        print(tcnt / k)
        print(est_ij * est_12)
        print(((act_ij * act_12) - (tcnt / k)) / (act_ij * act_12) * 100)
    else:
        # find the estimated covariance
        return tcnt / k - est_ij * est_12

p = 10000
k = 1000
reps = 100

# covs = []
# for i in range(reps):
#     covs += [one_sim(p, k)]
# print(sum(covs) / reps)
# print(weightstats.ztest(covs))

one_sim(p, k, True)
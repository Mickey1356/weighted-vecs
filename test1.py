import numpy as np
# import sympy as sp
from tqdm import tqdm
from math import pi

import warnings
warnings.filterwarnings('ignore')

# generate the expressions for c1 and c2 (for CV)
# tij, ti1, tj1 = sp.symbols('theta_ij theta_i1 theta_j1')
# e_a = 1 - tij / sp.pi
# e_b = 1 - ti1 / sp.pi
# e_c = 1 - tj1 / sp.pi

# var_b = e_b * (1 - e_b)
# var_c = e_c * (1 - e_c)

# three_way = 1 - (tij + ti1 + tj1) / (2 * sp.pi)
# cov_ab = three_way - e_a * e_b
# cov_ac = three_way - e_a * e_c
# cov_bc = three_way - e_b * e_c

# cov_mat = sp.Matrix([[var_b, cov_bc], [cov_bc, var_c]])
# inv_mat = cov_mat.inv()
# ys = sp.Matrix([[cov_ab], [cov_ac]])
# cs = -inv_mat * ys

# # set up the equation 1 - tij/pi = A + c1(B - uB) + c2(C - uC)
# sym_A, sym_B, sym_C = sp.symbols('A B C')
# sym_c1 = cs[0]
# sym_c2 = cs[1]
# eqn = sym_A + sym_c1 * (sym_B - e_b) + sym_c2 * (sym_C - e_c) - (1 - tij / sp.pi)
# diff_eqn = sp.diff(eqn, tij)

p = 2
k = 20
num_NR_iter = 20
reps = 1

# generate 3 random vectors (the first two are the test vectors, and the third is the extra vector)
set_angle = 0 / 180 * pi
x = np.cos(set_angle)
y = np.sin(set_angle)
vecs = np.array([[1, 0], [x, y]])
vecs = np.append(vecs, np.random.standard_normal(size=(1, 2)), axis=0)
# vecs = np.random.standard_normal(size=(3, p))
vecs = np.array([np.array(v) / np.linalg.norm(v) for v in vecs])
print(vecs)

# calculate the actual angles
angle_ij = np.arccos(np.dot(vecs[0], vecs[1]))
angle_i1 = np.arccos(np.dot(vecs[0], vecs[2]))
angle_j1 = np.arccos(np.dot(vecs[1], vecs[2]))

theta_i1 = angle_i1
theta_j1 = angle_j1

# pi = np.pi

def one_sim():
    # generate the projections
    R = np.random.standard_normal(size=(p, k))
    V = np.matmul(vecs, R)
    V = np.sign(V)

    # Pre-processing
    # count the required stuff
    t1 = t2 = t3 = 0
    n1 = n2 = n3 = n4 = 0
    for (vi, vj, v1) in np.transpose(V):
        if vi == vj: # counts A
            t1 += 1
        if vi == v1: # counts B
            t2 += 1
        if vj == v1: # counts C
            t3 += 1
        if vi != vj == v1:
            n1 += 1
        if vi == vj != v1:
            n2 += 1
        if vi == vj == v1:
            n3 += 1
        if vi == v1 != vj:
            n4 += 1

    # Control variates 1
    # calculate A, B, C
    A = t1 / k
    B = t2 / k
    C = t3 / k

    print(t1, t2, t3, n1, n2, n3, n4)

    # we need an initial estimate for tij
    # A = 1 - tij/pi => tij = pi (1 - A)
    est_ij = np.pi * (1 - A)

    # calculate the CV coefficients
    # act_cs = cs.evalf(subs={tij: est_ij, ti1: angle_i1, tj1: angle_j1})
    theta_ij = est_ij
    c1 = (-2*theta_i1**2*theta_j1 + pi*theta_i1**2 - 2*pi*theta_i1*theta_ij + 2*pi*theta_i1*theta_j1 - 2*theta_ij**2*theta_j1 + pi*theta_ij**2 + 2*pi*theta_ij*theta_j1 + 2*theta_j1**3 - 3*pi*theta_j1**2)/(pi*theta_i1**2 + 4*theta_i1*theta_ij*theta_j1 - 2*pi*theta_i1*theta_ij - 2*pi*theta_i1*theta_j1 + pi*theta_ij**2 - 2*pi*theta_ij*theta_j1 + pi*theta_j1**2)

    c2 = (2*theta_i1**3 - 3*pi*theta_i1**2 - 2*theta_i1*theta_ij**2 + 2*pi*theta_i1*theta_ij - 2*theta_i1*theta_j1**2 + 2*pi*theta_i1*theta_j1 + pi*theta_ij**2 - 2*pi*theta_ij*theta_j1 + pi*theta_j1**2)/(pi*theta_i1**2 + 4*theta_i1*theta_ij*theta_j1 - 2*pi*theta_i1*theta_ij - 2*pi*theta_i1*theta_j1 + pi*theta_ij**2 - 2*pi*theta_ij*theta_j1 + pi*theta_j1**2)


    # calculate the true mean values for CV
    u_b = 1 - angle_i1 / np.pi
    u_c = 1 - angle_j1 / np.pi

    # find the new value of our estimate
    new_est_ij = A + c1 * (B - u_b) + c2 * (C - u_c)

    # find the estimated angle
    cv_est_angle = np.pi * (1 - new_est_ij)

    # Control variates 2
    # substitute the known values in (ti1 and tj1)
    # subs = {ti1: angle_i1, tj1: angle_j1, sym_A: A, sym_B: B, sym_C: C}
    # partial_eqn = eqn.subs([(k, v) for k, v in subs.items()])
    # partial_diff_eqn = diff_eqn.subs([(k, v) for k, v in subs.items()])

    # find value for tij (using root finder)
    # checks for angle (0, 2pi)
    # def cv_f(x):
    #     return partial_eqn.evalf(subs={tij: x})
    
    # def cv_fp(x):
    #     return partial_diff_eqn.evalf(subs={tij: x})

    def cv1(theta_ij):
        return theta_ij*(pi*theta_i1**2 + 4*theta_i1*theta_ij*theta_j1 - 2*pi*theta_i1*theta_ij - 2*pi*theta_i1*theta_j1 + pi*theta_ij**2 - 2*pi*theta_ij*theta_j1 + pi*theta_j1**2) + pi*(A - 1)*(pi*theta_i1**2 + 4*theta_i1*theta_ij*theta_j1 - 2*pi*theta_i1*theta_ij - 2*pi*theta_i1*theta_j1 + pi*theta_ij**2 - 2*pi*theta_ij*theta_j1 + pi*theta_j1**2) + (theta_i1 + pi*(B - 1))*(-2*theta_i1**2*theta_j1 + pi*theta_i1**2 - 2*pi*theta_i1*theta_ij + 2*pi*theta_i1*theta_j1 - 2*theta_ij**2*theta_j1 + pi*theta_ij**2 + 2*pi*theta_ij*theta_j1 + 2*theta_j1**3 - 3*pi*theta_j1**2) + (theta_j1 + pi*(C - 1))*(2*theta_i1**3 - 3*pi*theta_i1**2 - 2*theta_i1*theta_ij**2 + 2*pi*theta_i1*theta_ij - 2*theta_i1*theta_j1**2 + 2*pi*theta_i1*theta_j1 + pi*theta_ij**2 - 2*pi*theta_ij*theta_j1 + pi*theta_j1**2)

    def cv1_p(theta_ij):
        return pi*(4*A*theta_i1*theta_j1 - 2*pi*A*theta_i1 + 2*pi*A*theta_ij - 2*pi*A*theta_j1 - 2*pi*B*theta_i1 - 4*B*theta_ij*theta_j1 + 2*pi*B*theta_ij + 2*pi*B*theta_j1 - 4*C*theta_i1*theta_ij + 2*pi*C*theta_i1 + 2*pi*C*theta_ij - 2*pi*C*theta_j1 - theta_i1**2 + 2*theta_i1*theta_ij - 2*theta_i1*theta_j1 + 2*pi*theta_i1 + 3*theta_ij**2 + 2*theta_ij*theta_j1 - 6*pi*theta_ij - theta_j1**2 + 2*pi*theta_j1)

    def cv_f2(theta_ij):
        x = (theta_ij*(pi*theta_i1**2 + 4*theta_i1*theta_ij*theta_j1 - 2*pi*theta_i1*theta_ij - 2*pi*theta_i1*theta_j1 + pi*theta_ij**2 - 2*pi*theta_ij*theta_j1 + pi*theta_j1**2) + pi*(A - 1)*(pi*theta_i1**2 + 4*theta_i1*theta_ij*theta_j1 - 2*pi*theta_i1*theta_ij - 2*pi*theta_i1*theta_j1 + pi*theta_ij**2 - 2*pi*theta_ij*theta_j1 + pi*theta_j1**2) + (theta_i1 + pi*(B - 1))*(-2*theta_i1**2*theta_j1 + pi*theta_i1**2 - 2*pi*theta_i1*theta_ij + 2*pi*theta_i1*theta_j1 - 2*theta_ij**2*theta_j1 + pi*theta_ij**2 + 2*pi*theta_ij*theta_j1 + 2*theta_j1**3 - 3*pi*theta_j1**2) + (theta_j1 + pi*(C - 1))*(2*theta_i1**3 - 3*pi*theta_i1**2 - 2*theta_i1*theta_ij**2 + 2*pi*theta_i1*theta_ij - 2*theta_i1*theta_j1**2 + 2*pi*theta_i1*theta_j1 + pi*theta_ij**2 - 2*pi*theta_ij*theta_j1 + pi*theta_j1**2))/(pi*(pi*theta_i1**2 + 4*theta_i1*theta_ij*theta_j1 - 2*pi*theta_i1*theta_ij - 2*pi*theta_i1*theta_j1 + pi*theta_ij**2 - 2*pi*theta_ij*theta_j1 + pi*theta_j1**2))
        return x
    
    def cv_fp2(theta_ij):
        x = (-2*(2*theta_i1*theta_j1 - pi*theta_i1 + pi*theta_ij - pi*theta_j1)*(theta_ij*(pi*theta_i1**2 + 4*theta_i1*theta_ij*theta_j1 - 2*pi*theta_i1*theta_ij - 2*pi*theta_i1*theta_j1 + pi*theta_ij**2 - 2*pi*theta_ij*theta_j1 + pi*theta_j1**2) + pi*(A - 1)*(pi*theta_i1**2 + 4*theta_i1*theta_ij*theta_j1 - 2*pi*theta_i1*theta_ij - 2*pi*theta_i1*theta_j1 + pi*theta_ij**2 - 2*pi*theta_ij*theta_j1 + pi*theta_j1**2) + (theta_i1 + pi*(B - 1))*(-2*theta_i1**2*theta_j1 + pi*theta_i1**2 - 2*pi*theta_i1*theta_ij + 2*pi*theta_i1*theta_j1 - 2*theta_ij**2*theta_j1 + pi*theta_ij**2 + 2*pi*theta_ij*theta_j1 + 2*theta_j1**3 - 3*pi*theta_j1**2) + (theta_j1 + pi*(C - 1))*(2*theta_i1**3 - 3*pi*theta_i1**2 - 2*theta_i1*theta_ij**2 + 2*pi*theta_i1*theta_ij - 2*theta_i1*theta_j1**2 + 2*pi*theta_i1*theta_j1 + pi*theta_ij**2 - 2*pi*theta_ij*theta_j1 + pi*theta_j1**2)) + (pi*theta_i1**2 + 4*theta_i1*theta_ij*theta_j1 - 2*pi*theta_i1*theta_ij - 2*pi*theta_i1*theta_j1 + pi*theta_ij**2 - 2*pi*theta_ij*theta_j1 + pi*theta_j1**2)*(pi*theta_i1**2 + 4*theta_i1*theta_ij*theta_j1 - 2*pi*theta_i1*theta_ij - 2*pi*theta_i1*theta_j1 + pi*theta_ij**2 - 2*pi*theta_ij*theta_j1 + 2*theta_ij*(2*theta_i1*theta_j1 - pi*theta_i1 + pi*theta_ij - pi*theta_j1) + pi*theta_j1**2 + 2*pi*(A - 1)*(2*theta_i1*theta_j1 - pi*theta_i1 + pi*theta_ij - pi*theta_j1) - 2*(theta_i1 + pi*(B - 1))*(pi*theta_i1 + 2*theta_ij*theta_j1 - pi*theta_ij - pi*theta_j1) - 2*(theta_j1 + pi*(C - 1))*(2*theta_i1*theta_ij - pi*theta_i1 - pi*theta_ij + pi*theta_j1)))/(pi*(pi*theta_i1**2 + 4*theta_i1*theta_ij*theta_j1 - 2*pi*theta_i1*theta_ij - 2*pi*theta_i1*theta_j1 + pi*theta_ij**2 - 2*pi*theta_ij*theta_j1 + pi*theta_j1**2)**2)
        return x

    def covp(p):
        return 2*pi**2*(-pi**3*n1*p**3 - 2*pi**2*n1*p**2*theta_i1 - pi**2*n1*p**2*theta_j1 + 2*pi**3*n1*p**2 - pi*n1*p*theta_i1**2 - pi*n1*p*theta_i1*theta_j1 + 2*pi**2*n1*p*theta_i1 + pi**2*n1*p*theta_j1 - pi**3*n1*p - pi**3*n2*p**3 - pi**2*n2*p**2*theta_i1 - pi**2*n2*p**2*theta_j1 + 2*pi**3*n2*p**2 - pi*n2*p*theta_i1*theta_j1 + pi**2*n2*p*theta_i1 + pi**2*n2*p*theta_j1 - pi**3*n2*p - pi**3*n3*p**3 - 2*pi**2*n3*p**2*theta_i1 - 2*pi**2*n3*p**2*theta_j1 + 3*pi**3*n3*p**2 - pi*n3*p*theta_i1**2 - 3*pi*n3*p*theta_i1*theta_j1 + 4*pi**2*n3*p*theta_i1 - pi*n3*p*theta_j1**2 + 4*pi**2*n3*p*theta_j1 - 3*pi**3*n3*p - n3*theta_i1**2*theta_j1 + pi*n3*theta_i1**2 - n3*theta_i1*theta_j1**2 + 3*pi*n3*theta_i1*theta_j1 - 2*pi**2*n3*theta_i1 + pi*n3*theta_j1**2 - 2*pi**2*n3*theta_j1 + pi**3*n3 - pi**3*n4*p**3 - pi**2*n4*p**2*theta_i1 - 2*pi**2*n4*p**2*theta_j1 + 2*pi**3*n4*p**2 - pi*n4*p*theta_i1*theta_j1 + pi**2*n4*p*theta_i1 - pi*n4*p*theta_j1**2 + 2*pi**2*n4*p*theta_j1 - pi**3*n4*p)

    def covp_p(p):
        return 2*pi**3*(-3*pi**2*n1*p**2 - 4*pi*n1*p*theta_i1 - 2*pi*n1*p*theta_j1 + 4*pi**2*n1*p - n1*theta_i1**2 - n1*theta_i1*theta_j1 + 2*pi*n1*theta_i1 + pi*n1*theta_j1 - pi**2*n1 - 3*pi**2*n2*p**2 - 2*pi*n2*p*theta_i1 - 2*pi*n2*p*theta_j1 + 4*pi**2*n2*p - n2*theta_i1*theta_j1 + pi*n2*theta_i1 + pi*n2*theta_j1 - pi**2*n2 - 3*pi**2*n3*p**2 - 4*pi*n3*p*theta_i1 - 4*pi*n3*p*theta_j1 + 6*pi**2*n3*p - n3*theta_i1**2 - 3*n3*theta_i1*theta_j1 + 4*pi*n3*theta_i1 - n3*theta_j1**2 + 4*pi*n3*theta_j1 - 3*pi**2*n3 - 3*pi**2*n4*p**2 - 2*pi*n4*p*theta_i1 - 4*pi*n4*p*theta_j1 + 4*pi**2*n4*p - n4*theta_i1*theta_j1 + pi*n4*theta_i1 - n4*theta_j1**2 + 2*pi*n4*theta_j1 - pi**2*n4)



    cv_2_est_angle = est_ij
    cv_2_est_angle2 = est_ij
    p3 = n3 / k
    for _ in range(num_NR_iter):
        # cv_2_est_angle = cv_2_est_angle - cv_f(cv_2_est_angle) / cv_fp(cv_2_est_angle)
        cv_2_est_angle = cv_2_est_angle - cv_f2(cv_2_est_angle) / cv_fp2(cv_2_est_angle)
        cv_2_est_angle2 = cv_2_est_angle2 - cv1(cv_2_est_angle2) / cv1_p(cv_2_est_angle2)
        p3 = p3 - covp(p3) / covp_p(p3)
    
    p2 = p3 - 1 + (angle_i1 + angle_j1) / np.pi
    cv_3 = np.pi * (1 - p2 - p3)


    # MLE
    # perform newton-raphson to find p3
    # cross-multiply for stability
    # def mle_f(x):
    #     # p1 = n1 * np.pi * (angle_i1 + np.pi(x - 1)) * (angle_i1 + angle_j1 + np.pi(x - 1)) * x
    #     # p2 = n2 * np.pi * (angle_i1 + np.pi(x - 1)) * (angle_j1 + np.pi(x - 1)) * x
    #     # p3 = n3 * (angle_i1 + np.pi(x - 1)) * (angle_j1 + np.pi(x - 1)) * (angle_i1 + angle_j1 + np.pi(x - 1))
    #     # p4 = n4 * np.pi * (angle_j1 + np.pi(x - 1)) * (angle_i1 + angle_j1 + np.pi(x - 1)) * x

    #     p1 = n1 * np.pi / (angle_j1 + np.pi * (x - 1))
    #     p2 = n4 * np.pi / (angle_i1 + np.pi * (x - 1))
    #     p3 = n2 * np.pi / (angle_i1 + angle_j1 + np.pi * (x - 1))
    #     p4 = n3 / x
    #     return p1 + p2 + p3 + p4

    # def mle_fp(x):
    #     p1 = n1 * np.pi * np.pi / (angle_j1 + np.pi * (x - 1)) ** 2
    #     p2 = n4 * np.pi * np.pi / (angle_i1 + np.pi * (x - 1)) ** 2
    #     p3 = n2 * np.pi * np.pi / (angle_i1 + angle_j1 + np.pi * (x - 1)) ** 2
    #     p4 = n3 / x ** 2
    #     return - p1 - p2 - p3 - p4

    def mle_o(p):
        return (pi**3*n1*p**3 + 2*pi**2*n1*p**2*theta_i1 + pi**2*n1*p**2*theta_j1 - 2*pi**3*n1*p**2 + pi*n1*p*theta_i1**2 + pi*n1*p*theta_i1*theta_j1 - 2*pi**2*n1*p*theta_i1 - pi**2*n1*p*theta_j1 + pi**3*n1*p + pi**3*n2*p**3 + pi**2*n2*p**2*theta_i1 + pi**2*n2*p**2*theta_j1 - 2*pi**3*n2*p**2 + pi*n2*p*theta_i1*theta_j1 - pi**2*n2*p*theta_i1 - pi**2*n2*p*theta_j1 + pi**3*n2*p + pi**3*n3*p**3 + 2*pi**2*n3*p**2*theta_i1 + 2*pi**2*n3*p**2*theta_j1 - 3*pi**3*n3*p**2 + pi*n3*p*theta_i1**2 + 3*pi*n3*p*theta_i1*theta_j1 - 4*pi**2*n3*p*theta_i1 + pi*n3*p*theta_j1**2 - 4*pi**2*n3*p*theta_j1 + 3*pi**3*n3*p + n3*theta_i1**2*theta_j1 - pi*n3*theta_i1**2 + n3*theta_i1*theta_j1**2 - 3*pi*n3*theta_i1*theta_j1 + 2*pi**2*n3*theta_i1 - pi*n3*theta_j1**2 + 2*pi**2*n3*theta_j1 - pi**3*n3 + pi**3*n4*p**3 + pi**2*n4*p**2*theta_i1 + 2*pi**2*n4*p**2*theta_j1 - 2*pi**3*n4*p**2 + pi*n4*p*theta_i1*theta_j1 - pi**2*n4*p*theta_i1 + pi*n4*p*theta_j1**2 - 2*pi**2*n4*p*theta_j1 + pi**3*n4*p)/(p*(pi**3*p**3 + 2*pi**2*p**2*theta_i1 + 2*pi**2*p**2*theta_j1 - 3*pi**3*p**2 + pi*p*theta_i1**2 + 3*pi*p*theta_i1*theta_j1 - 4*pi**2*p*theta_i1 + pi*p*theta_j1**2 - 4*pi**2*p*theta_j1 + 3*pi**3*p + theta_i1**2*theta_j1 - pi*theta_i1**2 + theta_i1*theta_j1**2 - 3*pi*theta_i1*theta_j1 + 2*pi**2*theta_i1 - pi*theta_j1**2 + 2*pi**2*theta_j1 - pi**3))

    
    def mle_op(p):
        return (pi*p*(pi**3*p**3 + 2*pi**2*p**2*theta_i1 + 2*pi**2*p**2*theta_j1 - 3*pi**3*p**2 + pi*p*theta_i1**2 + 3*pi*p*theta_i1*theta_j1 - 4*pi**2*p*theta_i1 + pi*p*theta_j1**2 - 4*pi**2*p*theta_j1 + 3*pi**3*p + theta_i1**2*theta_j1 - pi*theta_i1**2 + theta_i1*theta_j1**2 - 3*pi*theta_i1*theta_j1 + 2*pi**2*theta_i1 - pi*theta_j1**2 + 2*pi**2*theta_j1 - pi**3)*(3*pi**2*n1*p**2 + 4*pi*n1*p*theta_i1 + 2*pi*n1*p*theta_j1 - 4*pi**2*n1*p + n1*theta_i1**2 + n1*theta_i1*theta_j1 - 2*pi*n1*theta_i1 - pi*n1*theta_j1 + pi**2*n1 + 3*pi**2*n2*p**2 + 2*pi*n2*p*theta_i1 + 2*pi*n2*p*theta_j1 - 4*pi**2*n2*p + n2*theta_i1*theta_j1 - pi*n2*theta_i1 - pi*n2*theta_j1 + pi**2*n2 + 3*pi**2*n3*p**2 + 4*pi*n3*p*theta_i1 + 4*pi*n3*p*theta_j1 - 6*pi**2*n3*p + n3*theta_i1**2 + 3*n3*theta_i1*theta_j1 - 4*pi*n3*theta_i1 + n3*theta_j1**2 - 4*pi*n3*theta_j1 + 3*pi**2*n3 + 3*pi**2*n4*p**2 + 2*pi*n4*p*theta_i1 + 4*pi*n4*p*theta_j1 - 4*pi**2*n4*p + n4*theta_i1*theta_j1 - pi*n4*theta_i1 + n4*theta_j1**2 - 2*pi*n4*theta_j1 + pi**2*n4) - (4*pi**3*p**3 + 6*pi**2*p**2*theta_i1 + 6*pi**2*p**2*theta_j1 - 9*pi**3*p**2 + 2*pi*p*theta_i1**2 + 6*pi*p*theta_i1*theta_j1 - 8*pi**2*p*theta_i1 + 2*pi*p*theta_j1**2 - 8*pi**2*p*theta_j1 + 6*pi**3*p + theta_i1**2*theta_j1 - pi*theta_i1**2 + theta_i1*theta_j1**2 - 3*pi*theta_i1*theta_j1 + 2*pi**2*theta_i1 - pi*theta_j1**2 + 2*pi**2*theta_j1 - pi**3)*(pi**3*n1*p**3 + 2*pi**2*n1*p**2*theta_i1 + pi**2*n1*p**2*theta_j1 - 2*pi**3*n1*p**2 + pi*n1*p*theta_i1**2 + pi*n1*p*theta_i1*theta_j1 - 2*pi**2*n1*p*theta_i1 - pi**2*n1*p*theta_j1 + pi**3*n1*p + pi**3*n2*p**3 + pi**2*n2*p**2*theta_i1 + pi**2*n2*p**2*theta_j1 - 2*pi**3*n2*p**2 + pi*n2*p*theta_i1*theta_j1 - pi**2*n2*p*theta_i1 - pi**2*n2*p*theta_j1 + pi**3*n2*p + pi**3*n3*p**3 + 2*pi**2*n3*p**2*theta_i1 + 2*pi**2*n3*p**2*theta_j1 - 3*pi**3*n3*p**2 + pi*n3*p*theta_i1**2 + 3*pi*n3*p*theta_i1*theta_j1 - 4*pi**2*n3*p*theta_i1 + pi*n3*p*theta_j1**2 - 4*pi**2*n3*p*theta_j1 + 3*pi**3*n3*p + n3*theta_i1**2*theta_j1 - pi*n3*theta_i1**2 + n3*theta_i1*theta_j1**2 - 3*pi*n3*theta_i1*theta_j1 + 2*pi**2*n3*theta_i1 - pi*n3*theta_j1**2 + 2*pi**2*n3*theta_j1 - pi**3*n3 + pi**3*n4*p**3 + pi**2*n4*p**2*theta_i1 + 2*pi**2*n4*p**2*theta_j1 - 2*pi**3*n4*p**2 + pi*n4*p*theta_i1*theta_j1 - pi**2*n4*p*theta_i1 + pi*n4*p*theta_j1**2 - 2*pi**2*n4*p*theta_j1 + pi**3*n4*p))/(p**2*(pi**3*p**3 + 2*pi**2*p**2*theta_i1 + 2*pi**2*p**2*theta_j1 - 3*pi**3*p**2 + pi*p*theta_i1**2 + 3*pi*p*theta_i1*theta_j1 - 4*pi**2*p*theta_i1 + pi*p*theta_j1**2 - 4*pi**2*p*theta_j1 + 3*pi**3*p + theta_i1**2*theta_j1 - pi*theta_i1**2 + theta_i1*theta_j1**2 - 3*pi*theta_i1*theta_j1 + 2*pi**2*theta_i1 - pi*theta_j1**2 + 2*pi**2*theta_j1 - pi**3)**2)


    def mle2(p):
        x = pi*n1*p*(theta_i1 + pi*(p - 1))*(theta_i1 + theta_j1 + pi*(p - 1)) + pi*n2*p*(theta_i1 + pi*(p - 1))*(theta_j1 + pi*(p - 1)) + n3*(theta_i1 + pi*(p - 1))*(theta_j1 + pi*(p - 1))*(theta_i1 + theta_j1 + pi*(p - 1)) + pi*n4*p*(theta_j1 + pi*(p - 1))*(theta_i1 + theta_j1 + pi*(p - 1))
        return x
    
    def mle2_p(p):
        x = pi*(pi*n1*p*(theta_i1 + pi*(p - 1)) + pi*n1*p*(theta_i1 + theta_j1 + pi*(p - 1)) + n1*(theta_i1 + pi*(p - 1))*(theta_i1 + theta_j1 + pi*(p - 1)) + pi*n2*p*(theta_i1 + pi*(p - 1)) + pi*n2*p*(theta_j1 + pi*(p - 1)) + n2*(theta_i1 + pi*(p - 1))*(theta_j1 + pi*(p - 1)) + n3*(theta_i1 + pi*(p - 1))*(theta_j1 + pi*(p - 1)) + n3*(theta_i1 + pi*(p - 1))*(theta_i1 + theta_j1 + pi*(p - 1)) + n3*(theta_j1 + pi*(p - 1))*(theta_i1 + theta_j1 + pi*(p - 1)) + pi*n4*p*(theta_j1 + pi*(p - 1)) + pi*n4*p*(theta_i1 + theta_j1 + pi*(p - 1)) + n4*(theta_j1 + pi*(p - 1))*(theta_i1 + theta_j1 + pi*(p - 1)))
        return x

    p3 = n3 / k
    p3o = n3 / k
    # # just perform NR
    # # probability checks
    for _ in range(num_NR_iter):
        # p3 = p3 - mle_f(p3) / mle_fp(p3)
        p3o = p3o - mle_o(p3o) / mle_op(p3o)
        p3 = p3 - mle2(p3) / mle2_p(p3)
    # mle_est_angle = p3

    p2 = p3o - 1 + (angle_i1 + angle_j1) / np.pi
    mle_est_angle1 = np.pi * (1 - p2 - p3o)

    # # calculate the estimated angle
    p2 = p3 - 1 + (angle_i1 + angle_j1) / np.pi
    mle_est_angle2 = np.pi * (1 - p2 - p3)

    # print(mle2(p3), mle_o(p3))

    # return the estimates
    return cv_est_angle, cv_2_est_angle, cv_2_est_angle2, cv_3, mle_est_angle1, mle_est_angle2


# repeat the simulation many times
cv_ests, cv_ests2, cv_ests22, cv_ests3, mle_ests1, mle_ests = np.transpose(np.array([one_sim() for _ in (range(reps))]))
print(angle_ij, "actual")
diff_cv = (cv_ests - angle_ij) ** 2
diff_cv2 = (cv_ests2 - angle_ij) ** 2
diff_cv22 = (cv_ests22 - angle_ij) ** 2
diff_cv3 = (cv_ests3 - angle_ij) ** 2

diff_mle1 = (mle_ests1 - angle_ij) ** 2
diff_mle = (mle_ests - angle_ij) ** 2
print(np.mean(cv_ests), np.mean(diff_cv), "sub")
print(np.mean(cv_ests2), np.mean(diff_cv2), "cv cubic (raw)")
print(np.mean(cv_ests22), np.mean(diff_cv22), "cv cubic (cross)")
print(np.mean(cv_ests3), np.mean(diff_cv3), "cv cubic (subbed)")
print(np.mean(mle_ests1), np.mean(diff_mle1), "mle cubic (raw)")
print(np.mean(mle_ests), np.mean(diff_mle), "mle cubic (cross)")

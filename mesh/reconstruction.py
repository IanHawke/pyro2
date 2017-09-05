import mesh.patch as patch
import mesh.array_indexer as ai
import numpy as np

def limit(data, myg, idir, limiter):

    if limiter == 0:
        return nolimit(data, myg, idir)
    elif limiter == 1:
        return limit2(data, myg, idir)
    else:
        return limit4(data, myg, idir)


def nolimit(a, myg, idir):
    """ just a centered difference without any limiting """

    lda = myg.scratch_array()

    if idir == 1:
        lda.v(buf=2)[:,:] = 0.5*(a.ip(1, buf=2) - a.ip(-1, buf=2))
    elif idir == 2:
        lda.v(buf=2)[:,:] = 0.5*(a.jp(1, buf=2) - a.jp(-1, buf=2))

    return lda


def limit2(a, myg, idir):
    """ 2nd order monotonized central difference limiter """

    lda = myg.scratch_array()
    dc = myg.scratch_array()
    dl = myg.scratch_array()
    dr = myg.scratch_array()

    if idir == 1:
        dc.v(buf=2)[:,:] = 0.5*(a.ip(1, buf=2) - a.ip(-1, buf=2))
        dl.v(buf=2)[:,:] = a.ip(1, buf=2) - a.v(buf=2)
        dr.v(buf=2)[:,:] = a.v(buf=2) - a.ip(-1, buf=2)

    elif idir == 2:
        dc.v(buf=2)[:,:] = 0.5*(a.jp(1, buf=2) - a.jp(-1, buf=2))
        dl.v(buf=2)[:,:] = a.jp(1, buf=2) - a.v(buf=2)
        dr.v(buf=2)[:,:] = a.v(buf=2) - a.jp(-1, buf=2)

    d1 = 2.0*np.where(np.fabs(dl) < np.fabs(dr), dl, dr)
    dt = np.where(np.fabs(dc) < np.fabs(d1), dc, d1)
    lda.v(buf=myg.ng)[:,:] = np.where(dl*dr > 0.0, dt, 0.0)

    return lda


def limit4(a, myg, idir):
    """ 4th order monotonized central difference limiter """

    lda_tmp = limit2(a, myg, idir)

    lda = myg.scratch_array()
    dc = myg.scratch_array()
    dl = myg.scratch_array()
    dr = myg.scratch_array()

    if idir == 1:
        dc.v(buf=2)[:,:] = (2./3.)*(a.ip(1, buf=2) - a.ip(-1, buf=2) -
                                    0.25*(lda_tmp.ip(1, buf=2) + lda_tmp.ip(-1, buf=2)))
        dl.v(buf=2)[:,:] = a.ip(1, buf=2) - a.v(buf=2)
        dr.v(buf=2)[:,:] = a.v(buf=2) - a.ip(-1, buf=2)

    elif idir == 2:
        dc.v(buf=2)[:,:] = (2./3.)*(a.jp(1, buf=2) - a.jp(-1, buf=2) - \
                                    0.25*(lda_tmp.jp(1, buf=2) + lda_tmp.jp(-1, buf=2)))
        dl.v(buf=2)[:,:] = a.jp(1, buf=2) - a.v(buf=2)
        dr.v(buf=2)[:,:] = a.v(buf=2) - a.jp(-1, buf=2)

    d1 = 2.0*np.where(np.fabs(dl) < np.fabs(dr), dl, dr)
    dt = np.where(np.fabs(dc) < np.fabs(d1), dc, d1)
    lda.v(buf=myg.ng)[:,:] = np.where(dl*dr > 0.0, dt, 0.0)

    return lda

def flatten(myg, q, idir, ivars, rp):
    """ compute the 1-d flattening coefficients """

    xi = myg.scratch_array()
    z = myg.scratch_array()
    t1 = myg.scratch_array()
    t2 = myg.scratch_array()

    delta = rp.get_param("compressible.delta")
    z0 = rp.get_param("compressible.z0")
    z1 = rp.get_param("compressible.z1")
    smallp = 1.e-10

    if idir == 1:
        t1.v(buf=2)[:,:] = abs(q.ip(1, n=ivars.ip, buf=2) -
                               q.ip(-1, n=ivars.ip, buf=2))
        t2.v(buf=2)[:,:] = abs(q.ip(2, n=ivars.ip, buf=2) -
                               q.ip(-2, n=ivars.ip, buf=2))

        z[:,:] = t1/np.maximum(t2, smallp)

        t2.v(buf=2)[:,:] = t1.v(buf=2)/np.minimum(q.ip(1, n=ivars.ip, buf=2),
                                                  q.ip(-1, n=ivars.ip, buf=2))
        t1.v(buf=2)[:,:] = q.ip(-1, n=ivars.iu, buf=2) - q.ip(1, n=ivars.iu, buf=2)

    elif idir == 2:
        t1.v(buf=2)[:,:] = abs(q.jp(1, n=ivars.ip, buf=2) -
                               q.jp(-1, n=ivars.ip, buf=2))
        t2.v(buf=2)[:,:] = abs(q.jp(2, n=ivars.ip, buf=2) -
                               q.jp(-2, n=ivars.ip, buf=2))

        z[:,:] = t1/np.maximum(t2, smallp)

        t2.v(buf=2)[:,:] = t1.v(buf=2)/np.minimum(q.jp(1, n=ivars.ip, buf=2),
                                                  q.jp(-1, n=ivars.ip, buf=2))
        t1.v(buf=2)[:,:] = q.jp(-1, n=ivars.iv, buf=2) - q.jp(1, n=ivars.iv, buf=2)

    xi.v(buf=myg.ng)[:,:] = np.minimum(1.0, np.maximum(0.0, 1.0 - (z - z0)/(z1 - z0)))

    xi[:,:] = np.where(np.logical_and(t1 > 0.0, t2 > delta), xi, 1.0)

    return xi



def flatten_multid(myg, q, xi_x, xi_y, ivars):
    """ compute the multidimensional flattening coefficient """

    xi = myg.scratch_array()

    px = np.where(q.ip(1, n=ivars.ip, buf=2) -
                  q.ip(-1, n=ivars.ip, buf=2) > 0,
                  xi_x.ip(-1, buf=2), xi_x.ip(1, buf=2))

    py = np.where(q.jp(1, n=ivars.ip, buf=2) -
                  q.jp(-1, n=ivars.ip, buf=2) > 0,
                  xi_y.jp(-1, buf=2), xi_y.jp(1, buf=2))

    xi.v(buf=2)[:,:] = np.minimum(np.minimum(xi_x.v(buf=2), px),
                                  np.minimum(xi_y.v(buf=2), py))

    return xi


# Constants for the WENO reconstruction
# NOTE: integer division laziness means this WILL fail on python2
C_3 = np.array([1, 2]) / 3
a_3 = np.array([[3, -1], [1, 1]]) / 2
sigma_3 = np.array([[[1, 0], [-2, 1]], [[1, 0], [-2, 1]]])

C_5 = np.array([1, 6, 3]) / 10
a_5 = np.array([[11, -7, 2], [2, 5, -1], [-1, 5, 2]]) / 6
sigma_5 = np.array([[[40, 0, 0],
                        [-124, 100, 0],
                        [44, -76, 16] ],
                       [[16, 0, 0],
                        [-52, 52, 0],
                        [20, -52, 16] ],
                       [[16, 0, 0],
                        [-76, 44, 0],
                        [100, -124, 40] ] ]) / 12

C_7 = np.array([1, 12, 18, 4]) / 35
a_7 = np.array([ [25, -23, 13, -3],
                 [3, 13, -5, 1],
                 [-1, 7, 7, -1],
                 [1, -5, 13, 3] ]) / 12
sigma_7 = np.array([ [ [2107, 0, 0, 0],
                       [-9402, 11003, 0, 0],
                       [7042, -17246, 7043, 0],
                       [-1854, 4642, -3882, 547]
                     ],
                     [ [547, 0, 0, 0],
                       [-2522, 3443, 0, 0],
                       [1922, -5966, 2843, 0],
                       [-494, 1602, -1642, 267]
                     ],
                     [ [267, 0, 0, 0],
                       [-1642, 2843, 0, 0],
                       [1602, -5966, 3443, 0],
                       [-494, 1922, -2522, 547]
                     ],
                     [ [547, 0, 0, 0],
                       [-3882, 7043, 0, 0],
                       [4642, -17246, 11003, 0],
                       [-1854, 7042, -9402, 2107]
                     ]
                   ])

C_9 = np.array([1, 20, 60, 40, 5]) / 126
a_9 = np.array([ [137, -163, 137, -63, 12],
                 [12, 77, -43, 17, -3],
                 [-3, 27, 47, -13, 2],
                 [2, -13, 47, 27, -3],
                 [-3, 17, -43, 77, 12] ]) / 60
sigma_9 = np.array([ [ [107918, 0, 0, 0, 0],
                       [-649501, 1020563, 0, 0, 0],
                       [758823, -2462076, 1521393, 0, 0],
                       [-411487, 1358458, -1704396, 482963, 0],
                       [86329, -288007, 364863, -208501, 22658]
                     ],
                     [ [22658, 0, 0, 0, 0],
                       [-140251, 242723, 0, 0, 0],
                       [165153, -611976, 406293, 0, 0],
                       [-88297, 337018, -464976, 138563, 0],
                       [18079, -70237, 99213, -60871, 6908]
                     ],
                     [ [6908, 0, 0, 0, 0],
                       [-51001, 104963, 0, 0, 0],
                       [67923, -299076, 231153, 0, 0],
                       [-38947, 179098, -299076, 104963, 0],
                       [8209, -38947, 67923, -51001, 6908]
                     ],
                     [ [6908, 0, 0, 0, 0],
                       [-60871, 138563, 0, 0, 0],
                       [99213, -464976, 406293, 0, 0],
                       [-70237, 337018, -611976, 242723, 0],
                       [18079, -88297, 165153, -140251, 22658]
                     ],
                     [ [22658, 0, 0, 0, 0],
                       [-208501, 482963, 0, 0, 0],
                       [364863, -1704396, 1521393, 0, 0],
                       [-288007, 1358458, -2462076, 1020563, 0],
                       [86329, -411487, 758823, -649501, 107918]
                     ],
                   ])

C_all = { 2 : C_3,
          3 : C_5,
          4 : C_7,
          5 : C_9 }
a_all = { 2 : a_3,
          3 : a_5,
          4 : a_7,
          5 : a_9 }
sigma_all = { 2 : sigma_3,
              3 : sigma_5,
              4 : sigma_7,
              5 : sigma_9 }

def weno_upwind(q, order):
    """
    Perform upwinded (left biased) WENO reconstruction

    Parameters
    ----------

    q : np array
        input data
    order : int
        WENO order (k)

    Returns
    -------

    q_plus : np array
        data reconstructed to the right
    """
    a = a_all[order]
    C = C_all[order]
    sigma = sigma_all[order]
    epsilon = 1e-16
    alpha = np.zeros(order)
    beta = np.zeros(order)
    q_stencils = np.zeros(order)
    for k in range(order):
        for l in range(order):
            for m in range(l+1):
                beta[k] += sigma[k, l, m] * q[order-1+k-l] * q[order-1+k-m]
        alpha[k] = C[k] / (epsilon + beta[k]**2)
        for l in range(order):
            q_stencils[k] += a[k, l] * q[order-1+k-l]
    w = alpha / np.sum(alpha)

    return np.dot(w, q_stencils)

def weno(q, order):
    """
    Perform WENO reconstruction

    Parameters
    ----------

    q : np array
        input data with 3 ghost zones
    order : int
        WENO order (k)

    Returns
    -------

    q_plus, q_minus : np array
        data reconstructed to the right / left respectively
    """
    Npoints = q.shape
    q_minus = np.zeros_like(q)
    q_plus  = np.zeros_like(q)
    for i in range(order, Npoints-order):
        q_plus [i] = weno_upwind(q[i+1-order:i+order], order)
        q_minus[i] = weno_upwind(q[i+order-1:i-order:-1], order)
    return q_minus, q_plus

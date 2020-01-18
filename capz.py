import numpy as np


def capz(N, M, P, Q, h_r):
    ''' Calculates coefficients of common acoustic pole and zero modeling of room transfer functions.

    Estimated using least squares method.
    Reference:
    Y.Haneda, S.Makino, Y.Kaneda "Common acoustical pole and zero modeling of room transfer functions", IEEE Transactions on Speech and Audio Processing, vol.2, no.2, pp. 320-328, 1994

    Parameters
    ----------
    N : int
        Order of impulse response.
    M : int
        Number of measured impulse responses.
    P : int
        Order of AR coefficients.
    Q : int
        Order of MA coefficients.
    h_r: numpy ndarray (M, N)
        Measured impulse responses.

    Returns
    -------
    a : numpy ndarray (P, 1)
        Estimated common AR coefficients.
    B : numpy ndarray (M, Q)
        Estimated MA coefficients for each measured room impulse response.
    '''
    assert h_r.shape = (M, N), 'The shape of h_r needs to be ({}, {}), instead of '.format(M, N, h_r.shape)

    As = np.zeros((M, N + P + 1, P))
    for i in range(M):
        for j in range(P):
            As[i, j + 1:j + 1 + N, j] = h_r[i, :]

    D = np.zeros((N + P + 1, Q + 1))
    D[:Q + 1, :] = np.eye(Q+1)

    BigA = np.zeros((M * (N + P + 1), P + M * (Q + 1)))

    for m in range(M):
        BigA[m * (N + P + 1):(m+1)*(N + P + 1), :P] = As[m, :, :]

    for i in range(M):
        BigA[i * (N + P + 1):(i + 1) * (N + P + 1), P +
             i * (Q + 1): P + (i + 1) * (Q + 1)] = D

    h_pad = np.hstack((h_r, np.zeros((M, P + 1))))
    x = np.linalg.solve(np.dot(BigA.T, BigA), np.dot(BigA.T, h_pad.ravel()))

    a = -x[:P]
    B = x[P:].reshape(M, -1)

    return a, B

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from Cholesky import *
from model import *
from PHS_Kernel_func import *


def u(t):
    return 0.2 * t * np.sign(np.sin((t + 2) * np.pi / 2)) + 1


def G(x):
    return np.array([0, 1])


def H(x, m=1, k=10):
    return 1.0 / 2.0 / m * x[1] ** 2 + 1.0 / 2 * k * x[0] ** 2


def ode_fun(x, t, JR, H, G, u):
    # Compute state derivatives using PHS form with numerical gradients.
    dim = np.shape(x)[0]
    dH = np.zeros(dim)

    for i in range(dim):
        y = x.copy()
        y[i] = x[i] - 1e-5
        dH[i] = (H(x) - H(y)) / 1e-5

    dx = np.dot(JR, dH) + G(x) * u(t)
    return dx


# this function is called only once for graph production
# Figure1-- function is used for encapsulation of main driver
def graph1(t_span, x_org):
    # Plot the results
    plt.figure(1)
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(t_span, x_org[:, 0])
    plt.xlabel('time')
    plt.ylabel('position')
    plt.subplot(2, 1, 2)
    plt.plot(t_span, x_org[:, 1])
    plt.xlabel('time')
    plt.ylabel('momentum')


def base_init():
    # base initialization
    k = 10
    R = 1
    m = 1
    JR = np.array([[0, 1], [-1, -1 / R]])

    # Define the initial state
    x0 = np.array([1, 0])

    # Define the time span for the simulation
    t_span = np.arange(0, 20.01, 0.01)

    return k, R, m, JR, x0, t_span


def prepare_training_data(x_org, JR, H, G, u):
    # Input data
    t = np.array([0.01 * i for i in range(0, (20 * 100) + 1)])
    X = x_org[0::10, :].T
    t_mod = t[::10]
    n_data = X.shape[1]
    dX = np.zeros((2, n_data))

    # Get the derivatives (for the real system, we would need to use numerical gradients)
    for i in range(n_data):
        dX[:, i] = ode_fun(X[:, i], t_mod[i], JR, H, G, u) - np.dot(G(X[:, i]), u(t_mod[i]))

    dX = dX.T.flatten()
    # Output data / corrupted by some noise
    dX = dX + 1 * np.random.randn(*dX.shape)

    return t, X, t_mod, n_data, dX


def get_test_pred(X, JR, hyp, alpha):
    X_test, Y_test = np.meshgrid(np.arange(-2, 2.1, 0.1), np.arange(-2, 2.1, 0.1))
    x_test = np.vstack((Y_test.ravel(), X_test.ravel()))
    n_test = X_test.shape
    JRr = np.kron(np.eye(n_test[0] * n_test[1]), JR)

    k_matrix = PHS_kernel(x_test, X, hyp.get_SD(), hyp.get_L(), 1)
    temp = np.reshape(JRr.dot(k_matrix).T, (n_test[0] * n_test[1], len(alpha)))
    pred = temp.dot(alpha)
    return X_test, Y_test, n_test, x_test, JRr, pred


def ode_fun_gp(x, t, JR, JRX, alpha, G, u_tst, hyp, X):
    """
    Differential equation system function for odeint.
    """
    # Calculate the PHS kernel using PHSkernel_se

    PHS_kernel_res = PHS_kernel(x.reshape(len(x), 1), X, hyp.get_SD(), hyp.get_L(), 2)
    PHS_kernel_res.reshape(int(0.5 * PHS_kernel_res.shape[0]), int(2 * PHS_kernel_res.shape[1]))

    # Calculate the weighted sum
    weighted_sum = JR.dot(PHS_kernel_res.T).dot(JRX.T).dot(alpha)

    # Calculate the gradient of the system dynamics
    gradient = G(x)

    # Calculate the external input/control signal
    input_signal = u_tst(t)

    # Calculate the derivative of the state variables
    dxdt = weighted_sum + gradient * input_signal

    return dxdt


def getTrue_hamiltonian(n_test, x_test):
    H_true = np.zeros(n_test[0] * n_test[1])
    for i in range(n_test[0] * n_test[1]):
        H_true[i] = H(x_test[:, i])
    return H_true


def main():
    # base initialization:
    k, R, m, JR, x0, t_span = base_init()

    # Simulate the system using odeint, --Figure 1
    x_org = odeint(ode_fun, x0, t_span, args=(JR, H, G, u))
    graph1(t_span, x_org)

    # prepare for training data
    t, X, t_mod, n_data, dX = prepare_training_data(x_org, JR, H, G, u)

    lb = np.concatenate((1e-6 * np.ones((4, 1)), -10 * np.ones((1, 1))))
    ub = np.concatenate((1000 * np.ones((4, 1)), np.zeros((1, 1))))
    model = Model(lb=lb, ub=ub, X0=X, dX0=dX)

    # learning of parameters
    model.optimize_Hyp()

    JRX = np.kron(np.eye(n_data), JR)
    K = JRX.dot(PHS_kernel(X, X, model.get_Hyp_sd(), model.get_Hyp_l(), 2)).dot(JRX.T)
    L = Cholesky_decomp(K + (model.get_Hyp_sn() ** 2) * np.eye(K.shape[0]))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, dX))

    # test data and prediction
    X_test, Y_test, n_test, x_test, JRr, pred = get_test_pred(X, JR, model.get_Hyp(), alpha)

    # true hamiltonian
    H_true = getTrue_hamiltonian(n_test, x_test)

    const = np.min(pred)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X_test, Y_test, np.reshape(H_true, n_test), rstride=2, cstride=2, color='maroon', linewidth=0.3)
    ax.plot_surface(X_test, Y_test, np.reshape(pred - const, n_test), linewidth=0.3)
    ax.scatter3D(X[0, :], X[1, :], np.zeros(n_data), color='red', marker='o')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('Hamiltonian')
    plt.show()

    # Define the initial state
    x0 = np.array([0.5, 0]).T

    # Define the time span for the simulation
    t_span = np.arange(0, 20.01, 0.01)
    u_test = lambda t: 0 * t

    x_gp = odeint(ode_fun_gp, x0, t_span, args=(JR, JRX, alpha, G, u_test, model.get_Hyp(), X))
    x_org = odeint(ode_fun, x0, t_span, args=(JR, H, G, u_test))

    plt.subplot(2, 1, 1)
    plt.plot(t_span, x_org[:, 0], color='red', label='True Hamiltonian', linewidth=0.2)
    plt.plot(t_span, x_gp[:, 0], color='blue', label='GP model', linewidth=0.2)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t_span, x_org[:, 1], color='red', label='True Hamiltonian')
    plt.plot(t_span, x_gp[:, 1], color='blue', label='GP model')
    plt.legend()
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

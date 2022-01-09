"""
05/29/2020, to implement dgm for solving 1d helmholtz equation.
                loss = \int_{element} [equatin-residual]^2 +
                        similar term for bc and ck continuity
            this is implemented. works very well.

05/29/2020, to implement 1D helmholtz equation without integration
            by parts in weak form.

            this is implemented. performs well. seems to require larger network
            to get same accuracy than formulation with integration by parts.

05/28/2020, convert to tensorflow 2, using gradient tape to compute
            derivatives in eager mode.
            add normalization of input data
            add parameter and other files for output

            change optimizer to L-BFGS
            need to define train_step_lbfgs(data) in subclass of keras.Model
                where data = (train_x,train_y)

In this implementation:
(1) Sherwin-Karniadakis basis within each element.
    enforce equation within each element. different modes in different elements
        correspond to different equations
(2) Field function represented by local DNN within each element
    enforce C^k continuity across element boundaries for field function, where k=0,1,2,3,4, or -1

solve Helmholtz equation in 1D by
spectral element methods with DNN, on domain [a,b]
multiple elements, on domain [a,b]
Spectral element Petrov-Galerkin method DNN
Using Sherin-Karniadakis basis functions, within each element

function represented by multiple local DNN, or a DNN with multiple inputs and multiple outputs
enforce weak form of equations within each element in terms of polynomials (SK-basis)
only enforce equation within each element,
different modes in different elements correspond to different equations
enforce C^k continuity across element boundaries for field function, where k=0, 1, 2, 3, 4, or -1
    when k=-1, do not enforce any continuity for field function across element boundaries
local DNN representation for each element, enforce certain continuity across element boundary

need to extract data for element from global tensor for u and du/dx
use keras.backend.slice() or tensorflow.split() to do this

'adam' seems better than 'nadam' for this problem

another implementation:
  basis matrix dimension: [N_quad, N_modes]
  Weight matrix dimension: [N_elem, N_quad], etc
########################################################

09/18/2020, to implement uniform collocation points for solving 1D helmholtz equation
            DGM, least squares loss.
            this is implemented.

09/18/2020, to change L-BFGS optimizer to Adam.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import pandas as pd

import my_util
import polylib
import skbasis as skb
#import dk_lbfgs as klbfgs
from dparam import Dparam

#tf.random.set_seed(1234)

tf.config.experimental_run_functions_eagerly(False)
_epsilon = 1.0e-14

dpar = Dparam()

## ========================================= ##
# domain contains coordinates of element boundaries
domain = [ 0.0, 4.0 ]
#domain = [ 0.0, 2.0, 4.0, 5.0, 12.0, 13.0, 14.0, 14.5, 15.0 ]
#domain = [ 0.0, 2.0, 4.0, 5.0, 7.5, 9.5, 11.5, 13.0, 14.0, 14.5, 15.0 ]

# domain parameters
N_elem = len(domain) - 1  # number of elements

# C_k continuity
CK = 1  # C^k continuity

LAMBDA_COEFF = 10.0   # lambda constant
#aa = 3.0

LR_factor = 0.025 # learning rate factor, actual_lr = LR_factor * default_lr
MAX_EPOCHS = 50000 # max epochs
to_read_init_weight = True # flag

BC_ck_penalty_coeff = 0.9
Equ_penalty_coeff = 1.0 - BC_ck_penalty_coeff

## ========================================= ##
# NN parameters
layers = [ 1, 100, 100, 100, 100, 1 ]
activations = [ 'None', 'tanh', 'tanh', 'tanh', 'tanh', 'linear']
assert len(layers)==len(activations)
#layers = [ 1, 15, 15,  1 ]
#activations = [ 'None',  'tanh', 'tanh',  'linear']

N_colloc = 201 # number of collocation points per element

N_quad = N_colloc # number of quadrature points
N_modes = 20 # number of modes
N_pred = 801 # number of points for prediction

## ========================================= ##
# default, set to double precision
K.set_floatx('float64')
K.set_epsilon(_epsilon)

## =========================== ##
"""
def the_anal_soln(x):
    return x * np.cos(aa * x)

def the_anal_soln_deriv2(x):
    return -2.0 * aa * np.sin(aa * x) - aa * aa * x * np.cos(aa * x)
"""

# parameters for analytic solution
dpar.AA = 1.0
dpar.aa = 2.0*np.pi
dpar.bb = 3.0*np.pi

def the_anal_soln(x):
    return dpar.AA * np.sin(dpar.bb * (x + 0.05)) * np.cos(dpar.aa * (x + 0.05)) + 2.0

def the_anal_soln_deriv2(x):
    AA = dpar.AA
    aa = dpar.aa
    bb = dpar.bb
    return -AA*(aa*aa+bb*bb)*np.sin(bb*(x+0.05))*np.cos(aa*(x+0.05))\
           -2.0*AA*aa*bb*np.cos(bb*(x+0.05))*np.sin(aa*(x+0.05))

def the_source_term(lambda_coeff, x):
    value = the_anal_soln_deriv2(x) - lambda_coeff*the_anal_soln(x)
    return value

# ================================= #
def loss_manip_func(x):
    """
        loss manipulation function
        x -- tensor
    """
    y = x*x
    return y

## ===================================== ##
def calc_basis (id, z):
    """
    compute id-th basis function on given points z
    :param id: index of basis function
    :param z: points to evaluate on
    :return: values on points
    """
    # use Sherwin-Karniadakis basis
    return skb.calc_basis(id, z)

def calc_basis_deriv(id, z):
    """
    compute derivative of id-th basis fuction on given points
    :param id: index of basis function
    :param z: points to evaluate on
    :return: values on points
    """
    # use Sherwin-Karniadakis basis
    return skb.calc_basis_deriv(id, z)

def calc_zw (n_quads):
    """
    compute quadrature points and weights
    :param n_quads: number of qaudrature points
    :return:
    """
    z, w = polylib.zwgll(n_quads) # zeros and weights of Gauss-Labatto-Legendre quadrature
    return (z,w)

def calc_jacobian(this_domain):
    """
    compute the Jacobian for each element
    :param this_domain: vector containing element boundary coefficients
    :return: tuple with shape: (n_elem, 3). [:,0] -- J_0e; [:,1] -- J_1e; [:,2] -- J_2e, size of element
             where scaling is x = J_1e*xi + J_0e, where xi is in [-1,1]
                   element size is J_2e
    """
    n_elem = len(this_domain) - 1
    jacob = np.zeros((n_elem, 3))
    for i in range(n_elem):
        jacob[i, 0] = (this_domain[i+1]+this_domain[i])*0.5
        jacob[i, 1] = (this_domain[i+1]-this_domain[i])*0.5
        jacob[:, 2] = this_domain[i+1]-this_domain[i] # size of this element

    return jacob

def get_basis_mat(n_modes, n_quads):
    """
    compute matrix of basis functions and weight vector
    on Gauss-Lobatto-Legendre quadrature points
    :param n_modes: number of modes
    :param n_quads: number of quadrature points
    :return: tuple (B, w), where B[n_modes][n_quads] is the basis matrix on quadrature points
                           w[n_quads] is the vector of weights on the quadrature points
    """
    B = np.zeros((n_modes, n_quads))
    z, w = calc_zw(n_quads) #polylib.zwgll(n_quads)
    # (z,w) contains zeros and weights of Lobatto-Legendre

    # compute basis matrix
    for i in range(n_modes):
        B[i, :] = calc_basis(i, z) #polylib.legendref(z, i)
    # now B[n_modes,n_quads] contains matrix of basis functions

    return (B, w)

def get_basis_deriv_mat(n_modes, n_quads):
    """
    compute derivative matrix of basis functions on Gauss-Lobatto-Legendres quadrature points
    :param n_modes: number of modes
    :param n_quads: number of quadrature points
    :return: basis derivative matrix Bd[n_modes][n_quads] on quadrature points
    """
    Bd = np.zeros((n_modes, n_quads))
    z, _ = calc_zw (n_quads) #polylib.zwgll(n_quads)

    for i in range(n_modes):
        Bd[i, :] = calc_basis_deriv(i, z) # polylib.legendred(z, i)

    return Bd

def get_basis_boundary_mat(n_modes):
    """
    compute matrix of basis functions on boundaries
    :param n_modes: number of modes
    :return: values on boundaries 1, and -1
    """
    z = np.zeros(2)
    z[0] = -1.0
    z[1] = 1.0

    B_bound = np.zeros((n_modes, 2))
    for i in range(n_modes):
        B_bound[i, :] = calc_basis(i, z)

    return B_bound

def get_basis_info(n_modes, n_quads):
    """
    compute expansion basis data for each element
    :param n_modes: number of modes per element
    :param n_quads: number of quadrature points per element
    :return: tuple of basis data, (B, Bd, W, B_bound)
             where B : basis matrix on quadrature points, shape: (n_quads, n_modes)
                   Bd : basis derivative matrix on quadrature points, shape: (n_quads, n_modes)
                   W : weight matrix on quadrature points, shape: (1, n_quads)
                   B_bound : basis values on boundary x=-1, 1, shape: (2, n_modes)
                             B_bound[0, :] contains basis values on x=-1
                             B_bound[1, :] contains basis values on x=1
    """
    B, W = get_basis_mat(n_modes, n_quads)
    B_trans = np.transpose(B)  # B_trans shape: (n_quads, n_modes)
    # Now B contains matrix of bases on quadrature points, dimension [n_modes][n_quads]
    #     W contains vector of weights on quadrature points, dimension [n_quads]

    # basis function derivative matrix
    Bd = get_basis_deriv_mat(n_modes, n_quads)
    Bd_trans = np.transpose(Bd)  # shape: (n_quads, n_modes)
    # now Bd contains matrix of derivatives of bases on quadrature points, in standard element
    #     dimension [n_modes][n_quads]

    W_tmp = np.expand_dims(W, 0) # shape: (1, n_quads)

    B_bound = get_basis_boundary_mat(n_modes)
    # now B_bound[n_modes][0:2] contains basis values on -1 and 1, in standard element
    B_bound_trans = np.transpose(B_bound)
    # now B_bound_trans has shape: (2, n_modes)

    return (B_trans, Bd_trans, W_tmp, B_bound_trans)

def one_elem_loss_generator_quad(id_elem, elem_param, basis_info, model_info):
    """
    build loss function for one element
    :param id_elem: id of current element and sub-model
    :param elem_param: tuple, (n_modes, n_quads, lambda_coeff, bc, jacobian, domain, ck_k)
                       where n_modes is the number of modes in each element
                             n_quads is the number of quadrature points in each element
                             lambda_coeff is the lambda coefficient
                             bc : numpy array, shape: (2,2); bc[:, 0] contains coordinates of 2 boundaries
                                                             bc[:, 1] contains Dirivhlet BC values of 2 boundaries
                             jacobian: vector containing jacobians of each element, shape: (n_elem,3)
                             domain: vector contains element boundary coordinates, shape: (n_elem+1,)
                             ck_k : k value in C_k continuity
    :param basis_info: tuple, (B, Bd, W, B_bound)
                       where B : basis matrix, shape: (n_quads, n_modes)
                             Bd : basis derivative matrix, shape: (n_quads, n_modes)
                             W : weight matrix, shape: (1, n_quads)
                             B_bound : matrix of basis values on x=-1, 1; shape: (2, n_modes)
    :param model_info: tuple, (model, sub_model_set)
                       where model: multi_element_model for n_elem elements, overall model of n_elem sub-models
                             sub_model_set: list of sub-models for each element, shape: (n_elem,)
    :return: loss function for the sub-model of this element

    if this is the first element, will compute equation residual of this element, and also of element-boundary
        C^k continuity conditions, as well as global boundary condition loss
    if this is not the first element, will compute only the equation residual of this element
    """
    n_modes, n_quads, lambda_coeff, bc, jacob, this_domain, ck_k = elem_param
    n_elem = jacob.shape[0] # number of elements

    global_model, sub_model_list = model_info # global_model contains the overall model

    B_trans, Bd_trans, W_mat_A, B_bound = basis_info
    # B_trans contains basis function matrix, shape: (n_quads, n_modes)
    # Bd_trans contains basis derivative matrix, shape: (n_quads, n_modes)
    # W_mat_A contains weight matrix, shape: (1, n_quads)
    # B_bound contains boundary basis matrix, shape: (2, n_modes)

    W_mat = np.zeros((1, n_quads))
    W_mat[0, :] = W_mat_A[0, :] * jacob[id_elem, 1]
    # now W_mat contains jacobian * W matrix, shape: (1, n_quads)

    B_bound_trans = np.copy(B_bound) # shape: (2, n_modes)
    B_bound_trans[0, :] = -B_bound_trans[0, :]
    # negate value at x=-1, because in weak form we have -phi(-1)*du/dx|_{x=-1}

    ## ============================================##
    # generate Keras tensors from these matrices or vectors
    B_tensor = tf.constant(B_trans) # in standard element, shape: (n_quads, n_modes)
    W_mat_tensor = tf.constant(W_mat) # shape of W_mat_tensor: (1, n_quads)
    W_mat_A_tensor = tf.constant(W_mat_A) # shape : (1, n_quads)
    Bd_tensor = tf.constant(Bd_trans) # in standard element, shape; (n_quads, n_modes)

    B_bound_tensor = tf.constant(B_bound_trans) # shape: (2, n_modes)
    # these tensors are needed when computing losses

    del B_trans, Bd_trans, W_mat_A, W_mat, B_bound, B_bound_trans
    # delete these variables, no longer useful

    #+++++++++++++++++++++++++++++++++++#
    # element_size
    this_esize_inv = 1.0/jacob[id_elem,2] # this element_size inverse

    # ======================================= #
    BCL = np.zeros((1,1))
    BCL[0,0] = bc[0, 1] # left boundary condition
    BCL_tensor = tf.constant(BCL) # shape: (1,1)

    BCR = np.zeros((1,1))
    BCR[0,0] = bc[1,1] # right boundary condition
    BCR_tensor = tf.constant(BCR) # shape: (1,1)

    # ====================================== #
    # continuity across element boundaries

    # element-boundary value assembly
    bound_assembly = np.zeros((n_elem-1, n_elem*2))
    for i in range(n_elem-1):
        bound_assembly[i, 2*i+1] = 1.0 # right boundary of element on the left
        bound_assembly[i, 2*(i+1)] = -1.0 # left boundary of element on the right

    bound_assembly_tensor = tf.constant(bound_assembly) # shape: (n_elem-1, n_elem*2)
    # to be used for computing different between values across element boundaries
    del bound_assembly

    # ============================= #
    def loss_generator_Ck(ck):
        """
        Ck loss function generator
        :param ck: integer, >= 0, k value in C_k continuity
        :return: loss function for C_k
        """

        def loss_func_c0():
            """
            compute loss across element boundary for C^0 continuity across element boundaries
            :return: loss value
            """
            # now global_model.u_pred contains u, list of n_elem tensors of shape (n_quads,1)
            if n_elem <= 1:
                return tf.constant(0.0, dtype=tf.float64)

            u_pred = global_model.u_pred
            u_comb = tf.stack(u_pred, 0)  # shape: (n_elem,n_quads,1)
            output_tensor = u_comb[:, 0::n_quads - 1, 0:1]  # shape: (n_elem,2,1), u on element boundaries

            C0_concat = tf.reshape(output_tensor, (n_elem * 2, 1))  # shape: (n_elem*2,1)
            C0_residual = K.dot(bound_assembly_tensor, C0_concat)  # shape: (n_elem-1, 1)
            C0_res_func = loss_manip_func(C0_residual) # shape: (n_elem-1, 1)

            this_loss = tf.reduce_mean(C0_res_func)
            """
            if n_elem > 1:
                this_loss = tf.reduce_mean(C0_res_func)
            else:
                this_loss = tf.constant(0.0, dtype=tf.float64)
            """
            return this_loss

        def loss_func_c1():
            """
            compute loss across element boundary for C^1 continuity across element boundaries
            :return: loss value
            """
            if n_elem <= 1:
                return tf.constant(0.0, dtype=tf.float64)

            # now global_model.u_pred contains u, list of n_elem tensors of shape (n_quads,1)
            #     global_model.dudx contains du/dx, list of n_elem tensors of shape (n_quads,1)
            u_pred = global_model.u_pred
            dudx = global_model.dudx

            u_comb = tf.stack(u_pred, 0) # shape: (n_elem,n_quads,1)
            output_tensor = u_comb[:,0::n_quads-1,0:1] # shape: (n_elem,2,1), u on element boundaries
            dudx_comb = tf.stack(dudx, 0) # shape: (n_elem,n_quads,1)
            grad_dudx = dudx_comb[:,0::n_quads-1,0:1] # shape: (n_elem,2,1), du/dx on element boundaries

            C0_concat = tf.reshape(output_tensor, (n_elem*2,1)) # shape: (n_elem*2,1)
            C1_concat = tf.reshape(grad_dudx, (n_elem*2,1)) # shape: (n_elem*2,1)

            C0_residual = K.dot(bound_assembly_tensor, C0_concat) # shape: (n_elem-1, 1)
            C1_residual = K.dot(bound_assembly_tensor, C1_concat) # shape: (n_elem-1, 1)

            C0_res_func = loss_manip_func(C0_residual) # shape: (n_elem-1, 1)
            C1_res_func = loss_manip_func(C1_residual) # shape: (n_elem-1, 1)

            this_loss = tf.reduce_mean(C0_res_func) + tf.reduce_mean(C1_res_func)
            """
            if n_elem > 1:
                this_loss = tf.reduce_mean(C0_res_func) + tf.reduce_mean(C1_res_func)
            else:
                this_loss = tf.constant(0.0, dtype=tf.float64)
            """
            return this_loss

        def loss_func_c2():
            """
            loss for C^2 continuity across element boundaries
            :return: loss value
            """
            if n_elem <= 1:
                return tf.constant(0.0, dtype=tf.float64)

            # now global_model.u_pred contains u, list of n_elem tensors of shape (n_quads,1)
            #     global_model.dudx contains du/dx, list of n_elem tensors of shape (n_quads,1)
            #     global_model.d2u_dx2 contains d^2u/dx^2, list of n_elem tensors of shape (n_quads,1)
            u_pred = global_model.u_pred
            dudx_pred = global_model.dudx
            d2udx2_pred = global_model.d2u_dx2

            u_comb = tf.stack(u_pred, 0) # shape: (n_elem,n_quads,1)
            u_bound = u_comb[:,0::n_quads-1,0:1] #shape: (n_elem,2,1)
            u_b = tf.reshape(u_bound, (2*n_elem,1)) # shape: (2*n_elem,1)

            dudx_comb = tf.stack(dudx_pred, 0) # shape: (n_elem,n_quads,1)
            dudx_bound = dudx_comb[:,0::n_quads-1,0:1] # shape: (n_elem,2,1)
            dudx_b = tf.reshape(dudx_bound, (2*n_elem,1)) # shape: (2*n_elem,1)

            dudx_2nd_comb = tf.stack(d2udx2_pred, 0) # shape: (n_elem,n_quads,1)
            dudx_2nd_bound = dudx_2nd_comb[:,0::n_quads-1,0:1] # shape: (n_elem,2,1)
            dudx_2nd_b = tf.reshape(dudx_2nd_bound, (2*n_elem,1)) # shape: (2*n_elem,1)

            C0_res = K.dot(bound_assembly_tensor, u_b) # shape: (n_elem-1,1)
            C1_res = K.dot(bound_assembly_tensor, dudx_b) # shape (n_elem-1,1)
            C2_res = K.dot(bound_assembly_tensor, dudx_2nd_b) # shape: (n_elem-1,1)

            C0_res_func = loss_manip_func(C0_res) # shape: (n_elem-1,1)
            C1_res_func = loss_manip_func(C1_res) # shape: (n_elem-1,1)
            C2_res_func = loss_manip_func(C2_res) # shape: (n_elem-1,1)

            this_loss = tf.reduce_mean(C0_res_func) + tf.reduce_mean(C1_res_func) \
                        + tf.reduce_mean(C2_res_func)
            """
            if n_elem > 1:
                this_loss = tf.reduce_mean(C0_res_func) + tf.reduce_mean(C1_res_func) \
                            + tf.reduce_mean(C2_res_func)
            else:
                this_loss = tf.constant(0.0, dtype=tf.float64)
            """
            return this_loss

        def loss_func_ck_default():
            return tf.constant(0.0, dtype=tf.float64)

        # ++++++++++++ #
        if ck == 0:
            return loss_func_c0
        elif ck == 1:
            return loss_func_c1
        elif ck == 2:
            return loss_func_c2
        else:
            print("ERROR: loss_generator_ck() -- C^k continuity with (infinity > k > 2) is not implemented!\n")
            return loss_func_ck_default
        # +++++++++++++ #

    The_Loss_Func_Ck = loss_generator_Ck(ck_k)
    # now The_Loss_Func_Ck is the loss function for computing Ck continuity loss
    #     across element boundaries

    # ======================================= #
    def Equation_Residual(y_true, y_pred):
        """
        actual computation of residual tensor for equation of this element
        :param y_true: label data, shape: (n_quads,1)
        :param y_pred: preduction data, shape: (n_quads,1)
        :return: loss value
        """
        # now global_model.u_pred contains u prediction, list of n_elem tensors of shape (n_quads,1)
        #     global_model.dudx contains du/dx, list of n_elem tensors of shape (n_quads,1)
        #     global_model.d2u_dx2 contains d^2u/dx^2, list of n_elem tensors of shape (n_quads,1)
        d2u_dx2 = global_model.d2u_dx2[id_elem] # shape: (n_quads,1)
        dudx_2nd = tf.reshape(d2u_dx2, (1,n_quads)) # shape: (1,n_quads)

        u = tf.reshape(y_pred, (1,n_quads)) # shape: (1,n_quads)

        y_true_0 = y_true[:, 0]  # y_true_0 has shape (nquads,) or (?,)
        f = tf.reshape(y_true_0, (1, n_quads))  # shape: (1,n_quads)

        res = dudx_2nd - lambda_coeff*u - f # shape: (1,n_quads)
        res_func = loss_manip_func(res) # res_squared, shape: (1,n_quads)

        temp_v1 = res_func * W_mat_tensor # shape: (1,n_quads)
        this_loss = this_esize_inv * tf.reduce_sum(temp_v1) # shape: (), residual equared integral

        return this_loss

    def The_Loss_Func(y_true, y_pred):
        """
        actual computation of loss function for ordinary element, not the first element
        only compute the residual loss of equation for current element
        :param y_true: label data
        :param y_pred: preduction data
        :return: loss value
        """
        T_tot = Equation_Residual(y_true, y_pred)
        # T_tot contains residual-squared integral tensor for equation of this element, shape: ()

        this_loss = Equ_penalty_coeff*T_tot
        return this_loss

    def The_First_Loss_Func(y_true, y_pred):
        """
        actual computation of loss for first element
        compute residual loss of equation for current element, and loss in continuity across element
           boundary, and loss in boundary condition of domain
        :param y_true:
        :param y_pred:
        :return:
        """
        T_tot = Equation_Residual(y_true, y_pred)
        # now T_tot contains residual-squared integral tensor for equation of this element, shape: ()

        # ========================== #
        # domain boundary condition
        # now global_model.u_pred contains y_pred tensor, list of n_elem tensors of shape (n_quads,1)
        u_pred = global_model.u_pred # list of n_elem tensors of shape (n_quads,1)
        u_1st_elem = u_pred[0] # shape: (n_quads,1)
        u_last_elem = u_pred[-1] # shape: (n_quads,1)

        BCL_out = u_1st_elem[0:1,0:1] # u prediction on left boundary, shape: (1,1)
        T_bc_L = BCL_out - BCL_tensor # shape: (1,1)
        bc_res_func_l = loss_manip_func(T_bc_L)

        BCR_out = u_last_elem[n_quads-1:n_quads,0:1] # u prediction on right boundary, shape: (1,1)
        T_bc_R = BCR_out - BCR_tensor # shape: (1,1)
        bc_res_func_r = loss_manip_func(T_bc_R)

        # ========================= #
        # C^k continuity across element boundary
        Ck_loss = The_Loss_Func_Ck()
        # now Ck_loss contains loss value for C_k continuity across element boundaries

        # ========================= #
        this_loss = Equ_penalty_coeff*T_tot \
                    +  BC_ck_penalty_coeff*(tf.reduce_sum(bc_res_func_l)
                                            + tf.reduce_sum(bc_res_func_r)
                                            + Ck_loss ) * n_elem
        return this_loss

    # ====================================== #
    if id_elem == 0:
        return The_First_Loss_Func
    else:
        return The_Loss_Func
    # ====================================== #

def one_elem_loss_generator(id_elem, elem_param, model_info):
    """
    build loss function for one element
    :param id_elem: id of current element and sub-model
    :param elem_param: tuple, (n_modes, n_quads, lambda_coeff, bc, jacobian, domain, ck_k)
                       where n_modes is the number of modes in each element
                             n_quads is the number of quadrature points in each element
                             lambda_coeff is the lambda coefficient
                             bc : numpy array, shape: (2,2); bc[:, 0] contains coordinates of 2 boundaries
                                                             bc[:, 1] contains Dirivhlet BC values of 2 boundaries
                             jacobian: vector containing jacobians of each element, shape: (n_elem,3)
                             domain: vector contains element boundary coordinates, shape: (n_elem+1,)
                             ck_k : k value in C_k continuity
    :param model_info: tuple, (model, sub_model_set)
                       where model: multi_element_model for n_elem elements, overall model of n_elem sub-models
                             sub_model_set: list of sub-models for each element, shape: (n_elem,)
    :return: loss function for the sub-model of this element

    if this is the first element, will compute equation residual of this element, and also of element-boundary
        C^k continuity conditions, as well as global boundary condition loss
    if this is not the first element, will compute only the equation residual of this element
    """
    n_quads, lambda_coeff, bc, jacob, this_domain, ck_k = elem_param
    n_elem = jacob.shape[0] # number of elements

    global_model, sub_model_list = model_info # global_model contains the overall model

    #+++++++++++++++++++++++++++++++++++#
    # element_size
    #this_esize_inv = 1.0/jacob[id_elem,2] # this element_size inverse

    # ======================================= #
    BCL = np.zeros((1,1))
    BCL[0,0] = bc[0, 1] # left boundary condition
    BCL_tensor = tf.constant(BCL) # shape: (1,1)

    BCR = np.zeros((1,1))
    BCR[0,0] = bc[1,1] # right boundary condition
    BCR_tensor = tf.constant(BCR) # shape: (1,1)

    # ====================================== #
    # continuity across element boundaries

    # element-boundary value assembly
    bound_assembly = np.zeros((n_elem-1, n_elem*2))
    for i in range(n_elem-1):
        bound_assembly[i, 2*i+1] = 1.0 # right boundary of element on the left
        bound_assembly[i, 2*(i+1)] = -1.0 # left boundary of element on the right

    bound_assembly_tensor = tf.constant(bound_assembly) # shape: (n_elem-1, n_elem*2)
    # to be used for computing different between values across element boundaries
    del bound_assembly

    # ============================= #
    def loss_generator_Ck(ck):
        """
        Ck loss function generator
        :param ck: integer, >= 0, k value in C_k continuity
        :return: loss function for C_k
        """
        def loss_func_c0():
            """
            compute loss across element boundary for C^0 continuity across element boundaries
            :return: loss value
            """
            # now global_model.u_pred contains u, list of n_elem tensors of shape (n_quads,1)
            if n_elem <= 1:
                return tf.constant(0.0, dtype=tf.float64)

            u_pred = global_model.u_pred
            u_comb = tf.stack(u_pred, 0)  # shape: (n_elem,n_quads,1)
            output_tensor = u_comb[:, 0::n_quads - 1, 0:1]  # shape: (n_elem,2,1), u on element boundaries

            C0_concat = tf.reshape(output_tensor, (n_elem * 2, 1))  # shape: (n_elem*2,1)
            C0_residual = K.dot(bound_assembly_tensor, C0_concat)  # shape: (n_elem-1, 1)
            C0_res_func = loss_manip_func(C0_residual) # shape: (n_elem-1, 1)

            this_loss = tf.reduce_mean(C0_res_func)
            """
            if n_elem > 1:
                this_loss = tf.reduce_mean(C0_res_func)
            else:
                this_loss = tf.constant(0.0, dtype=tf.float64)
            """
            return this_loss

        def loss_func_c1():
            """
            compute loss across element boundary for C^1 continuity across element boundaries
            :return: loss value
            """
            if n_elem <= 1:
                return tf.constant(0.0, dtype=tf.float64)

            # now global_model.u_pred contains u, list of n_elem tensors of shape (n_quads,1)
            #     global_model.dudx contains du/dx, list of n_elem tensors of shape (n_quads,1)
            u_pred = global_model.u_pred
            dudx = global_model.dudx

            u_comb = tf.stack(u_pred, 0) # shape: (n_elem,n_quads,1)
            output_tensor = u_comb[:,0::n_quads-1,0:1] # shape: (n_elem,2,1), u on element boundaries
            dudx_comb = tf.stack(dudx, 0) # shape: (n_elem,n_quads,1)
            grad_dudx = dudx_comb[:,0::n_quads-1,0:1] # shape: (n_elem,2,1), du/dx on element boundaries

            C0_concat = tf.reshape(output_tensor, (n_elem*2,1)) # shape: (n_elem*2,1)
            C1_concat = tf.reshape(grad_dudx, (n_elem*2,1)) # shape: (n_elem*2,1)

            C0_residual = K.dot(bound_assembly_tensor, C0_concat) # shape: (n_elem-1, 1)
            C1_residual = K.dot(bound_assembly_tensor, C1_concat) # shape: (n_elem-1, 1)

            C0_res_func = loss_manip_func(C0_residual) # shape: (n_elem-1, 1)
            C1_res_func = loss_manip_func(C1_residual) # shape: (n_elem-1, 1)

            this_loss = tf.reduce_mean(C0_res_func) + tf.reduce_mean(C1_res_func)
            """
            if n_elem > 1:
                this_loss = tf.reduce_mean(C0_res_func) + tf.reduce_mean(C1_res_func)
            else:
                this_loss = tf.constant(0.0, dtype=tf.float64)
            """
            return this_loss

        def loss_func_c2():
            """
            loss for C^2 continuity across element boundaries
            :return: loss value
            """
            if n_elem <= 1:
                return tf.constant(0.0, dtype=tf.float64)

            # now global_model.u_pred contains u, list of n_elem tensors of shape (n_quads,1)
            #     global_model.dudx contains du/dx, list of n_elem tensors of shape (n_quads,1)
            #     global_model.d2u_dx2 contains d^2u/dx^2, list of n_elem tensors of shape (n_quads,1)
            u_pred = global_model.u_pred
            dudx_pred = global_model.dudx
            d2udx2_pred = global_model.d2u_dx2

            u_comb = tf.stack(u_pred, 0) # shape: (n_elem,n_quads,1)
            u_bound = u_comb[:,0::n_quads-1,0:1] #shape: (n_elem,2,1)
            u_b = tf.reshape(u_bound, (2*n_elem,1)) # shape: (2*n_elem,1)

            dudx_comb = tf.stack(dudx_pred, 0) # shape: (n_elem,n_quads,1)
            dudx_bound = dudx_comb[:,0::n_quads-1,0:1] # shape: (n_elem,2,1)
            dudx_b = tf.reshape(dudx_bound, (2*n_elem,1)) # shape: (2*n_elem,1)

            dudx_2nd_comb = tf.stack(d2udx2_pred, 0) # shape: (n_elem,n_quads,1)
            dudx_2nd_bound = dudx_2nd_comb[:,0::n_quads-1,0:1] # shape: (n_elem,2,1)
            dudx_2nd_b = tf.reshape(dudx_2nd_bound, (2*n_elem,1)) # shape: (2*n_elem,1)

            C0_res = K.dot(bound_assembly_tensor, u_b) # shape: (n_elem-1,1)
            C1_res = K.dot(bound_assembly_tensor, dudx_b) # shape (n_elem-1,1)
            C2_res = K.dot(bound_assembly_tensor, dudx_2nd_b) # shape: (n_elem-1,1)

            C0_res_func = loss_manip_func(C0_res) # shape: (n_elem-1,1)
            C1_res_func = loss_manip_func(C1_res) # shape: (n_elem-1,1)
            C2_res_func = loss_manip_func(C2_res) # shape: (n_elem-1,1)

            this_loss = tf.reduce_mean(C0_res_func) + tf.reduce_mean(C1_res_func) \
                        + tf.reduce_mean(C2_res_func)
            """
            if n_elem > 1:
                this_loss = tf.reduce_mean(C0_res_func) + tf.reduce_mean(C1_res_func) \
                            + tf.reduce_mean(C2_res_func)
            else:
                this_loss = tf.constant(0.0, dtype=tf.float64)
            """
            return this_loss

        def loss_func_ck_default():
            return tf.constant(0.0, dtype=tf.float64)

        # ++++++++++++ #
        if ck == 0:
            return loss_func_c0
        elif ck == 1:
            return loss_func_c1
        elif ck == 2:
            return loss_func_c2
        else:
            print("ERROR: loss_generator_ck() -- C^k continuity with (infinity > k > 2) is not implemented!\n")
            return loss_func_ck_default
        # +++++++++++++ #

    The_Loss_Func_Ck = loss_generator_Ck(ck_k)
    # now The_Loss_Func_Ck is the loss function for computing Ck continuity loss
    #     across element boundaries

    # ======================================= #
    def Equation_Residual(y_true, y_pred):
        """
        actual computation of residual tensor for equation of this element
        :param y_true: label data, shape: (n_quads,1)
        :param y_pred: preduction data, shape: (n_quads,1)
        :return: loss value
        """
        # now global_model.u_pred contains u prediction, list of n_elem tensors of shape (n_quads,1)
        #     global_model.dudx contains du/dx, list of n_elem tensors of shape (n_quads,1)
        #     global_model.d2u_dx2 contains d^2u/dx^2, list of n_elem tensors of shape (n_quads,1)
        d2u_dx2 = global_model.d2u_dx2[id_elem] # shape: (n_quads,1)
        dudx_2nd = tf.reshape(d2u_dx2, (1,n_quads)) # shape: (1,n_quads)

        u = tf.reshape(y_pred, (1,n_quads)) # shape: (1,n_quads)

        y_true_0 = y_true[:, 0]  # y_true_0 has shape (nquads,) or (?,)
        f = tf.reshape(y_true_0, (1, n_quads))  # shape: (1,n_quads)

        res = dudx_2nd - lambda_coeff*u - f # shape: (1,n_quads)
        res_func = loss_manip_func(res) # res_squared, shape: (1,n_quads)

        #temp_v1 = res_func * W_mat_tensor # shape: (1,n_quads)
        #this_loss = this_esize_inv * tf.reduce_sum(temp_v1) # shape: (), residual equared integral
        this_loss = tf.reduce_mean(res_func) # shape: ()

        return this_loss

    def The_Loss_Func(y_true, y_pred):
        """
        actual computation of loss function for ordinary element, not the first element
        only compute the residual loss of equation for current element
        :param y_true: label data
        :param y_pred: preduction data
        :return: loss value
        """
        T_tot = Equation_Residual(y_true, y_pred)
        # T_tot contains residual-squared integral tensor for equation of this element, shape: ()

        this_loss = Equ_penalty_coeff*T_tot
        return this_loss

    def The_First_Loss_Func(y_true, y_pred):
        """
        actual computation of loss for first element
        compute residual loss of equation for current element, and loss in continuity across element
           boundary, and loss in boundary condition of domain
        :param y_true:
        :param y_pred:
        :return:
        """
        T_tot = Equation_Residual(y_true, y_pred)
        # now T_tot contains residual-squared integral tensor for equation of this element, shape: ()

        # ========================== #
        # domain boundary condition
        # now global_model.u_pred contains y_pred tensor, list of n_elem tensors of shape (n_quads,1)
        u_pred = global_model.u_pred # list of n_elem tensors of shape (n_quads,1)
        u_1st_elem = u_pred[0] # shape: (n_quads,1)
        u_last_elem = u_pred[-1] # shape: (n_quads,1)

        BCL_out = u_1st_elem[0:1,0:1] # u prediction on left boundary, shape: (1,1)
        T_bc_L = BCL_out - BCL_tensor # shape: (1,1)
        bc_res_func_l = loss_manip_func(T_bc_L)

        BCR_out = u_last_elem[n_quads-1:n_quads,0:1] # u prediction on right boundary, shape: (1,1)
        T_bc_R = BCR_out - BCR_tensor # shape: (1,1)
        bc_res_func_r = loss_manip_func(T_bc_R)

        # ========================= #
        # C^k continuity across element boundary
        Ck_loss = The_Loss_Func_Ck()
        # now Ck_loss contains loss value for C_k continuity across element boundaries

        # ========================= #
        this_loss = Equ_penalty_coeff*T_tot \
                    +  BC_ck_penalty_coeff*(tf.reduce_sum(bc_res_func_l)
                                            + tf.reduce_sum(bc_res_func_r)
                                            + Ck_loss ) * n_elem
        return this_loss

    # ====================================== #
    if id_elem == 0:
        return The_First_Loss_Func
    else:
        return The_Loss_Func
    # ====================================== #


def multi_elem_loss_generator(elem_param, model_info):
    """
    build list of loss functions
    :param elem_param: tuple, (n_modes, n_quads, lambda_coeff, bc, jacobian, domain, ck_k)
                       where n_modes is the number of modes in each element
                             n_quads is the number of quadrature points in each element
                             lambda_coeff is the lambda coefficient
                             bc : numpy array, shape: (2,2); bc[:, 0] contains coordinates of 2 boundaries
                                                             bc[:, 1] contains Dirivhlet BC values of 2 boundaries
                             jacobian: vector containing jacobians of each element, shape: (n_elem,3)
                                       [:,0] -- J_0e; [:,1] -- J_1e; [:,2] -- J_2e
                             domain: vector contains element boundary coordinates, shape: (n_elem+1,)
                             ck_k : k value in C_k continuity
    :param model_info: tuple, (model, sub_model_set)
                       where model: multi_element_model for n_elem elements, overall model of n_elem sub-models
                             sub_model_set: list of sub-models for each element, shape: (n_elem,)
    :return: list of loss functions for the multi-element model
    """
    _, _, _, jacob, _, _ = elem_param
    n_elem = jacob.shape[0]
    norm_fac = 1.0/n_elem

    loss_func_set = []
    loss_weights = []
    for ix_elem in range(n_elem):
        loss_one = one_elem_loss_generator(ix_elem, elem_param, model_info)
        #loss_one = one_elem_loss_generator(ix_elem, elem_param, basis_info, model_info)
        loss_func_set.append(loss_one)
        loss_weights.append(norm_fac)
    # now loss_func_set contains the list of loss functions
    #     loss_weights contains the list of loss weights

    return (loss_func_set, loss_weights)

class DkModel(keras.Model):
    @tf.function
    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            with tf.GradientTape() as tape_1:
                tape_1.watch(x)
                with tf.GradientTape() as tape_2:
                    tape_2.watch(x)
                    y_pred = self(x, training=True)
                    if not isinstance(y_pred,list):
                        y_pred = [y_pred]
                dudx = tape_2.gradient(y_pred, x)  # list of tensors with shape: (n_quads,1)
            dudx_2nd = tape_1.gradient(dudx, x) # list of n_elem tensors of shape (n_quads,1)

            self.u_pred = y_pred  # store u in this model
            self.dudx = dudx  # store dudx in this model
            self.d2u_dx2 = dudx_2nd # store d2u_dx2 in this model
            loss = self.compiled_loss(y, y_pred)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients,trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def train_step_lbfgs(self, data):
        """
        compute loss and gradients
        :param data: tuple, (x, y), or (input,label)
        :return: tuple, (loss_value, gradients)
                 where loss_value is the loss value
                       gradients is the gradient w.r.t. training variables in model
        """
        x, y = data

        with tf.GradientTape() as tape:
            with tf.GradientTape() as tape_1:
                tape_1.watch(x)
                with tf.GradientTape() as tape_2:
                    tape_2.watch(x)
                    y_pred = self(x, training=True)
                    if not isinstance(y_pred,list):
                        y_pred = [y_pred]
                dudx = tape_2.gradient(y_pred, x)  # list of tensors with shape: (n_quads,1)
            dudx_2nd = tape_1.gradient(dudx, x) # list of n_elem tensors of shape (n_quads,1)

            self.u_pred = y_pred  # store u in this model
            self.dudx = dudx  # store dudx in this model
            self.d2u_dx2 = dudx_2nd # store d2u_dx2 in this model
            loss = self.compiled_loss(y, y_pred)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        return loss, gradients

def input_scale_func_generator(id_elem, jacob):
    """
    function generator for scaling input data to [-1, 1]
    :param id_elem: id of current element
    :param jacob: jacobian data for all elements, shape: (n_elem, 3)
                  jacob[:,0] contains J_0e
                  jacob[:,1] contains J_1e
                  jacob[:,2] contains J_2e
                  scaling is x = J_1e * xi + J_0e, where x is physical coordinate, xi is in [-1,1]
    :return: scaling function
    """
    J0 = jacob[id_elem, 0]
    J1 = jacob[id_elem, 1]
    J1_inv = 1.0/J1

    def scale_func(x):
        return J1_inv*(x-J0)

    return scale_func

def build_submodel_A(id_elem, layers, activ, jacob):
    """
    build sub-model for one element
    :param: id_elem, id of this sub-model
    :param layers: list of nodes for each layer
    :param activ: list of activation functions for each layer
    :param jacob: jacobian, shape: (n_elem,3)
    :return: the sub-model
    """
    n_layers = len(layers)
    this_input = keras.layers.Input(shape=[layers[0]], name="input_"+str(id_elem)) # input layer

    # +++++++++++++++++++++++++++++++++ #
    # scaling layer, lambda layer
    # add lambda layer to re-scale input data to [-1,1]
    func = input_scale_func_generator(id_elem, jacob)
    lambda_layer = keras.layers.Lambda(func, name="lambda_"+str(id_elem))(this_input)
    # now the lambda layer is added behind the input to re-scale data to [-1,1]

    # first hidden layer
    if n_layers == 2:
        this_name = "output_" + str(id_elem)
    else:
        this_name = "hidden_1_" + str(id_elem)
    this_layer = keras.layers.Dense(layers[1], activation=activ[1], name=this_name)(lambda_layer)

    for i in range(n_layers-2):
        if i == n_layers-3:
            this_name = "output_" + str(id_elem)
        else:
            this_name = "hidden_" + str(i+2) + "_" + str(id_elem)
        this_layer = keras.layers.Dense(layers[i+2], activation=activ[i+2],
                                        name=this_name)(this_layer)

    # now this_layer contains the output layer for sub-model
    this_model = DkModel(inputs=[this_input], outputs=[this_layer])
    return this_model

def build_multi_element_model(N_elem, layers, activ, jacob):
    """
    build model for multiple elements
    :param N_elem: number of elements, >=1
    :param layers: layers vector in each one-element model
    :param activ: activation vector in each one-element model
    :return: multi-element model
    """
    model_set = []
    input_set = []
    output_set = []
    for ix_elem in range(N_elem):
        elem_model = build_submodel_A(ix_elem, layers, activ, jacob)
        model_set.append(elem_model)
        input_set.append(elem_model.input)
        output_set.append(elem_model.output)
    # now model_set contains the list of element models
    #     input_set contains the list of input tensors
    #     output_set contains the list output tensors

    this_model = DkModel(inputs=input_set, outputs=output_set)
    # now this_model contains the multi-input multi-output model

    return (this_model, model_set)

def build_model(layers, activ, n_quads, lambda_coeff, bc, jacob, this_domain, ck_k):
    n_elem = jacob.shape[0]
    my_model, sub_model_set = build_multi_element_model(n_elem, layers, activ, jacob)
    # now my_model contains the multi-element DNN model

    #basis_info = get_basis_info(n_modes, n_quads) # basis data
    elem_param = (n_quads, lambda_coeff, bc, jacob, this_domain, ck_k)
    model_info = (my_model, sub_model_set)
    #loss_func_list, loss_weight_list = multi_elem_loss_generator(elem_param, basis_info, model_info)
    loss_func_list, loss_weight_list = multi_elem_loss_generator(elem_param, model_info)
    # now loss_func_list contains list of loss functions
    #     loss_weight_list contains list of loss weights

    # compile model
    my_model.compile(optimizer='adam',
                     loss=loss_func_list, loss_weights = loss_weight_list,
                     metrics=['accuracy'])
    return my_model

def get_learning_rate(model):
    lr = K.get_value(model.optimizer.lr)
    return float(lr)

def set_learning_rate(model, LR):
    K.set_value(model.optimizer.lr, LR)

def output_data(file, z_coord, v_pred, v_true):
    dim = len(z_coord)
    with open(file, 'w') as fileout:
        fileout.write("variables = x, value-predict, value-true, error\n")
        for line in range(dim):
            fileout.write("%.14e %.14e %.14e %.14e\n" % (z_coord[line], v_pred[line], v_true[line],
                                                      np.abs(v_pred[line]-v_true[line])))

def save_history(hist_file, hist_obj):
    """
    save training history
    :param hist_file: file name to store history data
    :param train_hist: history object from keras.model.fit()
    :return:
    """
    # convert the hist.history dict to pandas DataFrame
    hist_df = pd.DataFrame(hist_obj.history)

    # save to csv:
    with open(hist_file, mode='w') as f:
        hist_df.to_csv(f)

def save_run_param(param_file, param_data):
    """
    save parameters for this run
    :param param_file: file name to output to
    :param param_data: tuple, (domain, dnn_struct, mode_quad, to_read_init_weight,
                                _epsilon, Ck_cont, train_param, lr, elapsed_time,
                                penalty, coeff_b_logcosh)
                              where domain = (domain_x, domain_y)
                                    dnn_struct = (layers, activations)
                                    mode_quad = (n_modes, n_quads)
                                    to_read_init_weight : True or False
                                    _epsilon : epsilon
                                    Ck_cont : Ck continuity
                                    train_param = (batch_size, early_stop_patience, max_epochs)
                                    lr = (default_learning_rate, actual_learning_rate)
                                    elapsed_time = (training_elapsed_time, prediction_elapsed_time)
                                    soln_param = (lambda_coeff, aa)
                                    penalty = (BC_ck_Penalty_Coeff,Equ_Penalty_Coeff) : penlaty parameter for (bc,ck)-loss
                                    coeff_b_logcosh : coefficient in log(cosh(x)) function
    :return:
    """
    domain, dnn_struct, mode_quad, to_read_init_weight, eps, ck, train_param, \
    lr, elap_time, soln_param, penalty, err_soln = param_data
    Layers, Activ = dnn_struct
    n_modes, n_quads = mode_quad
    batch_size, patience, max_epochs = train_param
    default_lr, actual_lr = lr
    train_elapsed_time, prediction_elapsed_time = elap_time
    lambda_coeff, aa = soln_param
    bc_penalty, equ_penalty = penalty
    linf_err, l2_err = err_soln

    with open(param_file,"w") as fileout:
        fileout.write("domain = " + str(domain) + "\n")
        fileout.write("layers = " + str(Layers) + "\n")
        fileout.write("activations = " + str(Activ) + "\n")
        fileout.write("N_mode = %d\n" % (n_modes))
        fileout.write("N_quad = %d\n" % (n_quads))
        fileout.write("to_read_init_weight = " + str(to_read_init_weight) + "\n")
        fileout.write("_epsilon = %.14e\n" % eps)
        fileout.write("Ck_continuity = %d\n" % ck)
        fileout.write("batch_size = %d\n" % batch_size)
        fileout.write("early_stop_patience = %d\n" % patience)
        fileout.write("maximum_epochs = %d\n" % max_epochs)
        fileout.write("default_learning_rate = %.12e\n" % default_lr)
        fileout.write("actual_learning_rate = %.12e\n" % actual_lr)
        fileout.write("training_elapsed_time (seconds) = %.14e\n" % train_elapsed_time)
        fileout.write("prediction_elapsed_time (seconds) = %.14e\n" % pred_elapsed_time)
        fileout.write("solution parameter: lambda = %.12e\n" % lambda_coeff)
        fileout.write("solution parameter: aa = %.12e\n" % aa)
        fileout.write("bc/Ck-continuity penalty coefficient: %.12e\n" % bc_penalty)
        fileout.write("equation penalty coefficient: %.12e\n" % equ_penalty)
        fileout.write("solution errors -- linf-error = %.14e,  l2-error = %.14e\n" % (linf_err, l2_err))

def gen_training_data(n_colloc, jacob):
    """
    n_colloc: number of collocation points per element
    jacob: jacobian data, shape: (n_elem, 3)
    """
    n_elem = jacob.shape[0]

    In_data = []
    Label_data = []
    zz = np.linspace(-1.0, 1.0, n_colloc)

    for i in range(n_elem):
        A = jacob[i,1]
        B = jacob[i,0] #(domain[i] + domain[i+1])*0.5

        tmp_in_data = np.zeros((n_colloc, 1))
        tmp_in_data[:, 0] = zz * A + B
        In_data.append(tmp_in_data)

        tmp_label = the_source_term(Lambda_Coeff, tmp_in_data)
        Label_data.append(tmp_label)

    return (In_data, Label_data)

def gen_predict_data(n_pred, this_domain):
    """
    n_pred: number of prediction data points
    this_domain: domain, list
    """
    n_elem = len(this_domain) - 1

    data_in = []
    for i in range(n_elem):
        tmp_data = np.zeros((n_pred, 1))
        tmp_data[:, 0] = np.linspace(domain[i], domain[i+1], n_pred)
        data_in.append(tmp_data)

    return data_in


if __name__ == '__main__':

    # files
    base_path = './data/Helm1D_LocDNN_DGM_Adam_UniformColloc'
    my_util.mkdir(base_path,parent_flag=True)
    problem = 'helm1d_locdnn_'
    method = 'dgm_adam'
    file_base = base_path + '/' + problem + method

    solution_file = file_base + '_soln.dat'
    parameter_file = file_base + '_param.dat'
    model_weight_file = file_base + '_weights.hd5'
    model_init_weight_file = file_base + '_init_weights.hd5'
    train_history_file = file_base + '_history.csv'

    Ck_cont = CK
    Lambda_Coeff = LAMBDA_COEFF #np.float(2.0)

    # number of entries in input data: n_quads*n_elem
    #N_input = N_quad*N_elem
    batch_size = N_quad

    early_stop_patience = 1000
    max_epochs = MAX_EPOCHS

    bc_data = np.zeros((2,2))
    bc_data[0, 0] = domain[0] # coordinate of left boundary
    bc_data[0, 1] = the_anal_soln(bc_data[0,0]) # boundary value of left boundary
    bc_data[1, 0] = domain[-1] # coordinate of right boundary
    bc_data[1, 1] = the_anal_soln(bc_data[1, 0])

    jacobian = calc_jacobian(domain)
    # now jacobian contains the vector of jacobians, shape: (n_elem,3)

    the_DNN = build_model(layers, activations,
                          N_quad, Lambda_Coeff, bc_data, jacobian, domain, Ck_cont)
    if to_read_init_weight and my_util.is_file(model_init_weight_file):
        the_DNN.load_weights(model_init_weight_file)

    default_lr = get_learning_rate(the_DNN)
    print("Default learning rate: %f" % default_lr)
    if LR_factor > 1.0 or LR_factor < 1.0:
        LR = LR_factor * default_lr
        set_learning_rate(the_DNN, LR)
    else:
        LR = default_lr

    # generate training data
    In_data, Label_data = gen_training_data(N_quad, jacobian)

    from timeit import default_timer as timer
    begin_time = timer()

    #klbfgs.lbfgs_train(the_DNN, In_data, Label_data, max_epochs)

    early_stop = keras.callbacks.EarlyStopping(monitor='loss', mode='min',
                                     verbose=1,
                                     patience=early_stop_patience,
                                     restore_best_weights=True)
    train_hist = the_DNN.fit(In_data, Label_data,
                            epochs=max_epochs, batch_size=batch_size, shuffle=False,
                            callbacks=[early_stop])

    end_time = timer()
    train_elapsed_time = end_time - begin_time
    print("Training time elapsed (seconds): %.14e" % (train_elapsed_time))

    # +++++++++++++++++++++++++++++++ #
    # save weights to file
    time_stamp = my_util.get_timestamp()
    save_weight_file = model_weight_file + time_stamp
    save_param_file = parameter_file + time_stamp
    save_init_weight_file = model_init_weight_file + time_stamp

    the_DNN.save_weights(save_weight_file, save_format="h5")
    if to_read_init_weight == True:
        my_util.rename_file(model_init_weight_file, save_init_weight_file)
    my_util.copy_file(save_weight_file, model_init_weight_file)

    # save training history
    #hist_file = train_history_file + time_stamp
    #save_history(hist_file, train_hist)

    # prediction
    #N_pred = 201
    data_in = gen_predict_data(N_pred, domain)

    pred_start_time = timer()
    Soln = the_DNN.predict(data_in)
    pred_end_time = timer()
    pred_elapsed_time = pred_end_time - pred_start_time
    print("Prediction time elapsed (seconds): %.14e" % (pred_elapsed_time))

    A_in_data = np.concatenate(data_in, 0) # shape: (N_elem*N_quad, 1)
    pred_in_data = np.reshape(A_in_data, (N_elem*N_pred,))
    A_soln = np.concatenate(Soln, 0) # shape: (N_elem*N_quad, 1)
    Soln_flat = np.reshape(A_soln, (N_elem*N_pred,))

    Exact_soln = the_anal_soln(A_in_data)
    Exact_soln_flat = np.reshape(Exact_soln, (N_elem*N_pred,))

    soln_file = solution_file + time_stamp
    output_data(soln_file, pred_in_data, Soln_flat, Exact_soln_flat)

    error = np.abs(Soln_flat - Exact_soln_flat)
    linf_err = np.amax(error)
    l2_err = np.std(error)
    print("maximum error = %.14e,  l2-error = %.14e" % (linf_err, l2_err))

    #+++++++++++++++++++++++++++++++++++++++++++#
    # save parameters
    param_data = (domain, (layers, activations), (N_modes, N_quad), to_read_init_weight,
                  _epsilon, Ck_cont, (batch_size, early_stop_patience, max_epochs), (default_lr, LR),
                  (train_elapsed_time, pred_elapsed_time), (Lambda_Coeff, 0.0),
                  (BC_ck_penalty_coeff,Equ_penalty_coeff), (linf_err, l2_err)
                 )
    save_run_param(save_param_file, param_data)





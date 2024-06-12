import jax
import jax.numpy as jnp
import jax.random as jr
import jaxopt
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
import os
import random
import itertools

# Uncomment and modify the below variables accordingly depending on available resources
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

jax.config.update("jax_enable_x64", True)
seed = jr.PRNGKey(0)

# Initialize parameters
ELEM_Q = 1.602e-19
BOLTZ = 1.380662e-23
EPS_0 = 8.864-12 * 1e-9
EPS_S = 11.7 * EPS_0
T = 300
KT_Q = BOLTZ * T / ELEM_Q
NI = 1e10 * 1e-21

def MLP(layers: list[int] = [1, 64, 1], activation: callable = jnp.tanh):
    def init_params(key):
        def _init(key, d_in, d_out):
            w = jr.normal(key, shape=(d_in, d_out)) * jnp.sqrt(2 / (d_in + d_out))
            b = jnp.zeros((d_out,))
            return [w, b]

        keys = jr.split(key, len(layers) - 1)
        params = list(map(_init, keys, layers[:-1], layers[1:]))
        return params

    def apply(params, inputs):
        for W, b in params[:-1]:
            outputs = inputs @ W + b
            inputs = activation(outputs)
        W, b = params[-1]
        outputs = inputs @ W + b
        return outputs

    return init_params, apply

key, subkey = jr.split(seed)
init_params, apply = MLP([2, 100, 100, 100, 100, 2])
params = init_params(subkey)

@jax.jit
def pinn(params, x, Nd, Vd_vec):
    V_contact = KT_Q * jnp.log(Nsd / NI)

    input = jnp.hstack([jnp.atleast_1d(x), jnp.atleast_1d(Vd_vec)])
    pinn_temp = apply(params, input).squeeze()
    
    pot = pinn_temp[0]
    log_charge = jnp.log(Nsd) + x * (1-x) * pinn_temp[1]
    
    return jnp.hstack([pot, log_charge])

@jax.jit
def pinn_x(params, x, Nd, Vd_vec):
    return jax.jacfwd(pinn, 1)(params, x, Nd, Vd_vec) / L

@jax.jit
def pinn_xx(params, x, Nd, Vd_vec):
    return jax.jacfwd(pinn_x, 1)(params, x, Nd, Vd_vec) / L

@jax.jit
def compute_derivatives(f, delta_X):
    # Compute the first derivative using central differences for internal points
    delta_X = delta_X * L
    df = jnp.zeros_like(f)
    df = df.at[0].set((f[1] - f[0]) / delta_X)  # forward difference at the first point
    df = df.at[1:-1].set((f[2:] - f[:-2]) / (2 * delta_X))
    df = df.at[-1].set((f[-1] - f[-2]) / delta_X)  # backward difference at the last point
    
    return df

@jax.jit
def mse_cont(params, x, Nd, Vd_vec):
    pred =  jax.vmap(pinn, in_axes=(None, 0, 0, 0 ))(params, x, Nd, Vd_vec)
    pred_x =  jax.vmap(pinn_x, in_axes=(None, 0, 0, 0 ))(params, x, Nd, Vd_vec)
    pred_xx =  jax.vmap(pinn_xx, in_axes=(None, 0, 0, 0 ))(params, x, Nd, Vd_vec)   
    
    res = pred_x[:,1] * pred_x[:,0] + pred_xx[:,0] - KT_Q * ( (pred_x[:, 0])**2 + pred_xx[:, 1] )
    mse = ((res) ** 2).mean() * 1e2
    
    return mse

@jax.jit 
def mse_poi(params, x, Nd, Vd_vec):
    pred =  jax.vmap(pinn, in_axes=(None, 0, 0, 0))(params, x, Nd, Vd_vec)
    pred_xx =  jax.vmap(pinn_xx, in_axes=(None, 0, 0, 0))(params, x, Nd, Vd_vec)
    
    Nd_original = Nd * (np.log(Nsd_max) - np.log(Nsd_min)) + np.log(Nsd_min)
    Nd_original = jnp.exp(Nd_original)
    
    res = pred_xx[:, 0] + (ELEM_Q/EPS_S)  * ( Nd_original -  jnp.exp(pred[:, 1]))
    mse = (res ** 2).mean()
    
    return mse
    
@jax.jit
def mse_data(params, x, Nd, Vd_vec, train_charge_list, train_pot_list):
    mse_charge = 0
    mse_pot = 0
    for i in range(Nsample):
        pred = jax.vmap(pinn, in_axes=(None, 0, 0, 0))(params, x, Nd_input_list[:, i], Vd_input_list[i,:])
        mse_charge += (((jnp.exp(pred[:, 1]) - train_charge_list[:,i])/train_charge_list[:,i])**2).mean()
        mse_pot += (((pred[:, 0] - train_pot_list[:,i])/train_pot_list[:,i])**2).mean()

    mse_charge /= Nsample
    mse_pot /= Nsample

    return (mse_charge ** 0.5 + mse_pot ** 0.5)
    
@jax.jit
def mse_bc(params, x, Nd, Vd_vec, Nsd, Vd):
    pred = jax.vmap(pinn, in_axes=(None, 0, 0, 0))(params, x, Nd, Vd_vec)
    V_contact = KT_Q * jnp.log(Nsd / NI)
    
    mse_pot = ((pred[0,0] - V_contact))**2 + ((pred[-1, 0] - V_contact - Vd))**2 
    mse = (mse_pot)
    
    return mse

# Need Neumann BC at the S/C/D interfaces
@jax.jit
def mse_ifc(params, x, Nd, Vd_vec):
    pred = jax.vmap(pinn_x, in_axes=(None, 0, 0, 0))(params, x, Nd, Vd_vec)

    # Apply Neumann BC numerically
    mse_pot_ifc = (pred[interface_idx - 1, 0] - 2 * pred[interface_idx , 0] + pred[interface_idx + 1, 0])**2 + (pred[501 - interface_idx - 1, 0] - 2 * pred[501 - interface_idx , 0] +  pred[501 - interface_idx + 1, 0])**2
    mse_charge_ifc = (pred[interface_idx - 1, 1] - 2 * pred[interface_idx , 1] + pred[interface_idx + 1, 1])**2 + \
                     (pred[501 - interface_idx - 1, 1] - 2 * pred[501 - interface_idx , 1] + pred[501 - interface_idx + 1, 1])**2
    
    mse = mse_pot_ifc + mse_charge_ifc
    return mse

    
@jax.jit
def mse_total(params, x, Nd, Vd_vec, train_charge_list, train_pot_list, diff_train_charge_list, diff_train_pot_list, Nsd, Vd):
    w_cont = 1
    w_poi = 1
    w_bc = 1
    w_data = 1
    w_ifc = 1

    MSE_cont_val = mse_cont(params, x, Nd, Vd_vec)
    MSE_poi_val = mse_poi(params, x, Nd, Vd_vec)
    MSE_bc_val = mse_bc(params, x, Nd, Vd_vec, Nsd, Vd)
    MSE_data_val = mse_data(params, x, Nd, Vd_vec, train_charge_list, train_pot_list)
    MSE_ifc_val = mse_ifc(params, x, Nd, Vd_vec)
    
    mse = w_cont * MSE_cont_val + w_poi * MSE_poi_val + w_bc * MSE_bc_val 
    mse += w_data * MSE_data_val + w_ifc * MSE_ifc_val
    return mse
    
@jax.jit
def loss(params, x, Nd, Vd_vec, train_charge_list, train_pot_list,  diff_train_charge_list, diff_train_pot_list, Nsd, Vd):
    loss = mse_total(params, x, Nd, Vd_vec, train_charge_list, train_pot_list,  diff_train_charge_list, diff_train_pot_list, Nsd, Vd)
    return loss

x = jnp.linspace(0, 1, 501)
dx = x[1] - x[0]
opt = jaxopt.LBFGS(loss, history_size=50 )

@jax.jit
def step(params, state, x, Nd, Vd_vec, train_charge_list, train_pot_list,  diff_train_charge_list, diff_train_pot_list, Nsd, Vd):
    params, state = opt.update(params, state, x, Nd, Vd_vec, train_charge_list, train_pot_list,  diff_train_charge_list, diff_train_pot_list, Nsd, Vd)
    return params, state

def get_train_profile(train_data, Vd_list):
    
    train_pot_list = np.zeros((501, Nsample))
    train_charge_list = np.zeros((501, Nsample))
    diff_train_pot_list = np.zeros((501, Nsample))
    diff_train_charge_list = np.zeros((501, Nsample))
    
    for i in range(Nsample):
        Vd_temp = sample_list[i, 0]
        Nsd_temp = sample_list[i, 1]
        Nch_temp = sample_list[i, 2]
        
        # Get training data 
        curr_data = train_data[(train_data[:, 0] == Nsd_temp * 1e21) & (train_data[:, 1] == Nch_temp * 1e21) & \
                (train_data[:, 2] == Lch ) & (train_data[:, 3] == Vd_temp)]
        
        train_pot_list[:,i] = np.array(curr_data[0, 5:506])
        train_charge_list[:,i] = np.array(curr_data[0, 506:1008] * 1e-27)

        diff_train_pot_list[:,i] = compute_derivatives(train_pot_list[:,i], dx)
        diff_train_charge_list[:,i] = compute_derivatives(train_charge_list[:,i], dx)

    return jnp.array(train_pot_list), jnp.array(train_charge_list), jnp.array(diff_train_pot_list), jnp.array(diff_train_charge_list)


# Import training data
file_name = "DD_full_data_Lsd_20.dat"
train_data = np.loadtxt(file_name)

loss_traj = []
print("LBFGS running...")
tic = time.time()

Lsd = 20
log_file = open('log_1D_Poi_Cont_with_Neumann_BC.txt', 'w')

Vd_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
Nsd_list = [1e-2]
Nch_list = [1e-4]
Vd_input_list = jnp.vstack([jnp.linspace(0, Vd, 501) for Vd in Vd_list])
sample_list = jnp.array(list(itertools.product(Vd_list, Nsd_list, Nch_list)))
Nsample = len(sample_list)
it = 0

Lch = 30
L = Lch + 2 * Lsd
Nsd_max = (np.array(Nsd_list)).max()
Nsd_min = (np.array(Nch_list)).min()

interface_idx = round(Lsd / (dx * L))

Nd_input_list = np.zeros((501, Nsample))

for i in range(Nsample):
    Nd_input_list[:, i] = sample_list[i, 1] # Nsd
    Nd_input_list[interface_idx+1:501 - interface_idx, i] = sample_list[i, 2] # Nch

    # Interface Doping
    Nd_input_list[interface_idx+1, i] = (sample_list[i, 1] + sample_list[i, 2]) / 2 # S/C interface
    Nd_input_list[501 - interface_idx - 1, i] = (sample_list[i, 1] + sample_list[i, 2]) / 2 # C/D interfacce

Nd_input_list = jnp.array(  ( np.log(Nd_input_list) - np.log(Nsd_min) )  / (np.log(Nsd_max) - np.log(Nsd_min))  )

train_pot_list, train_charge_list, diff_train_pot_list, diff_train_charge_list = get_train_profile(train_data, sample_list)

batch_size = Nsample
    
isample_rand_list = np.array(range(0, Nsample))

while True:

    if it >= 20 * batch_size:
        break
    
    isample = it % batch_size
    if isample == 0:
        random.shuffle(isample_rand_list)
            
    Vd = sample_list[isample_rand_list[isample], 0]
    Nsd = sample_list[isample_rand_list[isample], 1]
    Nch = sample_list[isample_rand_list[isample], 2]
    
    Nd = Nd_input_list[:, isample_rand_list[isample]]
    
    Vd_input = Vd_input_list[round(Vd / 0.2), :]
    
    # Get training data 
    train_pot = train_pot_list[:, isample_rand_list[isample]]
    train_charge = train_charge_list[:, isample_rand_list[isample]]

    state = opt.init_state(params, x, Nd, Vd_input, train_charge_list, train_pot_list,  diff_train_charge_list, diff_train_pot_list, Nsd, Vd)

    for i in range(0, 1000):
        params, state = step(params, state, x, Nd, Vd_input, train_charge_list, train_pot_list,  diff_train_charge_list, diff_train_pot_list, Nsd, Vd)
    
    MSE_cont_val = mse_cont(params, x, Nd, Vd_input)
    MSE_poi_val = mse_poi(params, x, Nd, Vd_input)
    MSE_bc_val = mse_bc(params, x, Nd, Vd_input, Nsd, Vd)
    MSE_data_val = mse_data(params, x, Nd, Vd_input, train_charge_list, train_pot_list)
    MSE_ifc_val = mse_ifc(params, x, Nd, Vd_input)
    
    pinn_loss = state.value
    loss_traj.append(pinn_loss)

    log_msg= f"Nsd: {Nsd:.1e}, Nch: {Nch:.1e}, Vd: {Vd:.2f} it: {it}, loss: {pinn_loss:.5e} mse_cont: {MSE_cont_val:.5e}, mse_poi: {MSE_poi_val:.5e}, mse_bc: {MSE_bc_val:.5e}, mse_ifc: {MSE_ifc_val:.5e}, mse_data: {MSE_data_val:.5e}"
    print(log_msg)
    log_file.write(log_msg)
        
    if MSE_data_val < 1e-4:
        print("Stopping criteria met MSE_data_val < 1e-5")
        break;

    it += 1 

toc = time.time()
log_file.write("Elapsed time = {} s\n".format(toc - tic))
log_file.close()
print("Elapsed time = {} s".format(toc - tic))

# Save loss_trajectory and trained params 
with open('loss_traj_1D_Poi_Cont_with_Neumann_BC.pkl', 'wb') as f:
    pickle.dump(loss_traj, f)

with open('params_1D_Poi_Cont_with_Neumann_BC.pkl', 'wb') as f:
    pickle.dump(params, f)

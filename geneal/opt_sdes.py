import numpy as np
from tqdm import tqdm
import jax
import copy
import jax.numpy as jnp
from losses import *


def SGD(nit, eta, problem):
  #initializing variables
  name = problem.index
  x = []

  g_x = problem.g_x()
  g_x = jax.jit(g_x)
  
  Sigma = problem.Sigma 
  d = problem.x0.shape[0]
  comparison_stat = np.zeros((nit,))
  loss_hist = np.zeros((nit,))

  #appending first stats
  x_it = copy.deepcopy(problem.x0) 
  x.append(np.array(x_it).flatten())
  comparison_stat[0] = jnp.linalg.norm(g_x(x_it,name))+ jnp.linalg.norm(x_it)
  loss_hist[0] = problem.loss(x_it,name)

  #loop
  for k in tqdm(range(nit-1)):
    #update
    Z = jnp.array(Sigma@np.random.normal(size=(d,)))
    grad_local = g_x(x_it,name)
    x_it = x_it - eta*(grad_local+Z)

    #saving stats
    x.append(np.array(x_it).flatten())
    comparison_stat[k+1] = jnp.linalg.norm(grad_local) + jnp.linalg.norm(x_it)
    loss_hist[k+1] = problem.loss(x_it,name)
  return np.array(x),np.array(loss_hist), np.array(comparison_stat)


def SGD_SDE(nit, eta, dt,problem):
  name = problem.index
  nit_sde = int(nit*eta/dt)
  #initializing variables
  x = []
  comparison_stat = np.zeros((nit_sde,))
  loss_hist = np.zeros((nit_sde,))

  g_x = problem.g_x()
  g_x = jax.jit(g_x)

  d = problem.x0.shape[0]
  Sigma = problem.Sigma
  #appending first stats
  x_it = copy.deepcopy(problem.x0) 
  x.append(np.array(x_it).flatten())
  comparison_stat[0] = jnp.linalg.norm(g_x(x_it,name))+ jnp.linalg.norm(x_it)
  loss_hist[0] = problem.loss(x_it,name)

  #loop
  for k in tqdm(range(nit_sde-1)):
    #update
    delta_W = jnp.sqrt(dt)*np.random.normal(size=(d,))
    grad_local = g_x(x_it,name)
    x_it = x_it - dt*grad_local + jnp.array(jnp.sqrt(eta)*Sigma@delta_W)

    #saving stats
    x.append(np.array(x_it).flatten())
    comparison_stat[k+1] = jnp.linalg.norm(grad_local) + jnp.linalg.norm(x_it)
    loss_hist[k+1] = problem.loss(x_it,name)
  return np.array(x),np.array(loss_hist), np.array(comparison_stat)


def USAM(nit, eta, rho, problem): # THIS IS USAM

  name = problem.index
  #initializing variables
  x = []
  d = problem.x0.shape[0]
  comparison_stat = np.zeros((nit,))
  loss_hist = np.zeros((nit,))

  g_x = problem.g_x()
  g_x = jax.jit(g_x)

  #appending first stats
  x_it = copy.deepcopy(problem.x0) 
  x.append(np.array(x_it).flatten())
  comparison_stat[0] = jnp.linalg.norm(g_x(x_it,name))+ jnp.linalg.norm(x_it)
  loss_hist[0] = problem.loss(x_it,name)

  #loop
  for k in tqdm(range(nit-1)):
    #update
    Z = jnp.array(problem.Sigma@np.random.normal(size=(d,)))
    grad_local = g_x(x_it,name)
    x_it = x_it - eta*(g_x(x_it + rho*g_x(x_it,name) + Z,name)+Z)

    #saving stats
    x.append(np.array(x_it).flatten())
    comparison_stat[k+1] = jnp.linalg.norm(grad_local) + jnp.linalg.norm(x_it)
    loss_hist[k+1] = problem.loss(x_it,name)
  return np.array(x),np.array(loss_hist), np.array(comparison_stat)


def SAM(nit, eta, rho,problem): 
  #initializing variables
  name = problem.index
  x = []
  d = problem.x0.shape[0]
  comparison_stat = np.zeros((nit,))
  loss_hist = np.zeros((nit,))

  g_x = problem.g_x()
  g_x = jax.jit(g_x)

  #appending first stats
  x_it = copy.deepcopy(problem.x0) 
  x.append(np.array(x_it).flatten())
  comparison_stat[0] = jnp.linalg.norm(g_x(x_it,name))+ jnp.linalg.norm(x_it)
  loss_hist[0] = problem.loss(x_it,name)

  #loop
  for k in tqdm(range(nit-1)):
    #update
    Z = jnp.array(problem.Sigma@np.random.normal(size=(d,)))
    grad_local = g_x(x_it,name)
    x_it = x_it - eta*(g_x(x_it + (rho/jnp.linalg.norm(grad_local))*grad_local+ Z,name)+Z)

    #saving stats
    x.append(np.array(x_it).flatten())
    comparison_stat[k+1] = jnp.linalg.norm(grad_local) + jnp.linalg.norm(x_it)
    loss_hist[k+1] = problem.loss(x_it,name)
  return np.array(x),np.array(loss_hist), np.array(comparison_stat)




def USAM_SDE1(nit, eta, rho, dt,problem):
  name = problem.index
  nit_sde = int(nit*eta/dt)
  #initializing variables
  x = []
  comparison_stat = np.zeros((nit_sde,))
  loss_hist = np.zeros((nit_sde,))

  g_x = problem.g_x()
  g_x = jax.jit(g_x)
  H_x = problem.H_x()
  H_x = jax.jit(H_x)

  d = problem.x0.shape[0]
  #appending first stats
  x_it = copy.deepcopy(problem.x0) 
  x.append(np.array(x_it).flatten())
  comparison_stat[0] = jnp.linalg.norm(g_x(x_it,name))+ jnp.linalg.norm(x_it)
  loss_hist[0] = problem.loss(x_it,name)

  #loop
  for k in tqdm(range(nit_sde-1)):
    #update
    delta_W = np.sqrt(dt)*np.random.normal(size=(d,))
    grad_local = g_x(x_it,name)
    hess_local = H_x(x_it,name)
    grad_tilde = grad_local+rho*hess_local@grad_local
    x_it = x_it - dt*grad_tilde + jnp.array(np.sqrt(eta)*(np.eye(d)+rho*hess_local)@problem.Sigma@delta_W) # This is USAM: Original Antonio Version

    #saving stats
    x.append(np.array(x_it).flatten())
    comparison_stat[k+1] = jnp.linalg.norm(grad_local) + jnp.linalg.norm(x_it)
    loss_hist[k+1] = problem.loss(x_it,name)
  return np.array(x),np.array(loss_hist), np.array(comparison_stat)


def SAM_SDE1(nit, eta, rho, dt,problem):
  name = problem.index
  nit_sde = int(nit*eta/dt)
  #initializing variables
  x = []
  comparison_stat = np.zeros((nit_sde,))
  loss_hist = np.zeros((nit_sde,))

  g_x = problem.g_x()
  g_x = jax.jit(g_x)
  H_x = problem.H_x()
  H_x = jax.jit(H_x)

  #appending first stats
  d = problem.x0.shape[0]
  x_it = copy.deepcopy(problem.x0) 
  x.append(np.array(x_it).flatten())
  comparison_stat[0] = jnp.linalg.norm(g_x(x_it,name))+ jnp.linalg.norm(x_it)
  loss_hist[0] = problem.loss(x_it,name)

  #loop
  for k in tqdm(range(nit_sde-1)):
    #update
    delta_W = np.sqrt(dt)*np.random.normal(size=(d,))
    grad_local = g_x(x_it,name)
    hess_local = H_x(x_it,name)
    grad_tilde = grad_local+(rho/jnp.linalg.norm(grad_local))*hess_local@grad_local
    x_it = x_it - dt*grad_tilde + jnp.array(np.sqrt(eta)*(np.eye(d)+(rho/jnp.linalg.norm(grad_local))*hess_local)@problem.Sigma@delta_W) # This is USAM: Original Antonio Version

    #saving stats
    x.append(np.array(x_it).flatten())
    comparison_stat[k+1] = jnp.linalg.norm(grad_local) + jnp.linalg.norm(x_it)
    loss_hist[k+1] = problem.loss(x_it,name)
  return np.array(x),np.array(loss_hist), np.array(comparison_stat)




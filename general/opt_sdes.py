import numpy as np
from tqdm import tqdm
import jax
import copy
import jax.numpy as jnp
from losses import *
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)


def SGD(nit, eta, problem,seed):
  #initializing variables
  name = problem.index
  x = []

  g_x = problem.g_x()
  g_x = jax.jit(g_x)
  
  Sigma = problem.Sigma 
  d = problem.x0.shape[0]
  comparison_stat = np.zeros((nit,))
  loss_hist = np.zeros((nit,))
  grad_list = np.zeros((nit,))
  tr_list = np.zeros((nit,))

  #appending first stats
  x_it = copy.deepcopy(problem.x0) 
  x.append(np.array(x_it).flatten())
  if name == 3 and problem.use_analytical:
    grad_0 = problem.grad(x_it).flatten()
  else:
    grad_0 = g_x(x_it,name)
  comparison_stat[0] = jnp.linalg.norm(grad_0)+ jnp.linalg.norm(x_it)
  loss_hist[0] = problem.loss(x_it,name)
  grad_list[0] = jnp.linalg.norm(grad_0)
  if name == 3:
    tr_list[0] = problem.hess_trace(x_it)

  #loop
  for k in tqdm(range(nit-1)):
    #update
    np.random.seed(k*seed)
    Z = jnp.array(Sigma@np.random.normal(size=(d,)))
    if name == 3 and problem.use_analytical:
      grad_local = problem.grad(x_it).flatten()
    else:
      grad_local = g_x(x_it,name)
    x_it = x_it - eta*(grad_local+Z)

    #saving stats
    x.append(np.array(x_it).flatten())
    comparison_stat[k+1] = jnp.linalg.norm(grad_local) + jnp.linalg.norm(x_it)
    loss_hist[k+1] = problem.loss(x_it,name)
    grad_list[k+1] = jnp.linalg.norm(grad_local)
    if name == 3:
      tr_list[k+1] = problem.hess_trace(x_it)
  return np.array(x),np.array(loss_hist), np.array(comparison_stat), np.array(grad_list), np.array(tr_list)


def SGD_SDE(nit,eta,dt,problem,seed):
  name = problem.index
  nit_sde = int(nit*eta/dt)
  #initializing variables
  x = []
  comparison_stat = np.zeros((nit_sde,))
  loss_hist = np.zeros((nit_sde,))
  grad_list = np.zeros((nit_sde,))
  tr_list = np.zeros((nit_sde,))

  g_x = problem.g_x()
  g_x = jax.jit(g_x)

  d = problem.x0.shape[0]
  Sigma = problem.Sigma
  #appending first stats
  x_it = copy.deepcopy(problem.x0) 
  x.append(np.array(x_it).flatten())
  if name == 3 and problem.use_analytical:
    grad_0 = problem.grad(x_it).flatten()
  else:
    grad_0 = g_x(x_it,name)
  comparison_stat[0] = jnp.linalg.norm(grad_0)+ jnp.linalg.norm(x_it)
  loss_hist[0] = problem.loss(x_it,name)
  grad_list[0] = jnp.linalg.norm(grad_0)
  if name == 3:
    tr_list[0] = problem.hess_trace(x_it)

  #loop
  for k in tqdm(range(nit_sde-1)):
    #update
    np.random.seed(k*seed)
    delta_W = np.random.normal(size=(d,))  #TODO: should be multiplied by sqrt(dt)
    if name == 3 and problem.use_analytical:
      grad_local = problem.grad(x_it).flatten()
    else:
      grad_local = g_x(x_it,name)
    x_it = x_it - dt*(grad_local + jnp.array(Sigma@delta_W))  #TODO: not entirely correct: should be np.sqrt(eta*dt)

    #saving stats
    x.append(np.array(x_it).flatten())
    comparison_stat[k+1] = jnp.linalg.norm(grad_local) + jnp.linalg.norm(x_it)
    loss_hist[k+1] = problem.loss(x_it,name)
    grad_list[k+1] = jnp.linalg.norm(grad_local)
    if name == 3:
      tr_list[k+1] = problem.hess_trace(x_it)
  return np.array(x),np.array(loss_hist), np.array(comparison_stat),np.array(grad_list), np.array(tr_list)



def USAM(nit, eta, rho, problem,seed): # THIS IS USAM

  name = problem.index
  #initializing variables
  x = []
  d = problem.x0.shape[0]
  comparison_stat = np.zeros((nit,))
  loss_hist = np.zeros((nit,))
  grad_list = np.zeros((nit,))
  tr_list = np.zeros((nit,))

  g_x = problem.g_x()
  g_x = jax.jit(g_x)

  #appending first stats
  x_it = copy.deepcopy(problem.x0) 
  x.append(np.array(x_it).flatten())
  if name == 3 and problem.use_analytical:
    grad_0 = problem.grad(x_it).flatten()
  else:
    grad_0 = g_x(x_it,name)
  comparison_stat[0] = jnp.linalg.norm(grad_0)+ jnp.linalg.norm(x_it)
  loss_hist[0] = problem.loss(x_it,name)
  grad_list[0] = jnp.linalg.norm(grad_0)
  if name == 3:
    tr_list[0] = problem.hess_trace(x_it)

  #loop
  for k in tqdm(range(nit-1)):
    #update
    np.random.seed(k*seed)
    Z = jnp.array(problem.Sigma@np.random.normal(size=(d,)))
    if name == 3 and problem.use_analytical:
      grad_local = problem.grad(x_it).flatten()
      x_it = x_it - eta*(problem.grad(x_it + rho*(grad_local + Z)).flatten()+Z)
    else:
      grad_local = g_x(x_it,name)
      x_it = x_it - eta*(g_x(x_it + rho*(grad_local + Z),name)+Z)

    #saving stats
    x.append(np.array(x_it).flatten())
    comparison_stat[k+1] = jnp.linalg.norm(grad_local) + jnp.linalg.norm(x_it)
    loss_hist[k+1] = problem.loss(x_it,name)
    grad_list[k+1] = jnp.linalg.norm(grad_local)
    if name == 3:
      tr_list[k+1] = problem.hess_trace(x_it)
  return np.array(x),np.array(loss_hist), np.array(comparison_stat),np.array(grad_list),np.array(tr_list)



def SAM(nit, eta, rho,problem,seed): 
  #initializing variables
  name = problem.index
  x = []
  d = problem.x0.shape[0]
  comparison_stat = np.zeros((nit,))
  loss_hist = np.zeros((nit,))
  grad_list = np.zeros((nit,))
  tr_list = np.zeros((nit,))

  g_x = problem.g_x()
  g_x = jax.jit(g_x)

  #appending first stats
  x_it = copy.deepcopy(problem.x0) 
  x.append(np.array(x_it).flatten())
  if name == 3 and problem.use_analytical:
    grad_0 = problem.grad(x_it).flatten()
  else:
    grad_0 = g_x(x_it,name)
  comparison_stat[0] = jnp.linalg.norm(grad_0)+ jnp.linalg.norm(x_it)
  loss_hist[0] = problem.loss(x_it,name)
  grad_list[0] = jnp.linalg.norm(grad_0)
  if name == 3:
    tr_list[0] = problem.hess_trace(x_it)

  #loop
  for k in tqdm(range(nit-1)):
    #update
    np.random.seed(k*seed)
    Z = jnp.array(problem.Sigma@np.random.normal(size=(d,)))
    if name == 3 and problem.use_analytical:
      grad_local = problem.grad(x_it).flatten()
      x_it = x_it - eta*(problem.grad(x_it + (rho/jnp.linalg.norm(grad_local))*(grad_local+ Z)).flatten()+Z)
    else:
      grad_local = g_x(x_it,name)
      x_it = x_it - eta*(g_x(x_it + (rho/jnp.linalg.norm(grad_local))*(grad_local+ Z),name)+Z)

    #saving stats
    x.append(np.array(x_it).flatten())
    comparison_stat[k+1] = jnp.linalg.norm(grad_local) + jnp.linalg.norm(x_it)
    loss_hist[k+1] = problem.loss(x_it,name)
    grad_list[k+1] = jnp.linalg.norm(grad_local)
    if name == 3:
      tr_list[k+1] = problem.hess_trace(x_it)
  return np.array(x),np.array(loss_hist), np.array(comparison_stat),np.array(grad_list),np.array(tr_list)





def USAM_SDE1(nit, eta, rho, dt,problem,seed):
  name = problem.index
  nit_sde = int(nit*eta/dt)
  #initializing variables
  x = []
  comparison_stat = np.zeros((nit_sde,))
  loss_hist = np.zeros((nit_sde,))
  grad_list = np.zeros((nit_sde,))
  tr_list = np.zeros((nit_sde,))
  Sigma = problem.Sigma

  g_x = problem.g_x()
  g_x = jax.jit(g_x)
  H_x = problem.H_x()
  H_x = jax.jit(H_x)

  d = problem.x0.shape[0]
  #appending first stats
  x_it = copy.deepcopy(problem.x0) 
  x.append(np.array(x_it).flatten())
  if name == 3 and problem.use_analytical:
    grad_0 = problem.grad(x_it).flatten()
  else:
    grad_0 = g_x(x_it,name)
  comparison_stat[0] = jnp.linalg.norm(grad_0)+ jnp.linalg.norm(x_it)
  loss_hist[0] = problem.loss(x_it,name)
  grad_list[0] = jnp.linalg.norm(grad_0)
  if name == 3:
    tr_list[0] = problem.hess_trace(x_it)

  #loop
  for k in tqdm(range(nit_sde-1)):
    #update
    np.random.seed(k*seed)
    delta_W = np.random.normal(size=(d,))
    if name == 3 and problem.use_analytical:
      grad_local = problem.grad(x_it).flatten()
      hess_local = H_x(x_it,name)
      grad_tilde = grad_local+rho*hess_local@grad_local
      x_it = x_it - dt*(grad_tilde + jnp.array((np.eye(d)+rho*hess_local)@Sigma@delta_W))
    else:
      grad_local = g_x(x_it,name)
      hess_local = H_x(x_it,name)
      grad_tilde = grad_local+rho*hess_local@grad_local
      x_it = x_it - dt*(grad_tilde + jnp.array((np.eye(d)+rho*hess_local)@Sigma@delta_W))
    
    #saving stats
    x.append(np.array(x_it).flatten())
    comparison_stat[k+1] = jnp.linalg.norm(grad_local) + jnp.linalg.norm(x_it)
    loss_hist[k+1] = problem.loss(x_it,name)
    grad_list[k+1] = jnp.linalg.norm(grad_local)
    if name == 3:
      tr_list[k+1] = problem.hess_trace(x_it)
  return np.array(x),np.array(loss_hist), np.array(comparison_stat),np.array(grad_list),np.array(tr_list)


def SAM_SDE1(nit, eta, rho, dt,problem,seed):
  name = problem.index
  nit_sde = int(nit*eta/dt)
  #initializing variables
  x = []
  comparison_stat = np.zeros((nit_sde,))
  loss_hist = np.zeros((nit_sde,))
  grad_list = np.zeros((nit_sde,))
  tr_list = np.zeros((nit_sde,))
  Sigma = problem.Sigma

  g_x = problem.g_x()
  g_x = jax.jit(g_x)
  H_x = problem.H_x()
  H_x = jax.jit(H_x)

  #appending first stats
  d = problem.x0.shape[0]
  x_it = copy.deepcopy(problem.x0) 
  x.append(np.array(x_it).flatten())
  if name == 3 and problem.use_analytical:
    grad_0 = problem.grad(x_it).flatten()
  else:
    grad_0 = g_x(x_it,name)
  comparison_stat[0] = jnp.linalg.norm(grad_0)+ jnp.linalg.norm(x_it)
  loss_hist[0] = problem.loss(x_it,name)
  grad_list[0] = jnp.linalg.norm(grad_0)
  if name == 3:
    tr_list[0] = problem.hess_trace(x_it)

  #loop
  for k in tqdm(range(nit_sde-1)):
    #update
    np.random.seed(k*seed)
    delta_W = np.random.normal(size=(d,))
    if name == 3 and problem.use_analytical:
      grad_local = problem.grad(x_it).flatten()
    else:
      grad_local = g_x(x_it,name)
    hess_local = H_x(x_it,name)
    grad_tilde = grad_local+(rho/jnp.linalg.norm(grad_local))*hess_local@grad_local
    x_it = x_it - dt*(grad_tilde + jnp.array((np.eye(d)+(rho/jnp.linalg.norm(grad_local))*hess_local)@Sigma@delta_W))

    #saving stats
    x.append(np.array(x_it).flatten())
    comparison_stat[k+1] = jnp.linalg.norm(grad_local) + jnp.linalg.norm(x_it)
    loss_hist[k+1] = problem.loss(x_it,name)
    grad_list[k+1] = jnp.linalg.norm(grad_local)
    if name == 3:
      tr_list[k+1] = problem.hess_trace(x_it)
  return np.array(x),np.array(loss_hist), np.array(comparison_stat),np.array(grad_list),np.array(tr_list)



def USAM_SDE1_NO_COV(nit, eta, rho, dt,problem,seed):
  name = problem.index
  nit_sde = int(nit*eta/dt)
  #initializing variables
  x = []
  comparison_stat = np.zeros((nit_sde,))
  loss_hist = np.zeros((nit_sde,))
  grad_list = np.zeros((nit_sde,))
  tr_list = np.zeros((nit_sde,))
  Sigma = problem.Sigma

  g_x = problem.g_x()
  g_x = jax.jit(g_x)
  H_x = problem.H_x()
  H_x = jax.jit(H_x)

  d = problem.x0.shape[0]
  #appending first stats
  x_it = copy.deepcopy(problem.x0) 
  x.append(np.array(x_it).flatten())
  if name == 3 and problem.use_analytical:
    grad_0 = problem.grad(x_it).flatten()
  else:
    grad_0 = g_x(x_it,name)
  comparison_stat[0] = jnp.linalg.norm(grad_0)+ jnp.linalg.norm(x_it)
  loss_hist[0] = problem.loss(x_it,name)
  grad_list[0] = jnp.linalg.norm(grad_0)
  if name == 3:
    tr_list[0] = problem.hess_trace(x_it)

  #loop
  for k in tqdm(range(nit_sde-1)):
    #update
    np.random.seed(k*seed)
    delta_W = np.random.normal(size=(d,))
    if name == 3 and problem.use_analytical:
      grad_local = problem.grad(x_it).flatten()
      hess_local = H_x(x_it,name)
      grad_tilde = grad_local+rho*hess_local@grad_local
      x_it = x_it - dt*(grad_tilde + jnp.array(Sigma@delta_W))
    else:
      grad_local = g_x(x_it,name)
      hess_local = H_x(x_it,name)
      grad_tilde = grad_local+rho*hess_local@grad_local
      x_it = x_it - dt*(grad_tilde + jnp.array(Sigma@delta_W))
    
    #saving stats
    x.append(np.array(x_it).flatten())
    comparison_stat[k+1] = jnp.linalg.norm(grad_local) + jnp.linalg.norm(x_it)
    loss_hist[k+1] = problem.loss(x_it,name)
    grad_list[k+1] = jnp.linalg.norm(grad_local)
    if name == 3:
      tr_list[k+1] = problem.hess_trace(x_it)
  return np.array(x),np.array(loss_hist), np.array(comparison_stat),np.array(grad_list),np.array(tr_list)




def SAM_SDE1_NO_COV(nit, eta, rho, dt,problem,seed):
  name = problem.index
  nit_sde = int(nit*eta/dt)
  #initializing variables
  x = []
  comparison_stat = np.zeros((nit_sde,))
  loss_hist = np.zeros((nit_sde,))
  grad_list = np.zeros((nit_sde,))
  tr_list = np.zeros((nit_sde,))
  Sigma = problem.Sigma

  g_x = problem.g_x()
  g_x = jax.jit(g_x)
  H_x = problem.H_x()
  H_x = jax.jit(H_x)

  #appending first stats
  d = problem.x0.shape[0]
  x_it = copy.deepcopy(problem.x0) 
  x.append(np.array(x_it).flatten())
  if name == 3 and problem.use_analytical:
    grad_0 = problem.grad(x_it).flatten()
  else:
    grad_0 = g_x(x_it,name)
  comparison_stat[0] = jnp.linalg.norm(grad_0)+ jnp.linalg.norm(x_it)
  loss_hist[0] = problem.loss(x_it,name)
  grad_list[0] = jnp.linalg.norm(grad_0)
  if name == 3:
    tr_list[0] = problem.hess_trace(x_it)

  #loop
  for k in tqdm(range(nit_sde-1)):
    #update
    np.random.seed(k*seed)
    delta_W = np.random.normal(size=(d,))
    if name == 3 and problem.use_analytical:
      grad_local = problem.grad(x_it).flatten()
    else:
      grad_local = g_x(x_it,name)
    hess_local = H_x(x_it,name)
    grad_tilde = grad_local+(rho/jnp.linalg.norm(grad_local))*hess_local@grad_local
    x_it = x_it - dt*(grad_tilde + jnp.array(Sigma@delta_W))

    #saving stats
    x.append(np.array(x_it).flatten())
    comparison_stat[k+1] = jnp.linalg.norm(grad_local) + jnp.linalg.norm(x_it)
    loss_hist[k+1] = problem.loss(x_it,name)
    grad_list[k+1] = jnp.linalg.norm(grad_local)
    if name == 3:
      tr_list[k+1] = problem.hess_trace(x_it)
  return np.array(x),np.array(loss_hist), np.array(comparison_stat),np.array(grad_list),np.array(tr_list)



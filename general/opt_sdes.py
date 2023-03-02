import jax
import jax.numpy as jnp
#jax.config.update('jax_platform_name', 'cpu')
from jax import grad
from jax import vmap
jax.config.update("jax_enable_x64", True)

import cupy as cp

import numpy as np

import scipy

from tqdm import tqdm
import copy

from losses import *


def sqrtm(A):
    # Calculate the SVD of the matrix
    U, s, V = jnp.linalg.svd(A)

    # Take the square root of the singular values
    sqrt_s = jnp.sqrt(s)

    # Reconstruct the matrix using the square root of the singular values
    sqrtA = jax.lax.dot(U * sqrt_s, V)

    return sqrtA


# SGD optimizer
def SGD(nit, eta, problem,seed, fix_random_seed_it):
  # problem name
  name = problem.index


  # instantiate and jit gradient function
  g_x = problem.g_x()
  g_x = jax.jit(g_x)
  
  # noise and problem dimension
  Sigma = problem.Sigma 
  d = problem.x0.shape[0]

  # logs
  x = []
  comparison_stat = np.zeros((nit,))
  loss_hist = np.zeros((nit,))
  grad_list = np.zeros((nit,))
  tr_list = np.zeros((nit,))

  #appending first stats
  x_it = copy.deepcopy(problem.x0) 
  x.append(np.array(x_it).flatten())
  # for matrix sensing use analytical grads
  if name == 3 and problem.use_analytical:
    grad_0 = problem.grad(x_it).flatten()
  else:
    grad_0 = g_x(x_it,name)
  comparison_stat[0] = jnp.linalg.norm(grad_0)+ jnp.linalg.norm(x_it)
  loss_hist[0] = problem.loss(x_it,name)
  grad_list[0] = jnp.linalg.norm(grad_0)
  # for matrix sensing use analytical hessian
  if name == 3:
    tr_list[0] = problem.hess_trace(x_it)

  # main loop
  for k in tqdm(range(nit-1)):
    # fix seed at every iteration
    if fix_random_seed_it:
      np.random.seed(k*seed) 
    # noise matrix
    Z = jnp.array(Sigma@np.random.normal(size=(d,)))
    # for matrix sensing use analytical grads
    if name == 3 and problem.use_analytical:
      grad_local = problem.grad(x_it).flatten()
    else:
      grad_local = g_x(x_it,name)
    # update
    x_it = x_it - eta*(grad_local+Z)

    #saving stats ad return
    x.append(np.array(x_it).flatten())
    comparison_stat[k+1] = jnp.linalg.norm(grad_local) + jnp.linalg.norm(x_it)
    loss_hist[k+1] = problem.loss(x_it,name)
    grad_list[k+1] = jnp.linalg.norm(grad_local)
    if name == 3:
      tr_list[k+1] = problem.hess_trace(x_it)
  return np.array(x),np.array(loss_hist), np.array(comparison_stat), np.array(grad_list), np.array(tr_list)

# USAM optimizer
def USAM(nit, eta, rho, problem,seed,fix_random_seed_it): 
  # problem name
  name = problem.index

  # instantiate and jit gradient function
  g_x = problem.g_x()
  g_x = jax.jit(g_x)

  # noise and problem dimension
  Sigma = problem.Sigma 
  d = problem.x0.shape[0]

  #logs
  x = []
  comparison_stat = np.zeros((nit,))
  loss_hist = np.zeros((nit,))
  grad_list = np.zeros((nit,))
  tr_list = np.zeros((nit,))

  # appending first stats
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

  # main loop
  for k in tqdm(range(nit-1)):
    # fix seed at every iteration
    if fix_random_seed_it:
      np.random.seed(k*seed) 
    # noise sde
    Z = jnp.array(Sigma@np.random.normal(size=(d,)))
    if name == 3 and problem.use_analytical:
      grad_local = problem.grad(x_it).flatten()
      # update
      x_it = x_it - eta*(problem.grad(x_it + rho*(grad_local + Z)).flatten()+Z)
    else:
      grad_local = g_x(x_it,name)
      #update
      x_it = x_it - eta*(g_x(x_it + rho*(grad_local + Z),name)+Z)

    #saving stats and return
    x.append(np.array(x_it).flatten())
    comparison_stat[k+1] = jnp.linalg.norm(grad_local) + jnp.linalg.norm(x_it)
    loss_hist[k+1] = problem.loss(x_it,name)
    grad_list[k+1] = jnp.linalg.norm(grad_local)
    if name == 3:
      tr_list[k+1] = problem.hess_trace(x_it)
  return np.array(x),np.array(loss_hist), np.array(comparison_stat),np.array(grad_list),np.array(tr_list)

# SAM optimizer
def SAM(nit, eta, rho,problem,seed,fix_random_seed_it): 
  # problem name
  name = problem.index

  # instantiate and jit gradient function
  g_x = problem.g_x()
  g_x = jax.jit(g_x)

  # noise and problem dimension
  Sigma = problem.Sigma
  d = problem.x0.shape[0]

  #logs
  x = []
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
    # fix seed at every iteration
    if fix_random_seed_it:
      np.random.seed(k*seed) 
    # noise sde
    Z = jnp.array(Sigma@np.random.normal(size=(d,)))
    if name == 3 and problem.use_analytical:
      grad_local = problem.grad(x_it).flatten()
      # update
      x_it = x_it - eta*(problem.grad(x_it + (rho/jnp.linalg.norm(grad_local))*(grad_local+ Z)).flatten()+Z)
    else:
      grad_local = g_x(x_it,name)
      # update
      x_it = x_it - eta*(g_x(x_it + (rho/jnp.linalg.norm(grad_local))*(grad_local+ Z),name)+Z)

    #saving stats
    x.append(np.array(x_it).flatten())
    comparison_stat[k+1] = jnp.linalg.norm(grad_local) + jnp.linalg.norm(x_it)
    loss_hist[k+1] = problem.loss(x_it,name)
    grad_list[k+1] = jnp.linalg.norm(grad_local)
    if name == 3:
      tr_list[k+1] = problem.hess_trace(x_it)
  return np.array(x),np.array(loss_hist), np.array(comparison_stat),np.array(grad_list),np.array(tr_list)

# SAM TRUE optimizer
def SAM_true(nit, eta, rho,problem,seed,fix_random_seed_it): 

  # problem name
  name = problem.index

  # instantiate and jit gradient function
  g_x = problem.g_x()
  g_x = jax.jit(g_x)

  # noise and problem dimension
  Sigma = problem.Sigma
  d = problem.x0.shape[0]

  #logs
  x = []
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
    # fix seed at every iteration
    if fix_random_seed_it:
      np.random.seed(k*seed) 
    # noise sde
    Z = jnp.array(Sigma@np.random.normal(size=(d,)))
    if name == 3 and problem.use_analytical:
      grad_local = problem.grad(x_it).flatten()
      # update
      x_it = x_it - eta*(problem.grad(x_it + (rho/jnp.linalg.norm(grad_local+Z))*(grad_local+ Z)).flatten()+Z)
    else:
      grad_local = g_x(x_it,name)
      # update
      x_it = x_it - eta*(g_x(x_it + (rho/jnp.linalg.norm(grad_local+Z))*(grad_local+ Z),name)+Z)

    #saving stats and return
    x.append(np.array(x_it).flatten())
    comparison_stat[k+1] = jnp.linalg.norm(grad_local) + jnp.linalg.norm(x_it)
    loss_hist[k+1] = problem.loss(x_it,name)
    grad_list[k+1] = jnp.linalg.norm(grad_local)
    if name == 3:
      tr_list[k+1] = problem.hess_trace(x_it)
  return np.array(x),np.array(loss_hist), np.array(comparison_stat),np.array(grad_list),np.array(tr_list)




# SGD SDE
def SGD_SDE(nit,eta,dt,problem,seed,fix_random_seed_it):
  # problem name and number of its
  name = problem.index
  nit_sde = int(nit*eta/dt)

  # instantiate and jit gradient function
  g_x = problem.g_x()
  g_x = jax.jit(g_x)

  # noise and problem dimension
  d = problem.x0.shape[0]
  Sigma = problem.Sigma

  # logs
  x = []
  comparison_stat = np.zeros((nit_sde,))
  loss_hist = np.zeros((nit_sde,))
  grad_list = np.zeros((nit_sde,))
  tr_list = np.zeros((nit_sde,))

  #appending first stats
  x_it = copy.deepcopy(problem.x0) 
  x.append(np.array(x_it).flatten())
  # for matrix sensing use analytical grads
  if name == 3 and problem.use_analytical:
    grad_0 = problem.grad(x_it).flatten()
  else:
    grad_0 = g_x(x_it,name)
  comparison_stat[0] = jnp.linalg.norm(grad_0)+ jnp.linalg.norm(x_it)
  loss_hist[0] = problem.loss(x_it,name)
  grad_list[0] = jnp.linalg.norm(grad_0)
  if name == 3:
    tr_list[0] = problem.hess_trace(x_it)

  # main loop
  for k in tqdm(range(nit_sde-1)):
    # fix seed at every iteration
    if fix_random_seed_it:
      np.random.seed(k*seed) 
    # delta noise sde
    delta_W = np.random.normal(size=(d,))  #TODO: should be multiplied by sqrt(dt)
    # for matrix sensing use analytical grads
    if name == 3 and problem.use_analytical:
      grad_local = problem.grad(x_it).flatten()
    else:
      grad_local = g_x(x_it,name)
    # update
    x_it = x_it - dt*(grad_local + jnp.array(Sigma@delta_W))  #TODO: not entirely correct: should be np.sqrt(eta*dt)

    #saving stats and return
    x.append(np.array(x_it).flatten())
    comparison_stat[k+1] = jnp.linalg.norm(grad_local) + jnp.linalg.norm(x_it)
    loss_hist[k+1] = problem.loss(x_it,name)
    grad_list[k+1] = jnp.linalg.norm(grad_local)
    if name == 3:
      tr_list[k+1] = problem.hess_trace(x_it)
  return np.array(x),np.array(loss_hist), np.array(comparison_stat),np.array(grad_list), np.array(tr_list)

# USAM SDE
def USAM_SDE(nit, eta, rho, dt,problem,seed,fix_random_seed_it):
  # problem name and number of its
  name = problem.index
  nit_sde = int(nit*eta/dt)

  # instantiate and jit gradient function
  g_x = problem.g_x()
  g_x = jax.jit(g_x)

  # noise and problem dimension
  d = problem.x0.shape[0]
  Sigma = problem.Sigma

  # logs
  x = []
  comparison_stat = np.zeros((nit_sde,))
  loss_hist = np.zeros((nit_sde,))
  grad_list = np.zeros((nit_sde,))
  tr_list = np.zeros((nit_sde,))

  #appending first stats
  x_it = copy.deepcopy(problem.x0) 
  x.append(np.array(x_it).flatten())
  # for matrix sensing use analytical grads
  if name == 3 and problem.use_analytical:
    grad_0 = problem.grad(x_it).flatten()
  else:
    grad_0 = g_x(x_it,name)
  comparison_stat[0] = jnp.linalg.norm(grad_0)+ jnp.linalg.norm(x_it)
  loss_hist[0] = problem.loss(x_it,name)
  grad_list[0] = jnp.linalg.norm(grad_0)
  if name == 3:
    tr_list[0] = problem.hess_trace(x_it)

  # jitted hessian vector product
  @partial(jax.jit)
  def hvp(x, v):
      return grad(lambda x: jnp.vdot(g_x(x,name), v))(x)

  #loop
  for k in tqdm(range(nit_sde-1)):
    # fix seed at every iteration
    if fix_random_seed_it:
      np.random.seed(k*seed) 
    # delta noise sde
    delta_W = np.random.normal(size=(d,))  #TODO: should be multiplied by sqrt(dt)
    if name == 3 and problem.use_analytical:
      grad_local = problem.grad(x_it).flatten()
      hess_local = H_x(x_it,name)
      grad_tilde = grad_local+rho*hess_local@grad_local
      #update  
      x_it = x_it - dt*(grad_tilde + jnp.array((np.eye(d)+rho*hess_local)@Sigma@delta_W))
    else:
      grad_local = g_x(x_it,name)
      grad_tilde = grad_local+rho*hvp(x_it,grad_local)
      increment = Sigma@delta_W
      # update
      x_it = x_it - dt*(grad_tilde + increment + rho*hvp(x_it,increment)) 
    
    #saving stats and return
    x.append(np.array(x_it).flatten())
    comparison_stat[k+1] = jnp.linalg.norm(grad_local) + jnp.linalg.norm(x_it)
    loss_hist[k+1] = problem.loss(x_it,name)
    grad_list[k+1] = jnp.linalg.norm(grad_local)
    if name == 3:
      tr_list[k+1] = problem.hess_trace(x_it)
  return np.array(x),np.array(loss_hist), np.array(comparison_stat),np.array(grad_list),np.array(tr_list)
  
# SAM SDE
def SAM_SDE(nit, eta, rho, dt,problem,seed,fix_random_seed_it):
  # problem name and number of its
  name = problem.index
  nit_sde = int(nit*eta/dt)

  # instantiate and jit gradient function
  g_x = problem.g_x()
  g_x = jax.jit(g_x)

  # noise and problem dimension
  d = problem.x0.shape[0]
  Sigma = problem.Sigma

  # logs
  x = []
  comparison_stat = np.zeros((nit_sde,))
  loss_hist = np.zeros((nit_sde,))
  grad_list = np.zeros((nit_sde,))
  tr_list = np.zeros((nit_sde,))

  #appending first stats
  x_it = copy.deepcopy(problem.x0) 
  x.append(np.array(x_it).flatten())
  # for matrix sensing use analytical grads
  if name == 3 and problem.use_analytical:
    grad_0 = problem.grad(x_it).flatten()
  else:
    grad_0 = g_x(x_it,name)
  comparison_stat[0] = jnp.linalg.norm(grad_0)+ jnp.linalg.norm(x_it)
  loss_hist[0] = problem.loss(x_it,name)
  grad_list[0] = jnp.linalg.norm(grad_0)
  if name == 3:
    tr_list[0] = problem.hess_trace(x_it)

  # jitted hessian vector product
  @partial(jax.jit)
  def hvp(x, v):
      return grad(lambda x: jnp.vdot(g_x(x,name), v))(x)

  #loop
  for k in tqdm(range(nit_sde-1)):
    # fix seed at every iteration
    if fix_random_seed_it:
      np.random.seed(k*seed) 
    # delta noise sde
    delta_W = np.random.normal(size=(d,)) #TODO: should be multiplied by sqrt(dt)
    if name == 3 and problem.use_analytical:
      grad_local = problem.grad(x_it).flatten()
    else:
      grad_local = g_x(x_it,name)
    norm = jnp.linalg.norm(grad_local)
    grad_tilde = grad_local+rho*hvp(x_it,grad_local/norm)
    increment = Sigma@delta_W
    # update
    x_it = x_it - dt*(grad_tilde + jnp.array(increment+rho*hvp(x_it,increment/norm))) #TODO: Why jitting here and not for USAM in 358?


    #saving stats and return
    x.append(np.array(x_it).flatten())
    comparison_stat[k+1] = jnp.linalg.norm(grad_local) + jnp.linalg.norm(x_it)
    loss_hist[k+1] = problem.loss(x_it,name)
    grad_list[k+1] = jnp.linalg.norm(grad_local)
    if name == 3:
      tr_list[k+1] = problem.hess_trace(x_it)
  return np.array(x),np.array(loss_hist), np.array(comparison_stat),np.array(grad_list),np.array(tr_list)

# SAM TRUE SDE
def SAM_SDE_true(nit, eta, rho, dt,problem,seed,fix_random_seed_it,nr1,nr2,nr3):
  # problem name and number of its
  name = problem.index
  nit_sde = int(nit*eta/dt)

  # instantiate and jit gradient function
  g_x = problem.g_x()
  g_x = jax.jit(g_x)

  # noise and problem dimension
  d = problem.x0.shape[0]
  Sigma = problem.Sigma

  # logs
  x = []
  comparison_stat = np.zeros((nit_sde,))
  loss_hist = np.zeros((nit_sde,))
  grad_list = np.zeros((nit_sde,))
  tr_list = np.zeros((nit_sde,))

  #appending first stats
  x_it = copy.deepcopy(problem.x0) 
  x.append(np.array(x_it).flatten())
  # for matrix sensing use analytical grads
  if name == 3 and problem.use_analytical:
    grad_0 = problem.grad(x_it).flatten()
  else:
    grad_0 = g_x(x_it,name)
  comparison_stat[0] = jnp.linalg.norm(grad_0)+ jnp.linalg.norm(x_it)
  loss_hist[0] = problem.loss(x_it,name)
  grad_list[0] = jnp.linalg.norm(grad_0)
  if name == 3:
    tr_list[0] = problem.hess_trace(x_it)

  # jitted hessian vector product
  @partial(jax.jit)
  def hvp(x, v):
      return grad(lambda x: jnp.vdot(g_x(x,name), v))(x)

  #main loop
  for k in tqdm(range(nit_sde-1)):
    if name == 3 and problem.use_analytical:
      grad_local = problem.grad(x_it).flatten()
    else:
      grad_local = g_x(x_it,name)

    # These n_runj are the sampling size of the expected values below. Likely, the bigger, the more accurate the estimate, but also the higher the comput cost.
    n_run1=nr1
    n_run2=nr2
    n_run3=nr3
    
    # calculate sigma tilde. This loop is to estimate the most outer expected value
    grad_final = 0
    for i in range(n_run1):
      # This Z1 is the stochasticity of the elements used to estimate the outer EV. We average over n_run1 realizations of such a variable
      Z1 = jnp.array(Sigma@np.random.normal(size=(d,)))
      grad_tilde = 0
      
      ## This loop is for the estimation of the inner-most EV
      for j in range(n_run2):
        # This Z2 is the stochasticity of the elements used to estimate the inner EV. We average over n_run2 realizations of such a variable 
        Z2 = jnp.array(Sigma@np.random.normal(size=(d,)))
        grad_tilde += hvp(x_it,(grad_local+Z2)/jnp.linalg.norm(grad_local+Z2))
      grad_tilde = grad_tilde/n_run2
      grad_final += Z1*(grad_tilde - hvp(x_it,(grad_local+Z1)/jnp.linalg.norm(grad_local+Z1))).T
    sigma_tilde = grad_final/n_run1

    # We remember that Sigma is a proxy for the sqrt of the SGD Cov Matrix. Therefore, we need to square it to retreive the real Cov Matrix of SGD.
    sigma_sgd = Sigma**2
    # By def, we need to sum sigma_tilde to its transpose
    sigma_symm = sigma_tilde+sigma_tilde.T
    
    # Here we sum all the necessary matrices and take the sqrt as usual. WE use real because numerical due to negligible numerical issues we would have extremely small imaginary numbers and we get read of those.
    #cov = jnp.real(jax.scipy.linalg.sqrtm(sigma_sgd + rho*sigma_symm))
    cov = jnp.real(sqrtm(sigma_sgd + rho*sigma_symm))


    ### We estimate the bias term.
    grad_tilde_ = grad_local
    grad_tilde_exp = 0
    for j in range(n_run3):
      # This Z3 is the stochasticity of the elements used to estimate the EV. We average over n_run3 realizations of such a variable 
      Z3 = jnp.array(Sigma@np.random.normal(size=(d,)))
      grad_tilde_exp += hvp(x_it,(grad_local+Z3)/jnp.linalg.norm(grad_local+Z3))
    grad_tilde_exp=grad_tilde_exp/n_run3
    grad_tilde_+=rho*grad_tilde_exp

    # fix seed at every iteration
    if fix_random_seed_it:
      np.random.seed(k*seed) 
    # delta noise sde
    delta_W = np.random.normal(size=(d,)) #TODO: should be multiplied by sqrt(dt)

    # the actual update
    x_it = x_it - dt*(grad_tilde_ + cov@delta_W)  #TODO: not entirely correct: should be np.sqrt(eta*dt)

    #saving stats and return
    x.append(np.array(x_it).flatten())
    comparison_stat[k+1] = jnp.linalg.norm(grad_local) + jnp.linalg.norm(x_it)
    loss_hist[k+1] = problem.loss(x_it,name)
    grad_list[k+1] = jnp.linalg.norm(grad_local)
    if name == 3:
      tr_list[k+1] = problem.hess_trace(x_it)
  return np.array(x),np.array(loss_hist), np.array(comparison_stat),np.array(grad_list),np.array(tr_list)

# SAM TRUE SDE DRIFT
def SAM_SDE_true_NO_COV(nit, eta, rho, dt,problem,seed,fix_random_seed_it,nr3):
  # problem name and number of its
  name = problem.index
  nit_sde = int(nit*eta/dt)

  # instantiate and jit gradient function
  g_x = problem.g_x()
  g_x = jax.jit(g_x)

  # noise and problem dimension
  d = problem.x0.shape[0]
  Sigma = problem.Sigma

  # logs
  x = []
  comparison_stat = np.zeros((nit_sde,))
  loss_hist = np.zeros((nit_sde,))
  grad_list = np.zeros((nit_sde,))
  tr_list = np.zeros((nit_sde,))

  #appending first stats
  x_it = copy.deepcopy(problem.x0) 
  x.append(np.array(x_it).flatten())
  # for matrix sensing use analytical grads
  if name == 3 and problem.use_analytical:
    grad_0 = problem.grad(x_it).flatten()
  else:
    grad_0 = g_x(x_it,name)
  comparison_stat[0] = jnp.linalg.norm(grad_0)+ jnp.linalg.norm(x_it)
  loss_hist[0] = problem.loss(x_it,name)
  grad_list[0] = jnp.linalg.norm(grad_0)
  if name == 3:
    tr_list[0] = problem.hess_trace(x_it)

  # jitted hessian vector product
  @partial(jax.jit)
  def hvp(x, v):
      return grad(lambda x: jnp.vdot(g_x(x,name), v))(x)

  # main loop
  for k in tqdm(range(nit_sde-1)):
    if name == 3 and problem.use_analytical:
      grad_local = problem.grad(x_it).flatten()
    else:
      grad_local = g_x(x_it,name)

    n_run3=nr3

    # We estimate the bias term.
    grad_tilde_ = grad_local
    grad_tilde_exp = 0
    for j in range(n_run3):
      # This Z3 is the stochasticity of the elements used to estimate the EV. We average over n_run3 realizations of such a variable 
      Z3 = jnp.array(Sigma@np.random.normal(size=(d,)))
      grad_tilde_exp += hvp(x_it,(grad_local+Z3)/jnp.linalg.norm(grad_local+Z3))
    grad_tilde_exp=grad_tilde_exp/n_run3
    grad_tilde_+=rho*grad_tilde_exp

    # fix seed at every iteration
    if fix_random_seed_it:
      np.random.seed(k*seed) 
    # delta noise sde
    delta_W = np.random.normal(size=(d,)) #TODO: should be multiplied by sqrt(dt)

    # update
    x_it = x_it - dt*(grad_tilde_ + jnp.array(Sigma@delta_W)) #TODO: not entirely correct: should be np.sqrt(eta*dt)

    #saving stats and return
    x.append(np.array(x_it).flatten())
    comparison_stat[k+1] = jnp.linalg.norm(grad_local) + jnp.linalg.norm(x_it)
    loss_hist[k+1] = problem.loss(x_it,name)
    grad_list[k+1] = jnp.linalg.norm(grad_local)
    if name == 3:
      tr_list[k+1] = problem.hess_trace(x_it)
  return np.array(x),np.array(loss_hist), np.array(comparison_stat),np.array(grad_list),np.array(tr_list)

# SAM SDE DRIFT
def SAM_SDE_NO_COV(nit, eta, rho, dt,problem,seed,fix_random_seed_it):
  # problem name and number of its
  name = problem.index
  nit_sde = int(nit*eta/dt)

  # instantiate and jit gradient function
  g_x = problem.g_x()
  g_x = jax.jit(g_x)

  # noise and problem dimension
  d = problem.x0.shape[0]
  Sigma = problem.Sigma

  # logs
  x = []
  comparison_stat = np.zeros((nit_sde,))
  loss_hist = np.zeros((nit_sde,))
  grad_list = np.zeros((nit_sde,))
  tr_list = np.zeros((nit_sde,))

  #appending first stats
  x_it = copy.deepcopy(problem.x0) 
  x.append(np.array(x_it).flatten())
  # for matrix sensing use analytical grads
  if name == 3 and problem.use_analytical:
    grad_0 = problem.grad(x_it).flatten()
  else:
    grad_0 = g_x(x_it,name)
  comparison_stat[0] = jnp.linalg.norm(grad_0)+ jnp.linalg.norm(x_it)
  loss_hist[0] = problem.loss(x_it,name)
  grad_list[0] = jnp.linalg.norm(grad_0)
  if name == 3:
    tr_list[0] = problem.hess_trace(x_it)

  # jitted hessian vector product
  @partial(jax.jit)
  def hvp(x, v):
      return grad(lambda x: jnp.vdot(g_x(x,name), v))(x)

  #loop
  for k in tqdm(range(nit_sde-1)):
    # fix seed at every iteration
    if fix_random_seed_it:
      np.random.seed(k*seed) 
    # delta noise sde
    delta_W = np.random.normal(size=(d,))  #TODO: should be multiplied by sqrt(dt)
    if name == 3 and problem.use_analytical:
      grad_local = problem.grad(x_it).flatten()
    else:
      grad_local = g_x(x_it,name)
    grad_tilde = grad_local+rho*hvp(x_it,grad_local/jnp.linalg.norm(grad_local))
    # update
    x_it = x_it - dt*(grad_tilde + jnp.array(Sigma@delta_W))

    #saving stats and return
    x.append(np.array(x_it).flatten())
    comparison_stat[k+1] = jnp.linalg.norm(grad_local) + jnp.linalg.norm(x_it)
    loss_hist[k+1] = problem.loss(x_it,name)
    grad_list[k+1] = jnp.linalg.norm(grad_local)
    if name == 3:
      tr_list[k+1] = problem.hess_trace(x_it)
  return np.array(x),np.array(loss_hist), np.array(comparison_stat),np.array(grad_list),np.array(tr_list)

# USAM SDE DRIFT
def USAM_SDE_NO_COV(nit, eta, rho, dt,problem,seed,fix_random_seed_it):
  # problem name and number of its
  name = problem.index
  nit_sde = int(nit*eta/dt)

  # instantiate and jit gradient function
  g_x = problem.g_x()
  g_x = jax.jit(g_x)

  # noise and problem dimension
  d = problem.x0.shape[0]
  Sigma = problem.Sigma

  # logs
  x = []
  comparison_stat = np.zeros((nit_sde,))
  loss_hist = np.zeros((nit_sde,))
  grad_list = np.zeros((nit_sde,))
  tr_list = np.zeros((nit_sde,))

  #appending first stats
  x_it = copy.deepcopy(problem.x0) 
  x.append(np.array(x_it).flatten())
  # for matrix sensing use analytical grads
  if name == 3 and problem.use_analytical:
    grad_0 = problem.grad(x_it).flatten()
  else:
    grad_0 = g_x(x_it,name)
  comparison_stat[0] = jnp.linalg.norm(grad_0)+ jnp.linalg.norm(x_it)
  loss_hist[0] = problem.loss(x_it,name)
  grad_list[0] = jnp.linalg.norm(grad_0)
  if name == 3:
    tr_list[0] = problem.hess_trace(x_it)

  # jitted hessian vector product
  @partial(jax.jit)
  def hvp(x, v):
      return grad(lambda x: jnp.vdot(g_x(x,name), v))(x)

  #loop
  for k in tqdm(range(nit_sde-1)):
    # fix seed at every iteration 
    if fix_random_seed_it:
      np.random.seed(k*seed) 
    # delta noise sde
    delta_W = np.random.normal(size=(d,))  #TODO: should be multiplied by sqrt(dt)
    if name == 3 and problem.use_analytical:
      grad_local = problem.grad(x_it).flatten()
      hess_local = H_x(x_it,name)
      grad_tilde = grad_local+rho*hess_local@grad_local
      # update
      x_it = x_it - dt*(grad_tilde + jnp.array(Sigma@delta_W))
    else:
      grad_local = g_x(x_it,name)
      grad_tilde = grad_local+rho*hvp(x_it,grad_local)
      # update
      x_it = x_it - dt*(grad_tilde + jnp.array(Sigma@delta_W)) #TODO: not entirely correct: should be np.sqrt(eta*dt)

    #saving stats and return
    x.append(np.array(x_it).flatten())
    comparison_stat[k+1] = jnp.linalg.norm(grad_local) + jnp.linalg.norm(x_it)
    loss_hist[k+1] = problem.loss(x_it,name)
    grad_list[k+1] = jnp.linalg.norm(grad_local)
    if name == 3:
      tr_list[k+1] = problem.hess_trace(x_it)
  return np.array(x),np.array(loss_hist), np.array(comparison_stat),np.array(grad_list),np.array(tr_list)









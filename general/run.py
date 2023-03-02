import numpy as np
np.random.seed(42) # Added by Enea 

import jax
import jax.numpy as jnp
#jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler

import pickle 
from tqdm import tqdm
import hydra
import copy

from losses import *
from opt_sdes import *



@hydra.main(config_name='run')
def main(cfg):

  #experiment settings
  name = cfg.name
  nexp = cfg.nexp
  nit = cfg.nit
  fix_random_seed_it = cfg.fix_random_seed_it
  nr1,nr2,nr3 = cfg.n_run1,cfg.n_run2,cfg.n_run3

  # optimizer settings
  etas = cfg.etas
  rhos = cfg.rhos
  f_star = cfg.f_star
  sigma = cfg.sigma

  # network settings
  d_quadratic = cfg.d_quadratic
  HIDDEN = cfg.HIDDEN
  

  # initialize empty lists for logging errors
  # 1) SGD vs SGD_SDE
  error_mean_SGD_SGD_SDE = np.zeros((len(etas),len(rhos)))
  # 2) SAM_TRUE vs SGD_SDE
  error_mean_SAM_True_SGD_SDE = np.zeros((len(etas),len(rhos)))
  # 3) SAM_TRUE vs SAM_TRUE_SDE
  error_mean_SAM_True_SAM_True_SDE = np.zeros((len(etas),len(rhos)))
  # 4) SAM_TRUE vs SAM_SDE
  error_mean_SAM_True_SAM_SDE = np.zeros((len(etas),len(rhos)))
  # 5) SAM_TRUE vs USAM_SDE
  error_mean_SAM_True_USAM_SDE = np.zeros((len(etas),len(rhos)))
  # 6) SAM vs SGD_SDE
  error_mean_SAM_SGD_SDE = np.zeros((len(etas),len(rhos)))
  # 7) SAM vs SAM_SDE
  error_mean_SAM_SAM_SDE = np.zeros((len(etas),len(rhos)))
  # 8) USAM vs USAM_SDE
  error_mean_USAM_USAM_SDE = np.zeros((len(etas),len(rhos)))
  # 9) USAM vs SGD_SDE
  error_mean_USAM_SGD_SDE = np.zeros((len(etas),len(rhos)))
  # 10) USAM vs USAM_SDE_DRIFT
  error_mean_USAM_USAM_SDE_DRIFT = np.zeros((len(etas),len(rhos)))
  # 11) SAM vs SAM_SDE_DRIFT
  error_mean_SAM_SAM_SDE_DRIFT = np.zeros((len(etas),len(rhos)))
  # 12) SAM_TRUE vs SAM_TRUE_SDE_DRIFT
  error_mean_SAM_True_SAM_True_SDE_DRIFT = np.zeros((len(etas),len(rhos)))

  # 1) SGD vs SGD_SDE
  error_std_SGD_SGD_SDE = np.zeros((len(etas),len(rhos)))
  # 2) SAM_TRUE vs SGD_SDE
  error_std_SAM_True_SGD_SDE = np.zeros((len(etas),len(rhos)))
  # 3) SAM_TRUE vs SAM_TRUE_SDE
  error_std_SAM_True_SAM_True_SDE = np.zeros((len(etas),len(rhos)))
  # 4) SAM_TRUE vs SAM_SDE
  error_std_SAM_True_SAM_SDE = np.zeros((len(etas),len(rhos)))
  # 5) SAM_TRUE vs USAM_SDE
  error_std_SAM_True_USAM_SDE = np.zeros((len(etas),len(rhos)))
  # 6) SAM vs SGD_SDE
  error_std_SAM_SGD_SDE = np.zeros((len(etas),len(rhos)))
  # 7) SAM vs SAM_SDE
  error_std_SAM_SAM_SDE = np.zeros((len(etas),len(rhos)))
  # 8) USAM vs USAM_SDE
  error_std_USAM_USAM_SDE = np.zeros((len(etas),len(rhos)))
  # 9) USAM vs SGD_SDE
  error_std_USAM_SGD_SDE = np.zeros((len(etas),len(rhos)))
  # 10) USAM vs USAM_SDE_DRIFT
  error_std_USAM_USAM_SDE_DRIFT = np.zeros((len(etas),len(rhos)))
  # 11) SAM vs SAM_SDE_DRIFT
  error_std_SAM_SAM_SDE_DRIFT = np.zeros((len(etas),len(rhos)))
  # 12) SAM_TRUE vs SAM_TRUE_SDE_DRIFT
  error_std_SAM_True_SAM_True_SDE_DRIFT = np.zeros((len(etas),len(rhos)))

  # initialize empty lists for logging iterates
  # 1) SGD traj
  x_it_all_SGD = np.zeros((len(etas),len(rhos),nit))
  # 2) SGD_SDE traj
  x_it_all_SGD_SDE = np.zeros((len(etas),len(rhos),nit))
  # 3) USAM traj
  x_it_all_USAM = np.zeros((len(etas),len(rhos),nit))
  # 4) SAM traj
  x_it_all_SAM = np.zeros((len(etas),len(rhos),nit))
  # 5) SAM_TRUE traj
  x_it_all_SAM_True = np.zeros((len(etas),len(rhos),nit))
  # 6) SAM_TRUE_SDE traj
  x_it_all_SAM_True_SDE = np.zeros((len(etas),len(rhos),nit))
  # 7) SAM_SDE traj
  x_it_all_SAM_SDE = np.zeros((len(etas),len(rhos),nit))
  # 8) USAM_SDE traj
  x_it_all_USAM_SDE = np.zeros((len(etas),len(rhos),nit))
  # 9) USAM_SDE_DRIFT traj
  x_it_all_USAM_SDE_DRIFT = np.zeros((len(etas),len(rhos),nit))
  # 10) SAM_SDE_DRIFT traj
  x_it_all_SAM_SDE_DRIFT = np.zeros((len(etas),len(rhos),nit))
  # 11) SAM_SDE_TRUE_DRIFT traj
  x_it_all_SAM_SDE_True_DRIFT = np.zeros((len(etas),len(rhos),nit))

  # 1) SGD traj
  x_it_all_SGD_std = np.zeros((len(etas),len(rhos),nit))
  # 2) SGD_SDE traj
  x_it_all_SGD_SDE_std = np.zeros((len(etas),len(rhos),nit))
  # 3) USAM traj
  x_it_all_USAM_std = np.zeros((len(etas),len(rhos),nit))
  # 4) SAM traj
  x_it_all_SAM_std = np.zeros((len(etas),len(rhos),nit))
  # 5) SAM_TRUE traj
  x_it_all_SAM_True_std = np.zeros((len(etas),len(rhos),nit))
  # 6) SAM_TRUE_SDE traj
  x_it_all_SAM_True_SDE_std = np.zeros((len(etas),len(rhos),nit))
  # 7) SAM_SDE traj
  x_it_all_SAM_SDE_std = np.zeros((len(etas),len(rhos),nit))
  # 8) USAM_SDE traj
  x_it_all_USAM_SDE_std = np.zeros((len(etas),len(rhos),nit))
  # 9) USAM_SDE_DRIFT traj
  x_it_all_USAM_SDE_DRIFT_std = np.zeros((len(etas),len(rhos),nit))
  # 10) SAM_SDE_DRIFT traj
  x_it_all_SAM_SDE_DRIFT_std = np.zeros((len(etas),len(rhos),nit))
  # 11) SAM_SDE_TRUE_DRIFT traj
  x_it_all_SAM_SDE_True_DRIFT_std = np.zeros((len(etas),len(rhos),nit))
  
  # initialize empty lists for logging losses
  # 1) SGD loss
  loss_all_SGD = np.zeros((len(etas),len(rhos),nit))
  # 2) SGD_SDE loss
  loss_all_SGD_SDE = np.zeros((len(etas),len(rhos),nit))
  # 3) USAM loss
  loss_all_USAM = np.zeros((len(etas),len(rhos),nit))
  # 4) SAM loss
  loss_all_SAM = np.zeros((len(etas),len(rhos),nit))
  # 5) SAM_TRUE loss
  loss_all_SAM_True = np.zeros((len(etas),len(rhos),nit))
  # 6) SAM_TRUE_SDE loss
  loss_all_SAM_True_SDE = np.zeros((len(etas),len(rhos),nit))
  # 7) SAM_SDE loss
  loss_all_SAM_SDE = np.zeros((len(etas),len(rhos),nit))
  # 8) USAM_SDE loss
  loss_all_USAM_SDE = np.zeros((len(etas),len(rhos),nit))
  # 9) USAM_SDE_DRIFT loss
  loss_all_USAM_SDE_DRIFT = np.zeros((len(etas),len(rhos),nit))
  # 10) SAM_SDE_DRIFT loss
  loss_all_SAM_SDE_DRIFT = np.zeros((len(etas),len(rhos),nit))
  # 11) SAM_SDE_TRUE_DRIFT loss
  loss_all_SAM_SDE_True_DRIFT = np.zeros((len(etas),len(rhos),nit))

  # 1) SGD loss
  loss_all_SGD_std = np.zeros((len(etas),len(rhos),nit))
  # 2) SGD_SDE loss
  loss_all_SGD_SDE_std = np.zeros((len(etas),len(rhos),nit))
  # 3) USAM loss
  loss_all_USAM_std = np.zeros((len(etas),len(rhos),nit))
  # 4) SAM loss
  loss_all_SAM_std = np.zeros((len(etas),len(rhos),nit))
  # 5) SAM_TRUE loss
  loss_all_SAM_True_std = np.zeros((len(etas),len(rhos),nit))
  # 6) SAM_TRUE_SDE loss
  loss_all_SAM_True_SDE_std = np.zeros((len(etas),len(rhos),nit))
  # 7) SAM_SDE loss
  loss_all_SAM_SDE_std = np.zeros((len(etas),len(rhos),nit))
  # 8) USAM_SDE loss
  loss_all_USAM_SDE_std = np.zeros((len(etas),len(rhos),nit))
  # 9) USAM_SDE_DRIFT loss
  loss_all_USAM_SDE_DRIFT_std = np.zeros((len(etas),len(rhos),nit))
  # 10) SAM_SDE_DRIFT loss
  loss_all_SAM_SDE_DRIFT_std = np.zeros((len(etas),len(rhos),nit))
  # 11) SAM_SDE_TRUE_DRIFT loss
  loss_all_SAM_SDE_True_DRIFT_std = np.zeros((len(etas),len(rhos),nit))
  


  #loop over all etas and rhos
  for idx_eta in range(len(etas)):
    
    eta = etas[idx_eta]
    #simulation with finer grid
    dt = eta/cfg.resolution
    
    for idx_rho in tqdm(range(len(rhos))):
     
      # for each rho fix a different seed TODO: put in cfg and clarifxy why we do it
      np.random.seed(idx_rho)
      rho = rhos[idx_rho]
      print('simulating for eta='+str(eta)+', rho ='+str(rho))
     
      #init variables for each combination of eta and rho
      
      # 1) stats SGD
      stats_SGD = []
      # 2) stats SGD_SDE
      stats_SGD_SDE = []
      # 3) stats SAM_TRUE
      stats_SAM_True = []
      # 4) stats SAM_TRUE_SDE
      stats_SAM_True_SDE = []
      # 5) stats SAM
      stats_SAM = []
      # 6) stats SAM_SDE
      stats_SAM_SDE = []
      # 7) stats USAM
      stats_USAM = []
      # 8) stats USAM_SDE
      stats_USAM_SDE = []
      # 9) stats SAM_TRUE_SDE_DRIFT
      stats_SAM_True_SDE_DRIFT = []
      # 10) stats SAM_SDE_DRIFT
      stats_SAM_SDE_DRIFT = []
      # 11) stats USAM_SDE_DRIFT
      stats_USAM_SDE_DRIFT = []

      # 1) loss SGD
      loss_hist_SGD = []
      # 2) loss SGD_SDE
      loss_hist_SGD_SDE = []
      # 3) loss SAM_TRUE
      loss_hist_SAM_True = []
      # 4) loss SAM_TRUE_SDE
      loss_hist_SAM_True_SDE = []
      # 5) loss SAM
      loss_hist_SAM = []
      # 6) loss SAM_SDE
      loss_hist_SAM_SDE = []
      # 7) loss USAM
      loss_hist_USAM = []
      # 8) loss USAM_SDE
      loss_hist_USAM_SDE = []
      # 9) loss SAM_TRUE_SDE_DRIFT
      loss_hist_SAM_SDE_True_DRIFT = []
      # 10) loss SAM_SDE_DRIFT
      loss_hist_SAM_SDE_DRIFT = []
      # 11) loss USAM_SDE_DRIFT
      loss_hist_USAM_SDE_DRIFT = []
      
      # 1) traj SGD
      x_it_hist_SGD = []
      # 2) traj SGD_SDE
      x_it_hist_SGD_SDE = []
      # 3) traj SAM_TRUE
      x_it_hist_SAM_True = []
      # 4) traj SAM_TRUE_SDE
      x_it_hist_SAM_True_SDE = []
      # 5) traj SAM
      x_it_hist_SAM = []
      # 6) traj SAM_SDE
      x_it_hist_SAM_SDE = []
      # 7) traj USAM
      x_it_hist_USAM = []
      # 8) traj USAM_SDE
      x_it_hist_USAM_SDE = []
      # 9) traj SAM_TRUE_SDE_DRIFT
      x_it_hist_SAM_SDE_True_DRIFT = []
      # 10) traj SAM_SDE_DRIFT
      x_it_hist_SAM_SDE_DRIFT = []
      # 11) traj USAM_SDE_DRIFT
      x_it_hist_USAM_SDE_DRIFT = []


      # instantiate task based on cfg
      problem = Opt_problem(cfg)
      
      
      # loop over number of experiments (averaged over)
      for e in tqdm(range(nexp)):

        # 1) SGD
        if cfg.SGD:
          print('Running SGD...')
          x_it, loss_hist, stats, grad_hist,tr_hist = SGD(nit, eta, problem,e+1,fix_random_seed_it)
          stats_SGD.append(stats)      
          loss_hist_SGD.append(loss_hist)
          x_it_hist_SGD.append(np.linalg.norm(x_it,axis=1))

        # 2) SAM_TRUE
        if cfg.SAM_TRUE:
          print('Running SAM_True...')
          x_it, loss_hist, stats, grad_hist,tr_hist = SAM_true(nit, eta, rho,problem,e+1,fix_random_seed_it)
          stats_SAM_True.append(stats)
          loss_hist_SAM_True.append(loss_hist)
          x_it_hist_SAM_True.append(np.linalg.norm(x_it,axis=1))
        
        # 3) SAM
        if cfg.SAM:
          print('Running SAM...')
          x_it, loss_hist, stats, grad_hist,tr_hist = SAM(nit, eta, rho,problem,e+1,fix_random_seed_it)
          stats_SAM.append(stats)
          loss_hist_SAM.append(loss_hist)
          x_it_hist_SAM.append(np.linalg.norm(x_it,axis=1))

        # 4) USAM
        if cfg.USAM:
          print('Running USAM...')
          x_it, loss_hist, stats, grad_hist,tr_hist = USAM(nit, eta, rho,problem,e+1,fix_random_seed_it)
          stats_USAM.append(stats)
          loss_hist_USAM.append(loss_hist)
          x_it_hist_USAM.append(np.linalg.norm(x_it,axis=1))

        # 5) SAM_TRUE_SDE
        if cfg.SAM_TRUE_SDE:
          print('Running SAM_True SDE...')
          x_it, loss_hist, stats, grad_hist,tr_hist = SAM_SDE_true(nit,eta,rho, dt, problem,e+1,fix_random_seed_it,nr1,nr2,nr3)
          stats_SAM_True_SDE.append(stats)
          loss_hist_SAM_True_SDE.append(loss_hist)
          x_it_hist_SAM_True_SDE.append(np.linalg.norm(x_it,axis=1))

        # 6) SAM_TRUE_SDE_DRIFT
        if cfg.SAM_TRUE_SDE_DRIFT:
          print('Running SAM_True SDE no cov...')
          x_it, loss_hist, stats, grad_hist,tr_hist = SAM_SDE_true_NO_COV(nit,eta,rho, dt, problem,e+1,fix_random_seed_it,nr3)
          stats_SAM_True_SDE_DRIFT.append(stats)
          loss_hist_SAM_SDE_True_DRIFT.append(loss_hist)
          x_it_hist_SAM_SDE_True_DRIFT.append(np.linalg.norm(x_it,axis=1))

        # 7) SGD_SDE
        if cfg.SGD_SDE:
          print('Running SGD SDE...')
          x_it, loss_hist, stats, grad_hist,tr_hist = SGD_SDE(nit,eta,dt,problem,e+1,fix_random_seed_it)
          stats_SGD_SDE.append(stats)
          loss_hist_SGD_SDE.append(loss_hist)
          x_it_hist_SGD_SDE.append(np.linalg.norm(x_it,axis=1))

        # 8) SAM_SDE
        if cfg.SAM_SDE:
          print('Running SAM SDE...')
          x_it, loss_hist, stats, grad_hist,tr_hist = SAM_SDE(nit,eta,rho, dt, problem,e+1,fix_random_seed_it)
          stats_SAM_SDE.append(stats)
          loss_hist_SAM_SDE.append(loss_hist)
          x_it_hist_SAM_SDE.append(np.linalg.norm(x_it,axis=1))

        # 9) SAM_SDE_DRIFT
        if cfg.SAM_SDE_DRIFT:
          print('Running SAM SDE no cov...')
          x_it, loss_hist, stats, grad_hist,tr_hist = SAM_SDE_NO_COV(nit,eta,rho, dt, problem,e+1,fix_random_seed_it)
          stats_SAM_SDE_DRIFT.append(stats)
          loss_hist_SAM_SDE_DRIFT.append(loss_hist)
          x_it_hist_SAM_SDE_DRIFT.append(np.linalg.norm(x_it,axis=1))

        # 10) USAM_SDE
        if cfg.USAM_SDE:
          print('Running USAM SDE...')
          x_it, loss_hist, stats, grad_hist,tr_hist = USAM_SDE(nit,eta,rho, dt, problem,e+1,fix_random_seed_it)
          stats_USAM_SDE.append(stats)
          loss_hist_USAM_SDE.append(loss_hist)
          x_it_hist_USAM_SDE.append(np.linalg.norm(x_it,axis=1))

        # 11) USAM_SDE_DRIFT
        if cfg.USAM_SDE_DRIFT:
          print('Running USAM SDE no cov...')
          x_it, loss_hist, stats, grad_hist,tr_hist = USAM_SDE_NO_COV(nit,eta,rho, dt, problem,e+1,fix_random_seed_it)
          stats_USAM_SDE_DRIFT.append(stats)
          loss_hist_USAM_SDE_DRIFT.append(loss_hist)
          x_it_hist_USAM_SDE_DRIFT.append(np.linalg.norm(x_it,axis=1))


      #converting to numpy
      stats_SGD = np.array(stats_SGD)
      stats_SAM_True = np.array(stats_SAM_True)
      stats_SAM = np.array(stats_SAM)
      stats_USAM = np.array(stats_USAM)
      stats_SGD_SDE = np.array(stats_SGD_SDE)
      stats_SAM_True_SDE = np.array(stats_SAM_True_SDE)
      stats_SAM_SDE = np.array(stats_SAM_SDE)
      stats_USAM_SDE = np.array(stats_USAM_SDE)
      stats_USAM_SDE_DRIFT = np.array(stats_USAM_SDE_DRIFT)
      stats_SAM_SDE_DRIFT = np.array(stats_SAM_SDE_DRIFT)
      stats_SAM_SDE_True_DRIFT = np.array(stats_SAM_True_SDE_DRIFT)
 
      loss_hist_SGD = np.array(loss_hist_SGD)
      loss_hist_SAM_True = np.array(loss_hist_SAM_True)
      loss_hist_SAM = np.array(loss_hist_SAM)
      loss_hist_USAM = np.array(loss_hist_USAM)
      loss_hist_SGD_SDE = np.array(loss_hist_SGD_SDE)
      loss_hist_SAM_True_SDE = np.array(loss_hist_SAM_True_SDE)
      loss_hist_SAM_SDE = np.array(loss_hist_SAM_SDE)
      loss_hist_USAM_SDE = np.array(loss_hist_USAM_SDE)
      loss_hist_USAM_SDE_DRIFT = np.array(loss_hist_USAM_SDE_DRIFT)
      loss_hist_SAM_SDE_DRIFT = np.array(loss_hist_SAM_SDE_DRIFT)
      loss_hist_SAM_SDE_True_DRIFT = np.array(loss_hist_SAM_SDE_True_DRIFT)

      x_it_hist_SGD = np.array(x_it_hist_SGD)
      x_it_hist_SAM_True = np.array(x_it_hist_SAM_True)
      x_it_hist_SAM = np.array(x_it_hist_SAM)
      x_it_hist_USAM = np.array(x_it_hist_USAM)
      x_it_hist_SGD_SDE = np.array(x_it_hist_SGD_SDE)
      x_it_hist_SAM_True_SDE = np.array(x_it_hist_SAM_True_SDE)
      x_it_hist_SAM_SDE = np.array(x_it_hist_SAM_SDE)
      x_it_hist_USAM_SDE = np.array(x_it_hist_USAM_SDE)
      x_it_hist_USAM_SDE_DRIFT = np.array(x_it_hist_USAM_SDE_DRIFT)
      x_it_hist_SAM_SDE_DRIFT = np.array(x_it_hist_SAM_SDE_DRIFT)
      x_it_hist_SAM_SDE_True_DRIFT = np.array(x_it_hist_SAM_SDE_True_DRIFT)


      #average over runs
      if cfg.SGD:
        stats_SGD = np.mean(stats_SGD, axis=0)
      if cfg.SAM_TRUE:
        stats_SAM_True = np.mean(stats_SAM_True, axis=0)
      if cfg.SAM:
        stats_SAM = np.mean(stats_SAM, axis=0)
      if cfg.USAM:
        stats_USAM = np.mean(stats_USAM, axis=0)
      if cfg.SGD_SDE:
        stats_SGD_SDE = np.mean(stats_SGD_SDE, axis=0)[::int(eta/dt)]
      if cfg.SAM_TRUE_SDE:
        stats_SAM_True_SDE = np.mean(stats_SAM_True_SDE, axis=0)[::int(eta/dt)]
      if cfg.SAM_SDE:
        stats_SAM_SDE = np.mean(stats_SAM_SDE, axis=0)[::int(eta/dt)]
      if cfg.USAM_SDE:
        stats_USAM_SDE = np.mean(stats_USAM_SDE, axis=0)[::int(eta/dt)]
      if cfg.USAM_SDE_DRIFT:
        stats_USAM_SDE_DRIFT = np.mean(stats_USAM_SDE_DRIFT, axis=0)[::int(eta/dt)]
      if cfg.SAM_SDE_DRIFT:
        stats_SAM_SDE_DRIFT = np.mean(stats_SAM_SDE_DRIFT, axis=0)[::int(eta/dt)]
      if cfg.SAM_TRUE_SDE_DRIFT:
        stats_SAM_SDE_True_DRIFT = np.mean(stats_SAM_SDE_True_DRIFT, axis=0)[::int(eta/dt)]


      # 1) SGD vs SGD_SDE
      if cfg.SGD and cfg.SGD_SDE:
        error_mean_SGD_SGD_SDE[idx_eta, idx_rho] = np.max(np.abs(stats_SGD-stats_SGD_SDE))/(np.mean(np.abs(stats_SGD)))
        error_std_SGD_SGD_SDE[idx_eta, idx_rho] = np.std(np.abs(stats_SGD-stats_SGD_SDE))/(np.mean(np.abs(stats_SGD)))
      # 2) SAM_TRUE vs SGD_SDE
      if cfg.SAM_TRUE and cfg.SGD_SDE:
        error_mean_SAM_True_SGD_SDE[idx_eta, idx_rho] = np.max(np.abs(stats_SAM_True-stats_SGD_SDE))/(np.mean(np.abs(stats_SAM_True)))
        error_std_SAM_True_SGD_SDE[idx_eta, idx_rho] = np.std(np.abs(stats_SAM_True-stats_SGD_SDE))/(np.mean(np.abs(stats_SAM_True)))
      # 3) SAM_TRUE vs SAM_TRUE_SDE
      if cfg.SAM_TRUE and cfg.SAM_TRUE_SDE:
        error_mean_SAM_True_SAM_True_SDE[idx_eta, idx_rho] = np.max(np.abs(stats_SAM_True-stats_SAM_True_SDE))/(np.mean(np.abs(stats_SAM_True)))
        error_std_SAM_True_SAM_True_SDE[idx_eta, idx_rho] = np.std(np.abs(stats_SAM_True-stats_SAM_True_SDE))/(np.mean(np.abs(stats_SAM_True)))
      # 4) SAM_TRUE vs SAM_SDE
      if cfg.SAM_TRUE and cfg.SAM_SDE:
        error_mean_SAM_True_SAM_SDE[idx_eta, idx_rho] = np.max(np.abs(stats_SAM_True-stats_SAM_SDE))/(np.mean(np.abs(stats_SAM_True)))
        error_std_SAM_True_SAM_SDE[idx_eta, idx_rho] = np.std(np.abs(stats_SAM_True-stats_SAM_SDE))/(np.mean(np.abs(stats_SAM_True)))      
      # 5) SAM_TRUE vs USAM_SDE
      if cfg.SAM_TRUE and cfg.USAM_SDE:
        error_mean_SAM_True_USAM_SDE[idx_eta, idx_rho] = np.max(np.abs(stats_SAM_True-stats_USAM_SDE))/(np.mean(np.abs(stats_SAM_True)))
        error_std_SAM_True_USAM_SDE[idx_eta, idx_rho] = np.std(np.abs(stats_SAM_True-stats_USAM_SDE))/(np.mean(np.abs(stats_SAM_True)))      
      # 6) SAM vs SGD_SDE
      if cfg.SAM and cfg.SGD_SDE:
        error_mean_SAM_SGD_SDE[idx_eta, idx_rho] = np.max(np.abs(stats_SAM-stats_SGD_SDE))/(np.mean(np.abs(stats_SAM)))
        error_std_SAM_SGD_SDE[idx_eta, idx_rho] = np.std(np.abs(stats_SAM-stats_SGD_SDE))/(np.mean(np.abs(stats_SAM)))
      # 7) SAM vs SAM_SDE
      if cfg.SAM and cfg.SAM_SDE:
        error_mean_SAM_SAM_SDE[idx_eta, idx_rho] = np.max(np.abs(stats_SAM-stats_SAM_SDE))/(np.mean(np.abs(stats_SAM)))
        error_std_SAM_SAM_SDE[idx_eta, idx_rho] = np.std(np.abs(stats_SAM-stats_SAM_SDE))/(np.mean(np.abs(stats_SAM)))
      # 8) USAM vs USAM_SDE
      if cfg.USAM and cfg.USAM_SDE:
        error_mean_USAM_USAM_SDE[idx_eta, idx_rho] = np.max(np.abs(stats_USAM-stats_USAM_SDE))/(np.mean(np.abs(stats_USAM)))
        error_std_USAM_USAM_SDE[idx_eta, idx_rho] = np.std(np.abs(stats_USAM-stats_USAM_SDE))/(np.mean(np.abs(stats_USAM)))
      # 9) USAM vs SGD_SDE
      if cfg.USAM and cfg.SGD_SDE:
        error_mean_USAM_SGD_SDE[idx_eta, idx_rho] = np.max(np.abs(stats_USAM-stats_SGD_SDE))/(np.mean(np.abs(stats_USAM)))
        error_std_USAM_SGD_SDE[idx_eta, idx_rho] = np.std(np.abs(stats_USAM-stats_SGD_SDE))/(np.mean(np.abs(stats_USAM)))      
      # 10) USAM vs USAM_SDE_DRIFT
      if cfg.USAM and cfg.USAM_SDE_DRIFT:
        error_mean_USAM_USAM_SDE_DRIFT[idx_eta, idx_rho] = np.max(np.abs(stats_USAM-stats_USAM_SDE_DRIFT))/(np.mean(np.abs(stats_USAM)))
        error_std_USAM_USAM_SDE_DRIFT[idx_eta, idx_rho] = np.std(np.abs(stats_USAM-stats_USAM_SDE_DRIFT))/(np.mean(np.abs(stats_USAM)))      
      # 11) SAM vs SAM_SDE_DRIFT
      if cfg.SAM and cfg.SAM_SDE_DRIFT:
        error_mean_SAM_SAM_SDE_DRIFT[idx_eta, idx_rho] = np.max(np.abs(stats_SAM-stats_SAM_SDE_DRIFT))/(np.mean(np.abs(stats_SAM)))
        error_std_SAM_SAM_SDE_DRIFT[idx_eta, idx_rho] = np.std(np.abs(stats_SAM-stats_SAM_SDE_DRIFT))/(np.mean(np.abs(stats_SAM)))      
      # 12) SAM_TRUE vs SAM_TRUE_SDE_DRIFT
      if cfg.SAM_TRUE and cfg.SAM_TRUE_SDE_DRIFT:
        error_mean_SAM_True_SAM_True_SDE_DRIFT[idx_eta, idx_rho] = np.max(np.abs(stats_SAM_True-stats_SAM_SDE_True_DRIFT))/(np.mean(np.abs(stats_SAM_True)))
        error_std_SAM_True_SAM_True_SDE_DRIFT[idx_eta, idx_rho] = np.std(np.abs(stats_SAM_True-stats_SAM_SDE_True_DRIFT))/(np.mean(np.abs(stats_SAM_True)))


      # 1) SGD loss
      if cfg.SGD:
        loss_all_SGD[idx_eta, idx_rho,:] = np.mean(loss_hist_SGD, axis=0)
        loss_all_SGD_std[idx_eta, idx_rho,:] = np.std(loss_hist_SGD, axis=0)
      # 2) SAM_TRUE loss
      if cfg.SAM_TRUE:
        loss_all_SAM_True[idx_eta, idx_rho,:] = np.mean(loss_hist_SAM_True, axis=0)
        loss_all_SAM_True_std[idx_eta, idx_rho,:] = np.std(loss_hist_SAM_True, axis=0)
      # 3) SAM loss
      if cfg.SAM:
        loss_all_SAM[idx_eta, idx_rho,:] = np.mean(loss_hist_SAM, axis=0)
        loss_all_SAM_std[idx_eta, idx_rho,:] = np.std(loss_hist_SAM, axis=0)
      # 4) USAM loss
      if cfg.USAM:
        loss_all_USAM[idx_eta, idx_rho,:] = np.mean(loss_hist_USAM, axis=0)
        loss_all_USAM_std[idx_eta, idx_rho,:] = np.std(loss_hist_USAM, axis=0)
      # 5) SGD_SDE loss
      if cfg.SGD_SDE:
        loss_all_SGD_SDE[idx_eta, idx_rho,:] = np.mean(loss_hist_SGD_SDE, axis=0)[::int(eta/dt)]
        loss_all_SGD_SDE_std[idx_eta, idx_rho,:] = np.std(loss_hist_SGD_SDE, axis=0)[::int(eta/dt)]
      # 6) SAM_TRUE_SDE loss
      if cfg.SAM_TRUE_SDE:
        loss_all_SAM_True_SDE[idx_eta, idx_rho,:] = np.mean(loss_hist_SAM_True_SDE, axis=0)[::int(eta/dt)]
        loss_all_SAM_True_SDE_std[idx_eta, idx_rho,:] = np.std(loss_hist_SAM_True_SDE, axis=0)[::int(eta/dt)]
      # 7) SAM_SDE loss
      if cfg.SAM_SDE:
        loss_all_SAM_SDE[idx_eta, idx_rho,:] = np.mean(loss_hist_SAM_SDE, axis=0)[::int(eta/dt)]
        loss_all_SAM_SDE_std[idx_eta, idx_rho,:] = np.std(loss_hist_SAM_SDE, axis=0)[::int(eta/dt)]
      # 8) USAM_SDE loss
      if cfg.USAM_SDE:
        loss_all_USAM_SDE[idx_eta, idx_rho,:] = np.mean(loss_hist_USAM_SDE, axis=0)[::int(eta/dt)]
        loss_all_USAM_SDE_std[idx_eta, idx_rho,:] = np.std(loss_hist_USAM_SDE, axis=0)[::int(eta/dt)]
      # 9) USAM_SDE_DRIFT loss
      if cfg.USAM_SDE_DRIFT:
        loss_all_USAM_SDE_DRIFT[idx_eta, idx_rho,:] = np.mean(loss_hist_USAM_SDE_DRIFT, axis=0)[::int(eta/dt)]
        loss_all_USAM_SDE_DRIFT_std[idx_eta, idx_rho,:] = np.std(loss_hist_USAM_SDE_DRIFT, axis=0)[::int(eta/dt)]
      # 10) SAM_SDE_DRIFT loss
      if cfg.SAM_SDE_DRIFT:
        loss_all_SAM_SDE_DRIFT[idx_eta, idx_rho,:] = np.mean(loss_hist_SAM_SDE_DRIFT, axis=0)[::int(eta/dt)]
        loss_all_SAM_SDE_DRIFT_std[idx_eta, idx_rho,:] = np.std(loss_hist_SAM_SDE_DRIFT, axis=0)[::int(eta/dt)]
      # 11) SAM_SDE_TRUE_DRIFT loss
      if cfg.SAM_TRUE_SDE_DRIFT:
        loss_all_SAM_SDE_True_DRIFT[idx_eta, idx_rho,:] = np.mean(loss_hist_SAM_SDE_True_DRIFT, axis=0)[::int(eta/dt)]
        loss_all_SAM_SDE_True_DRIFT_std[idx_eta, idx_rho,:] = np.std(loss_hist_SAM_SDE_True_DRIFT, axis=0)[::int(eta/dt)]


      # 1) SGD traj
      if cfg.SGD:      
        x_it_all_SGD[idx_eta, idx_rho,:] = np.mean(x_it_hist_SGD, axis=0)
        x_it_all_SGD_std[idx_eta, idx_rho,:] = np.std(x_it_hist_SGD, axis=0)
      # 2) SAM_TRUE traj
      if cfg.SAM_TRUE:
        x_it_all_SAM_True[idx_eta, idx_rho,:] = np.mean(x_it_hist_SAM_True, axis=0)
        x_it_all_SAM_True_std[idx_eta, idx_rho,:] = np.std(x_it_hist_SAM_True, axis=0)
      # 3) SAM traj
      if cfg.SAM:
        x_it_all_SAM[idx_eta, idx_rho,:] = np.mean(x_it_hist_SAM, axis=0)
        x_it_all_SAM_std[idx_eta, idx_rho,:] = np.std(x_it_hist_SAM, axis=0)
      # 4) USAM traj
      if cfg.USAM:
        x_it_all_USAM[idx_eta, idx_rho,:] = np.mean(x_it_hist_USAM, axis=0)
        x_it_all_USAM_std[idx_eta, idx_rho,:] = np.std(x_it_hist_USAM, axis=0)
      # 5) SGD_SDE traj
      if cfg.SGD_SDE:
        x_it_all_SGD_SDE[idx_eta, idx_rho,:] = np.mean(x_it_hist_SGD_SDE, axis=0)[::int(eta/dt)]
        x_it_all_SGD_SDE_std[idx_eta, idx_rho,:] = np.std(x_it_hist_SGD_SDE, axis=0)[::int(eta/dt)]
      # 6) SAM_TRUE_SDE traj
      if cfg.SAM_TRUE_SDE:
        x_it_all_SAM_True_SDE[idx_eta, idx_rho,:] = np.mean(x_it_hist_SAM_True_SDE, axis=0)[::int(eta/dt)]
        x_it_all_SAM_True_SDE_std[idx_eta, idx_rho,:] = np.std(x_it_hist_SAM_True_SDE, axis=0)[::int(eta/dt)]
      # 7) SAM_SDE traj
      if cfg.SAM_SDE:
        x_it_all_SAM_SDE[idx_eta, idx_rho,:] = np.mean(x_it_hist_SAM_SDE, axis=0)[::int(eta/dt)]
        x_it_all_SAM_SDE_std[idx_eta, idx_rho,:] = np.std(x_it_hist_SAM_SDE, axis=0)[::int(eta/dt)]
      # 8) USAM_SDE traj
      if cfg.USAM_SDE:
        x_it_all_USAM_SDE[idx_eta, idx_rho,:] = np.mean(x_it_hist_USAM_SDE, axis=0)[::int(eta/dt)]
        x_it_all_USAM_SDE_std[idx_eta, idx_rho,:] = np.std(x_it_hist_USAM_SDE, axis=0)[::int(eta/dt)]
      # 9) USAM_SDE_DRIFT traj
      if cfg.USAM_SDE_DRIFT:
        x_it_all_USAM_SDE_DRIFT[idx_eta, idx_rho,:] = np.mean(x_it_hist_USAM_SDE_DRIFT, axis=0)[::int(eta/dt)]
        x_it_all_USAM_SDE_DRIFT_std[idx_eta, idx_rho,:] = np.std(x_it_hist_USAM_SDE_DRIFT, axis=0)[::int(eta/dt)]
      # 10) SAM_SDE_DRIFT traj
      if cfg.SAM_SDE_DRIFT:
        x_it_all_SAM_SDE_DRIFT[idx_eta, idx_rho,:] = np.mean(x_it_hist_SAM_SDE_DRIFT, axis=0)[::int(eta/dt)]
        x_it_all_SAM_SDE_DRIFT_std[idx_eta, idx_rho,:] = np.std(x_it_hist_SAM_SDE_DRIFT, axis=0)[::int(eta/dt)]
      # 11) SAM_SDE_TRUE_DRIFT traj
      if cfg.SAM_TRUE_SDE_DRIFT:
        x_it_all_SAM_SDE_True_DRIFT[idx_eta, idx_rho,:] = np.mean(x_it_hist_SAM_SDE_True_DRIFT, axis=0)[::int(eta/dt)]
        x_it_all_SAM_SDE_True_DRIFT_std[idx_eta, idx_rho,:] = np.std(x_it_hist_SAM_SDE_True_DRIFT, axis=0)[::int(eta/dt)]
      
      
    

      results={'error_mean_SGD_SGD_SDE':error_mean_SGD_SGD_SDE,'error_std_SGD_SGD_SDE':error_std_SGD_SGD_SDE,
               'error_mean_SAM_True_SGD_SDE':error_mean_SAM_True_SGD_SDE,'error_std_SAM_True_SGD_SDE':error_std_SAM_True_SGD_SDE,
               'error_mean_SAM_True_SAM_True_SDE':error_mean_SAM_True_SAM_True_SDE,'error_std_SAM_True_SAM_True_SDE':error_std_SAM_True_SAM_True_SDE,
               'error_mean_SAM_True_SAM_SDE':error_mean_SAM_True_SAM_SDE,'error_std_SAM_True_SAM_SDE':error_std_SAM_True_SAM_SDE,
               'error_mean_SAM_True_USAM_SDE':error_mean_SAM_True_USAM_SDE,'error_std_SAM_True_USAM_SDE':error_std_SAM_True_USAM_SDE,
               'error_mean_SAM_SGD_SDE':error_mean_SAM_SGD_SDE,'error_std_SAM_SGD_SDE':error_std_SAM_SGD_SDE,
               'error_mean_SAM_SAM_SDE':error_mean_SAM_SAM_SDE,'error_std_SAM_SAM_SDE':error_std_SAM_SAM_SDE,
               'error_mean_USAM_USAM_SDE':error_mean_USAM_USAM_SDE,'error_std_USAM_USAM_SDE':error_std_USAM_USAM_SDE,
               'error_mean_USAM_SGD_SDE':error_mean_USAM_SGD_SDE,'error_std_USAM_SGD_SDE':error_std_USAM_SGD_SDE,
               'error_mean_USAM_USAM_SDE_DRIFT':error_mean_USAM_USAM_SDE_DRIFT,'error_std_USAM_USAM_SDE_DRIFT':error_std_USAM_USAM_SDE_DRIFT,
               'error_mean_SAM_SAM_SDE_DRIFT':error_mean_SAM_SAM_SDE_DRIFT,'error_std_SAM_SAM_SDE_DRIFT':error_std_SAM_SAM_SDE_DRIFT,
               'error_mean_SAM_True_SAM_True_SDE_DRIFT':error_mean_SAM_True_SAM_True_SDE_DRIFT,'error_std_SAM_True_SAM_True_SDE_DRIFT':error_std_SAM_True_SAM_True_SDE_DRIFT,

               'loss_all_SGD':loss_all_SGD,'loss_all_SGD_std':loss_all_SGD_std,
               'loss_all_SGD_SDE':loss_all_SGD_SDE,'loss_all_SGD_SDE_std':loss_all_SGD_SDE_std, 
               'loss_all_SAM_True':loss_all_SAM_True,'loss_all_SAM_True_std':loss_all_SAM_True_std,
               'loss_all_SAM_True_SDE':loss_all_SAM_True_SDE,'loss_all_SAM_True_SDE_std':loss_all_SAM_True_SDE_std,
               'loss_all_USAM': loss_all_USAM,'loss_all_USAM_std':loss_all_USAM_std,
               'loss_all_USAM_SDE':loss_all_USAM_SDE,'loss_all_USAM_SDE_std':loss_all_USAM_SDE_std,
               'loss_all_SAM':loss_all_SAM,'loss_all_SAM_std':loss_all_SAM_std,
               'loss_all_SAM_SDE':loss_all_SAM_SDE,'loss_all_SAM_SDE_std':loss_all_SAM_SDE_std,
               'loss_all_USAM_SDE_DRIFT':loss_all_USAM_SDE_DRIFT,'loss_all_USAM_SDE_DRIFT_std':loss_all_USAM_SDE_DRIFT_std,
               'loss_all_SAM_SDE_DRIFT':loss_all_SAM_SDE_DRIFT,'loss_all_SAM_SDE_DRIFT_std':loss_all_SAM_SDE_DRIFT_std,
               'loss_all_SAM_SDE_True_DRIFT':loss_all_SAM_SDE_True_DRIFT,'loss_all_SAM_SDE_True_DRIFT_std':loss_all_SAM_SDE_True_DRIFT_std,

                'x_it_all_SGD':x_it_all_SGD,'x_it_all_SGD_std':x_it_all_SGD_std,
                'x_it_all_SGD_SDE':x_it_all_SGD_SDE,'x_it_all_SGD_SDE_std':x_it_all_SGD_SDE_std, 
                'x_it_all_SAM_True':x_it_all_SAM_True,'x_it_all_SAM_True_std':x_it_all_SAM_True_std,
                'x_it_all_SAM_True_SDE':x_it_all_SAM_True_SDE,'x_it_all_SAM_True_SDE_std':x_it_all_SAM_True_SDE_std,
                'x_it_all_USAM': x_it_all_USAM,'x_it_all_USAM_std':x_it_all_USAM_std,
                'x_it_all_USAM_SDE':x_it_all_USAM_SDE,'x_it_all_USAM_SDE_std':x_it_all_USAM_SDE_std,
                'x_it_all_SAM':x_it_all_SAM,'x_it_all_SAM_std':x_it_all_SAM_std,
                'x_it_all_SAM_SDE':x_it_all_SAM_SDE,'x_it_all_SAM_SDE_std':x_it_all_SAM_SDE_std,
                'x_it_all_USAM_SDE_DRIFT':x_it_all_USAM_SDE_DRIFT,'x_it_all_USAM_SDE_DRIFT_std':x_it_all_USAM_SDE_DRIFT_std,
                'x_it_all_SAM_SDE_DRIFT':x_it_all_SAM_SDE_DRIFT,'x_it_all_SAM_SDE_DRIFT_std':x_it_all_SAM_SDE_DRIFT_std,
                'x_it_all_SAM_SDE_True_DRIFT':x_it_all_SAM_SDE_True_DRIFT,'x_it_all_SAM_SDE_True_DRIFT_std':x_it_all_SAM_SDE_True_DRIFT_std,

                'etas':etas,'rhos':rhos, 'f_star':f_star, 'problem':problem}

      with open('res_{}.pkl'.format(name), 'wb') as f:
          pickle.dump(results, f)
      print("")   

      





if __name__ == "__main__":
  main()

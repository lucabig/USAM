import numpy as np
import jax
import copy
import jax.numpy as jnp
np.random.seed(42) # Added by Enea 
from sklearn.datasets import load_breast_cancer
from losses import *
from opt_sdes import *
from sklearn.preprocessing import MinMaxScaler
import pickle 
from tqdm import tqdm
import hydra
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)


@hydra.main(config_name='run')
def main(cfg):

  #training settings
  nexp = cfg.nexp
  nit = cfg.nit

  # optimizer settings
  etas = cfg.etas
  rhos = cfg.rhos
  f_star = cfg.f_star

  name = cfg.name
  d_quadratic = cfg.d_quadratic
  HIDDEN = cfg.HIDDEN
  sigma = cfg.sigma

  error_mean_SGD_SGD_SDE = np.zeros((len(etas),len(rhos)))
  error_mean_SAM_True_SGD_SDE = np.zeros((len(etas),len(rhos)))
  error_mean_SAM_True_SAM_True_SDE = np.zeros((len(etas),len(rhos)))
  error_mean_SAM_True_SAM_SDE = np.zeros((len(etas),len(rhos)))
  error_mean_SAM_True_USAM_SDE = np.zeros((len(etas),len(rhos)))
  error_mean_SAM_SGD_SDE = np.zeros((len(etas),len(rhos)))
  error_mean_SAM_SAM_SDE = np.zeros((len(etas),len(rhos)))
  error_mean_USAM_USAM_SDE = np.zeros((len(etas),len(rhos)))
  error_mean_USAM_SGD_SDE = np.zeros((len(etas),len(rhos)))
  error_mean_USAM_USAM_SDE_DRIFT = np.zeros((len(etas),len(rhos)))
  error_mean_SAM_SAM_SDE_DRIFT = np.zeros((len(etas),len(rhos)))
  error_mean_SAM_True_SAM_True_SDE_DRIFT = np.zeros((len(etas),len(rhos)))


  error_std_SGD_SGD_SDE = np.zeros((len(etas),len(rhos)))
  error_std_SAM_True_SGD_SDE = np.zeros((len(etas),len(rhos)))
  error_std_SAM_True_SAM_True_SDE = np.zeros((len(etas),len(rhos)))
  error_std_SAM_True_SAM_SDE = np.zeros((len(etas),len(rhos)))
  error_std_SAM_True_USAM_SDE = np.zeros((len(etas),len(rhos)))
  error_std_SAM_SGD_SDE = np.zeros((len(etas),len(rhos)))
  error_std_SAM_SAM_SDE = np.zeros((len(etas),len(rhos)))
  error_std_USAM_USAM_SDE = np.zeros((len(etas),len(rhos)))
  error_std_USAM_SGD_SDE = np.zeros((len(etas),len(rhos)))
  error_std_USAM_USAM_SDE_DRIFT = np.zeros((len(etas),len(rhos)))
  error_std_SAM_SAM_SDE_DRIFT = np.zeros((len(etas),len(rhos)))
  error_std_SAM_True_SAM_True_SDE_DRIFT = np.zeros((len(etas),len(rhos)))



  x_it_all_SGD = np.zeros((len(etas),len(rhos),nit))
  x_it_all_SGD_SDE = np.zeros((len(etas),len(rhos),nit))
  x_it_all_USAM = np.zeros((len(etas),len(rhos),nit))
  x_it_all_SAM = np.zeros((len(etas),len(rhos),nit))
  x_it_all_SAM_True = np.zeros((len(etas),len(rhos),nit))
  x_it_all_SAM_True_SDE = np.zeros((len(etas),len(rhos),nit))
  x_it_all_SAM_SDE = np.zeros((len(etas),len(rhos),nit))
  x_it_all_USAM_SDE = np.zeros((len(etas),len(rhos),nit))
  x_it_all_USAM_SDE_DRIFT = np.zeros((len(etas),len(rhos),nit))
  x_it_all_SAM_SDE_DRIFT = np.zeros((len(etas),len(rhos),nit))
  x_it_all_SAM_True_SDE_DRIFT = np.zeros((len(etas),len(rhos),nit))

  x_it_all_SGD_std = np.zeros((len(etas),len(rhos),nit))
  x_it_all_SGD_SDE_std = np.zeros((len(etas),len(rhos),nit))
  x_it_all_USAM_std = np.zeros((len(etas),len(rhos),nit))
  x_it_all_SAM_std = np.zeros((len(etas),len(rhos),nit))
  x_it_all_SAM_True_std = np.zeros((len(etas),len(rhos),nit))
  x_it_all_SAM_True_SDE_std = np.zeros((len(etas),len(rhos),nit))
  x_it_all_SAM_SDE_std = np.zeros((len(etas),len(rhos),nit))
  x_it_all_USAM_SDE_std = np.zeros((len(etas),len(rhos),nit))
  x_it_all_USAM_SDE_DRIFT_std = np.zeros((len(etas),len(rhos),nit))
  x_it_all_SAM_SDE_DRIFT_std = np.zeros((len(etas),len(rhos),nit))
  x_it_all_SAM_True_SDE_DRIFT_std = np.zeros((len(etas),len(rhos),nit))





  loss_all_SGD = np.zeros((len(etas),len(rhos),nit))
  loss_all_SGD_SDE = np.zeros((len(etas),len(rhos),nit))
  loss_all_USAM = np.zeros((len(etas),len(rhos),nit))
  loss_all_SAM = np.zeros((len(etas),len(rhos),nit))
  loss_all_SAM_True = np.zeros((len(etas),len(rhos),nit))
  loss_all_SAM_True_SDE = np.zeros((len(etas),len(rhos),nit))
  loss_all_SAM_SDE = np.zeros((len(etas),len(rhos),nit))
  loss_all_USAM_SDE = np.zeros((len(etas),len(rhos),nit))
  loss_all_USAM_SDE_DRIFT = np.zeros((len(etas),len(rhos),nit))
  loss_all_SAM_SDE_DRIFT = np.zeros((len(etas),len(rhos),nit))
  loss_all_SAM_True_SDE_DRIFT = np.zeros((len(etas),len(rhos),nit))

  loss_all_SGD_std = np.zeros((len(etas),len(rhos),nit))
  loss_all_SGD_SDE_std = np.zeros((len(etas),len(rhos),nit))
  loss_all_USAM_std = np.zeros((len(etas),len(rhos),nit))
  loss_all_SAM_std = np.zeros((len(etas),len(rhos),nit))
  loss_all_SAM_True_std = np.zeros((len(etas),len(rhos),nit))
  loss_all_SAM_True_SDE_std = np.zeros((len(etas),len(rhos),nit))
  loss_all_SAM_SDE_std = np.zeros((len(etas),len(rhos),nit))
  loss_all_USAM_SDE_std = np.zeros((len(etas),len(rhos),nit))
  loss_all_USAM_SDE_DRIFT_std = np.zeros((len(etas),len(rhos),nit))
  loss_all_SAM_SDE_DRIFT_std = np.zeros((len(etas),len(rhos),nit))
  loss_all_SAM_True_SDE_DRIFT_std = np.zeros((len(etas),len(rhos),nit))



  #big loop over all etas and rhos
  for idx_eta in range(len(etas)):
    eta = etas[idx_eta]
    dt = eta/cfg.resolution #simulation with finer grid
    for idx_rho in tqdm(range(len(rhos))):
      np.random.seed(idx_rho)
      rho = rhos[idx_rho]
      print('simulating for eta='+str(eta)+', rho ='+str(rho))
      #init variables
      stats_SGD = []
      stats_SGD_SDE = []
      stats_SAM_True = []
      stats_SAM_True_SDE = []
      stats_SAM = []
      stats_SAM_SDE = []
      stats_USAM = []
      stats_USAM_SDE = []
      stats_SAM_True_SDE_DRIFT = []
      stats_SAM_SDE_DRIFT = []
      stats_USAM_SDE_DRIFT = []

      loss_hist_SGD = []
      loss_hist_SGD_SDE = []
      loss_hist_SAM_True = []
      loss_hist_SAM_True_SDE = []
      loss_hist_SAM = []
      loss_hist_SAM_SDE = []
      loss_hist_USAM = []
      loss_hist_USAM_SDE = []
      loss_hist_SAM_True_SDE_DRIFT = []
      loss_hist_SAM_SDE_DRIFT = []
      loss_hist_USAM_SDE_DRIFT = []

      x_it_hist_SGD = []
      x_it_hist_SGD_SDE = []
      x_it_hist_SAM_True = []
      x_it_hist_SAM_True_SDE = []
      x_it_hist_SAM = []
      x_it_hist_SAM_SDE = []
      x_it_hist_USAM = []
      x_it_hist_USAM_SDE = []
      x_it_hist_SAM_True_SDE_DRIFT = []
      x_it_hist_SAM_SDE_DRIFT = []
      x_it_hist_USAM_SDE_DRIFT = []


      problem = Opt_problem(cfg)

      #do experiments
      for e in tqdm(range(nexp)):

        if not cfg.sdes_only:  
          print('Running SGD...')
          x_it, loss_hist, stats, grad_hist,tr_hist = SGD(nit, eta, problem,e+1)
          stats_SGD.append(stats)      
          loss_hist_SGD.append(loss_hist)
          x_it_hist_SGD.append(np.linalg.norm(x_it,axis=1))


          print('Running SAM_True...')
          x_it, loss_hist, stats, grad_hist,tr_hist = SAM_true(nit, eta, rho,problem,e+1)
          stats_SAM_True.append(stats)
          loss_hist_SAM_True.append(loss_hist)
          x_it_hist_SAM_True.append(np.linalg.norm(x_it,axis=1))
          
          print('Running SAM...')
          x_it, loss_hist, stats, grad_hist,tr_hist = SAM(nit, eta, rho,problem,e+1)
          stats_SAM.append(stats)
          loss_hist_SAM.append(loss_hist)
          x_it_hist_SAM.append(np.linalg.norm(x_it,axis=1))

          print('Running USAM...')
          x_it, loss_hist, stats, grad_hist,tr_hist = USAM(nit, eta, rho,problem,e+1)
          stats_USAM.append(stats)
          loss_hist_USAM.append(loss_hist)
          x_it_hist_USAM.append(np.linalg.norm(x_it,axis=1))



        if cfg.sdes:

          print('Running SAM_True SDE...')
          x_it, loss_hist, stats, grad_hist,tr_hist = SAM_SDE1_true(nit,eta,rho, dt, problem,e+1)
          stats_SAM_True_SDE.append(stats)
          loss_hist_SAM_True_SDE.append(loss_hist)
          x_it_hist_SAM_True_SDE.append(np.linalg.norm(x_it,axis=1))

          print('Running SAM_True SDE no cov...')
          x_it, loss_hist, stats, grad_hist,tr_hist = SAM_SDE1_true_NO_COV(nit,eta,rho, dt, problem,e+1)
          stats_SAM_True_SDE_DRIFT.append(stats)
          loss_hist_SAM_True_SDE_DRIFT.append(loss_hist)
          x_it_hist_SAM_True_SDE_DRIFT.append(np.linalg.norm(x_it,axis=1))

          print('Running SGD SDE...')
          x_it, loss_hist, stats, grad_hist,tr_hist = SGD_SDE(nit,eta,dt,problem,e+1)
          stats_SGD_SDE.append(stats)
          loss_hist_SGD_SDE.append(loss_hist)
          x_it_hist_SGD_SDE.append(np.linalg.norm(x_it,axis=1))

          print('Running SAM SDE...')
          x_it, loss_hist, stats, grad_hist,tr_hist = SAM_SDE1(nit,eta,rho, dt, problem,e+1)
          stats_SAM_SDE.append(stats)
          loss_hist_SAM_SDE.append(loss_hist)
          x_it_hist_SAM_SDE.append(np.linalg.norm(x_it,axis=1))

          print('Running SAM SDE no cov...')
          x_it, loss_hist, stats, grad_hist,tr_hist = SAM_SDE1_NO_COV(nit,eta,rho, dt, problem,e+1)
          stats_SAM_SDE_DRIFT.append(stats)
          loss_hist_SAM_SDE_DRIFT.append(loss_hist)
          x_it_hist_SAM_SDE_DRIFT.append(np.linalg.norm(x_it,axis=1))

          print('Running USAM SDE...')
          x_it, loss_hist, stats, grad_hist,tr_hist = USAM_SDE1(nit,eta,rho, dt, problem,e+1)
          stats_USAM_SDE.append(stats)
          loss_hist_USAM_SDE.append(loss_hist)
          x_it_hist_USAM_SDE.append(np.linalg.norm(x_it,axis=1))

          print('Running USAM SDE no cov...')
          x_it, loss_hist, stats, grad_hist,tr_hist = USAM_SDE1_NO_COV(nit,eta,rho, dt, problem,e+1)
          stats_USAM_SDE_DRIFT.append(stats)
          loss_hist_USAM_SDE_DRIFT.append(loss_hist)
          x_it_hist_USAM_SDE_DRIFT.append(np.linalg.norm(x_it,axis=1))


      #converting to numpy
      stats_SGD = np.array(stats_SGD)
      stats_SAM_True = np.array(stats_SAM_True)
      stats_SAM = np.array(stats_SAM)
      stats_USAM = np.array(stats_USAM)
      if cfg.sdes:
        stats_SGD_SDE = np.array(stats_SGD_SDE)
        stats_SAM_True_SDE = np.array(stats_SAM_True_SDE)
        stats_SAM_SDE = np.array(stats_SAM_SDE)
        stats_USAM_SDE = np.array(stats_USAM_SDE)
        stats_USAM_SDE_DRIFT = np.array(stats_USAM_SDE_DRIFT)
        stats_SAM_SDE_DRIFT = np.array(stats_SAM_SDE_DRIFT)
        stats_SAM_SDE_True_DRIFT = np.array(stats_SAM_SDE_True_DRIFT)
 
      loss_hist_SGD = np.array(loss_hist_SGD)
      loss_hist_SAM_True = np.array(loss_hist_SAM_True)
      loss_hist_SAM = np.array(loss_hist_SAM)
      loss_hist_USAM = np.array(loss_hist_USAM)
      if cfg.sdes:
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
      if cfg.sdes:
        x_it_hist_SGD_SDE = np.array(x_it_hist_SGD_SDE)
        x_it_hist_SAM_True_SDE = np.array(x_it_hist_SAM_True_SDE)
        x_it_hist_SAM_SDE = np.array(x_it_hist_SAM_SDE)
        x_it_hist_USAM_SDE = np.array(x_it_hist_USAM_SDE)
        x_it_hist_USAM_SDE_DRIFT = np.array(x_it_hist_USAM_SDE_DRIFT)
        x_it_hist_SAM_SDE_DRIFT = np.array(x_it_hist_SAM_SDE_DRIFT)
        x_it_hist_SAM_SDE_True_DRIFT = np.array(x_it_hist_SAM_SDE_True_DRIFT)


      #average over runs
      stats_SGD = np.mean(stats_SGD, axis=0)
      stats_SAM_True = np.mean(stats_SAM_True, axis=0)
      stats_SAM = np.mean(stats_SAM, axis=0)
      stats_USAM = np.mean(stats_USAM, axis=0)
      if cfg.sdes:
        stats_SGD_SDE = np.mean(stats_SGD_SDE, axis=0)[::int(eta/dt)]
        stats_SAM_True_SDE = np.mean(stats_SAM_True_SDE, axis=0)[::int(eta/dt)]
        stats_SAM_SDE = np.mean(stats_SAM_SDE, axis=0)[::int(eta/dt)]
        stats_USAM_SDE = np.mean(stats_USAM_SDE, axis=0)[::int(eta/dt)]
        stats_USAM_SDE_DRIFT = np.mean(stats_USAM_SDE_DRIFT, axis=0)[::int(eta/dt)]
        stats_SAM_SDE_DRIFT = np.mean(stats_SAM_SDE_DRIFT, axis=0)[::int(eta/dt)]
        stats_SAM_SDE_True_DRIFT = np.mean(stats_SAM_SDE_True_DRIFT, axis=0)[::int(eta/dt)]


        error_mean_SGD_SGD_SDE[idx_eta, idx_rho] = np.max(np.abs(stats_SGD-stats_SGD_SDE))/(np.mean(np.abs(stats_SGD)))
        error_mean_SAM_True_SGD_SDE[idx_eta, idx_rho] = np.max(np.abs(stats_SAM_True-stats_SGD_SDE))/(np.mean(np.abs(stats_SAM_True)))
        error_mean_SAM_True_SAM_True_SDE[idx_eta, idx_rho] = np.max(np.abs(stats_SAM_True-stats_SAM_True_SDE))/(np.mean(np.abs(stats_SAM_True)))
        error_mean_SAM_SGD_SDE[idx_eta, idx_rho] = np.max(np.abs(stats_SAM-stats_SGD_SDE))/(np.mean(np.abs(stats_SAM)))
        error_mean_SAM_SAM_SDE[idx_eta, idx_rho] = np.max(np.abs(stats_SAM-stats_SAM_SDE))/(np.mean(np.abs(stats_SAM)))
        error_mean_USAM_SGD_SDE[idx_eta, idx_rho] = np.max(np.abs(stats_USAM-stats_SGD_SDE))/(np.mean(np.abs(stats_USAM)))
        error_mean_USAM_USAM_SDE[idx_eta, idx_rho] = np.max(np.abs(stats_USAM-stats_USAM_SDE))/(np.mean(np.abs(stats_USAM)))
        error_mean_SAM_True_SAM_SDE[idx_eta, idx_rho] = np.max(np.abs(stats_SAM_True-stats_SAM_SDE))/(np.mean(np.abs(stats_SAM_True)))
        error_mean_SAM_True_USAM_SDE[idx_eta, idx_rho] = np.max(np.abs(stats_SAM_True-stats_USAM_SDE))/(np.mean(np.abs(stats_SAM_True)))

        error_std_SGD_SGD_SDE[idx_eta, idx_rho] = np.std(np.abs(stats_SGD-stats_SGD_SDE))/(np.mean(np.abs(stats_SGD)))
        error_std_SAM_True_SGD_SDE[idx_eta, idx_rho] = np.std(np.abs(stats_SAM_True-stats_SGD_SDE))/(np.mean(np.abs(stats_SAM_True)))
        error_std_SAM_True_SAM_True_SDE[idx_eta, idx_rho] = np.std(np.abs(stats_SAM_True-stats_SAM_True_SDE))/(np.mean(np.abs(stats_SAM_True)))
        error_std_SAM_SGD_SDE[idx_eta, idx_rho] = np.std(np.abs(stats_SAM-stats_SGD_SDE))/(np.mean(np.abs(stats_SAM)))
        error_std_SAM_SAM_SDE[idx_eta, idx_rho] = np.std(np.abs(stats_SAM-stats_SAM_SDE))/(np.mean(np.abs(stats_SAM)))




      loss_all_SGD[idx_eta, idx_rho,:] = np.mean(loss_all_SGD, axis=0)
      loss_all_SAM_True[idx_eta, idx_rho,:] = np.mean(loss_all_SAM_True, axis=0)
      loss_all_SAM[idx_eta, idx_rho,:] = np.mean(loss_all_SAM, axis=0)
      loss_all_USAM[idx_eta, idx_rho,:] = np.mean(loss_all_USAM, axis=0)
      if cfg.sdes:
        loss_all_SGD_SDE[idx_eta, idx_rho,:] = np.mean(loss_all_SGD_SDE, axis=0)[::int(eta/dt)]
        loss_all_SAM_True_SDE[idx_eta, idx_rho,:] = np.mean(loss_all_SAM_True_SDE, axis=0)[::int(eta/dt)]
        loss_all_SAM_SDE[idx_eta, idx_rho,:] = np.mean(loss_all_SAM_SDE, axis=0)[::int(eta/dt)]
        loss_all_USAM_SDE[idx_eta, idx_rho,:] = np.mean(loss_all_USAM_SDE, axis=0)[::int(eta/dt)]
        loss_all_USAM_SDE_DRIFT[idx_eta, idx_rho,:] = np.mean(loss_all_USAM_SDE_DRIFT, axis=0)[::int(eta/dt)]
        loss_all_SAM_SDE_DRIFT[idx_eta, idx_rho,:] = np.mean(loss_all_SAM_SDE_DRIFT, axis=0)[::int(eta/dt)]
        loss_all_SAM_SDE_True_DRIFT[idx_eta, idx_rho,:] = np.mean(loss_all_SAM_SDE_True_DRIFT, axis=0)[::int(eta/dt)]

      loss_all_SGD_std[idx_eta, idx_rho,:] = np.std(loss_all_SGD, axis=0)
      loss_all_SAM_True_std[idx_eta, idx_rho,:] = np.std(loss_all_SAM_True, axis=0)
      loss_all_SAM_std[idx_eta, idx_rho,:] = np.std(loss_all_SAM, axis=0)
      loss_all_USAM_std[idx_eta, idx_rho,:] = np.std(loss_all_USAM, axis=0)
      if cfg.sdes:
        loss_all_SGD_SDE_std[idx_eta, idx_rho,:] = np.std(loss_all_SGD_SDE, axis=0)[::int(eta/dt)]
        loss_all_SAM_True_SDE_std[idx_eta, idx_rho,:] = np.mean(loss_all_SAM_True_SDE, axis=0)[::int(eta/dt)]
        loss_all_SAM_SDE_std[idx_eta, idx_rho,:] = np.std(loss_all_SAM_SDE, axis=0)[::int(eta/dt)]
        loss_all_USAM_SDE_std[idx_eta, idx_rho,:] = np.std(loss_all_USAM_SDE, axis=0)[::int(eta/dt)]
        loss_all_USAM_SDE_DRIFT_std[idx_eta, idx_rho,:] = np.std(loss_all_USAM_SDE_DRIFT, axis=0)[::int(eta/dt)]
        loss_all_SAM_SDE_DRIFT_std[idx_eta, idx_rho,:] = np.std(loss_all_SAM_SDE_DRIFT, axis=0)[::int(eta/dt)]
        loss_all_SAM_SDE_True_DRIFT_std[idx_eta, idx_rho,:] = np.std(loss_all_SAM_SDE_True_DRIFT, axis=0)[::int(eta/dt)]


      x_it_all_SGD[idx_eta, idx_rho,:] = np.mean(x_it_all_SGD, axis=0)
      x_it_all_SAM_True[idx_eta, idx_rho,:] = np.mean(x_it_all_SAM_True, axis=0)
      x_it_all_SAM[idx_eta, idx_rho,:] = np.mean(x_it_all_SAM, axis=0)
      x_it_all_USAM[idx_eta, idx_rho,:] = np.mean(x_it_all_USAM, axis=0)
      if cfg.sdes:
        x_it_all_SGD_SDE[idx_eta, idx_rho,:] = np.mean(x_it_all_SGD_SDE, axis=0)[::int(eta/dt)]
        x_it_all_SAM_True_SDE[idx_eta, idx_rho,:] = np.mean(x_it_all_SAM_True_SDE, axis=0)[::int(eta/dt)]
        x_it_all_SAM_SDE[idx_eta, idx_rho,:] = np.mean(x_it_all_SAM_SDE, axis=0)[::int(eta/dt)]
        x_it_all_USAM_SDE[idx_eta, idx_rho,:] = np.mean(x_it_all_USAM_SDE, axis=0)[::int(eta/dt)]
        x_it_all_USAM_SDE_DRIFT[idx_eta, idx_rho,:] = np.mean(x_it_all_USAM_SDE_DRIFT, axis=0)[::int(eta/dt)]
        x_it_all_SAM_SDE_DRIFT[idx_eta, idx_rho,:] = np.mean(x_it_all_SAM_SDE_DRIFT, axis=0)[::int(eta/dt)]
        x_it_all_SAM_SDE_True_DRIFT[idx_eta, idx_rho,:] = np.mean(x_it_all_SAM_SDE_True_DRIFT, axis=0)[::int(eta/dt)]

      x_it_all_SGD_std[idx_eta, idx_rho,:] = np.std(x_it_all_SGD, axis=0)
      x_it_all_SAM_True_std[idx_eta, idx_rho,:] = np.std(x_it_all_SAM_True, axis=0)
      x_it_all_SAM_std[idx_eta, idx_rho,:] = np.std(x_it_all_SAM, axis=0)
      x_it_all_USAM_std[idx_eta, idx_rho,:] = np.std(x_it_all_USAM, axis=0)
      if cfg.sdes:
        x_it_all_SGD_SDE_std[idx_eta, idx_rho,:] = np.std(x_it_all_SGD_SDE, axis=0)[::int(eta/dt)]
        x_it_all_SAM_True_SDE_std[idx_eta, idx_rho,:] = np.mean(x_it_all_SAM_True_SDE, axis=0)[::int(eta/dt)]
        x_it_all_SAM_SDE_std[idx_eta, idx_rho,:] = np.std(x_it_all_SAM_SDE, axis=0)[::int(eta/dt)]
        x_it_all_USAM_SDE_std[idx_eta, idx_rho,:] = np.std(x_it_all_USAM_SDE, axis=0)[::int(eta/dt)]
        x_it_all_USAM_SDE_DRIFT_std[idx_eta, idx_rho,:] = np.std(x_it_all_USAM_SDE_DRIFT, axis=0)[::int(eta/dt)]
        x_it_all_SAM_SDE_DRIFT_std[idx_eta, idx_rho,:] = np.std(x_it_all_SAM_SDE_DRIFT, axis=0)[::int(eta/dt)]
        x_it_all_SAM_SDE_True_DRIFT_std[idx_eta, idx_rho,:] = np.std(x_it_all_SAM_SDE_True_DRIFT, axis=0)[::int(eta/dt)]


      results={'error_mean_SGD_SGD_SDE':error_mean_SGD_SGD_SDE,'error_mean_SAM_True_SGD_SDE':error_mean_SAM_True_SGD_SDE,
              'error_mean_SAM_True_SAM_True_SDE':error_mean_SAM_True_SAM_True_SDE,'error_std_SGD_SGD_SDE':error_std_SGD_SGD_SDE,
              'error_std_SAM_True_SGD_SDE':error_std_SAM_True_SGD_SDE,'error_std_SAM_True_SAM_True_SDE':error_std_SAM_True_SAM_True_SDE,
              'loss_all_SGD':loss_all_SGD,'loss_all_SGD_SDE':loss_all_SGD_SDE,'loss_all_SAM_True':loss_all_SAM_True,'loss_all_SAM_True_SDE':loss_all_SAM_True_SDE,
              'loss_all_SGD_std':loss_all_SGD_std,'loss_all_SGD_SDE_std':loss_all_SGD_SDE_std, 
              'loss_all_SAM_True_std':loss_all_SAM_True_std,'loss_all_SAM_True_SDE_std':loss_all_SAM_True_SDE_std,
              'x_it_all_SGD':x_it_all_SGD,'x_it_all_SGD_SDE':x_it_all_SGD_SDE,
              'x_it_all_SAM_True':x_it_all_SAM_True,'x_it_all_SAM_True_SDE':x_it_all_SAM_True_SDE,
              'x_it_all_SGD_std':x_it_all_SGD_std,'x_it_all_SGD_SDE_std':x_it_all_SGD_SDE_std,
              'x_it_all_SAM_True_std':x_it_all_SAM_True_std,'x_it_all_SAM_True_SDE_std':x_it_all_SAM_True_SDE_std,
              'error_mean_SAM_SGD_SDE':error_mean_SAM_SGD_SDE,'error_mean_SAM_SAM_SDE':error_mean_SAM_SAM_SDE,
            'error_std_SAM_SGD_SDE':error_std_SAM_SGD_SDE,'error_std_SAM_SAM_SDE':error_std_SAM_SAM_SDE,
            'loss_all_SAM':loss_all_SAM,'loss_all_SAM_SDE':loss_all_SAM_SDE, 
            'loss_all_SAM_std':loss_all_SAM_std,'loss_all_SAM_SDE_std':loss_all_SAM_SDE_std,
            'x_it_all_SAM':x_it_all_SAM,'x_it_all_SAM_SDE':x_it_all_SAM_SDE,
            'x_it_all_SAM_std':x_it_all_SAM_std,'x_it_all_SAM_SDE_std':x_it_all_SAM_SDE_std,
              'etas':etas,'rhos':rhos, 'f_star':f_star, 'problem':problem}

      with open('res_{}.pkl'.format(name), 'wb') as f:
          pickle.dump(results, f)
      print("")   

      





if __name__ == "__main__":
  main()

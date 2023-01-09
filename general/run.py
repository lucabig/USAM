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

  problem = Opt_problem(cfg)

  error_mean_SGD_SGD_SDE = np.zeros((len(etas),len(rhos)))
  error_mean_USAM_SGD_SDE = np.zeros((len(etas),len(rhos)))
  error_mean_USAM_USAM_SDE = np.zeros((len(etas),len(rhos)))
  error_mean_SAM_SGD_SDE = np.zeros((len(etas),len(rhos)))
  error_mean_SAM_SAM_SDE = np.zeros((len(etas),len(rhos)))
  error_std_SGD_SGD_SDE = np.zeros((len(etas),len(rhos)))
  error_std_USAM_SGD_SDE = np.zeros((len(etas),len(rhos)))
  error_std_USAM_USAM_SDE = np.zeros((len(etas),len(rhos)))
  error_std_SAM_SGD_SDE = np.zeros((len(etas),len(rhos)))
  error_std_SAM_SAM_SDE = np.zeros((len(etas),len(rhos)))
  loss_all_SGD = np.zeros((len(etas),len(rhos),nit))
  loss_all_SGD_SDE = np.zeros((len(etas),len(rhos),nit))
  loss_all_USAM = np.zeros((len(etas),len(rhos),nit))
  loss_all_USAM_SDE = np.zeros((len(etas),len(rhos),nit))
  loss_all_SAM = np.zeros((len(etas),len(rhos),nit))
  loss_all_SAM_SDE = np.zeros((len(etas),len(rhos),nit))

  #big loop over all etas and rhos
  for idx_eta in range(len(etas)):
    eta = etas[idx_eta]
    dt = eta/1 #simulation with finer grid
    for idx_rho in tqdm(range(len(rhos))):
      rho = rhos[idx_rho]
      print('simulating for eta='+str(eta)+', rho ='+str(rho))
      #init variables
      stats_SGD = []
      stats_SGD_SDE = []
      stats_USAM = []
      stats_USAM_SDE = []
      stats_SAM = []
      stats_SAM_SDE = []
      loss_hist_SGD = []
      loss_hist_SGD_SDE = []
      loss_hist_USAM = []
      loss_hist_USAM_SDE = []
      loss_hist_SAM = []
      loss_hist_SAM_SDE = []
      #do experiments
      for e in tqdm(range(nexp)):
        _, loss_hist, stats = SGD(nit, eta, problem)
        stats_SGD.append(stats)      
        loss_hist_SGD.append(loss_hist)

        _, loss_hist, stats = SGD_SDE(nit,eta,dt,problem)
        stats_SGD_SDE.append(stats)
        loss_hist_SGD_SDE.append(loss_hist)

        _, loss_hist, stats = USAM(nit, eta, rho,problem)
        stats_USAM.append(stats)
        loss_hist_USAM.append(loss_hist)

        _, loss_hist, stats = USAM_SDE1(nit, eta, rho, dt,problem)
        stats_USAM_SDE.append(stats)
        loss_hist_USAM_SDE.append(loss_hist)

        _, loss_hist, stats = SAM(nit, eta, rho,problem)
        stats_SAM.append(stats)
        loss_hist_SAM.append(loss_hist)
        
        _, loss_hist, stats = SAM_SDE1(nit,eta,rho, dt, problem)
        stats_SAM_SDE.append(stats)
        loss_hist_SAM_SDE.append(loss_hist)

      #converting to numpy
      stats_SGD = np.array(stats_SGD)
      stats_USAM = np.array(stats_USAM)
      stats_SAM = np.array(stats_SAM)
      stats_SGD_SDE = np.array(stats_SGD_SDE)
      stats_USAM_SDE = np.array(stats_USAM_SDE)
      stats_SAM_SDE = np.array(stats_SAM_SDE)

      loss_hist_SGD = np.array(loss_hist_SGD)
      loss_hist_USAM = np.array(loss_hist_USAM)
      loss_hist_SAM = np.array(loss_hist_SAM)
      loss_hist_SGD_SDE = np.array(loss_hist_SGD_SDE)
      loss_hist_USAM_SDE = np.array(loss_hist_USAM_SDE)
      loss_hist_SAM_SDE = np.array(loss_hist_SAM_SDE)

      #average over runs
      stats_SGD = np.mean(stats_SGD, axis=0)
      stats_USAM = np.mean(stats_USAM, axis=0)
      stats_SAM = np.mean(stats_SAM, axis=0)
      stats_SGD_SDE = np.mean(stats_SGD_SDE, axis=0)[::int(eta/dt)]
      stats_USAM_SDE = np.mean(stats_USAM_SDE, axis=0)[::int(eta/dt)]
      stats_SAM_SDE = np.mean(stats_SAM_SDE, axis=0)[::int(eta/dt)]

      error_mean_SGD_SGD_SDE[idx_eta, idx_rho] = np.max(np.abs(stats_SGD-stats_SGD_SDE))/(np.mean(np.abs(stats_SGD)))
      error_mean_USAM_SGD_SDE[idx_eta, idx_rho] = np.max(np.abs(stats_USAM-stats_SGD_SDE))/(np.mean(np.abs(stats_USAM)))
      error_mean_USAM_USAM_SDE[idx_eta, idx_rho] = np.max(np.abs(stats_USAM-stats_USAM_SDE))/(np.mean(np.abs(stats_USAM)))
      error_mean_SAM_SGD_SDE[idx_eta, idx_rho] = np.max(np.abs(stats_SAM-stats_SGD_SDE))/(np.mean(np.abs(stats_SAM)))
      error_mean_SAM_SAM_SDE[idx_eta, idx_rho] = np.max(np.abs(stats_SAM-stats_SAM_SDE))/(np.mean(np.abs(stats_SAM)))

      error_std_SGD_SGD_SDE[idx_eta, idx_rho] = np.std(np.abs(stats_SGD-stats_SGD_SDE))/(np.mean(np.abs(stats_SGD)))
      error_std_USAM_SGD_SDE[idx_eta, idx_rho] = np.std(np.abs(stats_USAM-stats_SGD_SDE))/(np.mean(np.abs(stats_USAM)))
      error_std_USAM_USAM_SDE[idx_eta, idx_rho] = np.std(np.abs(stats_USAM-stats_USAM_SDE))/(np.mean(np.abs(stats_USAM)))
      error_std_SAM_SGD_SDE[idx_eta, idx_rho] = np.std(np.abs(stats_SAM-stats_SGD_SDE))/(np.mean(np.abs(stats_SAM)))
      error_std_SAM_SAM_SDE[idx_eta, idx_rho] = np.std(np.abs(stats_SAM-stats_SAM_SDE))/(np.mean(np.abs(stats_SAM)))

      loss_all_SGD[idx_eta, idx_rho,:] = np.mean(loss_hist_SGD, axis=0)
      loss_all_USAM[idx_eta, idx_rho,:] = np.mean(loss_hist_USAM, axis=0)
      loss_all_SAM[idx_eta, idx_rho,:] = np.mean(loss_hist_SAM, axis=0)
      loss_all_SGD_SDE[idx_eta, idx_rho,:] = np.mean(loss_hist_SGD_SDE, axis=0)[::int(eta/dt)]
      loss_all_USAM_SDE[idx_eta, idx_rho,:] = np.mean(loss_hist_USAM_SDE, axis=0)[::int(eta/dt)]
      loss_all_SAM_SDE[idx_eta, idx_rho,:] = np.mean(loss_hist_SAM_SDE, axis=0)[::int(eta/dt)]

    results={'error_mean_SGD_SGD_SDE':error_mean_SGD_SGD_SDE,'error_mean_USAM_SGD_SDE':error_mean_USAM_SGD_SDE,
            'error_mean_USAM_USAM_SDE':error_mean_USAM_USAM_SDE, 'error_mean_SAM_SGD_SDE':error_mean_SAM_SGD_SDE,
            'error_mean_SAM_SAM_SDE':error_mean_SAM_SAM_SDE,'error_std_SGD_SGD_SDE':error_std_SGD_SGD_SDE,'error_std_USAM_SGD_SDE':error_std_USAM_SGD_SDE,
            'error_std_USAM_USAM_SDE':error_std_USAM_USAM_SDE,'error_std_SAM_SGD_SDE':error_std_SAM_SGD_SDE,'error_std_SAM_SAM_SDE':error_std_SAM_SAM_SDE,
            'loss_all_SGD':loss_all_SGD,'loss_all_SGD_SDE':loss_all_SGD_SDE,'loss_all_USAM':loss_all_USAM, 'loss_all_USAM_SDE':loss_all_USAM_SDE,
            'loss_all_SAM':loss_all_SAM,'loss_all_SAM_SDE':loss_all_SAM_SDE, 'etas':etas,'rhos':rhos, 'f_star':f_star, 'problem':problem}

    with open('res_{}.pkl'.format(name), 'wb') as f:
        pickle.dump(results, f)
    print("")   

      





if __name__ == "__main__":
  main()






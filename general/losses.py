import jax
import optax
import jax.numpy as jnp
from jax.scipy.special import expit as sigmoid
#jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import load_breast_cancer


from functools import partial





class Opt_problem:
  def __init__(self, cfg):
    # name of the experiment
    self.name = cfg.name
    # feature dimension for quadratic
    self.d_quadratic = cfg.d_quadratic
    # dimension hidden space network (for classification problems)
    self.HIDDEN = cfg.HIDDEN
    # noise variance
    self.sigma = cfg.sigma
    # analytical hessian: valid only for matrix sensing
    self.use_analytical = cfg.use_analytical
    # featire dimension teacher network
    self.teac_feature_dim = cfg.teac_feature_dim
    # number of layers teacher network
    self.teac_n_layers = cfg.teac_n_layers
    # number of nodes teacher network
    self.teac_n_nodes = cfg.teac_n_nodes
    # use nonlinearity teacher network
    self.teac_linear = cfg.teac_linear
    # number of layers student network
    self.stud_n_layers = cfg.stud_n_layers
    # number of nodes student network
    self.stud_n_nodes = cfg.stud_n_nodes
    # use nonlinearity student network
    self.stud_linear = cfg.stud_linear

    # general settings for classification problems
    if self.name == 'class_shallow_linear' or self.name == 'class_deep_linear' or self.name == 'class_deep_nonlinear':
        # scale features dataset
        scaler = StandardScaler()
        X, y = load_breast_cancer(return_X_y=True)
        self.y=y
        self.X = scaler.fit_transform(X)
        # number of features classification
        self.num_feat_class = X.shape[1]
        # number of data classification
        self.N_class = X.shape[0]
        # number of classes classification
        self.num_classes = len(np.unique(y))-1
        # number of parameters for linear case (#features x #classes)
        d_class_shallow = int(self.num_feat_class*self.num_classes)
        

        # general settings deep classification problems
        if self.name == 'class_deep_linear' or self.name == 'class_deep_nonlinear':

            # assign unique indeces to tasks
            if not self.name == 'class_deep_nonlinear':
              self.index = 2
            else:
              self.index = 4


            # number of params input to first hidden layer (#features x #hidden nodes)
            d1 = int(self.num_feat_class*self.HIDDEN)
            # number of params hidden to output layer (#hidden nodes x #classes)
            d = int(self.HIDDEN*self.num_classes)
            # concatenate numbers of params
            ds = [d1,d]
            # instantiate empty list of params
            x0s = []
            # initialize params
            for i,k in enumerate(ds):
                x0 = (jnp.array(np.random.normal(size=(k,))/np.sqrt(self.HIDDEN))).flatten() # TODO: check init
                x0s.append(x0)
            # save params in a 1d vector
            self.x0 = jnp.concatenate(x0s)


            # initialize noise covariance (two ways)
            if not cfg.cov_diag: # TODO: check
              M = int(2*(sum(ds)))
              A = np.random.normal(size=(sum(ds),M))
              Sigma_squared = A@A.T/N_class
              U,S,V = np.linalg.svd(Sigma_squared)
              self.Sigma = self.sigma*jnp.array(U@np.diag(np.sqrt(S))@V) 
            else:
              self.Sigma = self.sigma*jnp.array(np.eye(sum(ds))) 



        # general settings shallow classification problems
        else: 

            # assign unique indeces to tasks
            self.index=1

            # initalize parameters
            self.x0 = jnp.array(np.random.normal(size=(d_class_shallow,))/np.sqrt(self.num_feat_class)) # TODO: check init
            d = self.x0.shape[0]
            
            # initialize noise covariance (two ways)
            if not cfg.cov_diag: # TODO: check
              M = int(2*d)
              A = np.random.normal(size=(d,M))
              Sigma_squared = A@A.T/N_class
              U,S,V = np.linalg.svd(Sigma_squared)
              self.Sigma = self.sigma*jnp.array(U@np.diag(np.sqrt(S))@V)
            else:   
              self.Sigma = self.sigma*jnp.array(np.eye(d)) 



    # general settings shallow classification problem
    elif self.name == 'quadratic':

        # assign unique indeces to tasks
        self.index=0

        # initialize parameters
        self.x0 = jnp.array(np.random.normal(size=(self.d_quadratic,)))
        
        # initialize number of data
        self.N_quadratic = int(2*self.d_quadratic) # TODO: check
        
        # initialize Hessian
        A = np.random.normal(size=(self.d_quadratic,self.N_quadratic))
        self.H = jnp.array(A@A.T/self.N_quadratic) 

        # initialize noise covariance (two ways)
        if not cfg.cov_diag: # TODO: check
          M = int(2*self.d_quadratic)
          A = np.random.normal(size=(self.d_quadratic,M))
          Sigma_squared = A@A.T/self.N_quadratic
          U,S,V = np.linalg.svd(Sigma_squared)
          self.Sigma = self.sigma*jnp.array(U@np.diag(np.sqrt(S))@V) #cast to jnp
        else:
          self.Sigma = self.sigma*jnp.array(np.eye(self.d_quadratic))
    



    # general settings matrix sensing problem
    elif self.name == 'matrix_sensing':

        # assign unique indeces to tasks
        self.index=3

        # dimension solution
        n_sensing = cfg.n_sensing
        self.n_sensing = n_sensing
        # rank solution
        U_rank = cfg.U_rank
        # number of train test samples
        num_samples = cfg.num_samples 
        self.num_samples = num_samples
        num_samples_test = cfg.num_samples_test

        ### create solution
        np.random.seed(0)
        U_sol = np.random.normal(0, 1, (n_sensing , U_rank))
        U_sol = U_sol/np.linalg.norm(U_sol,2)
        X_sol = U_sol@U_sol.T

        # create probes
        A_vec = np.zeros((num_samples,int(n_sensing*n_sensing)))
        A = []
        np.random.seed(0)
        for i in range(num_samples):
            A_i= np.random.normal(0, 1, (n_sensing , n_sensing))
            A.append((A_i+A_i.T)/2)
            A_vec[i,:] = A[i].flatten()
        self.A = A
        self.A_vec = A_vec

        # create observations
        y = A_vec@X_sol.flatten()
        # add noise
        self.y = y+0.01*np.random.normal(0, 1, y.shape) #### TODO: put in cfg

        #initialization params
        x0=np.random.normal(0, 1/np.sqrt(n_sensing), (n_sensing , n_sensing))
        self.x0 = x0.flatten()

        # noise
        self.Sigma = self.sigma*jnp.array(np.eye(self.x0.flatten().shape[0]))
      



  # general settings teacher stusent problem
    elif self.name == 'teacher_student':

      # assign unique indeces to tasks
      self.index=5
      # instantiate teacher and generate data with teacher
      W_opt = []
      # input
      self.X = np.random.normal(0, 1/np.sqrt(self.teac_feature_dim), (2*self.teac_feature_dim , self.teac_feature_dim)) # TODO: check
      # deep case
      if self.teac_n_layers > 0:
        for l in range(self.teac_n_layers):
          if l == 0:
            # first layer weights
            W0 = np.random.normal(0, 1/np.sqrt(self.teac_feature_dim), (self.teac_feature_dim , self.teac_n_nodes))
            W_opt.append(W0)
            if self.teac_linear:
              y = self.X@W0
            else:
              y = sigmoid(self.X@W0)
          else:
            # deeper layers
            W_l = np.random.normal(0, 1/np.sqrt(self.teac_n_nodes), (self.teac_n_nodes , self.teac_n_nodes))
            W_opt.append(W_l)
            if self.teac_linear:
              y = y@W_l
            else:
              y = sigmoid(y@W_l)  
        # readout weights
        W_out = np.random.normal(0, 1/np.sqrt(self.teac_n_nodes), (self.teac_n_nodes ,1))
        W_opt.append(W_out)
        # create outputs
        y_ = y@W_out
        # add noise
        self.y = y_ + np.random.normal(0, cfg.noise_)

      # shallow case
      else:
        W_out = np.random.normal(0, 1/np.sqrt(self.teac_feature_dim), (self.teac_feature_dim ,1))
        W_opt.append(W_out)
        if self.teac_linear:
          self.y = self.X@W_out
        else:
          self.y = sigmoid(self.X@W_out)


      # instantiate student 
      W_opt_stud = []
      # deep case
      if self.stud_n_layers > 0:
        for l in range(self.stud_n_layers):
          # first hidden layer wieghts
          if l == 0:
            W0 = np.random.normal(0, 1/np.sqrt(self.teac_feature_dim), (self.teac_feature_dim*self.stud_n_nodes))
            W_opt_stud.append(W0)
          # deeper weights
          else:
            W_l = np.random.normal(0, 1/np.sqrt(self.stud_n_nodes), (self.stud_n_nodes*self.stud_n_nodes))
            W_opt_stud.append(W_l)
        # redout
        W_out = np.random.normal(0, 1/np.sqrt(self.stud_n_nodes), (self.stud_n_nodes*1))
        W_opt_stud.append(W_out)
      # shallow case
      else:
        W_out = np.random.normal(0, 1/np.sqrt(self.teac_feature_dim), (self.teac_feature_dim*1))
        W_opt_stud.append(W_out)


      # count params student
      if self.stud_n_layers > 0:
        # number of params (sum weights dims flattened)
        n_params = self.teac_feature_dim*self.stud_n_nodes + self.stud_n_nodes*self.stud_n_nodes*(self.stud_n_layers-1)+self.stud_n_nodes 
        self.n_params = n_params
      else:
        # number of params (feature dim x 1)
        n_params = self.teac_feature_dim
        self.n_params = n_params

      self.x0 = np.concatenate(W_opt_stud)
      self.Sigma = self.sigma*jnp.array(np.eye(n_params))



## losses

  # loss teacher student
  @partial(jax.jit, static_argnums=(0,))
  def loss_teach_stud(self,x):
    # feature dim
    num_feat = self.teac_feature_dim
    # number of hidden nodes
    HIDDEN = self.stud_n_nodes
    # deep case
    if self.stud_n_layers > 0:
      for l in range(self.stud_n_layers):
        if l == 0:
          # recreate first hidden weights
          W = jnp.reshape(x[:num_feat*HIDDEN],(num_feat,HIDDEN))
          if self.stud_linear:
            y = self.X@W
          else:
            y = sigmoid(self.X@W)
        else:
          # recreate deeper hidden weights
          W_l = jnp.reshape(x[(num_feat*HIDDEN + (l-1)*HIDDEN*HIDDEN):(num_feat*HIDDEN + (l)*HIDDEN*HIDDEN)],(HIDDEN,HIDDEN))
          if self.stud_linear:
            y = y@W_l
          else:
            y = sigmoid(y@W_l)
      # redout
      last = self.n_params-HIDDEN
      W_out = jnp.reshape(x[last:],(HIDDEN,1))
      y = y@W_out
    # shallow case
    else:
      W_out = jnp.reshape(x,(self.n_params,1))
      if self.stud_linear:
        y = self.X@W_out 
      else:
        y = sigmoid(self.X@W_out) 
    # regularized mean squared error
    return jnp.mean((y[:,0]-self.y[:,0])**2) + (0.01/2)*np.sum(x**2) # TODO: put reg in cfg
        

  # loss quadratic
  @partial(jax.jit, static_argnums=(0,))
  def loss_quadratic(self,x): 
    return 0.5*x.T@self.H@x

  # loss shallow linear classification
  @partial(jax.jit, static_argnums=(0,))
  def loss_class_shallow_linear(self,x):
    # num features
    num_feat = self.num_feat_class
    # num classes
    num_classes = self.num_classes
    # weights
    W = jnp.reshape(x,(num_feat,num_classes))
    # logits
    logits = self.X@W
    # regularized cross entropy
    return jnp.sum(jnp.log(1+jnp.exp(-logits[:,0] * self.y)))/(2*self.N_class) +  (0.1/2)*np.sum(x**2) # TODO: put reg in cfg


  # loss deep linear classification
  @partial(jax.jit, static_argnums=(0,))
  def loss_class_deep_linear(self,x):
    # num features
    num_feat = self.num_feat_class
    # num classes
    num_classes = self.num_classes
    # hidden nodes
    HIDDEN = self.HIDDEN
    # first layer params number
    d1_squared= int(num_feat*HIDDEN)
    # initialize
    W1 = jnp.reshape(x[:d1_squared],(num_feat,HIDDEN))
    # redout
    W = jnp.reshape(x[d1_squared:],(HIDDEN,num_classes))
    #logits
    logits = self.X@W1@W
    # regularized cross entropy
    return jnp.sum(jnp.log(1+jnp.exp(-logits[:,0] * self.y)))/(2*self.N_class) +  (0.1/2)*np.sum(x**2) # TODO: put reg in cfg


  # loss deep nonlinear classification
  @partial(jax.jit, static_argnums=(0,))
  def loss_class_deep_nonlinear(self,x):
    # num features
    num_feat = self.num_feat_class
    # num classes
    num_classes = self.num_classes
    # hidden nodes
    HIDDEN = self.HIDDEN
    # first layer params number
    d1_squared= int(num_feat*HIDDEN)
    # second layer params number
    h_squared = int(HIDDEN*HIDDEN)
    # initialize
    W1 = jnp.reshape(x[:d1_squared],(num_feat,HIDDEN))
    # redout
    W = jnp.reshape(x[d1_squared:],(HIDDEN,num_classes))
    #logits (nonlinear case)
    logits = sigmoid(sigmoid(self.X@W1)@W)
    # regularized cross entropy
    return jnp.sum(jnp.log(1+jnp.exp(-logits[:,0] * self.y)))/(2*self.N_class) +  (0.1/2)*np.sum(x**2) # TODO: put reg in cfg
  

  # loss matrix sensing
  @partial(jax.jit, static_argnums=(0,))
  def loss_sensing(self,x):
    # reshape params
    x_ = jnp.reshape(x,(self.n_sensing,self.n_sensing))
    # construct low rank matrix
    X = x_@x_.T
    return jnp.linalg.norm(self.y-self.A_vec@X.flatten())**2/(2*self.num_samples)


  # utility function to instantiate the problem
  def loss(self,x,name):
    if self.index == 0:
        return self.loss_quadratic(x)
    elif self.index == 1:
        return self.loss_class_shallow_linear(x)
    elif self.index == 2:
        return self.loss_class_deep_linear(x)
    elif self.index == 3:
        return self.loss_sensing(x)
    elif self.index == 4:
        return self.loss_class_deep_nonlinear(x)
    elif self.index == 5:
      return self.loss_teach_stud(x)


  # analytical hessian for matrix sensing
  @partial(jax.jit, static_argnums=(0,))
  def hess_trace(self,x):
      temp_dim = self.n_sensing
      x = jnp.reshape(x,(temp_dim,temp_dim))
      X = x@x.T
      tr = 0
      ind = len(self.A)
      for i in range(ind):
          AU = self.A[i]@x
          res = (self.y[i]-self.A_vec[i]@X.flatten())
          for j in range(x.shape[0]):
              tr = tr+(4*AU[j,j]**2 - 2*res*self.A[i][j,j])/(self.num_samples)
      return tr   


  # analyiticl grad for matrix sensing
  @partial(jax.jit, static_argnums=(0,))
  def grad(self,U):
      U = jnp.reshape(U,(self.n_sensing,self.n_sensing))
      X = U@U.T
      grad = np.zeros(U.shape)
      for i in range(self.num_samples):
          grad = grad - 2*(self.y[i]-self.A_vec[i]@X.flatten())*(self.A[i]@U)/self.num_samples
      return grad


  # general gradient (jitted later)
  def g_x(self):
    return jax.grad(self.loss, argnums=0)


  # general hessian (jitted later)
  def H_x(self):
    return jax.hessian(self.loss, argnums=0)



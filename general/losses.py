import numpy as np
import jax
import copy
import optax
import jax.numpy as jnp
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import load_breast_cancer
from functools import partial
from jax.scipy.special import expit as sigmoid
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)




class Opt_problem:
  def __init__(self, cfg):
    self.name = cfg.name
    self.d_quadratic = cfg.d_quadratic
    self.HIDDEN = cfg.HIDDEN
    self.sigma = cfg.sigma
    self.use_analytical = cfg.use_analytical
    
    if self.name == 'class_shallow_linear' or self.name == 'class_deep_linear' or self.name == 'class_deep_nonlinear':
        scaler = StandardScaler()
        X, y = load_breast_cancer(return_X_y=True)
        self.y=y
        self.X = scaler.fit_transform(X)
        self.num_feat_class = X.shape[1]
        self.N_class = X.shape[0]
        self.num_classes = len(np.unique(y))-1
        d_class_shallow = int(self.num_feat_class*self.num_classes)
        if self.name == 'class_deep_linear' or self.name == 'class_deep_nonlinear':
            if not self.name == 'class_deep_nonlinear':
              self.index = 2
            else:
              self.index = 4
            d1 = int(self.num_feat_class*self.HIDDEN)
            d2 = int(self.HIDDEN*self.HIDDEN)
            d = int(self.HIDDEN*self.num_classes)
            ds = [d1,d2,d]
            x0s = []
            for i,k in enumerate(ds):
                x0 = (jnp.array(np.random.normal(size=(k,))/np.sqrt(self.HIDDEN))).flatten() # sqrt dim input
                x0s.append(x0)
            self.x0 = jnp.concatenate(x0s)
            # random noise covariance
            M = int(2*(sum(ds)))
            #A = np.random.normal(size=(sum(ds),M))
            #Sigma_squared = A@A.T/N_class
            #U,S,V = np.linalg.svd(Sigma_squared)
            #self.Sigma = self.sigma*jnp.array(U@np.diag(np.sqrt(S))@V) #cast to jnp
            #breakpoint()
            self.Sigma = self.sigma*jnp.array(np.eye(sum(ds))) 
        else: 
            self.index=1
            #self.x0 = jnp.array(np.random.normal(size=(d_class_shallow,))/np.sqrt(self.num_feat_class)) # Glorot init
            self.x0=1*np.random.randn(d_class_shallow,)
            d = self.x0.shape[0]
            M = int(2*d)
            #A = np.random.normal(size=(d,M))
            #Sigma_squared = A@A.T/N_class
            #U,S,V = np.linalg.svd(Sigma_squared)
            #self.Sigma = self.sigma*jnp.array(U@np.diag(np.sqrt(S))@V) #cast to jnp
            self.Sigma = self.sigma*jnp.array(np.eye(d)) 

    elif self.name == 'quadratic':
        self.index=0
        self.x0 = jnp.array(np.random.normal(size=(self.d_quadratic,)))
        self.N_quadratic = int(2*self.d_quadratic)
        A = np.random.normal(size=(self.d_quadratic,self.N_quadratic))
        self.H = jnp.array(A@A.T/self.N_quadratic) 
        # random noise covariance
        #M = int(2*self.d_quadratic)
        #A = np.random.normal(size=(self.d_quadratic,M))
        #Sigma_squared = A@A.T/self.N_quadratic
        #U,S,V = np.linalg.svd(Sigma_squared)
        #self.Sigma = self.sigma*jnp.array(U@np.diag(np.sqrt(S))@V) #cast to jnp
        self.Sigma = self.sigma*jnp.array(np.eye(self.d_quadratic))
    
    elif self.name == 'matrix_sensing':
        self.index=3
        n_sensing = cfg.n_sensing
        self.n_sensing = n_sensing
        U_rank = cfg.U_rank
        num_samples = cfg.num_samples 
        self.num_samples = num_samples
        num_samples_test = cfg.num_samples_test

        ### create solution
        np.random.seed(0)
        U_sol = np.random.normal(0, 1, (n_sensing , U_rank))
        U_sol = U_sol/np.linalg.norm(U_sol,2)
        X_sol = U_sol@U_sol.T

        # training data
        A_vec = np.zeros((num_samples,int(n_sensing*n_sensing)))
        A = []
        np.random.seed(0)
        for i in range(num_samples):
            #A_i= np.random.normal(0, 1/np.sqrt(n_sensing), (n_sensing , n_sensing))
            A_i= np.random.normal(0, 1, (n_sensing , n_sensing))
            A.append((A_i+A_i.T)/2)
            A_vec[i,:] = A[i].flatten()
        self.A = A
        self.A_vec = A_vec
   
        y = A_vec@X_sol.flatten()
        self.y = y+1*np.random.normal(0, 1, y.shape) #### TODO: put in cfg

        #initialization
        x0=np.random.normal(0, 1/np.sqrt(n_sensing), (n_sensing , n_sensing))
        self.x0 = x0.flatten()
        #x0 = np.random.normal(0, 1, (n_sensing , n_sensing))
        #x0 = x0/np.linalg.norm(x0,2)
        #self.x0 = x0.flatten()


        # noise
        self.Sigma = self.sigma*jnp.array(np.eye(self.x0.flatten().shape[0]))


  @partial(jax.jit, static_argnums=(0,))
  def loss_quadratic(self,x): 
    return 0.5*x.T@self.H@x


  @partial(jax.jit, static_argnums=(0,))
  def loss_class_shallow_linear(self,x):
    num_feat = self.num_feat_class
    num_classes = self.num_classes
    W = jnp.reshape(x,(num_feat,num_classes))
    logits = self.X@W
    return jnp.sum(jnp.log(1+jnp.exp(-logits[:,0] * self.y)))/(2*self.N_class) +  (0.1/2)*np.sum(x**2)
    #target = jax.nn.one_hot(self.y, num_classes)
    #return jnp.mean(optax.softmax_cross_entropy(logits, target))


  @partial(jax.jit, static_argnums=(0,))
  def loss_class_deep_linear(self,x):
    num_feat = self.num_feat_class
    num_classes = self.num_classes
    HIDDEN = self.HIDDEN
    d1_squared= int(num_feat*HIDDEN)
    h_squared = int(HIDDEN*HIDDEN)
    W1 = jnp.reshape(x[:d1_squared],(num_feat,HIDDEN))
    W2 = jnp.reshape(x[d1_squared:d1_squared+h_squared],(HIDDEN,HIDDEN))
    W = jnp.reshape(x[d1_squared+h_squared:],(HIDDEN,num_classes))
    logits = self.X@W1@W2@W
    #target = jax.nn.one_hot(self.y, num_classes)
    return jnp.sum(jnp.log(1+jnp.exp(-logits[:,0] * self.y)))/(2*self.N_class) +  (0.1/2)*np.sum(x**2)


  @partial(jax.jit, static_argnums=(0,))
  def loss_class_deep_nonlinear(self,x):
    num_feat = self.num_feat_class
    num_classes = self.num_classes
    HIDDEN = self.HIDDEN
    d1_squared= int(num_feat*HIDDEN)
    h_squared = int(HIDDEN*HIDDEN)
    W1 = jnp.reshape(x[:d1_squared],(num_feat,HIDDEN))
    W2 = jnp.reshape(x[d1_squared:d1_squared+h_squared],(HIDDEN,HIDDEN))
    W = jnp.reshape(x[d1_squared+h_squared:],(HIDDEN,num_classes))
    logits = sigmoid(sigmoid(sigmoid(self.X@W1)@W2)@W)
    #target = jax.nn.one_hot(self.y, num_classes)
    return jnp.sum(jnp.log(1+jnp.exp(-logits[:,0] * self.y)))/(2*self.N_class) +  (0.1/2)*np.sum(x**2)
  
  @partial(jax.jit, static_argnums=(0,))
  def loss_sensing(self,x):
    x_ = jnp.reshape(x,(self.n_sensing,self.n_sensing))
    X = x_@x_.T
    return jnp.linalg.norm(self.y-self.A_vec@X.flatten())**2/(2*self.num_samples)


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


  @partial(jax.jit, static_argnums=(0,))
  def grad(self,U):
      U = jnp.reshape(U,(self.n_sensing,self.n_sensing))
      X = U@U.T
      grad = np.zeros(U.shape)
      for i in range(self.num_samples):
          grad = grad - 2*(self.y[i]-self.A_vec[i]@X.flatten())*(self.A[i]@U)/self.num_samples
      return grad


  def g_x(self):
    return jax.grad(self.loss, argnums=0)

  def H_x(self):
    return jax.hessian(self.loss, argnums=0)

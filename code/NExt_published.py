"""
Archived verbatim copy of the implementation released with

    Zhibin Pu, Shufei Ge, Shijia Wang, "NExt-LMM: a two-stage LMM
    framework with near-exact variance components for genome-wide
    association studies", arXiv:2508.05278v2.

Kept so the published behaviour stays reproducible and the rewritten
modules can be checked against it. Do not use for new analyses: the
recursive branch of HInverse raises NameError (``HoDLR`` is undefined),
so only the leaf path has ever run, and the association statistic
divides by the spread of beta across markers rather than by a standard
error. See the module docstrings of HodlrInverse.py and NExt.py.
"""

from HodlrInverse_published import *
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression


class NExt:


   def __init__(self, Y, S_inv = []):
      
      
      self.N = len(Y)
      self.Y = Y
      self.S_inv = S_inv

   def hApproximate(self, K):
      
      y_std = (self.Y - np.mean(self.Y)) / np.std(self.Y)
      i, j = np.triu_indices_from(K, k=1)
      y_products = y_std[i] * y_std[j]
      K_vals = K[i, j]
      
      model = LinearRegression().fit(K_vals.reshape(-1, 1), y_products)
      h2 = model.coef_[0]

      return h2


   def hodler(self, Sigma):
      
      Hodler = HInverse(Sigma)

      return Hodler.inverse()


   def invApproximate(self, K, h):


      if 0 < h <= 1:
         self.h = h
      else:
         self.h = 0.1
      
      Sigma = self.h*K + (1-self.h) * np.eye(self.N)
      S_i = self.hodler(Sigma)

      return S_i

   
   def fit(self, W, K, h = None):

      self.h = h

      if self.h is None: 
         self.h = self.hApproximate(K) 
      if len(self.S_inv) == 0:
         self.S_inv = self.invApproximate(K, self.h)
 

      beta_pred = []

      for i in range(W.shape[1]):
         x = W[:, i].reshape((self.N,1))
         x = np.hstack([x, np.ones((self.N,1))])
         beta_ = np.linalg.inv(x.T @ self.S_inv @ x) @ x.T @ self.S_inv @ self.Y
         beta_pred.append(beta_[0])

      beta_pred = np.array(beta_pred)
      res = self.Y - W  @ beta_pred


      Q = res.T @ self.S_inv @ res 
      sigma_pred = Q / (self.N-W.shape[1])
      self.beta = beta_pred
      self.sigma = sigma_pred

   def association(self):
      
      T = self.beta / np.std(self.beta)
      p = 2.0*(stats.t.sf(np.abs(T), self.N-1))

      return p
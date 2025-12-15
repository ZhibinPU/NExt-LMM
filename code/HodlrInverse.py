import numpy as np

def compress_matrix_svd(A, threshold= pow(10, -3)):
   
   u, s, v = np.linalg.svd(A, full_matrices = True)

   k = 0
   for i in s:
      if(abs(i) > s[0] * threshold):
         k += 1
   smatrix = np.diagflat(s)
   u = np.matmul(u[:,:k], smatrix[:k, :k])
   v = v[:k,:] 

   return u, v



class HInverse:

    def __init__(self, X, block_min = 3):

        self.size = len(X)
        
        tmp = self.size//2

        if self.size <= block_min or tmp <= 2:
            
            self.F = X
            self.A11 = []
            self.A22 = []
            self.U = []
            self.V = []

        else:
            self.F = []
            self.U, self.V = compress_matrix_svd(X[:tmp, tmp:])
            self.A11 = HoDLR(X[:tmp,:tmp]) 
            self.A22 = X[tmp:,tmp:]

    def is_leafnode(self):
      return False if self.A11 else True
    
    def inverse(self):
       
       if self.is_leafnode():
          return np.linalg.inv(self.F)
       
       else:
          
          N11 = self.A11.inverse()
          S = self.A22 - self.V.T @ self.U.T @ N11 @ self.U @ self.V
          S_inv =  np.linalg.inv(S)

          K11 =  N11 + N11 @  self.U @ self.V @ S_inv @  self.V.T @ self.U.T @ N11
          K12 = -N11@  self.U @ self.V  @ S_inv
          K21 = -S_inv @ self.V.T @ self.U.T @ N11
          return np.block([[K11, K12],[K21, S_inv]])
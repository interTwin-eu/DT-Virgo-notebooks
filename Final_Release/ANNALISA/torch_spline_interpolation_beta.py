import torch
import numpy as np
import time 
import time
import torch
from matplotlib import pyplot as plt
from matplotlib import gridspec


import torch
import torch.nn as nn
import numpy as np

# Spline interpolation for 1D
class SplineInterpolate1D(nn.Module):
    def __init__(self, num_x_bins, kx=3, s=0.001):
        super().__init__()
        self.num_x_bins = num_x_bins
        self.kx = kx
        self.s = s

    def forward(self, Z, xin=None, xout=None):
        
        self.device=Z.device
        
        while len(Z.shape)<3:
           Z=Z.unsqueeze(0) 

        nx_points = Z.shape[-1]

        
        if xin is None:
            x = torch.linspace(-1, 1, nx_points, device=Z.device)
        else:
            x = xin.to(self.device)

        coef, tx = self.spline_fit_natural_torch(x, Z, self.kx, self.s)

        if xout is None:
            x_eval = torch.linspace(-1, 1, self.num_x_bins, device=Z.device)
        else:
            x_eval = xout.to(self.device)

        Z_interp = self.evaluate_spline_torch(x_eval, coef, tx, self.kx).view(-1, self.num_x_bins)
        return Z_interp
        
    def generate_natural_knots(self,x, k):
        """
        Generates a natural knot sequence for B-spline interpolation.
        Natural knot sequence means that 2*k knots are added to the beginning and end of datapoints as replicas of first and last datapoint respectively in order to enforce natural boundary conditions, i.e. second derivative =0.
        the other n nodes are placed in correspondece of thedata points.
    
        Args:
            x: Tensor of data point positions.
            k: Degree of the spline.
    
        Returns:
            Tensor of knot positions.
        """
        n = x.shape[0]
        t = torch.zeros(n + 2 * k, device=self.device)
        t[:k] = x[0]
        t[k:-k] = x
        t[-k:] = x[-1]
        return t
    
    def compute_L_R(self, x, t, d, m, k):
        
        '''
        Compute the L and R values for B-spline basis functions.
        L and R are respectively the firs and second coefficient multiplying B_{i,p-1}(x) and B_{i+1,p-1}(x) in De Boor's recursive formula for Bspline basis funciton computation:
        #{\displaystyle B_{i,p}(x):={\frac {x-t_{i}}{t_{i+p}-t_{i}}}B_{i,p-1}(x)+{\frac {t_{i+p+1}-x}{t_{i+p+1}-t_{i+1}}}B_{i+1,p-1}(x).}
        See https://en.wikipedia.org/wiki/De_Boor%27s_algorithm for further details
    
        Args:
            x: Tensor of data point positions.
            t: Tensor of knot positions.
            d: Current degree of the basis function.
            m: Number of intervals (n - k - 1, where n is the number of knots and k is the degree).
            k: Degree of the spline.
    
        Returns:
            L: Tensor containing left values for the B-spline basis functions.
            R: Tensor containing right values for the B-spline basis functions.
        '''
        left_num = x.unsqueeze(1) - t[:m].unsqueeze(0)
        left_den = t[d:m+d] - t[:m]
        L = left_num / left_den.unsqueeze(0)
        
        right_num = t[d+1:m+d+1] - x.unsqueeze(1)
        right_den = t[d+1:m+d+1] - t[1:m+1]
        R = right_num / right_den.unsqueeze(0)
    
        #handle zero denominator case
        zero_left = left_den == 0
        zero_right = right_den == 0
        zero_left_stacked = zero_left.tile(x.shape[0], 1)
        zero_right_stacked = zero_right.tile(x.shape[0], 1)
        
        L[zero_left_stacked] = 0
        R[zero_right_stacked] = 0
        
        return L, R
    
    def zeroth_order(self, x, k, t, n, m):
        
        """
        Compute the zeroth-order B-spline basis functions. Accoring to de Boors recursive formula:
        {\displaystyle B_{i,0}(x):={\begin{cases}1&{\text{if }}\quad t_{i}\leq x<t_{i+1}\\0&{\text{otherwise}}\end{cases}}}
        See https://en.wikipedia.org/wiki/De_Boor%27s_algorithm for reference
    
        Args:
            x: Tensor of data point positions.
            k: Degree of the spline.
            t: Tensor of knot positions.
            n: Number of data points.
            m: Number of intervals (n - k - 1, where n is the number of knots and k is the degree).
    
        Returns:
            b: Tensor containing the zeroth-order B-spline basis functions.
        """
        b = torch.zeros((n, m, k + 1),device=self.device)
        
        mask_lower = t[:m+1].unsqueeze(0)[:, :-1] <= x.unsqueeze(1)
        mask_upper = x.unsqueeze(1) < t[:m+1].unsqueeze(0)[:, 1:]
    
        b[:, :, 0] = mask_lower & mask_upper
        b[:, 0, 0] = torch.where(x < t[1], torch.ones_like(x,device=self.device), b[:, 0, 0])
        b[:, -1, 0] = torch.where(x >= t[-2], torch.ones_like(x,device=self.device), b[:, -1, 0])
        return b
    
    def bspline_basis_natural_torch(self, x, k, t):
        
        '''
        Compute bspline basis function using de Boor's recursive formula (See https://en.wikipedia.org/wiki/De_Boor%27s_algorithm for reference)
        Args:
            x: Tensor of data point positions.
            k: Degree of the spline.
            t: Tensor of knot positions.
    
        Returns:
            b[:,:,-1]: Tensor containing the kth-order B-spline basis functions.
        '''
        
        n = x.shape[0]
        m = t.shape[0] - k - 1
    
        #calculate seroth order basis funciton
        b = self.zeroth_order(x, k, t, n, m)
    
        #recursive de Boors formula for bspline basis functions
        for d in range(1, k + 1):
            L, R = self.compute_L_R(x, t, d, m, k)
            left = L * b[:, :, d-1]
    
            zeros_tensor = torch.zeros(b.shape[0], 1,device=self.device)
            temp_b = torch.cat([b[:, 1:, d-1], zeros_tensor], dim=1)
        
            right = R * temp_b
            b[:, :, d] = left + right
    
        return b[:, :, -1]
    
    def spline_fit_natural_torch(self, x, z, kx, s):
        """
      This function computes the B-spline coefficients for natural spline fitting of 1D data.
    
      Args:
          x: A 1D tensor representing the positions of the data points.
          z: A 1D tensor representing the data values.
          kx: The degree of the B-spline (integer).
          s: The smoothing parameter (positive float).
    
      Returns:
          coef: A 1D tensor representing the B-spline coefficients.
          tx: A 1D tensor representing the knot positions for the B-spline.
      """
    
        #generate natural knots
        tx = self.generate_natural_knots(x, kx)
    
        #compute basis functions
        bx = self.bspline_basis_natural_torch(x, kx, tx).to(self.device)
    
        #add regularizing term
        m = bx.size(1)
        I = torch.eye(m, device=self.device)
    
        #convert to float in case double
        z = z.float()
        bx = bx.float()
    
        # Expand bx to have a batch dimension: (1, n, m)
        bx = bx.unsqueeze(0)  # (1, n, m)
    
        # Compute B_T_B and B_T_z for each batch
        B_T_B = bx.transpose(1, 2) @ bx + s * I  # (1, m, m)
        B_T_z = bx.transpose(1, 2) @ z.unsqueeze(-1)  # (batch_size, m, 1)
    
        # Solve the linear system for each batch
        coef = torch.linalg.solve(B_T_B.expand(z.size(0), -1, -1), B_T_z).squeeze(-1)
        
        return coef.to(z.device), tx
    

    def evaluate_spline_torch(self, x, coef, tx, kx):
        """
      This function evaluates a pre-computed 1D spline at given points.
    
      Args:
          x: A 1D tensor representing the positions for evaluation.
          coef: A 1D tensor representing the pre-computed B-spline coefficients.
          tx: A 1D tensor representing the knot positions for the B-spline.
          kx: The order of the B-spline.
    
      Returns:
          z_eval: A 1D tensor containing the interpolated values at the specified points (x).
      """
    
        # Compute B-spline basis functions
        bx = self.bspline_basis_natural_torch(x, kx, tx).to(self.device)  
        
        # Expand bx to allow batch computation:
        bx = bx.unsqueeze(0)  
    
        # Perform batched matrix multiplication: 
        z_eval = (bx @ coef.unsqueeze(-1)).squeeze(-1)
        
        return z_eval
#---------------------------------------------------------------------

#2D

class SplineInterpolate2D(nn.Module):
    def __init__(self, num_t_bins, num_f_bins, kx=3, ky=3, sx=0.001, sy=0.001, logf=False, frange=(10, 100)):
        super().__init__()
        self.num_t_bins = num_t_bins
        self.num_f_bins = num_f_bins
        self.kx = kx
        self.ky = ky
        self.sx = sx
        self.sy = sy
        self.logf = logf
        self.frange = frange

    def forward(self, Z, freqs=None, xin=None, xout=None, yin=None, yout=None):

        self.device=Z.device
        
        while len(Z.shape)<4:
           Z=Z.unsqueeze(0) 
        
        batch_size,channel_size, nx_points, ny_points = Z.shape[0],  Z.shape[1], Z.shape[-2], Z.shape[-1]

        if xin is None:
            x = torch.linspace(-1, 1, nx_points, device=self.device)
        else:
            x = xin.to(self.device)

        if freqs is not None:
            y = freqs
        elif yin is None:
            if self.logf:
                y = torch.tensor(np.geomspace(self.frange[0], self.frange[1], ny_points), device=self.device)
            else:
                y = torch.linspace(-1, 1, ny_points, device=self.device)
        else:
            y = yin.to(self.device)

        coef, tx, ty = self.bivariate_spline_fit_natural_torch(x, y, Z, self.kx, self.ky, self.sx, self.sy)

        if xout is None:
            x_eval = torch.linspace(-1, 1, self.num_t_bins, device=self.device)
        else:
            x_eval = xout.to(self.device)

        if yout is None:
            if self.logf:
                y_eval = torch.tensor(np.geomspace(self.frange[0], self.frange[1], self.num_f_bins), device=self.device)
            else:
                y_eval = torch.linspace(-1, 1, self.num_f_bins, device=self.device)
        else:
            y_eval = yout.to(self.device)

        Z_interp = self.evaluate_bivariate_spline_torch(x_eval, y_eval, coef, tx, ty, self.kx, self.ky)
        return Z_interp

    def generate_natural_knots(self, x, k):
        n = x.shape[0]
        t = torch.zeros(n + 2 * k, device=self.device)
        t[:k] = x[0]
        t[k:-k] = x
        t[-k:] = x[-1]
        return t

    def bspline_basis_natural_torch(self, x, k, t):
        n = x.shape[0]
        m = t.shape[0] - k - 1

        b = self.zeroth_order(x, k, t, n, m)

        for d in range(1, k + 1):
            L, R = self.compute_L_R(x, t, d, m, k)
            left = L * b[:, :, d-1]

            zeros_tensor = torch.zeros(b.shape[0], 1, device=self.device)
            temp_b = torch.cat([b[:, 1:, d-1], zeros_tensor], dim=1)

            right = R * temp_b
            b[:, :, d] = left + right

        return b[:, :, -1]

    def zeroth_order(self, x, k, t, n, m):
        b = torch.zeros((n, m, k + 1), device=self.device)


        mask_lower = t[:m+1].unsqueeze(0)[:, :-1] <= x.unsqueeze(1)
        mask_upper = x.unsqueeze(1) < t[:m+1].unsqueeze(0)[:, 1:]

        b[:, :, 0] = mask_lower & mask_upper
        b[:, 0, 0] = torch.where(x < t[1], torch.ones_like(x,device=self.device), b[:, 0, 0])
        b[:, -1, 0] = torch.where(x >= t[-2], torch.ones_like(x,device=self.device), b[:, -1, 0])
        return b

    def compute_L_R(self, x, t, d, m, k):
        left_num = x.unsqueeze(1) - t[:m].unsqueeze(0)
        left_den = t[d:m+d] - t[:m]
        L = left_num / left_den.unsqueeze(0)

        right_num = t[d+1:m+d+1] - x.unsqueeze(1)
        right_den = t[d+1:m+d+1] - t[1:m+1]
        R = right_num / right_den.unsqueeze(0)

        zero_left = left_den == 0
        zero_right = right_den == 0
        zero_left_stacked = zero_left.tile(x.shape[0], 1)
        zero_right_stacked = zero_right.tile(x.shape[0], 1)

        L[zero_left_stacked] = 0
        R[zero_right_stacked] = 0

        return L, R

    def bivariate_spline_fit_natural_torch(self, x, y, z, kx, ky, sx, sy):
        tx = self.generate_natural_knots(x, kx)

        ty = self.generate_natural_knots(y, ky)


        Bx = self.bspline_basis_natural_torch(x, kx, tx).to(self.device)
        By = self.bspline_basis_natural_torch(y, ky, ty).to(self.device)
        


        mx = Bx.size(1)
        my = By.size(1)
        Ix = torch.eye(mx, device=self.device)
        Iy = torch.eye(my, device=self.device)
        
        # Adding batch dimension handling
        ByT_By = By.T.unsqueeze(0).unsqueeze(0) @ By.unsqueeze(0).unsqueeze(0) + (sy * Iy).unsqueeze(0).unsqueeze(0) 
        ByT_Z_Bx =  By.T.unsqueeze(0).unsqueeze(0)@ z.transpose(2,3) @ Bx.unsqueeze(0).unsqueeze(0)  
        E = torch.linalg.solve(ByT_By, ByT_Z_Bx) 

        BxT_Bx = Bx.T.unsqueeze(0).unsqueeze(0) @ Bx.unsqueeze(0).unsqueeze(0) + (sx * Ix).unsqueeze(0).unsqueeze(0)  
        C = torch.linalg.solve(BxT_Bx, E.transpose(2,3))

        return C.to(self.device), tx, ty
        
    def evaluate_bivariate_spline_torch(self, x, y, C, tx, ty, kx, ky):
        """
        Evaluate a bivariate spline on a grid of x and y points.
        
        Args:
            x: Tensor of x positions to evaluate the spline.
            y: Tensor of y positions to evaluate the spline.
            C: Coefficient tensor of shape (batch_size, mx, my).
            tx: Knot positions for x.
            ty: Knot positions for y.
            kx: Degree of spline in x.
            ky: Degree of spline in y.
            
        Returns:
            Z_interp: Interpolated values at the grid points.
        """
        Bx = self.bspline_basis_natural_torch(x, kx, tx).unsqueeze(0).unsqueeze(0).to(self.device)  
        By = self.bspline_basis_natural_torch(y, ky, ty).unsqueeze(0).unsqueeze(0).to(self.device)  
            
        # Perform matrix multiplication using einsum to get Z_interp
        Z_interp = By @ C.transpose(2,3) @ Bx.transpose(2,3)
        
        return Z_interp

        
    '''def evaluate_bivariate_spline_torch_old(self, x, y, C, tx, ty, kx, ky):
        Bx = self.bspline_basis_natural_torch(x, kx, tx)  # (num_x_eval_points, mx)
        By = self.bspline_basis_natural_torch(y, ky, ty)  # (num_y_eval_points, my)
        
        # Bx: (num_x_eval_points, mx)
        # By: (num_y_eval_points, my)
        # C: (batch_size, mx, my)
        
        # Need to expand dimensions to handle batch correctly
        Bx = Bx.unsqueeze(0)  # (1, num_x_eval_points, mx)
        By = By.unsqueeze(0)  # (1, num_y_eval_points, my)
        
        # Performing batch-wise evaluation
        Z_interp = torch.einsum('bnm,bmp,bpq->bnq', Bx, C, By.transpose(1, 2))
        return Z_interp
    '''



import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class SplineInterpolate1D_batch(nn.Module):
    def __init__(self, num_x_bins, kx=3, s=0.001, device='cuda'):
        super().__init__()
        self.num_x_bins = num_x_bins
        self.kx = kx
        self.s = s
        self.device=device

    def forward(self, Z, xout=None, xmin=-1,xmax=1):
        """
    The forward pass of the SplineInterpolate1D network. This function performs B-spline interpolation on a batch of 1D tensors.

    Args:
        Z: A list of 1D tensors (input data), each representing a signal to be interpolated.
        xin: Optional tensor of x-values for each signal. If None, x-values are generated between xmin and xmax.
        xmin: Minimum x-value for interpolation (used when xin is None).
        xmax: Maximum x-value for interpolation (used when xin is None).
        s: Regularization parameter for spline fitting. Controls the smoothness of the spline.
        kx: Degree of the B-spline (default is 3, for cubic splines).

    Returns:
        z_eval: A batched tensor of interpolated values at the x-values for each input tensor.
    """
        
        
        if type(Z) == list:
            
            #1. convert input to batched tensor and calculate batched knot vectors
            x_batched,t_batched,Z_batched=self.process_list_input(Z,xmin,xmax)       
            #print(f'{Z_batched.shape=}')
            
            #2. Compute batched Bspline basis functions
            Bx=self.bspline_basis_natural_torch_gamma(x_batched, self.kx, t_batched)
            #print(f'{Bx.shape=}')
        
            #3. Compute batched spline coefficients
            coef = self.spline_fit_natural_torch_gamma(x_batched,t_batched, Z_batched,Bx ,self.kx, self.s)
            
            #4. Compute batched evaluation grid
            if xout is None:
                x_eval = torch.arange(xmin, xmax, step=(xmax-xmin)/self.num_x_bins, device=self.device).repeat(x_batched.shape[0], 1)
            else:
                x_eval = xout.repeat(x_batched.shape[0], 1).to(self.device)
                
            #5. Interpolate batched input
            Z_interp = self.evaluate_spline_torch_gamma(x_eval, coef, t_batched, self.kx)#.view(-1, self.num_x_bins)
            
            #print(f'{Z_interp.shape=}')
            
            #6. Filter the tensor to keep only non-zero subtensors, i.e. undo batch padding
            non_zero_mask =(Z_interp != 0).any(dim=-1)
            #print(f'{non_zero_mask.shape=}')

            filtered_tensor = Z_interp[non_zero_mask].view(Z_batched.shape[2],Z_batched.shape[3],-1, self.num_x_bins)
            #print(f'{filtered_tensor.shape=}')



            return filtered_tensor
            
        
        

        
    def process_list_input(self,Z,xmin,xmax):
        """
    Converts a list of tensors into a batched tensor with padding and calculates corresponding knot vectors.

    Args:
        Z: A list of 1D tensors (input data).
        xmin: Minimum x-value for interpolation (used to generate x-points).
        xmax: Maximum x-value for interpolation (used to generate x-points).

    Returns:
        x_batched: A batched tensor containing the x-values for each tensor in Z, padded with values > xmax.
        t_batched: A batched tensor containing the knot vectors for each tensor in Z, padded with values < xmin.
        z_padded: A batched tensor containing the padded Z tensors.
    """
        
        # Get lengths and batch size
        lengths = [tensor.shape[-1] for tensor in Z]  
        
        if Z[0].shape[0]==1:
            Z=[tensor.transpose(0,1) for tensor in Z]
        
            
        batch_size_ts=Z[0].shape[0]
        
        channel_size_ts=Z[0].shape[1]
        
        
        
        
                    
        # Use OrderedDict to maintain order and count occurrences in one pass
        #There might be something more efficient than this. torch.unique() uses GPU but does not preserve order
        length_counts = OrderedDict()
        for length in lengths:

            if length in length_counts:
                length_counts[length] += 1
            else:
                length_counts[length] = 1
                
        # Extract unique lengths
        unique_lengths = list(length_counts.keys())  
        
        #Extract maximum tile size and max batch size
        max_length_z = max(lengths)  # Maximum length of tensors in Z
        batch_size_qtile_max=max(length_counts.values())
        
        # Preallocate tensor for padded input
        z_padded = torch.zeros((len(unique_lengths),batch_size_qtile_max,batch_size_ts,channel_size_ts,max_length_z), device=self.device)
        #print(f'{z_padded.shape=}')

        x_list, t_list = [], []
        

        for j,length in enumerate(unique_lengths): 
    
            # Collect tensors of the same length
            same_length_tensors = [tensor for tensor in Z if tensor.shape[-1] == length]

            # Fill preallocated tensor with input data. the resulting tensor will be padded with zeros
            for i, tensor in enumerate(same_length_tensors):
                
                z_padded[j, i,:,:,:tensor.shape[-1]] = tensor  # Fill with original tensor

            # Create x and t for this length
            x = torch.arange(xmin, xmax, (xmax-xmin)/length, device=self.device)  # Create x tensor
            t = self.generate_natural_knots_gamma(x, self.kx)  # Create t tensor
            x_list.append(x)
            t_list.append(t)


        #Extract max length for x and t tensors
        max_length_x = max([x.shape[0] for x in x_list])  # Maximum length for x
        max_length_t = max([t.shape[0] for t in t_list])  # Maximum length for t

        # Pad x and t tensors
        # padding x and t with values respectively greather that xmax and smaller than xmin is a mathematical trick in the for the computation
        #of Bspline basis functions that results into a bock diagonal matrix B'=[[B,0],[0,0]], where B is the Bspline basis function computed without padding
        x_padded = [torch.cat([x, torch.full((max_length_x - x.shape[0],), xmax+1, device=self.device)]) for x in x_list]
        t_padded = [torch.cat([t, torch.full((max_length_t - t.shape[0],), xmin-1, device=self.device)]) for t in t_list]

        # Stack tensors
        x_batched = torch.stack(x_padded)
        t_batched = torch.stack(t_padded)
        
        
        return x_batched,t_batched,z_padded

        
    def generate_natural_knots_gamma(self,x, k):
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
    
    def compute_L_R_gamma(self, x, t, d, m, k):
        
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
        
        left_num = x.unsqueeze(2) - t[:,:m].unsqueeze(1)
        left_den = t[:,d:m+d] - t[:,:m]
        L = left_num / left_den.unsqueeze(1)
        
        right_num = t[:,d+1:m+d+1].unsqueeze(1) - x.unsqueeze(2)
        right_den = t[:,d+1:m+d+1] - t[:,1:m+1]
        R = right_num / right_den.unsqueeze(1)
        
    
        #handle zero denominator case
        zero_left = left_den == 0
        zero_right = right_den == 0
        zero_left_stacked = zero_left.unsqueeze(1).expand(-1, x.shape[1], -1)  # Shape (batch_size, n, m)
        zero_right_stacked = zero_right.unsqueeze(1).expand(-1, x.shape[1], -1)  # Shape (batch_size, n, m)

        
        L[zero_left_stacked] = 0
        R[zero_right_stacked] = 0
        
        return L, R

    
    def zeroth_order_gamma(self, x, k, t, n, m):
        """
        Compute the zeroth-order B-spline basis functions for batched input.
        Args:
            x: Batched tensor of data point positions with shape (batch_size, n).
            k: Degree of the spline.
            t: Batched tensor of knot positions with shape (batch_size, m).
            n: Number of data points (can be batched).
            m: Number of intervals (n - k - 1, where n is the number of knots and k is the degree).

        Returns:
            b: Batched tensor containing the zeroth-order B-spline basis functions with shape (batch_size, n, m, k + 1).
        """
        batch_size = x.size(0)
        b = torch.zeros((batch_size, n, m, k + 1), device=self.device)  # Updated for batch size

        # Create masks to determine where the basis functions are 1
    
        mask_lower = t[:,:m+1].unsqueeze(1)[:,:, :-1] <= x.unsqueeze(2)
        mask_upper = x.unsqueeze(2) < t[:,:m+1].unsqueeze(1)[:, :,1:]
        

        # Combine masks to get the zeroth-order basis function values
        basic_mask = mask_lower & mask_upper  

        # Assign the basic_mask to the zeroth-order basis functions
        b[:, :, :, 0] = basic_mask 

        return b

    
    def bspline_basis_natural_torch_gamma(self, x, k, t):
        '''
        Compute B-spline basis function using de Boor's recursive formula for batched input.
        Args:
            x: Batched tensor of data point positions with shape (batch_size, n).
            k: Degree of the spline.
            t: Tensor of knot positions.

        Returns:
            b[:, :, :, -1]: Batched tensor containing the kth-order B-spline basis functions.
        '''
        batch_size = x.size(0)
        n = x.shape[1]  # Number of data points
        m = t.shape[1] - k - 1            

        # Calculate zeroth-order basis function
        b = self.zeroth_order_gamma(x, k, t, n, m)

        # Recursive de Boor's formula for B-spline basis functions
        for d in range(1, k + 1):
            L, R = self.compute_L_R_gamma(x, t, d, m, k)
            left = L * b[:, :, :, d-1]  

            zeros_tensor = torch.zeros(batch_size, x.shape[1], 1, device=self.device)
            temp_b = torch.cat([b[:,:, 1:, d-1], zeros_tensor], dim=2)  

            right = R * temp_b
            b[:, :, :, d] = left + right

        return b[:, :, :, -1]  # Return only the last order of basis functions

    
    def spline_fit_natural_torch_gamma(self, x, t, z, bx, kx, s):
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
    
        #add regularizing term
        m = bx.size(2)
        I = torch.eye(m, device=self.device)
    
        #convert to float in case double
        z = z.float()
        bx = bx.float()
    
        # Compute B_T_B and B_T_z for each batch
        
        B_T_B = bx.unsqueeze(1).unsqueeze(1).transpose(-2, -1) @ bx.unsqueeze(1).unsqueeze(1) + (s * I).unsqueeze(0).unsqueeze(0)  

        B_T_z = bx.unsqueeze(1).unsqueeze(1).transpose(-2, -1) @ z.permute(0,2,3,4,1) 
        
        
       
    
        # Solve the linear system for each batch
        coef = torch.linalg.solve(B_T_B, B_T_z).squeeze(-1) #B_T_B.expand(z.size(0), -1,-1, -1,-1)

        
        return coef.to(z.device)
    

    def evaluate_spline_torch_gamma(self, x, coef, tx, kx):
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
        bx = self.bspline_basis_natural_torch_gamma(x, kx, tx).to(self.device)  
        
        # Expand bx to allow batch computation
        bx = bx.unsqueeze(1).unsqueeze(1)  
        

        # Perform batched matrix multiplication: 
        z_eval = torch.matmul(bx, coef) #.permute(0,1,3,2)

        return z_eval.permute(1,2,0,4,3)
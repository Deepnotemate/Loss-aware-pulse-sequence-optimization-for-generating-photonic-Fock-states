import numpy as np
from scipy.sparse import csr_matrix, diags


class Loss_Aware_Optimization:
    def __init__(self, Omega, dim, cutoff, dt_val, loss_type, gamma, target_N):
            """
            Initializes Optmization scheme.

            Args: 
                Omega: single-photon Rabi Frequency
                dim: Truncation dimension for both cavity modes (Truncation at (dim,dim-1,e))
                cutoff: max order of couplings due to loss
                dt_val: time stepping for calculation of interaction between pulses
                loss_type: 'atom' or 'cavity' ... specifies whether we are optimizing for atom-decay or cavity-decay
                gamma: decay rate
                target_N: Fock State, we are optimizing for
            """
            self.Omega = Omega
            self.dim = dim
            self.cutoff = cutoff
            self.dt_val = dt_val
            self.loss_type = loss_type
            if loss_type == 'atom':
                self.label_dics, self.label_list = self.create_labels_atom(dim, cutoff)
                self.loss_step = self.loss_step_atom
            elif loss_type == 'cavity':
                self.label_dics, self.label_list = self.create_labels_cavity(dim, cutoff)
                self.loss_step = self.loss_step_cavity
            self.gamma = gamma
            self.target_N = target_N
            self.indices_N = self.get_pt_indices()


            # create relevant operators
            self.a_ia_s, self.a_sigma_dag, self.sigma, self.a_s, self.N_s = self.create_operators(self.label_dics)
            self.N_s = np.diag(self.N_s) # create array representing diagonal matrix N_s
            # setup Hamiltonians
            self.H_JC = 1j * self.Omega/2 * (self.a_sigma_dag - self.a_sigma_dag.T)
            self.H_NL = self.a_ia_s.T + self.a_ia_s # a_ia_s is real
            # digaonalize Hamiltonians:
            self.ev_JC, self.V_JC = np.linalg.eigh(self.H_JC)
            self.ev_NL, self.V_NL = np.linalg.eigh(self.H_NL)

            # calculate relevant sparse operators 
            self.H_JC_sparse = csr_matrix( 1j * self.Omega/2 * (self.a_sigma_dag - self.a_sigma_dag.T) )
            self.H_NL_sparse = csr_matrix( self.H_NL )

            # jump operator
            if self.loss_type == 'atom':
                self.sigma_sparse = csr_matrix( self.sigma )
                self.Lindblad_terms = [self.sigma_sparse * np.sqrt(self.gamma)]
            elif self.loss_type == 'cavity':
                self.a_s_sparse = csr_matrix( self.a_s )
                self.Lindblad_terms = [self.a_s_sparse * np.sqrt(self.gamma)]

    @staticmethod
    def create_labels_atom(dim, cutoff):
        '''
        Creates labels for truncated basis of sub-Hilbertspace for Hybdrid System regadring atom decay.
        Basis states are ordered (atom, signal, idler): (g,N,N), (g,N,N-1), (g, N, N-2), ... (e, N, N-1), (e, N, N-2), (e, N, N-3), ...
        Args:
            dim: single photon fock space truncation dimension.
            cutoff: max orderes of couplings due to atom-decay loss.
        Returns:
            labels: list of dictionaries of basis states
            label_list: list of label-tuples
        '''
        labels = []
        label_list = []
        index_counter = 0
        # g-labels
        for nn_g in range(dim+1): # first list all nng-states
            label = {"atom": "g", "signal": nn_g, "idler": nn_g, "index": index_counter} # create dictionary, with index in basis
            label_list.append(f'(g,{nn_g},{nn_g})')
            labels.append(label)
            index_counter +=1
            
        for i in range(dim+2): # list all idler, signal, g - states
            for j in range(1, i):
                n = i-j
                label = {"atom": "g", "signal": i-1, "idler": n-1, "index": index_counter} # create dictionary, with index in basis
                label_list.append(f'(g,{i-1},{n-1})')
                labels.append(label)
                index_counter +=1

                if j == cutoff:
                    break
        # e-labels
        nume = 0
        for i in range(1, dim+1): # list all idler, signal, e - states
            for j in range(i):
                nume +=1
                n = i-j
                label = {"atom": "e", "signal": i, "idler": n-1, "index": index_counter} # create dictionary, with index in basis
                label_list.append(f'(e,{i},{n-1})')
                labels.append(label)
                index_counter +=1

                if j == cutoff - 1: # we need the minus one to cut off at same value, as we cut off the g,n,n - states!
                    break
        return labels, label_list
    
    @staticmethod
    def create_labels_cavity(dim, cutoff):
        '''
        Creates labels for truncated basis of sub-Hilbertspace for Hybdrid System regadring cavity decay (photon loss).
        Basis states are ordered (atom, signal, idler): (g,N,N), (g,N-1,N), (g, N-2, N), ... (e, N, N-1), (e, N-1, N-1), (e, N-2, N-1), ...
        Args:
            dim: single photon fock space truncation dimension.
            cutoff: max orderes of couplings due to photon loss.
        Returns:
            labels: list of dictionaries of basis states
            label_list: list of label-tuples
        '''
        label_dics = [{"atom": "g", "signal": 0, "idler": 0}]    
        label_list = [f'(g,{0},{0})']
        for i in range(1, dim):
            label_list.append(f'(e,{i},{i-1})')
            label_dics.append({"atom": "e", "signal": i, "idler": i-1})
            for j in range(i):
                if j == cutoff:
                    break
                label_list.append(f'(e,{i-(j+1)},{i-1})')
                label_dics.append({"atom": "e", "signal": i-(j+1), "idler": i-1})

            label_list.append(f'(g,{i},{i})')
            label_dics.append({"atom": "g", "signal": i, "idler": i})

            for j in range(i):
                if j == cutoff:
                    break
                label_list.append(f'(g,{i-(j+1)},{i})')
                label_dics.append({"atom": "g", "signal": i-(j+1), "idler": i})

        return label_dics, label_list

    @staticmethod
    def create_operators(label_dics):
        '''
        Creates operators a_ia_s and a_i_sigma^{\dagger}, sigma, a_s and N_s in the basis defined by label_dics.
        Args:
            label_dics: list of dictionary-elements denoting the basis states.
        Returns:
            a_i_as: 2-photon two-mode annihilation operator.
            a_sigma_dag: Jaynes Cummings interaction operator, destroying a photon in the idler mode and creating an excitation in the 2LS.
            sigma: annihilation operator of the 2LS degree of freedom.
            a_s: signal-mode annihilation operator
            N_s: signal-mode number operator
        '''
        a_ia_s = np.zeros((len(label_dics), len(label_dics)))
        # create a_ia_s operator
        for i in range(len(label_dics)):
            for j in range(len(label_dics)):
                if label_dics[j]['signal'] == (label_dics[i]['signal']+1) and label_dics[j]['idler'] == (label_dics[i]['idler']+1) and label_dics[j]['atom'] == label_dics[i]['atom']:
                    a_ia_s[i, j] = np.sqrt((label_dics[i]['signal']+1)*(label_dics[i]['idler']+1))

                
        # create a_i_sigma_dagger operator
        a_sigma_dag = np.zeros((len(label_dics), len(label_dics)))
        for i in range(len(label_dics)):
            for j in range(len(label_dics)):
                if label_dics[j]['atom'] == 'g' and label_dics[i]['atom'] == 'e' and label_dics[j]['signal'] == label_dics[i]['signal'] and label_dics[j]['idler'] == (label_dics[i]['idler']+1):
                    a_sigma_dag[i, j] = np.sqrt(label_dics[j]['idler'])
                    
        # create sigma operator
        sigma = np.zeros((len(label_dics), len(label_dics)))
        for i in range(len(label_dics)):
            for j in range(len(label_dics)):
                if label_dics[j]['atom'] == 'e' and label_dics[i]['atom'] == 'g' and label_dics[j]['signal'] == label_dics[i]['signal'] and label_dics[j]['idler'] == label_dics[i]['idler']:
                    sigma[i, j] = 1
                    
        # create a_s operator
        a_s = np.zeros((len(label_dics), len(label_dics)))
        for i in range(len(label_dics)):
            for j in range(len(label_dics)):
                if label_dics[j]['atom'] == label_dics[i]['atom'] and label_dics[j]['signal'] == label_dics[i]['signal']+1 and label_dics[j]['idler'] == label_dics[i]['idler']:
                    a_s[i, j] = np.sqrt(label_dics[j]['signal'])
        
    # create N_s operator
        N_s = np.zeros((len(label_dics), len(label_dics)))
        for i in range(len(label_dics)):
            for j in range(len(label_dics)):
                if label_dics[j]['atom'] == label_dics[i]['atom'] and label_dics[j]['signal'] == label_dics[i]['signal'] and label_dics[j]['idler'] == label_dics[i]['idler']:
                    N_s[i, j] = label_dics[j]['signal']
        return a_ia_s, a_sigma_dag, sigma, a_s, N_s

    def create_U_NL(self, r, phi):
        '''
        Creates the operator U_NL depending on r and phi, by using the diagonalization to pass the parameters.
        Args:
            r: parametric gain magnitude.
            phi: phase, can be set to 0 or pi.
        Returns:
            U_NL: U_NL operator.
        '''
        sign = 1 if phi == 0 else -1
        U_NL = self.V_NL @ np.diag(np.exp(-1j * r * self.ev_NL * sign)) @ np.conj(self.V_NL).T 
        return U_NL

    def create_U_JC(self, t):
        '''
        Creates the operator U_JC depending on t, by using the diagonalization to pass the parameters.
        Args:
            t: time-delay parameter.
        Returns:
            U_JC: U_JC operator.
        '''
        U_JC = self.V_JC @ np.diag(np.exp(-1j * t * self.ev_JC)) @ np.conj(self.V_JC).T
        return U_JC

    def loss_step_atom(self, rho, dt):
        '''
        Applies the transformation to the density matrix during a dt-step that corresponds to the effect of
        atom-decay loss.
        Args:
            rho: density matrix of composite system. (structure: atom,idler,signal) [special structure due to pre-listing of N,N,g-states]
            dt: time-step 
        Returns:
            rho: transformed density matrix.
        '''
        dim, gamma_a = self.dim, self.gamma
        dim_sub = rho.shape[0] - (dim+1) # dimension of submatrix that is affected by loss-transformation
        
        rho[dim+1:dim_sub//2+dim+1, dim+1:dim_sub//2+dim+1] += ( rho[dim+1+dim_sub//2:, dim+1+dim_sub//2:] * (1-np.exp(-gamma_a*dt)) ) # rho_gg transformation
        # (rho_gg = rho_gg + rho_ee*(1-exp(-gamma_a*dt)))
        rho[dim+1+dim_sub//2:, dim+1+dim_sub//2:] *= np.exp(-gamma_a*dt) # rho_ee transformation
        rho[dim+1:dim_sub//2+dim+1, dim+1+dim_sub//2:] *= np.exp(-gamma_a/2*dt) # rho_eg transformation (upper right sector)
        rho[dim+1+dim_sub//2:, dim+1:dim_sub//2+dim+1] *= np.exp(-gamma_a/2*dt) # rho_ge transformation (lower left sector)
        rho[:dim+1, dim+1:] *= np.exp(-gamma_a/2*dt) # rho_eg transformation (very upper block)
        rho[dim+1:, :dim+1] *= np.exp(-gamma_a/2*dt) # rho_ge transformation (very left block)
        return rho

    def loss_step_cavity(self, rho, dt):
        '''
        Applies the transformation to the density matrix during a dt-step that corresponds to the effect of
        signal mode photon loss.
        Args:
            rho: density matrix of composite system.
            dt: time-step 
        Returns:
            rho: transformed density matrix.
        '''
        gamma_s = self.gamma

        T = 1.0 - np.exp(-gamma_s*dt)
        m0 = np.exp(-1/2 * dt * gamma_s*self.N_s) 

        # --- apply transformation rho = M0 @ rho @ np.conj(M0).T + M1 @ rho @ np.conj(M1).T, we use element-wise broadcasting for efficiency ---
        rho = m0[:, None] * (rho + T * (self.a_s_sparse @ rho @ self.a_s_sparse.T)) * m0[None, :]

        return rho

    def get_pt_indices(self):
        '''
        Calculates indices of basis that correspond to states that contribute to the fidelity of
        the reduced signal mode density matrix with the target N fock state.
        Returns:
            indices: indices of relevant basis states.
        '''
        indices_N = []
        for i in range(len(self.label_dics)):
            if self.label_dics[i]['signal'] == self.target_N:
                indices_N.append(i)
        return indices_N

    def calculate_Fidelity(self, rho):
        '''
        Calculates fidelity of reduced density matrix of signal mode with target fock state.
        Args:
            rho: density matrix represented in truncated basis. (structure: atom, idler, signal).
        Returns:
            Fidelity
        '''
        F_trunc = sum([rho[self.indices_N[i], self.indices_N[i]].real for i in range(len(self.indices_N))])
        return F_trunc
    
    @staticmethod
    def ME_RHS(rho, Hmat, Lindblad_terms):
        '''
        Calculates the r.h.s. of the Lindblad Master Equation by transforming the density matrix by the
        operator sum representation of the Lindbladian.
        Args:
            rho: density matrix
            Hmat: Hamiltonian
            Lindblad_terms: List of jump operators describing loss
        Returns:
            rho_out: transformed density matrix
        '''
        rho_out = -1j*(Hmat @ rho - rho @ Hmat)
        for i in range(len(Lindblad_terms)):
            L = Lindblad_terms[i]
            rho_out += L @ rho @ L.T.conjugate() - 1/2*(L.T.conjugate() @ L @ rho + rho @ L.T.conjugate() @ L)
        return rho_out

    @staticmethod
    def non_linear_derivative(rho, H_NL):
        '''
        Transforms the density matrix in a way that corresponds to taking the derivative 
        w.r.t. to the parameter r.
        Args:
            rho: density matrix
            H_NL: non-linear Hamiltonian describing pulse interaction for phase fixed to 0 or pi
        Returns:
            rho_out: transformed density matrix
        '''
        rho_out = -1j * H_NL @ rho
        rho_out += rho_out.conjugate().T
        return rho_out

    @staticmethod
    def Adam(iteration_number, hyperparams, params, s, m, gradient, epsilon = 10**-7):
            """ 
            Computes one step of Gradient Descent using the Adam Optimizer.
            Args:
                iteration_number: required for the two update steps (m_hat and s_hat)
                hyperparams: beta1, beta2, learning_rate ... hyperparameters required for Adam
                params: parameters, expects array of the form [r1,phi1,t1,r2,phi2,t2,r3,...]
                s: scaling vector (exponentially decaying average of past squared gradients)
                m: momentum vector (exponentially decaying average of past gradients)
                gradient: gradient of cost function at current point in parameter space
                epsilon: necessary to avoid singularities
            Returns:
                m: updated momentum vector
                s: updated scaling vector
                out: updated parameters
            """
            beta1, beta2, learning_rate = hyperparams
            m = beta1 * m - (1-beta1)*(-gradient) # changed the sign here, so technically we are computing a gradient-ascent scheme
            s = beta2 * s + (1-beta2)*gradient**2
            m_hat = m/(1-beta1**iteration_number)
            s_hat = s/(1-beta2**iteration_number)
            out = params + learning_rate*m_hat / np.sqrt(s_hat+epsilon) 
            return m, s, out
        

    def time_integrate(self, start_time, end_time, dt_val, rho, save_intermediate=False):
        '''
        Calculate Dynamics between pulses.
        Args:
            start_time: start time of integration
            end_time: end time of integration
            dt_val: tstep-value
            rho: density matrix
            save_intermediate: if True, saves intermediate density matrices at each time step
        Return:
            if save_intermediate is False: 
                rho: density matrix after interpulse dynamics
            if save_intermediate is True:
                timelist: list of time points
                rhos_intermediate: list of intermediate density matrices 

        '''
        dt_points = int((end_time-start_time)//dt_val)
        timelist = np.linspace(start_time, end_time, dt_points)
        dt = timelist[1]-timelist[0]
        U_JC = csr_matrix( self.create_U_JC(dt) )
        U_JC_H = U_JC.getH() # Hermitian conjugate of U_JC 
        rhos_intermediate = []  # List to store intermediate density matrices

        rho = self.loss_step(rho, dt/2)
        for j in range(len(timelist)-1): # for j in range(len(timelist)-1) because we use the symmetric splitting scheme 
            rho = U_JC @ rho @ U_JC_H
            rho = self.loss_step(rho, dt)
            if save_intermediate:
                rhos_intermediate.append(rho.copy())  # Save a copy of the current density matrix
        rho = U_JC @ rho @ U_JC_H
        rho = self.loss_step(rho, dt/2)
        if save_intermediate:
            rhos_intermediate.append(rho.copy())  # Save the final density matrix
            return timelist, rhos_intermediate
        else:
            return None, rho
  
    def dynamics_sub_unitaries(self, params, U_NLs, t_params):
        '''
        Calculates the Dynamics of the pulsed hybrid system for a given set of pulse-parameters and 
        returns all sub-density matrices.
        Args:
            params: parameter-array
            U_NLs: list of Unitaries describing the pulse-interactions
            t_params: cummulative array of time-delays
        Returns:
            rhos_out: list of density matrices after each pulse-interaction/Rabi-oscillation period
        '''
        squeeze = U_NLs[0][:,0] # state-vector after applying first squeezing operator
        rho = np.outer(squeeze, np.conj(squeeze))
        rhos_out = [rho] 
        for i in range(len(t_params)-1):
            start_time, end_time = t_params[i], t_params[i+1]
            _, rho = self.time_integrate(start_time, end_time, self.dt_val, rho, save_intermediate=False)
            rhos_out.append(rho)
            rho = U_NLs[i+1] @ rho @ np.conj(U_NLs[i+1].T)   
            rhos_out.append(rho)
        return rhos_out

    def dynamics_all_time_steps(self, params, U_NLs, t_params):
        '''
        Calculates the Dynamics of the pulsed hybrid system for a given set of pulse-parameters and 
        returns the density matrices at every time step.
        Args:
            params: parameter-array ... we assume (p+1) pulses for a p-pulse preparation protocol with the last pulse having r=0, for plotting purposes   
            U_NLs: list of Unitaries describing the pulse-interactions
            t_params: cummulative array of time-delays
        Returns:
            timelists: list of time points at every time step
            rhos_out: list of diagonal elements of the density matrix at every time step
            rho_after_last_pulse: density matrix after the last pulse of the preparation protocol, for later calculation of fidelity
        '''
        squeeze = U_NLs[0][:,0] # state-vector after applying first squeezing operator
        rho = np.outer(squeeze, np.conj(squeeze))
        rhos_out, timelists = [], []
        for i in range(len(t_params)-1):
            start_time, end_time = t_params[i], t_params[i+1]
            timelist, rhos_intermediate = self.time_integrate(start_time, end_time, self.dt_val, rho, save_intermediate=True)
            rhos_intermediate[-1] = U_NLs[i+1] @ rhos_intermediate[-1] @ np.conj(U_NLs[i+1].T)
            rho = rhos_intermediate[-1].copy() 
            if i == len(U_NLs)-3: # save the state after the last pulse, for later calculation of fidelity
                rho_after_last_pulse = rho

            rhos_out = rhos_out + [np.diag(rho).real for rho in rhos_intermediate]
            timelists = timelists + timelist.tolist()  # convert numpy array to list and concatenate
        return timelists, rhos_out, rho_after_last_pulse

    def compute_gradient(self, params):
        ''' 
        Computes the gradient of the fidelity w.r.t. all optimization parameters using
        a forward-mode differentiation scheme.

        Phases phi_i are kept fixed (binary 0 or pi), so their gradient components are set to 0.

        Args:
            params: parameter-array of the form [r1, phi1, t1, r2, phi2, t2, ..., r_p, phi_p]
                    where r_i: squeezing strength of pulse i, phi_i: phase (0 or pi), t_i: evolution duration after pulse i
        Returns:
            rhos_out_NL[-1]: final density matrix (after the last pulse)
            gradient_components: gradient array, structured as [dF/dr1, 0, dF/dt1, dF/dr2, 0, dF/dt2, ..., dF/dr_p, 0]
        '''
        
        # unpack parameters: squeezing strengths, phases, and cumulative time-delays 
        r_params, phi_params, t_params = params[::3], params[1::3], np.cumsum(np.append(0, params[2::3])) # t_params are cumulative in order to define time-intervalls
        U_NLs = [self.create_U_NL(r_params[i], phi_params[i]) for i in range(len(r_params))]
    
        # run forward pass: collect density matrices after each pulse (rhos_out_NL) and after each inter-pulse delay period (rhos_out_2LS)
        rhos_out = self.dynamics_sub_unitaries(params, U_NLs, t_params)
        rhos_out_NL, rhos_out_2LS = rhos_out[::2], rhos_out[1::2] # split into density matrices after pulse-interactions and after Rabi-oscillation periods
        signs = (-1) ** (phi_params== np.pi) # returns array encoding the phase of pulses

        g_t, g_r = [], [] # t-gradients, r-gradients
            
        for i in range(len(r_params)-1):
            # --- gradient w.r.t. r_i (squeezing strength of pulse i) ---
            g_r_comp = self.non_linear_derivative(rhos_out_NL[i], signs[i]*self.H_NL_sparse)

            # --- gradient w.r.t. t_i (time-delay after pulse i) ---
            g_t_comp = self.ME_RHS(rhos_out_2LS[i], self.H_JC_sparse, self.Lindblad_terms)

            # --- propagate through remaining pulses and time-delays ---
            g_t_comp = U_NLs[i+1] @ g_t_comp @ np.conj(U_NLs[i+1].T) 
            for j in range(i, len(r_params)-2):
                _, g_t_comp = self.time_integrate(t_params[j+1], t_params[j+2], self.dt_val, g_t_comp)
                g_t_comp = U_NLs[j+2] @ g_t_comp @ np.conj(U_NLs[j+2].T)
                _, g_r_comp = self.time_integrate(t_params[j], t_params[j+1], self.dt_val, g_r_comp)
                g_r_comp = U_NLs[j+1] @ g_r_comp @ np.conj(U_NLs[j+1].T)
            _, g_r_comp = self.time_integrate(t_params[len(r_params)-2], t_params[len(r_params)-1], self.dt_val, g_r_comp)
            g_r_comp = U_NLs[len(r_params)-1] @ g_r_comp @ np.conj(U_NLs[len(r_params)-1].T)
                
            g_t.append(g_t_comp)
            g_r.append(g_r_comp)

        # gradient w.r.t. r of the last pulse: no forward propagation needed since it acts directly on the final state
        g_r.append(self.non_linear_derivative(rhos_out_NL[len(r_params)-1], signs[len(r_params)-1]*self.H_NL_sparse)) # last pulse

        # assemble gradient vector: [dF/dr_1, 0, dF/dt_1, dF/dr_2, 0, dF/dt_2, ..., dF/dr_p, 0]
        gradient_components = []
        for comp in range(len(r_params)-1):
            gradient_components.append( self.calculate_Fidelity(g_r[comp]) ) # dF/dr_i
            gradient_components.append( 0 ) # for phase, which is not being modified
            gradient_components.append( self.calculate_Fidelity(g_t[comp]) ) # dF/dt_i
        gradient_components.append( self.calculate_Fidelity(g_r[comp+1]) ) # dF/dr_p (last pulse)
        gradient_components.append( 0 ) # phase of last pulse, not modified

        return rhos_out_NL[-1], np.array(gradient_components)


    def single_GD_run(self, num_iterations, init_parameters, hyperparams, stopping_cond):
        '''
        Algorithm for a single GD run for a given initial parameter-configuration.
        Args:
            num_iterations: number of iteratons
            init_parameters: initialization
            hyperparams: hyperparameters for Adam Optimization
            stopping_cond: small value to determine when to stop iterating 
        Returns:
            Probs: Fidelity with target state at every iteraiton 
            Params: parameter-configuration at every iteration
            stopping_condition_met: if True, stopping conditon was met
        '''
        t_fitted_to_zero = False
        stopping_condition_met = False
        Probs = []
        Params = []
        params = init_parameters
        m, s = np.zeros(init_parameters.size), np.zeros(init_parameters.size)
        for i in range(num_iterations):
            
            if (params[2::3] < (2*self.dt_val) ).any(): # check if time-integration will fail, 2*dt_val, because at least 2 values are necessary to define our timelist in the time_integrate - function
                t_fitted_to_zero = True
                return np.array(Probs), np.array(Params), stopping_condition_met, t_fitted_to_zero
            
            rho_f, gradient = self.compute_gradient(params)
            Probs.append(self.calculate_Fidelity(rho_f))
            m, s, params = self.Adam(i+1, hyperparams, params, s, m, gradient, epsilon = 10**-7)
            Params.append(params)
            if np.sqrt(gradient @ gradient) < stopping_cond:
                stopping_condition_met = True
                return np.array(Probs), np.array(Params), stopping_condition_met, t_fitted_to_zero
        return np.array(Probs), np.array(Params), stopping_condition_met, t_fitted_to_zero


import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class Unitary_Optimization:
    def __init__(self, Omega, dim, fixed_phases, optimizer, hyperparams, pulse_num, target_N, num_iterations, verbose=True):
        """
        Initializes Optmization scheme.

        Args: 
            Omega: single-photon Rabi Frequency
            dim: Truncation dimension for both cavity modes (Truncation at (dim,dim-1,e))
            optimizer: chooses type of optmizer used (Options are: 'Vanilla GD', 'Adam')
            hyperparams: array of [learning_rate, beta1, beta2] ... beta1 and beta2 are required for Adam, can be set to None in case we use Vanilla GD
            pulse_num: specifies number of pulses (N pulses -> 2N-1 params, if phases are fixed)
            target_N: Fock State, we are optimizing for
            num_iterations: number of iteration that will be used for a single GD-run
            fixed_phases: if False, phases will be updated during GD optimization;
                        phase-array can also be passed with phase values modolo pi (in this case, the phases will be fixed to those values during optimization)
            verbose: display time-updates during GD-run
        """
        self.Omega = Omega
        self.dim = dim
        self.phases_fixed = fixed_phases 
        self.optimizer = optimizer
        self.hyperparams = hyperparams
        self.pulse_num = pulse_num 
        self.labels = self.create_labels() # labels for basis states
        self.operators = self.initialize_operators() # operators representing pulse-interactions (with phases fixed to 0, pi) and Rabi Oscillation
        self.diagonalize = self.diagonalize_operators() # eigenvalues and eigenstates for all relevant operators
        self.target_N = target_N
        self.num_iterations = num_iterations
        self.verbose = verbose
    
    def initialize_operators(self):
        """
        Initializes relevant Operators depending on the self.phases_fixed input.
        If self.phases_fixed is an array, we are initializing all relevant operators with the respective fixed phases,
        otherwise we only initialize the H_2LS operator, as the non-linear Operators have to be diagonalized parameter-dependent.
        """
        if self.phases_fixed == False:
            return  [self.H_2LS()]
        else:
            out = [self.H_NL(np.pi * self.phases_fixed[i]) for i in range(self.pulse_num)]
            out.append(self.H_2LS())
            return out
        
    def create_labels(self):
        """
        Initializes labels for the composite basis states.
        Basis consists of states: {(N,N,g), (N,N-1,e)}, 
        where composite states describe (signal, idler, atom)
        """
        labels = []
        for i in range(self.dim):
            labels.append(fr'$| {i}, {i}, g\rangle$')
            labels.append(fr'$|{i},{i+1}, e\rangle$')
        return labels
    
    def ao_dag(self):
        """
        Initializes Operator $a_i \sigma^{\dagger}$.
        """
        counter = 0
        array = np.zeros(2*self.dim-1)
        for i in range(1, 2*self.dim-1, 2):
            array[i] = np.sqrt(counter + 1)
            counter += 1
        return np.diag(array, +1)
    
    def downdown(self):
        """
        Initializes Operator $a_i a_s$.
        """
        array = []
        for i in range(1, self.dim):
            array.append(i)
            array.append(np.sqrt(i+1)*np.sqrt(i))
        return np.diag(array, +2)
   
    def H_NL(self, phi):
        """ 
        Args:
            phi: phase of pulse
        Returns:
            exponent of unitary Operator U_NL 
            describing the pulse-interaction (up to -1j*r), 
            does not depend on r
        """
        down_down = self.downdown()
        up_up = down_down.T # real valued operator -> no need for complex conjugation
        return np.exp(1j*phi)*up_up + np.exp(-1j*phi)*down_down
    
    def H_2LS(self):
        """
        Returns:
            exponent of unitary Operator U_2LS 
            describing the Rabi-Oscillation period (up to -1j*t), 
            does not depend on t
        """
        rabi_up = self.ao_dag()
        rabi_down = rabi_up.T # real valued operator -> no need for complex conjugation
        return self.Omega * 1j * (rabi_up - rabi_down) / 2

    def diagonalize_operators(self):
        """
        Returns:
            eigenvalue-arrays, eigenstate-matrices for all relevant operators
        """
        diago = [np.linalg.eigh(self.operators[i]) for i in range(len(self.operators))] # eigenvalues and eigenvectors of operators
        evs = np.array([diago[i][0] for i in range(len(self.operators))])
        ess = np.array([diago[i][1] for i in range(len(self.operators))])
        return evs, ess
    
    def create_unitaries(self, params):
        """
        Creates all relevant unitary Operators required to 
        calculate the Dynamics.
        
        Args:
            params: parameters, expects array of the form [r1,t1,r2,t2,r3,...]
        Returns:
            U_matrices: array of unitary matrices, [U_NL1, U_2LS1, U_NL2, U_2LS2, U_NL3, ...];
            U_tot: total Unitary Dynamics Operator
        """
        if self.phases_fixed == False:
            # paramaters should be passed as (r1, t1, phi1, r2, t2, phi2, ...)
            r_params, t_params, phi_params = params[::3], params[2::3], params[1::3]
            # apply the diagonalization here:
            diago = [np.linalg.eigh(self.H_NL(phi_params[i])) for i in range(len(phi_params))]
            ev_arrays = [diago[i][0] for i in range(len(phi_params))]
            ev_arrays.append(self.diagonalize[0][0])
            V_mats = [diago[i][1] for i in range(len(phi_params))]
            V_mats.append(self.diagonalize[1][0])
        else: 
            # params should be passed as (r1, t1, r2, t2, ...)
            # diagonalizations have already been done
            V_mats = self.diagonalize[1] # eigenstate matrices
            ev_arrays = self.diagonalize[0] # eigenvalue vectors
            r_params, t_params = params[::2], params[1::2]
        
        U_matrices = [V_mats[0] @ np.diag(np.exp(-1j * ev_arrays[0] * r_params[0])) @ np.conj(V_mats[0].T)]
        # append unitary operators representing different interaction-sequences
        for i in range(len(t_params)):
            U_matrices.append(V_mats[-1] @ np.diag(np.exp(-1j * ev_arrays[-1] * t_params[i])) @ np.conj(V_mats[-1].T))
            U_matrices.append(V_mats[i+1] @ np.diag(np.exp(-1j * ev_arrays[i+1] * r_params[i+1])) @ np.conj(V_mats[i+1].T))

        U_tot = U_matrices[0]
        for i in range(1, len(U_matrices)):
            U_tot = U_matrices[i] @ U_tot
                 
        return U_matrices, U_tot
            
    def gradient(self, params):
        """
        Computes the gradient of the Loss function at the point in parameter-space
        given by params.
        Args:
            params: parameters, expects array of the form [r1,t1,r2,t2,r3,...]
        Returns:
            grad: array with all the gradient components.
            U_tot: total Unitary Dynamics Operator
        """
        if self.phases_fixed == False:
            # paramaters should be passed as (r1, t1, phi1, r2, t2, phi2, ...)
            r_params, t_params, phi_params = params[::3], params[2::3], params[1::3]
            U_matrices, U_tot = self.create_unitaries(params)            
            
            grad_r_matrices = []
            for i in range(len(r_params)):
                U_copy = U_matrices.copy()
                U_copy.insert( 2*i+1, (-1j * self.H_NL(phi_params[i])) )
                grad_matrix = U_copy[0]
                for mat in U_copy[1:]:
                    grad_matrix = mat @ grad_matrix 
                grad_r_matrices.append(grad_matrix)
                
            grad_phi_matrices = []
            for i in range(len(phi_params)):
                U_copy = U_matrices.copy()
                U_copy.insert(2*i+1, (r_params[i] * (np.exp(1j*phi_params[i])*self.downdown().T - np.exp(-1j*phi_params[i])*self.downdown())))
                grad_matrix = U_copy[0]
                for mat in U_copy[1:]:
                    grad_matrix = mat @ grad_matrix
                grad_phi_matrices.append(grad_matrix)
            
        else:
            # paramaters should be passed as (r1, t1, r2, t2, ...)
            r_params, t_params = params[::2], params[1::2]
            U_matrices, U_tot = self.create_unitaries(params)
            grad_r_matrices = []
            for i in range(len(r_params)):
                U_copy = U_matrices.copy()
                U_copy.insert( 2*i+1, (-1j * self.operators[i]) )
                grad_matrix = U_copy[0]
                for mat in U_copy[1:]:
                    grad_matrix = mat @ grad_matrix 
                grad_r_matrices.append(grad_matrix)
            
        grad_t_matrices = []
        H_JC = self.operators[-1]
        for i in range(len(t_params)):
            U_copy = U_matrices.copy()
            U_copy.insert(2*(i+1), (-1j * H_JC) )
            grad_matrix = U_copy[0]
            for mat in U_copy[1:]:
                grad_matrix = mat @ grad_matrix
            grad_t_matrices.append(grad_matrix)
            
        gradient_unitaries = [grad_r_matrices[0]]
        if self.phases_fixed == False:
            gradient_unitaries.append(grad_phi_matrices[0])
            for i in range(len(t_params)):
                gradient_unitaries.append(grad_t_matrices[i])
                gradient_unitaries.append(grad_r_matrices[i+1])
                gradient_unitaries.append(grad_phi_matrices[i+1])
        else:
            for i in range(len(t_params)):
                gradient_unitaries.append(grad_t_matrices[i])
                gradient_unitaries.append(grad_r_matrices[i+1])
        
        # matrix elements:
        grad = np.zeros(len(params))
        for i in range(len(grad)):
            g1 = - 2 * ( gradient_unitaries[i][2*self.target_N, 0] * np.conj(U_tot[2*self.target_N, 0]) ).real
            g2 = - 2 * ( gradient_unitaries[i][2*self.target_N-1, 0] * np.conj(U_tot[2*self.target_N-1, 0]) ).real
            grad[i] = g1 + g2
        return grad, U_tot 
    
    def Adam(self, iteration_number, params, s, m, gradient, epsilon = 10**-7):
        """ 
        Computes one step of Gradient Descent using the Adam Optimizer.
        Args:
            iteration_number: required for the two update steps (m_hat and s_hat)
            params: params: parameters, expects array of the form [r1,t1,r2,t2,r3,...]
            s: scaling vector (exponentially decaying average of past squared gradients)
            m: momentum vector (exponentially decaying average of past gradients)
            gradient: gradient of Loss function at current point in parameter space
            epsilon: necessary to avoid singularities
        Returns:
            m: updated momentum vector
            s: updated scaling vector
            out: updated parameters
        """
        beta1, beta2, learning_rate = self.hyperparams[1], self.hyperparams[2], self.hyperparams[0]
        m = beta1 * m - (1-beta1)*gradient
        s = beta2 * s + (1-beta2)*gradient**2
        m_hat = m/(1-beta1**iteration_number)
        s_hat = s/(1-beta2**iteration_number)
        out = params + learning_rate*m_hat / np.sqrt(s_hat+epsilon)
        return m, s, out
    
    def Vanilla_GD(self, params, gradient, epsilon = 10**-7):
        """
        Computes one step of Gradient Descent using the Vanilla Gradient Descent Optimizer.
        Args:
            params: params: parameters, expects array of the form [r1,t1,r2,t2,r3,...]
            gradient: gradient of Loss function at current point in parameter space
        Returns:
            out: updated parameters
        """
        learning_rate = self.hyperparams[0]
        return params - learning_rate * gradient**2 / (gradient + epsilon)
    
    
    def Adagrad(self, params, s, gradient, epsilon = 10**-7):
        """ 
        Computes one step of Gradient Descent using the Adagrad Optimizer.
        Args:
            params: parameters, expects array of the form [r1,t1,r2,t2,r3,...]
            s: scaling vector (exponentially decaying average of past squared gradients)
            gradient: gradient of Loss function at current point in parameter space
            epsilon: necessary to avoid singularities
        Returns:
            s: updated scaling vector
            out: updated parameters
        """
        s = s + gradient**2
        out = params - self.hyperparams[0] * gradient / np.sqrt(s + epsilon)
        return s, out

    def RMSprop(self, params, s, gradient, epsilon = 10**-7):
        """ 
        Computes one step of Gradient Descent using the RMSprop Optimizer.
        Args:
            params: parameters, expects array of the form [r1,t1,r2,t2,r3,...]
            s: scaling vector (exponentially decaying average of past squared gradients)
            gradient: gradient of Loss function at current point in parameter space
            epsilon: necessary to avoid singularities
        Returns:
            s: updated scaling vector
            out: updated parameters
        """
        learning_rate, beta = self.hyperparams[0], self.hyperparams[1]
        s = beta*s + (1-beta)*gradient**2
        out = params - learning_rate*gradient / np.sqrt(s+epsilon)
        return s, out
    
    def calculate_Fidelity(self, state, params, from_params = False):
        """
        Calculates the fidelity of a state to the target Fock State.
        Args:
            state: state vector of which we want to calculate the fidelity.
            params: parameters, can be passed if we directly want to calculate fidelity from params. Can be None otherwise
            from_params: gives the option to directly calculate the fidelity from the parameters as input.
        Returns:
            Fidelity to out target Fock State
        """
        if from_params == True:
            U_tot = self.create_unitaries(params)[1]
            state = U_tot[:,0]
        return np.sum( (np.abs(state)**2)[2*self.target_N-1:2*self.target_N+1] )
    
    def run_gradient_descent(self, initial_params, stopping_cond, save_Probs, save_params):
        """
        Gradient Descent Algorithm for a single initialization.
        Args:
            intial_params: initialization point in parameter space
            stopping_cond: determines when algorithm should stop (if sqrt(grad@grad) < stopping_cond)
            save_Probs: if True: we will save all Probabilities for every iteration
            save_params: if True: we will save all parameters for every iteration
        Returns:
            params: final parameters
            Probs: Fidelities to the target fock state at each iteration
            Param_array: array of all params at each iteration
            True/False: if True: stopping_conditoin was met 
        """
        params = initial_params.copy()
        s = np.zeros(len(params))
        m = np.zeros(len(params))
        Probs = []
        Param_array = []
        for i in range(self.num_iterations):
            gradient, U_tot = self.gradient(params)
            if self.optimizer == 'Adam':
                m, s, params = self.Adam(i+1, params, s, m, gradient)
            elif self.optimizer == 'Vanilla GD':
                params = self.Vanilla_GD(params, gradient)
            elif self.optimizer == 'Adagrad':
                s, params = self.Adagrad(params, s, gradient)
            elif self.optimizer == 'RMSprop':
                s, params = self.RMSprop(params, s, gradient)
            if save_Probs:
                psi_f = U_tot @ np.eye(2*self.dim)[0]
                Probs.append(self.calculate_Fidelity(psi_f, None, False))
            if save_params:
                Param_array.append(params)
            if self.verbose and i % 100 == 0:
                tqdm.write(f"Iteration #{i} | Probability: {self.calculate_Fidelity(U_tot[:, 0], None, False)}")
            if np.sqrt(gradient@gradient) < stopping_cond:
                return params, Probs, Param_array, True
        return params, Probs, Param_array, False
    
    def decibel_converter(self, r):
        """ 
        takes r in dB and returns r in normal units.
        Args:
            r: r-parameter (can be an array too) [dB]
        Returns:
            out: r-parameter (can be an array too) [normal Units]
        """
        out = -np.log(10**(-r/10)) / 2
        return out
    
    def to_decibel_converter(self, r):
        """ 
        takes r in normal units and returns r in dB.
        Args:
            r: r-parameter (can be an array too) [normal Units]
        Returns:
            out: r-parameter (can be an array too) [dB]
        """
        out = -10*np.log10( np.exp( -2*r ) )
        return out

    
    def phase_differences(self, params):
        """
        Returns phase differences.
        Args:
            params: parameter-array, expects array of the form [r1,phi1,t1,r2,phi2,t2,r3,...]
        Returns:
            array of phase differences modolo pi
        """
        if self.phases_fixed == False:
            phi_params = params[1::3]
            phase_diffs = [phi_params[i+1]-phi_params[i] for i in range(len(phi_params)-1)]
            return np.array(phase_diffs)/np.pi
        else: 
            return self.phases_fixed
        
        
    def create_grid(self, time_delays):
        """
        Creates the time-grid (and dt-values) for plotting the Dynamics given a set of parameters.
        Args:
            time_delays: array with the time-delay parameters
        Returns:
            time_grids: returns the time-grids for each of the Rabi-Oscillation-Periods.
            delta_ts: returns the time-steps used in each of the Rabi-Oscillation-Periods.
        """
        # time_delays is an array
        # all times are given in normal units
        time_points = np.append(0, np.cumsum(time_delays))
        num_points = 10000
        delta_ts = [(time_points[i+1]-time_points[i])/num_points for i in range(len(time_delays))]
        time_grids = np.array([np.arange(time_points[i], time_points[i+1], delta_ts[i]) for i in range(len(time_points)-1)])
        return time_grids, delta_ts
    
    def run_dynamics(self, params):
        """
        Computes states at all time steps in order to enable plotting of probabilities.
        Args:
            params: parameter-array
        Returns:
            states: state-vectors at all time-steps.
            times: time-grid.
        """
        if self.phases_fixed == False:
            r_params = params[::3]
            phi_params = params[1::3]
            time_delays = params[2::3]           
        else:
            r_params = params[::2]
            time_delays = params[1::2]

        t_grid, delta_ts = self.create_grid(time_delays)
        params_for_U_creation = [r_params[0]]
        if self.phases_fixed == False:
            params_for_U_creation.append(phi_params[0])
            for i in range(len(time_delays)-1): # time_delays array also includes last Rabi-Oscillation period, which we will handle separately
                params_for_U_creation.append(delta_ts[i])
                params_for_U_creation.append(r_params[i+1])
                params_for_U_creation.append(phi_params[i+1])
        else:
            for i in range(len(time_delays)-1):
                params_for_U_creation.append(delta_ts[i])
                params_for_U_creation.append(r_params[i+1])

        Unitaries, U_tot = self.create_unitaries(params_for_U_creation)
        Unitaries.append(self.diagonalize[1][-1] @ np.diag(np.exp(-1j * self.diagonalize[0][-1] * delta_ts[-1])) @ np.conj(self.diagonalize[1][-1].T)) # append last Rabi-Oscillation period
        # run dynamics:
        states = []
        vac = np.eye(2*self.dim, dtype = 'complex128')[0]
        psi_f = vac
        for time_span in range(len(t_grid)):
            psi_f = Unitaries[2*time_span] @ psi_f # here we only pick the U_NL matrices
            for tstep in range(t_grid[time_span].size):
                psi_f = Unitaries[2*time_span+1] @ psi_f # Unitaries are [U_NL1, U_2LS1, U_NL2, ...], therefore we pick only the U_2LS-matrices here
                states.append(psi_f)

        times = t_grid.reshape(t_grid.shape[0]*t_grid.shape[1])
        return np.array(states), times 
    
    def plot_dynamics(self, num_states, params, figsize, extra_time_delay, dpi, legend_coords, show_legend = True, save_fig = False):
        """
        Plots the Probabilities for the composite states over time, as well as the pulses indicated as dashed lines.
        Args:
            num_states: number of lowest-N states that we want to plot.
            params: parameter-configuration used to calculate the dynamics.
            figsize: tuple (f1,f2), describing the figsize of the plot.
            extra_time_delay: appended Rabi-Oscillation period for showing the final state oscillations.
            dpi: dpi-value for plotted figure.
            legend_coords: tuple (c1,c2), describing where to plot the legend.
            show_legend: if True, we depict the legend in the plot.
            save_fig: if True, we save the figure in the local directory
        """
        fig1, fig2 = figsize
        # append extra time_delay:
        params = np.append(params, extra_time_delay)
        
        # run dynamics:
        states, times = self.run_dynamics(params)
        
        time_scaler = 2*np.pi/self.Omega 
        fig, ax = plt.subplots(figsize=(fig1,fig2), dpi = dpi)
        for i in range(num_states):
            ax.plot(times / time_scaler, np.abs(states[:, i])**2, label = f'{self.labels[i]}')
            ax.set_xlabel(r'time [$\frac{2\pi}{\Omega}$]')
            ax.set_ylabel(r'$\langle N_i, N_s, a| \rho_f | N_i, N_s, a\rangle$')
        if show_legend: 
            ax.legend( loc='upper right', bbox_to_anchor=(legend_coords))
        # ax.set_title(f'P({self.target_N})={np.round((np.abs(states)**2)[-1, 2*self.target_N] + (np.abs(states)**2)[-1, 2*self.target_N-1], 3)}')
        # plot pulses as kicks:
        if self.phases_fixed == False:
            r_array = params[::3]
            t_array = np.append(0, np.cumsum(params[2::3]))
        else: 
            r_array = params[::2]
            t_array = np.append(0, np.cumsum(params[1::2]))
            
        for j in range(len(r_array)):
            if j == 0: # only for the legend
                ax.plot(np.ones(10)*t_array[j] / (2*np.pi), np.linspace(0,1,10), linestyle = '--', linewidth = 2, alpha = 0.5, c = 'black', label = 'Pulses')
            else:
                ax.plot(np.ones(10)*t_array[j] / (2*np.pi), np.linspace(0,1,10), linestyle = '--', linewidth = 2, alpha = 0.5, c = 'black', label = '')
        plt.tight_layout()
        if save_fig:
            plt.savefig(f'Plotted_Dynamics_N={self.target_N}.png')
        plt.show()
        
    def investigate_states(self, params, figsize, xlimit, save_fig):
        """
        Plots state vector of dynamics after each Unitary in several several subplots.
        Args:
            params: parameter-array
            figsize: tuple (figsize1, figsize2), determines figsize of plot
            xlimit: determins xlim of plot
            save_fig: saves figure if true
        """
        if self.phases_fixed == False:
            print('currently this function only works for fixed phases!')
            return None
        Units = self.create_unitaries(params)[0]
        state = np.eye(2*self.dim)[0]
        states = [np.abs(state)**2]
        for i in range(len(Units)):
            state = Units[i] @ state
            states.append(np.abs(state)**2)
        states = np.array(states)
        fig, ax = plt.subplots(len(states), figsize = figsize)
        matrix_labels = [r'vac', r'$U_{NL}:$'] 
        for i in range(len(states)-2):
            matrix_labels += [r'$U_{2LS}:$', r'$U_{NL}:$']
        for i in range(len(states)):
            ax[i].bar(np.arange(len(states[0])), states[i], label = r'$\langle N,N,g|\rho|N,N,g\rangle$')
            ax[i].bar(np.arange(1, len(states[0]), 2), states[i, 1::2], color = 'magenta', label = r'$\langle N-1,N,e|\rho|N-1,N,e\rangle$')
            if i == 0:
                param_label = ''
            elif i % 2 != 0:
                if self.phases_fixed[(i-1)//2] == 1:
                    param_label = rf'$r_{1+i//2}$='+str(np.round(params[i-1], 2))+rf',$\phi_{1+i//2}=\pi$'
                elif self.phases_fixed[(i-1)//2] == 0:
                    param_label = rf'$r_{1+i//2}$='+str(np.round(params[i-1], 2))+rf',$\phi_{1+i//2}=0$'
            else:
                param_label = rf'$t_{i//2}$='+str(np.round(params[i-1], 2))
            ax[i].scatter(0,0, color = 'black', s = 0.00001, label = str(matrix_labels[i], )+param_label)
            ax[i].legend()
            ax[i].set_xlim(-1, xlimit)
            ax[i].set_ylabel(rf'P')
        ax[i].set_xlabel(r'$N_{tot}$')
        plt.tight_layout()
        if save_fig:
            plt.savefig(f'states_after_unitaries_N={self.target_N}{len(states)//2}pulses.png')
        plt.show()

    def loss_landscape(self, params, r_var, t_var, phi_var, num_points_r, num_points_phi, num_points_t):
        '''
        Calculates slices through the loss landscape, varying one of the parameters while having the others fixed.
        Only works, if self.phases_fixed is initialized as false! Otherwise returns None.
        Args:
            params: parameters
            r_var: region arround r-parameters within Fidelity is varied
            phi_var: region arround phi-parameters within Fidelity is varied
            t_var: region arround t-parameters within Fidelity is varied
            num_points_r: number of points calculated along r-range
            num_points_phi: number of points calculated along phi-range
            num_points_t: number of points calculated along t-range
        Returns:
            slices: list of lists of arrays for the slices, ordered as r1,phi1,t1,r2,...
            all_ranges: list of lists of arrays for the ranges
        '''
        if self.phases_fixed == False:
            r_params, phi_params, t_params = params[::3], params[1::3], params[2::3]
        else:
            print('This function only works for phases_fixed = False!!!')
            return None
        slices = []
        
         
        t0 = []
        for i in range(len(t_params)):
            if t_params[i] - t_var < 0:
                t0.append(0)
            else:
                t0.append(t_params[i] - t_var)
                
        r_ranges = [np.linspace(r_params[i]-r_var, r_params[i]+r_var, num_points_r) for i in range(len(r_params))]
        t_ranges = [np.linspace(t0[i], t_params[i]+t_var, num_points_t) for i in range(len(t_params))]
        phi_ranges = [np.linspace(phi_params[i]-phi_var, phi_params[i]+phi_var, num_points_phi) for i in range(len(phi_params))]
        all_ranges = [r_ranges[0], phi_ranges[0]]
        for i in range(len(t_params)):
            all_ranges.append(t_ranges[i])
            all_ranges.append(r_ranges[i+1])
            all_ranges.append(phi_ranges[i+1])

        for j in range(len(params)):
            v_params = params.copy()
            Fs = []
            for l in range(len(all_ranges[j])):
                v_params[j] = all_ranges[j][l]
                F = self.calculate_Fidelity(r_params, v_params, True) # r_params is being passed as 'state' but un-used
                Fs.append(F)
            slices.append(Fs)
        return slices, all_ranges
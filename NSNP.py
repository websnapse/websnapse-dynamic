import json
import itertools
import numpy as np
from pprint import pprint


class NumericalSNPSystem:
    def __init__(self, system_description):
        self.neurons = system_description['neurons']
        self.synapses = system_description['syn']
        self.in_neuron = system_description['in_']
        self.out_neuron = system_description['out']

        # ORDERING SYSTEM DETAILS
        self.get_nrn_order()
        self.get_var_order()
        self.get_prf_order()

        # INITIAL MATRICES
        self.get_init_cmatrix()
        self.get_fmatrix()
        self.get_lmatrix()

        # MAPPING DETAILS
        self.map_neuron_to_var()
        self.map_neuron_to_neuron()
        self.map_func_to_var()

        # CONFIGURATION DEPENDENT MATRICES
        # spiking_mx = self.get_smatrix(self.config_mx[0])
        # production_mx = self.get_pmatrix(self.config_mx[0])
        # variable_mx = self.get_vmatrix(spiking_mx,self.config_mx[0])

    # ===========================================================================
    # Get the order of the variables and functions
    # ---------------------------------------------------------------------------
    # get_var_order() - Variables are ordered by the order of the neurons
    # get_prf_order() - Functions are ordered by the order of the neurons
    # ===========================================================================
    def get_var_order(self):
        self.variables = {var: self.neurons[neuron]['var'][var]
                          for neuron in self.neurons
                          for var in self.neurons[neuron]['var']}

        self.variable_keys = list(self.variables.keys())

    def get_prf_order(self):
        self.functions = {prf: self.neurons[neuron]['prf'][prf]
                          for neuron in self.neurons
                          for prf in self.neurons[neuron]['prf']}

        self.function_keys = list(self.functions.keys())

    def get_nrn_order(self):
        self.neuron_keys = list(self.neurons.keys())

    # ===========================================================================
    # Helper functions for building the Spiking Matrix
    # ---------------------------------------------------------------------------
    # check_threshold() - Check if the a function satisfies the threshold
    # compute_active_functions() - Active functions clusterred to each neuron
    # get_active_functions() - Get the indices of active functions in a neuron
    # ===========================================================================
    def check_threshold(self, config, function):
        function = self.functions[list(self.functions.keys())[function]]

        return all(config[list(self.variables.keys()).index(var)] >= function['thld']
                   if var != 'thld' and function['thld'] != None else True
                   for var in function)

    def compute_active_functions(self, config):
        active = np.zeros(
            (self.f_location_mx.shape[0], self.f_location_mx.shape[1]))
        active_count = np.zeros(self.f_location_mx.shape[1])

        for index_j, neuron in enumerate(self.f_location_mx.T):
            for index_i, function in enumerate(neuron):
                if self.check_threshold(config, index_i) and function:
                    active[index_i, index_j] = 1
                    active_count[index_j] += 1

        return (active_count, active)

    def get_active_functions(self, neuron, active):
        active = active.T[neuron]
        indices = [index for index, function in enumerate(active) if function]
        return indices

    # ===========================================================================
    # Helper functions for building the Production Matrix
    # ---------------------------------------------------------------------------
    # map_neuron_to_var() - Get the mapping of neurons to variables
    # map_neuron_to_neuron() - Get the mapping of neurons to neurons
    # map_func_to_var() - Get the mapping of functions to variables
    # ===========================================================================
    def map_neuron_to_var(self):
        self.neuron_to_var = {self.neuron_keys.index(neuron):
                              [self.variable_keys.index(
                                  var) for var in self.neurons[neuron]['var']]
                              for neuron in self.neurons}

    def map_neuron_to_neuron(self):
        self.neuron_to_neuron = {self.neuron_keys.index(neuron): []
                                 for neuron in self.neurons}

        for synapse in self.synapses:
            self.neuron_to_neuron[self.neuron_keys.index(
                synapse[0])] += [self.neuron_keys.index(synapse[1])]

    def map_func_to_var(self):
        self.func_to_var = {self.function_keys.index(prf): []
                            for prf in self.functions}

        for index, prf in enumerate(self.f_location_mx):
            belongs_to = np.where(np.isclose(prf, 1))[0][0]
            for neuron in self.neuron_to_neuron[belongs_to]:
                self.func_to_var[index] += self.neuron_to_var[neuron]

    # ===========================================================================
    # Get the matrices for the NSN P system
    # ---------------------------------------------------------------------------
    # get_init_cmatrix() - initial Configuration matrix
    # get_fmatrix() - Function matrix
    # get_lmatrix() - Function Location matrix
    # get_smatrix() - Spiking matrix
    # get_pmatrix() - Production matrix
    # ===========================================================================
    def get_init_cmatrix(self):
        self.config_mx = np.array([[self.neurons[neuron]['var'][var]
                                    for neuron in self.neurons
                                    for var in self.neurons[neuron]['var']
                                    ]], dtype=float)

    def get_fmatrix(self):
        self.function_mx = np.zeros((len(self.functions), len(self.variables)))

        for index_i, function in enumerate(self.functions):
            for index_j, variable in enumerate(self.variables):
                if variable in self.functions[function]:
                    self.function_mx[index_i,
                                     index_j] = self.functions[function][variable]

    def get_lmatrix(self):
        self.f_location_mx = np.zeros((len(self.functions), len(self.neurons)))

        for index_i, function in enumerate(self.functions):
            for index_j, neuron in enumerate(self.neurons):
                if function in self.neurons[neuron]['prf']:
                    self.f_location_mx[index_i, index_j] = 1
                else:
                    self.f_location_mx[index_i, index_j] = 0

    def get_smatrix(self, config, branch=None):
        active_count, active = self.compute_active_functions(config)
        comb_count = np.prod(active_count, where=active_count > 0)
        print(active_count)
        spiking_mx = np.zeros((int(comb_count), len(self.functions)))
        temp_comb_count = comb_count

        for index_m, neuron in enumerate(self.neurons):
            if active_count[index_m] == 0:
                continue

            functions = self.get_active_functions(index_m, active)
            amount = temp_comb_count/active_count[index_m]
            repeats = comb_count/temp_comb_count
            index_i = 0

            for index_k in range(int(repeats)):
                for index_j in functions:
                    counter = 0
                    while counter < amount:
                        spiking_mx[index_i, index_j] = 1
                        counter = counter + 1
                        index_i = index_i + 1

            temp_comb_count /= active_count[index_m]

        if branch is not None:
            temp_branch = branch if branch < comb_count else int(comb_count)
            indices = np.random.choice(
                spiking_mx.shape[0], temp_branch, replace=False)
            spiking_mx = spiking_mx[indices]

        return spiking_mx

    def get_pmatrix(self, config):
        production_mx = np.zeros((len(self.functions), len(self.variables)))

        for index_i, function in enumerate(self.function_mx):
            sum = 0
            for index_j, coefficient in enumerate(function):
                sum += coefficient * config[index_j]

            for index_j in self.func_to_var[index_i]:
                production_mx[index_i, index_j] = sum

        return production_mx

    def get_vmatrix(self, spiking_mx, config):
        variable_mx = np.array([config for i in range(spiking_mx.shape[0])])

        for index_i, row in enumerate(spiking_mx):
            for index_k, function in enumerate(row):
                if function == 1:
                    function_uid = self.function_keys[index_k]
                    for variable in self.functions[function_uid]:
                        if variable not in self.variable_keys:
                            continue
                        variable_index = self.variable_keys.index(variable)
                        variable_mx[index_i][variable_index] = 0

        return variable_mx

    # ===========================================================================
    # Main algorithm for simulating NSN P Systems
    # ---------------------------------------------------------------------------
    # simulate() - Simulate the NSN P system up to a certain depth
    # ===========================================================================
    def simulate(self, branch=None, cur_depth=0, sim_depth=10):
        unexplored_states = self.config_mx.tolist()
        explored_states = []
        depth = cur_depth
        state_graph = {
            'nrn_ord': self.neuron_keys,
            'fnc_ord': self.function_keys,
            'var_ord': self.variable_keys,
            'nodes': [],
        }

        while depth < sim_depth:
            next_states = []

            if not unexplored_states:
                break

            for config in unexplored_states:
                S = self.get_smatrix(np.array(config), branch)
                P = self.get_pmatrix(np.array(config))
                V = self.get_vmatrix(S, np.array(config))
                NG = np.matmul(S, P)
                C = np.add(NG, V)

                for state in C.tolist():
                    if state not in explored_states \
                            + unexplored_states + next_states:
                        next_states.append(state)

                state_graph['nodes'].append({
                    'conf': config,
                    'next': C.tolist(),  # list(k for k,_ in itertools.groupby(C.tolist()))
                    'spike': S.tolist(),
                })

                explored_states.append(config)
                unexplored_states.remove(config)

            unexplored_states += next_states
            depth = depth + 1

        return state_graph

# ACTIVE FUNCTIONS MATRIX EXAMPLE
# 		1 | 0 | 0
# 		1 | 0 | 0
# 		0 | 1 | 0
# 		0 | 1 | 0
# 		0 | 0 | 1
# 		0 | 0 | 1
# SPIKING MATRIX EXAMPLE
# 		1 0 1 0 1 0
# 		1 0 1 0 0 1
# 		1 0 0 1 1 0
# 		1 0 0 1 0 1
# 		0 1 1 0 1 0
# 		0 1 1 0 0 1
# 		0 1 0 1 1 0
# 		0 1 0 1 0 1


if __name__ == '__main__':
    with open('tests/NSNP_2.json', 'r') as f:
        data = json.load(f)

    NSNP = NumericalSNPSystem(data['NSNP'])
    state_graph = NSNP.simulate(
        data['branch'], data['cur_depth'], data['sim_depth'])

    for i in state_graph:
        if i == 'nodes':
            for j in state_graph[i]:
                print('CONFIGURATION:\t', j['conf'])
                print('SPIKING MATRIX:\t', j['spike'])
                print('NEXT STATES:\t', j['next'])
                print("====="*10)

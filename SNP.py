from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
import numpy as np
import regex as re
import json
import random

app = FastAPI()


class Rule(BaseModel):
    definition: str


class Neuron(BaseModel):
    type: str
    content: str
    rules: Dict[str, Rule]


class Synapse(BaseModel):
    source: str
    target: str
    weight: float


class SNPSystem(BaseModel):
    neurons: Dict[str, Neuron]
    synapses: Dict[str, Synapse]


class MatrixSNPSystem:
    def __init__(self, system: SNPSystem):
        self.neurons = system.neurons
        self.synapses = system.synapses

        self.set_neuron_order()
        self.set_rule_order()
        self.map_neuron_to_rule()

        self.neuron_count = len(self.neuron_keys)
        self.rule_count = len(self.rule_keys)

        self.config_vct = self.init_config_mx()
        self.adj_mx = self.init_adj_mx()
        self.init_trans_mx()
        self.delay_status_vct = self.init_delay_status_vct()
        self.neuron_status_vct = self.init_neuron_status_vct()
        self.delayed_indicator_vct = self.init_delayed_indicator_vct()

        self.print_system()
        self.simulate(1)

    def print_system(self):
        print("Neurons:")
        print(self.neuron_keys)
        print("Rules:")
        print(self.rule_keys)
        print("Initial Configuration:")
        print(self.config_vct)
        print("Neuron Rule Map:")
        print(self.neuron_rule_map)
        print(self.adj_mx)
        print("Spiking Transition Matrix:")
        print(self.trans_mx)
        print("Rule Delays:")
        print(self.rule_delay_vct)

    def simulate(self, iteration: int):
        print(f"= {iteration} ===")
        self.set_spikeable_mx()
        # print('SM:', end=' ')
        # print(self.spikeable_mx)

        self.choose_spiking_vct()
        # print('SV:', end=' ')
        # print(self.decision_vct)

        self.update_delay_status_vct()
        # print('DS:', end=' ')
        # print(self.delay_status_vct)

        self.compute_indicator_vct()
        # print('IV:', end=' ')
        # print(self.indicator_vct)

        self.update_delayed_indicator_vct()
        # print('DIV:', end=' ')
        # print(self.delayed_indicator_vct)

        self.update_neuron_status_vct()
        # print('NS:', end=' ')
        # print(self.neuron_status_vct)

        net_gain = np.dot(self.indicator_vct,
                          self.trans_mx).astype(int)
        # print('Net Gain:')
        # print(net_gain)
        print('CV:', end=' ')
        print(self.config_vct, end=' -> ')
        self.config_vct += np.multiply(
            self.neuron_status_vct, net_gain).astype(int)
        print(self.config_vct, self.config_vct[-1])

        while self.config_vct.sum() > 0:
            self.simulate(iteration+1)
        # print(depth)

    def choose_spiking_vct(self):
        possible_spiking_vct = len(self.spikeable_mx)
        decision_vct_idx = random.randint(1, possible_spiking_vct)-1
        # decision_vct_idx = int(input("Choose a spiking vector: "))

        self.decision_vct = self.spikeable_mx[decision_vct_idx]
    
    def update_delayed_indicator_vct(self):
        self.delayed_indicator_vct = np.logical_or(
            np.logical_and(self.delayed_indicator_vct, np.logical_not(self.indicator_vct)), 
            np.logical_and(self.decision_vct, self.delay_status_vct > 0)
        ).astype(int)

    def compute_indicator_vct(self):
        """
        Computes the indicator vector by multiplying 
        the rule status vector with the chosen 
        spiking vector
        """

        rule_status = np.logical_not(self.delay_status_vct).astype(int)

        self.indicator_vct = np.multiply(self.decision_vct + self.delayed_indicator_vct, rule_status)

    def update_neuron_status_vct(self):
        """
        Updates the neuron status vector by setting the value to 0
        if the corresponding rule is not activatable
        """
        self.neuron_status_vct = np.zeros((self.neuron_count, )).astype(int)
        existing_delays = self.delay_status_vct

        for rule_idx in range(self.rule_count):
            neuron_idx = self.get_mapped_neuron(rule_idx)
            self.neuron_status_vct[neuron_idx] = existing_delays[rule_idx] == 0

    def update_delay_status_vct(self):
        """
        Updates the delay status vector by decrementing each value by 1
        """

        for rule_idx in range(len(self.delayed_indicator_vct)):
            if self.delayed_indicator_vct[rule_idx] == 1:
                neuron_idx = self.get_mapped_neuron(rule_idx)
                delay = self.rule_delay_vct[rule_idx]
                for rule in self.neuron_rule_map[neuron_idx]:
                    self.delay_status_vct[rule] -= 1
                
        
        for rule_idx in range(self.rule_count):
            if self.decision_vct[rule_idx] == 1:
                neuron_idx = self.get_mapped_neuron(rule_idx)
                delay = self.rule_delay_vct[rule_idx]
                for rule in self.neuron_rule_map[neuron_idx]:
                    self.delay_status_vct[rule] = delay
        
    def init_delayed_indicator_vct(self):
        """
        Initializes the delayed spikeable vector to all zeros
        with a length equal to the number of rules
        """
        delayed_indicator_vct = np.zeros((self.rule_count, ))
        return delayed_indicator_vct

    def init_neuron_status_vct(self):
        """
        Initializes the neuron status vector to all ones
        with a length equal to the number of neurons
        """
        neuron_status_vct = np.ones((self.neuron_count, ))
        return neuron_status_vct

    def init_delay_status_vct(self):
        """
        Initializes the delay status vector to all zeros 
        with a length equal to the number of rules
        """
        delay_status_vct = np.zeros((self.rule_count, )).astype(int)
        return delay_status_vct

    def get_activatable_rules(self):
        """
        Returns a matrix of rules that are activatable by the current configuration
        """
        activatable_rules_mx = np.zeros(
            (self.rule_count, self.neuron_count, ))
        active_rules_in_neuron = np.zeros((self.neuron_count, ))
        for rule_idx, rule in enumerate(self.rules):
            neuron_idx = next(
                key for key, val in self.neuron_rule_map.items() if rule_idx in val)

            bound = self.rules[rule]['bound']
            spikes = self.config_vct[neuron_idx]
            neuron_status = self.neuron_status_vct[neuron_idx]
            rule_validity = np.multiply(neuron_status, self.check_rule_validity(
                bound, spikes)).astype(int)
            activatable_rules_mx[rule_idx, neuron_idx] = rule_validity
            active_rules_in_neuron[neuron_idx] += rule_validity
        return activatable_rules_mx, active_rules_in_neuron

    def set_spikeable_mx(self):
        """
        Creates a matrix of all possible combinations of rules that can be activated
        """
        activatable_rules, activatable_count = self.get_activatable_rules()
        comb_count = np.prod(activatable_count, where=activatable_count > 0)
        self.spikeable_mx = np.zeros((int(comb_count), self.rule_count))
        temp_comb_count = comb_count

        for neuron_idx, _ in enumerate(self.neurons):
            if activatable_count[neuron_idx] == 0:
                continue

            rules = self.get_active_rules(
                neuron_idx, activatable_rules)
            amount = temp_comb_count/activatable_count[neuron_idx]
            repeats = comb_count/temp_comb_count
            index_i = 0

            for _ in range(int(repeats)):
                for index_j in rules:
                    counter = 0
                    while counter < amount:
                        self.spikeable_mx[index_i, index_j] = 1
                        counter = counter + 1
                        index_i = index_i + 1

            temp_comb_count /= activatable_count[neuron_idx]

    def init_trans_mx(self):
        """
        Creates the Spiking Transition Matrix 
        by parsing the rules provided
        """
        self.trans_mx = np.zeros(
            (self.rule_count, self.neuron_count, )).astype(int)
        self.rule_delay_vct = np.zeros((self.rule_count, )).astype(int)

        for rule in self.rules:
            rule_idx = self.rule_keys.index(rule)
            consumption, production, delay = self.parse_rule(
                self.rules[rule]["definition"])
            self.rule_delay_vct[rule_idx] = delay

            neuron_idx = next(
                key for key, val in self.neuron_rule_map.items() if rule_idx in val)
            self.trans_mx[rule_idx] = np.dot(
                self.adj_mx[neuron_idx], production)
            self.trans_mx[rule_idx, neuron_idx] = consumption

    def init_config_mx(self):
        """
        Creates a matrix of initial neuron configurations
        """
        config_vct = np.array(
            [self.parse_spike(self.neurons[neuron].content) for neuron in self.neurons])
        return config_vct

    def init_adj_mx(self):
        """
        Creates an adjacency matrix of the neurons of the system
        """
        adj_mx = np.zeros((self.neuron_count, self.neuron_count)).astype(int)

        for synapse in self.synapses:
            source = self.neuron_keys.index(
                self.synapses[synapse].source)
            target = self.neuron_keys.index(self.synapses[synapse].target)
            adj_mx[source, target] = self.synapses[synapse].weight
        return adj_mx

    def parse_bound(self, definition: str):
        """
        Performs regex matching on the rule definition to get its bound
        """
        pattern = r"^((?P<bound>.*)\/)?(?P<consumption_bound>[a-z](\^((?P<consumed_single>[^0,1,\D])|({(?P<consumed_multiple>[2-9]|[1-9][0-9]+)})))?)->(?P<production>([a-z]((\^((?P<produced_single>[^0,1,\D])|({(?P<produced_multiple>[2-9]|[1-9][0-9]+]*)})))?;(?P<delay>[0-9]|[1-9][0-9]*))|(?P<forgot>0)))$"
        result = re.match(pattern, definition)
        if result is None:
            return ""
        return result.group('bound') or result.group('consumption_bound')

    def parse_spike(self, content: str):
        """
        Performs regex matching on the initial contents 
        of the neurons to get the number of spikes
        """
        pattern = r"^[a-z](\^((?P<spikes_single>[^0,1,\D])|{(?P<spikes_multiple>[2-9]|[1-9][0-9]+)}))?$"
        result = re.match(pattern, content)

        if result is not None:
            spike = int(result.group('spikes_single')
                        or result.group('spikes_multiple') or 1)
            return spike
        return 0

    def parse_rule(self, definition: str):
        """
        Performs regex matching on the rule definition to get 
        the consumption, production and delay values
        """
        pattern = r"^((?P<bound>.*)\/)?(?P<consumption_bound>[a-z](\^((?P<consumed_single>[^0,1,\D])|({(?P<consumed_multiple>[2-9]|[1-9][0-9]+)})))?)->(?P<production>([a-z]((\^((?P<produced_single>[^0,1,\D])|({(?P<produced_multiple>[2-9]|[1-9][0-9]+]*)})))?;(?P<delay>[0-9]|[1-9][0-9]*))|(?P<forgot>0)))$"

        result = re.match(pattern, definition)

        if result is None:
            return tuple((0, 0, 0, 0))

        consumption = result.group('consumed_single') or result.group(
            'consumed_multiple') or 1
        production = result.group('produced_single') or result.group(
            'produced_multiple') or 1 if not result.group('forgot') else 0
        delay = int(result.group('delay') or 0)

        consumption = -int(consumption)
        production = int(production)

        return consumption, production, delay

    def check_rule_validity(self, bound: str, spikes: int):
        """
        Performs regex matching to check 
        if the number of spikes inside neuron
        satisfies the bound of the rule
        """
        bound = re.sub("\\^(\\d)", "^{\\1}", bound)
        parsedBound = f"^{bound.replace('^', '')}$"
        validity = re.match(parsedBound, 'a'*spikes)
        return validity is not None

    def set_neuron_order(self):
        """
        Creates a list of neuron keys in the order of their appearance
        """
        self.neuron_keys = list(self.neurons.keys())

    def set_rule_order(self):
        """
        Creates a list of rule keys in the order of their appearance
        """
        self.rules = {rule: {"definition": self.neurons[neuron].rules[rule].definition,
                             "bound": self.parse_bound(self.neurons[neuron].rules[rule].definition)}
                      for neuron in self.neurons
                      for rule in self.neurons[neuron].rules}
        self.rule_keys = list(self.rules.keys())

    def map_neuron_to_rule(self):
        """
        Creates a dictionary mapping neuron index to rule index
        """
        self.neuron_rule_map = {self.neuron_keys.index(neuron):
                                [self.rule_keys.index(
                                    rule) for rule in self.neurons[neuron].rules]
                                for neuron in self.neurons}

    def get_active_rules(self, neuron, active):
        """
        Returns the indices of the active rules of a neuron
        """
        active = active.T[neuron]
        indices = [index for index, rule in enumerate(active) if rule]
        return indices

    def get_mapped_neuron(self, rule_idx: str):
        neuron_idx = next(
            key for key, val in self.neuron_rule_map.items() if rule_idx in val)
        return neuron_idx


if __name__ == '__main__':
    with open('tests/test3.json', 'r') as f:
        raw = json.load(f)
        data = SNPSystem(**raw)  # type: ignore

    SNP = MatrixSNPSystem(data)
    # print(SNP)
    # # state_graph = SNP.simulate(data['branch'],data['cur_depth'],data['sim_depth'])

    # for i in state_graph:
    # 	if i == 'nodes':
    # 		for j in state_graph[i]:
    # 			print('CONFIGURATION:\t',j['conf'])
    # 			print('SPIKING MATRIX:\t',j['spike'])
    # 			print('NEXT STATES:\t',j['next'])
    # 			print("====="*10)

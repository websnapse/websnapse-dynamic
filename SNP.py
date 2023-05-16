from models import SNPSystem
from utils import check_rule_validity, parse_rule
import numpy as np
import random
from typing import Dict, Union


class MatrixSNPSystem:
    def __init__(self, system: SNPSystem):
        self.neurons = system.nodes
        self.synapses = system.edges

        self.__set_neuron_order()
        self.__set_rule_order()

        self.config_vct = self.__init_config_vct()
        self.spike_train_vct = self.__init_spike_train_vct()
        self.adj_mx = self.__init_adj_mx()
        self.trans_mx = self.__init_trans_mx()
        self.delay_status_vct = self.__init_delay_status_vct()
        self.neuron_status_vct = self.__init_neuron_status_vct()
        self.delayed_indicator_vct = self.__init_delayed_indicator_vct()

        self.contents = np.empty((0, self.neuron_count), dtype=object)
        self.states = np.zeros([1, self.neuron_count], dtype=object)
        self.halted = False

        self.print_system()
        print("++")

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

    def simulate(self):
        self.__update_content()
        self.__add_input_spiketrain()
        self.__compute_spikeable_mx()
        self.__choose_decision_vct()
        self.__update_delay_status_vct()
        self.__compute_indicator_vct()
        self.__update_delayed_indicator_vct()
        self.__update_neuron_status_vct()
        self.__update_config_vct()
        self.__update_output_spiketrain()
        self.__update_neuron_states()
        self.__check_halt_conditions()

    def simulate_all(self):
        """
        Simulates the system for a single time step
        """

        iteration = 0
        while True:
            self.simulate()

            if self.halted:
                break

            iteration += 1

    def __check_halt_conditions(self):
        self.halted = (
            np.all(self.indicator_vct == 0)
            and np.all(self.neuron_status_vct == 1)
            and np.all(self.spike_train_vct[self.input_keys] == "")
        )

    def __update_content(self):
        self.content = self.config_vct.copy().astype(object)
        self.content[self.input_keys] = self.spike_train_vct[self.input_keys]
        self.content[self.output_keys] = self.spike_train_vct[self.output_keys]

        self.contents = np.append(self.contents, [self.content], axis=0).astype(object)

    def __add_input_spiketrain(self):
        for i in self.input_keys:
            spike = self.spike_train_vct[i][-1:]
            self.config_vct[i] = int(spike) if spike else 0
            self.spike_train_vct[i] = self.spike_train_vct[i][:-1]

    def __update_output_spiketrain(self):
        spike = np.char.add(
            self.spike_train_vct[self.output_keys].astype(str),
            self.config_vct[self.output_keys].astype(str),
        )
        self.spike_train_vct[self.output_keys] = spike

    def __update_neuron_states(self):
        self.state = np.where(self.neuron_status_vct == 0, -1, 0)

        for rule_idx in range(len(self.indicator_vct)):
            if self.indicator_vct[rule_idx] == 1:
                neuron_idx = self.__get_mapped_neuron(rule_idx)
                self.state[neuron_idx] = 1

        self.states = np.append(self.states, [self.state], axis=0).astype(object)

    def __update_config_vct(self):
        net_gain = np.dot(self.indicator_vct, self.trans_mx).astype(int)
        self.config_vct += np.multiply(self.neuron_status_vct, net_gain).astype(int)

    def __choose_decision_vct(self):
        possible_spiking_vct = len(self.spikeable_mx)
        decision_vct_idx = random.randint(1, possible_spiking_vct) - 1

        self.decision_vct = self.spikeable_mx[decision_vct_idx]

    def __update_delayed_indicator_vct(self):
        self.delayed_indicator_vct = np.logical_or(
            np.logical_and(
                self.delayed_indicator_vct, np.logical_not(self.indicator_vct)
            ),
            np.logical_and(self.decision_vct, self.delay_status_vct > 0),
        ).astype(int)

    def __compute_indicator_vct(self):
        """
        Computes the indicator vector by multiplying
        the rule status vector with the chosen
        spiking vector
        """

        rule_status = np.logical_not(self.delay_status_vct).astype(int)

        self.indicator_vct = np.multiply(
            self.decision_vct + self.delayed_indicator_vct, rule_status
        )

    def __update_neuron_status_vct(self):
        """
        Updates the neuron status vector by setting the value to 0
        if the corresponding rule is not activatable
        """
        self.neuron_status_vct = np.ones((self.neuron_count,)).astype(int)
        existing_delays = self.delay_status_vct

        for rule_idx in range(self.rule_count):
            neuron_idx = self.__get_mapped_neuron(rule_idx)
            self.neuron_status_vct[neuron_idx] = existing_delays[rule_idx] == 0

    def __update_delay_status_vct(self):
        """
        Updates the delay status vector by decrementing each value by 1
        """

        for rule_idx in range(len(self.delayed_indicator_vct)):
            if self.delayed_indicator_vct[rule_idx] == 1:
                neuron_idx = self.__get_mapped_neuron(rule_idx)
                delay = self.rule_delay_vct[rule_idx]
                for rule in self.neuron_rule_map[neuron_idx]:
                    self.delay_status_vct[rule] -= 1

        for rule_idx in range(self.rule_count):
            if self.decision_vct[rule_idx] == 1:
                neuron_idx = self.__get_mapped_neuron(rule_idx)
                delay = self.rule_delay_vct[rule_idx]
                for rule in self.neuron_rule_map[neuron_idx]:
                    self.delay_status_vct[rule] = delay

    def __compute_spikeable_mx(self):
        """
        Creates a matrix of all possible combinations of rules that can be activated
        """
        activatable_rules, activatable_count = self.__get_activatable_rules()
        comb_count = np.prod(activatable_count, where=activatable_count > 0)
        self.spikeable_mx = np.zeros((int(comb_count), self.rule_count))
        temp_comb_count = comb_count

        for neuron_idx, _ in enumerate(self.neurons):
            if activatable_count[neuron_idx] == 0:
                continue

            rules = self.__get_active_rules(neuron_idx, activatable_rules)
            amount = temp_comb_count / activatable_count[neuron_idx]
            repeats = comb_count / temp_comb_count
            index_i = 0

            for _ in range(int(repeats)):
                for index_j in rules:
                    counter = 0
                    while counter < amount:
                        self.spikeable_mx[index_i, index_j] = 1
                        counter = counter + 1
                        index_i = index_i + 1

            temp_comb_count /= activatable_count[neuron_idx]

    def __get_active_rules(self, neuron, active):
        """
        Returns the indices of the active rules of a neuron
        """
        active = active.T[neuron]
        indices = [index for index, rule in enumerate(active) if rule]
        return indices

    def __init_delayed_indicator_vct(self):
        """
        Initializes the delayed spikeable vector to all zeros
        with a length equal to the number of rules
        """
        delayed_indicator_vct = np.zeros((self.rule_count,))
        return delayed_indicator_vct

    def __init_neuron_status_vct(self):
        """
        Initializes the neuron status vector to all ones
        with a length equal to the number of neurons
        """
        neuron_status_vct = np.ones((self.neuron_count,))
        return neuron_status_vct

    def __init_trans_mx(self):
        """
        Creates the Spiking Transition Matrix
        by parsing the rules provided
        """
        trans_mx = np.zeros(
            (
                self.rule_count,
                self.neuron_count,
            )
        ).astype(int)
        self.rule_delay_vct = np.zeros((self.rule_count,)).astype(int)

        for rule in self.rules:
            rule_idx = self.rule_keys.index(rule)
            self.rule_delay_vct[rule_idx] = self.rules[rule]["delay"]

            neuron_idx = self.__get_mapped_neuron(rule_idx)
            trans_mx[rule_idx] = np.dot(
                self.adj_mx[neuron_idx], self.rules[rule]["production"]
            )
            trans_mx[rule_idx, neuron_idx] = self.rules[rule]["consumption"]

        return trans_mx

    def __init_delay_status_vct(self):
        """
        Initializes the delay status vector to all zeros
        with a length equal to the number of rules
        """
        delay_status_vct = np.zeros((self.rule_count,)).astype(int)
        return delay_status_vct

    def __init_spike_train_vct(self):
        """
        Initializes the spike train vector to all zeros
        with a length equal to the number of neurons
        """
        spike_train_vct = np.array(
            [n.spiketrain if n.nodeType == "input" else "" for n in self.neurons]
        ).astype(object)
        return spike_train_vct

    def __init_config_vct(self):
        """
        Creates a matrix of initial neuron configurations
        """
        # config_vct = np.array([n.content for n in self.neurons if n.type == 'regular' else n.spiketrain[:-1] if n.type == 'input' else 0])

        config_vct = np.array(
            [n.content if n.nodeType == "regular" else 0 for n in self.neurons]
        )
        return config_vct

    def __set_neuron_order(self):
        """
        Creates a list of neuron keys in the order of their appearance
        """
        self.neuron_keys = [n.id for n in self.neurons]
        self.regular_keys = [
            neuron_idx
            for neuron_idx, n in enumerate(self.neurons)
            if n.nodeType == "regular"
        ]
        self.input_keys = [
            neuron_idx
            for neuron_idx, n in enumerate(self.neurons)
            if n.nodeType == "input"
        ]
        self.output_keys = [
            neuron_idx
            for neuron_idx, n in enumerate(self.neurons)
            if n.nodeType == "output"
        ]
        self.neuron_count = len(self.neuron_keys)

    def __set_rule_order(self):
        """
        Creates a dictionary of rules in the order of their appearance
        """
        self.rule_count = 0
        self.rules: Dict[str, Dict[str, Union[str, int]]] = {}
        self.neuron_rule_map: Dict[int, list[int]] = {}
        for neuron_idx, n in enumerate(self.neurons):
            if n.nodeType == "regular":
                self.neuron_rule_map[neuron_idx] = []
                for rule in n.rules:
                    bound, consumption, production, delay = parse_rule(rule)
                    self.rules[f"r{self.rule_count}"] = {
                        "bound": bound,
                        "consumption": consumption,
                        "production": production,
                        "delay": delay,
                    }
                    self.neuron_rule_map[neuron_idx].append(self.rule_count)
                    self.rule_count += 1
            elif n.nodeType == "input":
                self.neuron_rule_map[neuron_idx] = []
                self.rules[f"r{self.rule_count}"] = {
                    "bound": ".+",
                    "consumption": -1,
                    "production": 1,
                    "delay": 0,
                }
                self.neuron_rule_map[neuron_idx].append(self.rule_count)
                self.rule_count += 1
            elif n.nodeType == "output":
                self.neuron_rule_map[neuron_idx] = []
                self.rules[f"r{self.rule_count}"] = {
                    "bound": ".+",
                    "consumption": -1,
                    "production": 0,
                    "delay": 0,
                }
                self.neuron_rule_map[neuron_idx].append(self.rule_count)
                self.rule_count += 1

        self.rule_keys = list(self.rules.keys())

    def __init_adj_mx(self):
        """
        Creates an adjacency matrix of the neurons of the system
        """
        adj_mx = np.zeros((self.neuron_count, self.neuron_count)).astype(int)

        for synapse in self.synapses:
            source = self.neuron_keys.index(synapse.source)
            target = self.neuron_keys.index(synapse.target)
            adj_mx[source, target] = synapse.label
        return adj_mx

    def __get_activatable_rules(self):
        """
        Returns a matrix of rules that are activatable by the current configuration
        """
        activatable_rules_mx = np.zeros(
            (
                self.rule_count,
                self.neuron_count,
            )
        )
        active_rules_in_neuron = np.zeros((self.neuron_count,))
        for rule_idx, rule in enumerate(self.rules):
            neuron_idx = self.__get_mapped_neuron(rule_idx)

            bound: str = str(self.rules[rule]["bound"])
            spikes: int = self.config_vct[neuron_idx]
            neuron_status: int = self.neuron_status_vct[neuron_idx]
            rule_validity = np.multiply(
                neuron_status, check_rule_validity(bound, spikes)
            ).astype(int)
            activatable_rules_mx[rule_idx, neuron_idx] = rule_validity
            active_rules_in_neuron[neuron_idx] += rule_validity
        return activatable_rules_mx, active_rules_in_neuron

    def __get_mapped_neuron(self, rule_idx: int):
        neuron_idx = next(
            key for key, val in self.neuron_rule_map.items() if rule_idx in val
        )
        return neuron_idx

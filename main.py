from fastapi import FastAPI
from models import SNPSystem
from utils import check_rule_validity, parse_rule
import numpy as np
import regex as re
from fastapi.middleware.cors import CORSMiddleware
import json
import random

app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:5173",
]
app.add_middleware(  # type: ignore
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MatrixSNPSystem:
    def __init__(self, system: SNPSystem):
        self.neurons = system.nodes
        self.synapses = system.edges

        self.set_neuron_order()
        self.set_rule_order()

        self.config_vct = self.init_config_mx()
        self.adj_mx = self.init_adj_mx()
        self.trans_mx = self.init_trans_mx()
        self.delay_status_vct = self.init_delay_status_vct()
        self.neuron_status_vct = self.init_neuron_status_vct()
        self.delayed_indicator_vct = self.init_delayed_indicator_vct()

        self.configurations = np.array([self.config_vct], dtype=object)
        self.neuron_states = np.zeros([1, self.neuron_count], dtype=object)

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
        """
        Simulates the system for a single time step
        """
        # print(f"= {iteration} ===")
        self.set_spikeable_mx()
        # print("SM:", end=" ")
        # print(self.spikeable_mx)

        self.choose_spiking_vct()
        # print("SV:", end=" ")
        # print(self.decision_vct)

        self.update_delay_status_vct()
        # print("DS:", end=" ")
        # print(self.delay_status_vct)

        self.compute_indicator_vct()
        # print("IV:", end=" ")
        # print(self.indicator_vct)

        self.update_delayed_indicator_vct()
        # print("DIV:", end=" ")
        # print(self.delayed_indicator_vct)

        self.update_neuron_status_vct()
        # print("NS:", end=" ")
        # print(self.neuron_status_vct)

        net_gain = np.dot(self.indicator_vct, self.trans_mx).astype(int)
        # print("Net Gain:")
        # print(net_gain)
        # print("CV:", end=" ")
        # print(self.config_vct, end=" -> ")
        self.prev_config_vct = self.config_vct.copy()
        self.config_vct += np.multiply(self.neuron_status_vct, net_gain).astype(int)
        # print(self.config_vct, self.config_vct[-1])

        # get the sum of the configuration vector with the output neuron from the output_keys removed

        regular_cfg = self.config_vct.copy()
        regular_cfg = np.delete(regular_cfg, self.output_keys)
        regular_cfg = np.delete(regular_cfg, self.input_keys)

        self.configurations = np.append(
            self.configurations, [self.config_vct], axis=0
        ).astype(object)

        # for updating neuron states, if the neuron status is 0, then the state is -1
        # else, get the neuron mapped to the rules in the indicator vector and set them to 1
        # if their value is 1 else, 0

        neuron_state = np.where(self.neuron_status_vct == 0, -1, 0)
        for rule_idx in range(len(self.indicator_vct)):
            if self.indicator_vct[rule_idx] == 1:
                neuron_idx = self.get_mapped_neuron(rule_idx)
                neuron_state[neuron_idx] = 1

        self.neuron_states = np.append(
            self.neuron_states, [neuron_state], axis=0
        ).astype(object)

        for i in range(self.config_vct.shape[0]):
            if i in self.output_keys:
                self.configurations[iteration][
                    i
                ] = f"{self.configurations[iteration-1][i]}{int(self.prev_config_vct[i] != self.config_vct[i])}"

        if regular_cfg.sum() == 0:
            iteration += 1

            neuron_state = np.zeros((self.neuron_count,)).astype(int)

            self.neuron_states = np.append(
                self.neuron_states, [neuron_state], axis=0
            ).astype(object)

            self.configurations = np.append(
                self.configurations, [self.config_vct], axis=0
            ).astype(object)

            for i in range(self.config_vct.shape[0]):
                if i in self.output_keys:
                    self.configurations[iteration][
                        i
                    ] = f"{self.configurations[iteration-1][i]}0"
            return

        self.simulate(iteration + 1)
        #     print(regular_cfg.sum())
        # print(depth)

    def choose_spiking_vct(self):
        possible_spiking_vct = len(self.spikeable_mx)
        decision_vct_idx = random.randint(1, possible_spiking_vct) - 1
        # decision_vct_idx = int(input("Choose a spiking vector: "))

        self.decision_vct = self.spikeable_mx[decision_vct_idx]

    def update_delayed_indicator_vct(self):
        self.delayed_indicator_vct = np.logical_or(
            np.logical_and(
                self.delayed_indicator_vct, np.logical_not(self.indicator_vct)
            ),
            np.logical_and(self.decision_vct, self.delay_status_vct > 0),
        ).astype(int)

    def compute_indicator_vct(self):
        """
        Computes the indicator vector by multiplying
        the rule status vector with the chosen
        spiking vector
        """

        rule_status = np.logical_not(self.delay_status_vct).astype(int)

        self.indicator_vct = np.multiply(
            self.decision_vct + self.delayed_indicator_vct, rule_status
        )

    def update_neuron_status_vct(self):
        """
        Updates the neuron status vector by setting the value to 0
        if the corresponding rule is not activatable
        """
        self.neuron_status_vct = np.ones((self.neuron_count,)).astype(int)
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

            rules = self.get_active_rules(neuron_idx, activatable_rules)
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

    def get_active_rules(self, neuron, active):
        """
        Returns the indices of the active rules of a neuron
        """
        active = active.T[neuron]
        indices = [index for index, rule in enumerate(active) if rule]
        return indices

    def init_delayed_indicator_vct(self):
        """
        Initializes the delayed spikeable vector to all zeros
        with a length equal to the number of rules
        """
        delayed_indicator_vct = np.zeros((self.rule_count,))
        return delayed_indicator_vct

    def init_delayed_spikeable(self):
        """
        Initializes the delayed spikeable vector to all zeros
        with a length equal to the number of rules
        """
        delayed_spikeable = np.zeros((self.rule_count,))
        return delayed_spikeable

    def init_neuron_status_vct(self):
        """
        Initializes the neuron status vector to all ones
        with a length equal to the number of neurons
        """
        neuron_status_vct = np.ones((self.neuron_count,))
        return neuron_status_vct

    def init_trans_mx(self):
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

            neuron_idx = self.get_mapped_neuron(rule_idx)
            trans_mx[rule_idx] = np.dot(
                self.adj_mx[neuron_idx], self.rules[rule]["production"]
            )
            trans_mx[rule_idx, neuron_idx] = self.rules[rule]["consumption"]

        return trans_mx

    def init_delay_status_vct(self):
        """
        Initializes the delay status vector to all zeros
        with a length equal to the number of rules
        """
        delay_status_vct = np.zeros((self.rule_count,)).astype(int)
        return delay_status_vct

    def init_config_mx(self):
        """
        Creates a matrix of initial neuron configurations
        """
        config_vct = np.array([n.content for n in self.neurons])
        return config_vct

    def set_neuron_order(self):
        """
        Creates a list of neuron keys in the order of their appearance
        """
        self.neuron_keys = [n.id for n in self.neurons]
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

    def set_rule_order(self):
        """
        Creates a dictionary of rules in the order of their appearance
        """
        self.rule_count = 0
        self.rules = {}
        self.neuron_rule_map = {}
        for neuron_idx, n in enumerate(self.neurons):
            if n.rules:
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
        self.rule_keys = list(self.rules.keys())

    def init_adj_mx(self):
        """
        Creates an adjacency matrix of the neurons of the system
        """
        adj_mx = np.zeros((self.neuron_count, self.neuron_count)).astype(int)

        for synapse in self.synapses:
            source = self.neuron_keys.index(synapse.source)
            target = self.neuron_keys.index(synapse.target)
            adj_mx[source, target] = synapse.label
        return adj_mx

    def get_activatable_rules(self):
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
            neuron_idx = self.get_mapped_neuron(rule_idx)

            bound = self.rules[rule]["bound"]
            spikes = self.config_vct[neuron_idx]
            neuron_status = self.neuron_status_vct[neuron_idx]
            rule_validity = np.multiply(
                neuron_status, check_rule_validity(bound, spikes)
            ).astype(int)
            activatable_rules_mx[rule_idx, neuron_idx] = rule_validity
            active_rules_in_neuron[neuron_idx] += rule_validity
        return activatable_rules_mx, active_rules_in_neuron

    def get_mapped_neuron(self, rule_idx: int):
        neuron_idx = next(
            key for key, val in self.neuron_rule_map.items() if rule_idx in val
        )
        return neuron_idx


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/simulate")
async def simulate(system: SNPSystem):
    matrixSNP = MatrixSNPSystem(system)
    print(matrixSNP.neuron_states)
    print(matrixSNP.configurations)
    return {
        "states": matrixSNP.neuron_states.tolist(),
        "configurations": matrixSNP.configurations.tolist(),
        "keys": matrixSNP.neuron_keys,
    }


# if __name__ == "__main__":
#     with open("tests/test3.json", "r") as f:
#         raw = json.load(f)
#         data = SNPSystem(**raw)

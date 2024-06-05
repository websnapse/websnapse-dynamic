class SATSystem:
    def __init__(self, exp):
        self.valid = None
        self.clauses = None
        self.unique = None
        self.variables = None
        self.neurons = []
        self.synapses = []
        self.rule_dict = []
        self.ts = []
        self.fs = []

        self.system = {}

        self.process_expression(exp)
        self.get_sat_config()
        
    def process_expression(self, exp):
        try:
            self.valid = True
            self.clauses = [i.strip() for i in exp.split("and")]

            self.variables = []
            for clause in self.clauses:
                new_vars = clause.split(" or ")
                self.variables.append(new_vars)

            unique_vars_set = set()
            for clause in self.variables:
                for var in clause:
                    unique_vars_set.add(var if var[0] != "-" else var[1:])

            self.unique = sorted(list(unique_vars_set))
        except:
            self.valid = False

    def gen_neuron(self, _id, _type, _content, _rules):
        if _type == "regular":
            return {
                "id": str(_id),
                "type": str(_type),
                "content": str(_content),
                "rules": _rules,
                "position": {"x": 0, "y": 0},
            }

        return {
            "id": str(_id),
            "type": str(_type),
            "content": str(_content),
            "position": {"x": 0, "y": 0},
        }


    def gen_synapse(self, f, t):
        return {"from": f, "to": t, "weight": 1}


    def gen_rule(self, n, rule, div):
        if div:
            return rule
        else:
            return f"\\left[{rule} \\right]_{{" + n + "}"


    def generate_tf(self, n, i=0, cur=""):
        if i == n:
            if cur[0] == "t":
                self.ts[n].append(cur)
            else:
                self.fs[n].append(cur)

            return

        self.generate_tf(n, i + 1, f"t_{i + 1}{cur}")
        self.generate_tf(n, i + 1, f"f_{i + 1}{cur}")


    def generate_tfs(self, n):
        for idx in range(n + 2):
            self.ts.append([])
            self.fs.append([])

            if idx > 0:
                self.generate_tf(idx)


    def generate_neurons(self, n, m, isp):
        self.neurons.append(self.gen_neuron("ISP", "input", isp, []))
        self.neurons.append(self.gen_neuron("OSP", "output", "", []))
        self.neurons.append(self.gen_neuron("in", "regular", "0", ["a^2\\to a^2", "a\\to a"]))
        self.neurons.append(
            self.gen_neuron("d_0", "regular", "4", ["a^4/a^3\\to a^3;8", f"a\\to a;{n * m - 1}"])
        )
        self.neurons.append(
            self.gen_neuron(
                "0",
                "regular",
                "0",
                [
                    "a^2\\to\\lambda",
                    "\\left[a\\right]\\to\\left[\\right]_{t_1}\\left|\\right|\\left[\\right]_{f_1}",
                ],
            )
        )

        self.neurons.append(self.gen_neuron("1", "regular", "0", ["a\\to a", "a^2\\to a^2"]))
        self.neurons.append(
            self.gen_neuron(
                "2",
                "regular",
                "2",
                ["a\\to a", "a^2\\to a^2", "a^3\\to\\lambda", "a^4\\to a"],
            )
        )
        self.neurons.append(
            self.gen_neuron(
                "3",
                "regular",
                "7",
                [
                    f"a^7/a^2\\to a^2;{2 * n - 3}",
                    f"a^5/a^2\\to a^2;{2 * n - 1}",
                    f"a^3\\to a^3;{n * m + 2}",
                ],
            )
        )
        self.neurons.append(
            self.gen_neuron(
                "4", "regular", "0", ["a^2\\to\\lambda", "a^3\\to a;1", "a^6\\to a^2;1"]
            )
        )

        if n <= 2:
            self.neurons.append(
                self.gen_neuron(
                    f"b_1",
                    "regular",
                    "2",
                    [
                        "\\left[a^2\\right]\\to\\left[\\right]_{d_1}\\left|\\right|\\left[\\right]_{d_2}"
                    ],
                )
            )
            self.neurons.append(
                self.gen_neuron(
                    f"e_1",
                    "regular",
                    "2",
                    [
                        "\\left[a^2\\right]\\to\\left[\\right]_{Cx_1}\\left|\\right|\\left[\\right]_{Cx_2}"
                    ],
                )
            )
            self.neurons.append(
                self.gen_neuron(
                    f"g_1",
                    "regular",
                    "2",
                    [
                        "\\left[a^2\\right]\\to\\left[\\right]_{h_1}\\left|\\right|\\left[\\right]_{h_2}"
                    ],
                )
            )

        else:
            self.neurons.append(
                self.gen_neuron(
                    f"b_1",
                    "regular",
                    "0",
                    [
                        "\\left[a^2\\right]\\to\\left[\\right]_{d_1}\\left|\\right|\\left[\\right]_{b_2}"
                    ],
                )
            )
            self.neurons.append(
                self.gen_neuron(
                    f"e_1",
                    "regular",
                    "0",
                    [
                        "\\left[a^2\\right]\\to\\left[\\right]_{Cx_1}\\left|\\right|\\left[\\right]_{e_2}"
                    ],
                )
            )
            self.neurons.append(
                self.gen_neuron(
                    f"g_1",
                    "regular",
                    "0",
                    [
                        "\\left[a^2\\right]\\to\\left[\\right]_{h_1}\\left|\\right|\\left[\\right]_{g_2}"
                    ],
                )
            )

        self.neurons.append(self.gen_neuron("out", "regular", "0", ["\\left(a^2\\right)^+/a\\to a"]))


    def generate_synapses(self, n):
        self.synapses.append(self.gen_synapse("ISP", "in"))
        self.synapses.append(self.gen_synapse("out", "OSP"))

        self.synapses.append(self.gen_synapse("1", "b_1"))
        self.synapses.append(self.gen_synapse("1", "e_1"))
        self.synapses.append(self.gen_synapse("1", "g_1"))

        self.synapses.append(self.gen_synapse("1", "2"))
        self.synapses.append(self.gen_synapse("3", "4"))
        self.synapses.append(self.gen_synapse("4", "0"))
        self.synapses.append(self.gen_synapse("0", "out"))

        self.synapses.append(self.gen_synapse("1", "0"))
        self.synapses.append(self.gen_synapse("2", "1"))
        self.synapses.append(self.gen_synapse("3", "2"))

        self.synapses.append(self.gen_synapse(f"d_{n}", "d_1"))
        self.synapses.append(self.gen_synapse(f"d_{n}", "4"))

        for i in range(n):
            self.synapses.append(self.gen_synapse(f"d_{i}", f"d_{i + 1}"))

        for i in range(1, n + 1):
            self.synapses.append(self.gen_synapse("in", f"Cx_{i}"))

        for i in range(1, n + 1):
            self.synapses.append(self.gen_synapse(f"d_{i}", f"Cx_{i}"))

        for i in range(1, n + 1):
            self.synapses.append(self.gen_synapse(f"Cx_{i}", f"h_{i}"))

        for i in range(1, n + 1):
            for t_i in self.ts[i]:
                self.synapses.append(self.gen_synapse(f"Cx_{i}1", t_i))

        for i in range(1, n + 1):
            for f_i in self.fs[i]:
                self.synapses.append(self.gen_synapse(f"Cx_{i}0", f_i))


    def generate_rule_dict(self, n, m):
        # A: Generation
        for i in range(1, n - 1):
            self.rule_dict.append(
                self.gen_rule(
                    f"b_{i}",
                    f"\\left[a^2\\right]_{{b_{i}}}\\to\\left[\\right]_{{d_{i}}}\\left|\\right|\\left[\\right]_{{b_{i + 1}}}",
                    1,
                )
            )
        self.rule_dict.append(
            self.gen_rule(
                f"b_{n - 1}",
                f"\\left[a^2\\right]_{{b_{n - 1}}}\\to\\left[\\right]_{{d_{n - 1}}}\\left|\\right|\\left[\\right]_{{d_{n}}}",
                1,
            )
        )

        for i in range(1, n - 1):
            self.rule_dict.append(
                self.gen_rule(
                    f"e_{i}",
                    f"\\left[a^2\\right]_{{e_{i}}}\\to\\left[\\right]_{{Cx_{i}}}\\left|\\right|\\left[\\right]_{{e_{i + 1}}}",
                    1,
                )
            )
        self.rule_dict.append(
            self.gen_rule(
                f"e_{n - 1}",
                f"\\left[a^2\\right]_{{e_{n - 1}}}\\to\\left[\\right]_{{Cx_{n - 1}}}\\left|\\right|\\left[\\right]_{{Cx_{n}}}",
                1,
            )
        )

        for i in range(1, n - 1):
            self.rule_dict.append(
                self.gen_rule(
                    f"g_{i}",
                    f"\\left[a^2\\right]_{{g_{i}}}\\to\\left[\\right]_{{h_{i}}}\\left|\\right|\\left[\\right]_{{g_{i + 1}}}",
                    1,
                )
            )
        self.rule_dict.append(
            self.gen_rule(
                f"g_{n - 1}",
                f"\\left[a^2\\right]_{{g_{n - 1}}}\\to\\left[\\right]_{{h_{n - 1}}}\\left|\\right|\\left[\\right]_{{h_{n}}}",
                1,
            )
        )

        for i in range(1, n + 1):
            self.rule_dict.append(
                self.gen_rule(
                    f"h_{i}",
                    f"\\left[a^2\\right]_{{h_{i}}}\\to\\left[\\right]_{{Cx_{i}1}}\\left|\\right|\\left[\\right]_{{Cx_{i}0}}",
                    1,
                )
            )

        for i in range(1, n + 1):
            self.rule_dict.append(self.gen_rule(f"d_{i}", "a\\to\\lambda", 0))
            self.rule_dict.append(self.gen_rule(f"d_{i}", "a^2\\to\\lambda", 0))

        for i in range(1, n + 1):
            self.rule_dict.append(self.gen_rule(f"Cx_{i}", "a\\to\\lambda", 0))
            self.rule_dict.append(self.gen_rule(f"Cx_{i}", "a^2\\to\\lambda", 0))

        for i in range(1, n + 1):
            self.rule_dict.append(self.gen_rule(f"Cx_{i}1", "a\\to\\lambda", 0))
            self.rule_dict.append(self.gen_rule(f"Cx_{i}1", "a^2\\to\\lambda", 0))

        for i in range(1, n + 1):
            self.rule_dict.append(self.gen_rule(f"Cx_{i}0", "a\\to\\lambda", 0))
            self.rule_dict.append(self.gen_rule(f"Cx_{i}0", "a^2\\to\\lambda", 0))

        self.rule_dict.append(self.gen_rule(f"{1}", "a\\to a", 0))
        self.rule_dict.append(self.gen_rule(f"{2}", "a\\to a", 0))

        self.rule_dict.append(self.gen_rule(f"{1}", "a^2\\to a^2", 0))
        self.rule_dict.append(self.gen_rule(f"{2}", "a^2\\to a^2", 0))

        self.rule_dict.append(self.gen_rule(f"{2}", "a^3\\to\\lambda", 0))
        self.rule_dict.append(self.gen_rule(f"{2}", "a^4\\to a", 0))

        self.rule_dict.append(self.gen_rule(f"{3}", f"a^7/a^2\\to a^2;{2 * n - 3}", 0))
        self.rule_dict.append(self.gen_rule(f"{3}", f"a^5/a^2\\to a^2;{2 * n - 1}", 0))

        self.rule_dict.append(self.gen_rule(f"{4}", "a^2\\to\\lambda", 0))
        self.rule_dict.append(self.gen_rule(f"{0}", "a^2\\to\\lambda", 0))

        self.rule_dict.append(
            self.gen_rule(
                f"{0}",
                "\\left[a\\right]_{0}\\to\\left[\\right]_{t_1}\\left|\\right|\\left[\\right]_{f_1}",
                1,
            )
        )

        for i in range(1, n):
            for t_i in self.ts[i]:
                self.rule_dict.append(
                    self.gen_rule(
                        t_i,
                        f"\\left[a\\right]_{{{t_i}}}\\to\\left[\\right]_{{t_{i + 1}{t_i}}}\\left|\\right|\\left[\\right]_{{f_{i + 1}{t_i}}}",
                        1,
                    )
                )

        for i in range(1, n):
            for f_i in self.fs[i]:
                self.rule_dict.append(
                    self.gen_rule(
                        f_i,
                        f"\\left[a\\right]_{{{f_i}}}\\to\\left[\\right]_{{t_{i + 1}{f_i}}}\\left|\\right|\\left[\\right]_{{f_{i + 1}{f_i}}}",
                        1,
                    )
                )

        # B: Input
        self.rule_dict.append(self.gen_rule("in", "a\\to a", 0))
        self.rule_dict.append(self.gen_rule("in", "a^2\\to a^2", 0))

        self.rule_dict.append(self.gen_rule("d_0", f"a^4/a^3\\to a^3;{4 * n}", 0))
        self.rule_dict.append(self.gen_rule("d_0", f"a\\to a;{n * m - 1}", 0))

        for i in range(1, n + 1):
            self.rule_dict.append(self.gen_rule(f"d_{i}", "a^3\\to a^3", 0))

        self.rule_dict.append(self.gen_rule(f"d_1", "a^4\\to\\lambda", 0))

        for i in range(1, n + 1):
            self.rule_dict.append(self.gen_rule(f"Cx_{i}", "a^3\\to\\lambda", 0))
            self.rule_dict.append(self.gen_rule(f"Cx_{i}", f"a^4\\to a^4;{n - i}", 0))
            self.rule_dict.append(self.gen_rule(f"Cx_{i}", f"a^5\\to a^5;{n - i}", 0))

        # C: SAT Checking
        for i in range(1, n + 1):
            self.rule_dict.append(self.gen_rule(f"Cx_{i}1", "a^4\\to a^3", 0))
            self.rule_dict.append(self.gen_rule(f"Cx_{i}1", "a^5\\to\\lambda", 0))

        for i in range(1, n + 1):
            self.rule_dict.append(self.gen_rule(f"Cx_{i}0", "a^4\\to\\lambda", 0))
            self.rule_dict.append(self.gen_rule(f"Cx_{i}0", "a^5\\to a^3", 0))

        self.rule_dict.append(self.gen_rule(f"{3}", f"a^3\\to a^3;{n * m + 2}", 0))

        self.rule_dict.append(self.gen_rule(f"{4}", f"a^3\\to a;1", 0))
        self.rule_dict.append(self.gen_rule(f"{4}", f"a^6\\to a^2;1", 0))

        for t_n in self.ts[n]:
            self.rule_dict.append(
                self.gen_rule(
                    t_n,
                    f"\\left[a\\right]_{{{t_n}}}\\to\\left[\\right]_{{t_{n + 1}{t_n}}}\\left|\\right|\\left[\\right]_{{f_{n + 1}{t_n}}}",
                    1,
                )
            )

        for i in range(1, n + 1):
            for t_i in self.ts[i]:
                for k in range(1, n + 1):
                    self.rule_dict.append(self.gen_rule(t_i, f"a^{{{3 * k + 1}}}\\to\\lambda", 0))
                    self.rule_dict.append(self.gen_rule(t_i, f"a^{{{3 * k + 2}}}/a^2\\to a^2", 0))

        for f_n in self.fs[n]:
            self.rule_dict.append(
                self.gen_rule(
                    f_n,
                    f"\\left[a\\right]_{{{f_n}}}\\to\\left[\\right]_{{t_{n + 1}{f_n}}}\\left|\\right|\\left[\\right]_{{f_{n + 1}{f_n}}}",
                    1,
                )
            )

        for i in range(1, n + 1):
            for f_i in self.fs[i]:
                for k in range(1, n + 1):
                    self.rule_dict.append(self.gen_rule(f_i, f"a^{{{3 * k + 1}}}\\to\\lambda", 0))
                    self.rule_dict.append(self.gen_rule(f_i, f"a^{{{3 * k + 2}}}/a^2\\to a^2", 0))

        for t_i in self.ts[n + 1]:
            for k in range(n + 1):
                if k == 0:
                    self.rule_dict.append(self.gen_rule(t_i, "a\\to\\lambda", 0))
                else:
                    self.rule_dict.append(self.gen_rule(t_i, f"a^{{{3 * k + 1}}}\\to\\lambda", 0))
                self.rule_dict.append(self.gen_rule(t_i, f"a^{{{3 * k + 2}}}\\to\\lambda", 0))

        for f_i in self.fs[n + 1]:
            for k in range(n + 1):
                if k == 0:
                    self.rule_dict.append(self.gen_rule(f_i, "a\\to\\lambda", 0))
                else:
                    self.rule_dict.append(self.gen_rule(f_i, f"a^{{{3 * k + 1}}}\\to\\lambda", 0))
                self.rule_dict.append(self.gen_rule(f_i, f"a^{{{3 * k + 2}}}\\to\\lambda", 0))

        # D: Output
        self.rule_dict.append(self.gen_rule("out", "\\left(a^2\\right)^+/a\\to a", 0))


    def get_sat_config(self):

        if self.valid:
            unique_idx = {}
            for idx, i in enumerate(self.unique):
                unique_idx[i] = idx

            clause_values = [[0 for i in range(len(self.unique))] for j in range(len(self.variables))]

            for idx, clause in enumerate(self.variables):
                for var in clause:
                    if var[0] == "-":
                        clause_values[idx][unique_idx[var[1:]]] = 2
                    else:
                        clause_values[idx][unique_idx[var]] = 1

            isp = "0000" * len(self.unique) + "".join(
                str(i) for clause in clause_values for i in clause
            )
            n = len(self.unique)
            m = len(self.variables)

            self.generate_tfs(n)
            self.generate_neurons(n, m, isp)
            self.generate_synapses(n)
            self.generate_rule_dict(n, m)

            self.system = {"neurons": self.neurons, "synapses": self.synapses, "rule_dict": self.rule_dict}
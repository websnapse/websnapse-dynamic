import { watch, reactive } from "vue";
import graph from "./graph";
import sample from "../data.json";
import { exportSystem } from "@/graph/utils/parse-system";
import { useStorage } from "@vueuse/core";

const system = reactive({
  mode: "pseudorandom",
  states: [],
  configuration: [],
  speed: 1.5,
  tick: 0,
  history: [],
  labels: [],
  backup: useStorage("system", sample),
  reset: null,
  ws: null,

  data() {
    return graph.value ? exportSystem(graph.value) : this.backup;
  },
  backupSystem() {
    this.backup = system.reset ? system.reset : exportSystem(graph.value);
  },
});

export default system;

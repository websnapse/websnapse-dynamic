<template>
  <TransitionRoot appear :show="dialog.createNeuron" as="template">
    <CreateNeuronDialog
      :isOpen="dialog.createNeuron"
      :closeModal="() => (dialog.createNeuron = false)"
    />
  </TransitionRoot>
  <TransitionRoot appear :show="dialog.createRule" as="template">
    <CreateRuleDialog
      :isOpen="dialog.createRule"
      :closeModal="() => (dialog.createRule = false)"
    />
  </TransitionRoot>
  <TransitionRoot appear :show="dialog.choiceHistory" as="template">
    <ChoiceHistoryDialog
      :isOpen="dialog.choiceHistory"
      :closeModal="() => (dialog.choiceHistory = false)"
    />
  </TransitionRoot>
  <TransitionRoot appear :show="dialog.rulebook" as="template">
    <RulebookDialog
      :isOpen="dialog.rulebook"
      :closeModal="() => (dialog.rulebook = false)"
    />
  </TransitionRoot>

  <TransitionRoot appear :show="dialog.createEdge" as="template">
    <CreateEdgeDialog
      :isOpen="dialog.createEdge"
      :closeModal="() => (dialog.createEdge = false)"
    />
  </TransitionRoot>
  <TransitionRoot appear :show="dialog.editNeuron" as="template">
    <EditNeuronDialog
      :isOpen="dialog.editNeuron"
      :closeModal="() => (dialog.editNeuron = false)"
      :details="dialog.details"
    />
  </TransitionRoot>
  <TransitionRoot appear :show="dialog.editSynapse" as="template">
    <EditSynapseDialog
      :isOpen="dialog.editSynapse"
      :closeModal="() => (dialog.editSynapse = false)"
      :details="dialog.details"
    />
  </TransitionRoot>
  <TransitionRoot appear :show="dialog.chooseRule" as="template">
    <ChooseRuleDialog
      :isOpen="dialog.chooseRule"
      :closeModal="() => (dialog.chooseRule = false)"
      :details="dialog.details"
      :socket="system.ws"
    />
  </TransitionRoot>
</template>

<script setup>
import { TransitionRoot } from "@headlessui/vue";
import CreateNeuronDialog from "@/components/CreateNeuronDialog.vue";
import EditNeuronDialog from "@/components/EditNeuronDialog.vue";
import EditSynapseDialog from "@/components/EditSynapseDialog.vue";
import ChooseRuleDialog from "@/components/ChooseRuleDialog.vue";
import ChoiceHistoryDialog from "@/components/ChoiceHistoryDialog.vue";
import RulebookDialog from "@/components/RulebookDialog.vue";
import CreateRuleDialog from "@/components/CreateRuleDialog.vue";
import dialog from "@/stores/dialog";
import system from "@/stores/system";
import CreateEdgeDialog from "./CreateEdgeDialog.vue";
</script>

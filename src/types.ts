export type SNNModelType = "Pure SNN" | "Hybrid SNN" | "ANN-to-SNN" | "Bio-inspired Graph" | "NLU SNN";

export interface SNNModel {
  id: string;
  name: string;
  author: string;
  parameters: string;
  type: SNNModelType;
  github: string;
  paper: string;
  huggingface: string;
  description: string;
  characteristics: string[];
  pros: string[];
  cons: string[];
  bioPlausibility: number; // 1 to 10
  energyEfficiency: number; // 1 to 10 (e.g. 10x, 100x)
  defaultThreshold: number; // V_th
  defaultDecay: number;     // tau decay
  defaultLeak: number;      // leakage
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: string;
  metrics?: SNNInferenceMetrics;
}

export interface SNNInferenceMetrics {
  spikeCount: number;
  synapticOps: number; // in SOPs (Synaptic Operations)
  flopsEquivalent: number; // in FLOPs
  energyJoulesANN: number;
  energyJoulesSNN: number;
  energySavedPercent: number;
  averageFiringRate: number; // Hz / neuron
  latencyMs: number;
}

export interface NeuronState {
  id: number;
  x: number;
  y: number;
  membranePotential: number; // 0 to V_th
  threshold: number;
  isFiring: boolean;
  type: "input" | "hidden" | "output";
}

export interface SynapseState {
  from: number;
  to: number;
  weight: number;
  lastSpikeTime: number; // relative
  isStimulated: boolean;
}

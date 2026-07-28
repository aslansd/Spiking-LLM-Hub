export type SNNModelType =
  | "Pure SNN"
  | "Hybrid SNN"
  | "ANN-to-SNN"
  | "Bio-inspired Graph"
  | "NLU SNN";

/**
 * What a visitor can actually obtain today. The previous schema implied every
 * entry was a downloadable chat model, which is not true for any of the
 * method papers or architecture releases.
 */
export type ReleaseStatus =
  | "weights-released" // downloadable checkpoints exist
  | "code-only" // implementation published, no checkpoints
  | "method"; // a technique applied to someone else's base model

export interface SNNModel {
  id: string;
  name: string;

  /** Author list as printed on the paper. Affiliation only where verified. */
  author: string;
  affiliation: string | null;

  /** Year of first public release of the paper or code. */
  year: number;

  /** Human-readable parameter description, e.g. "45M and 216M". */
  parameters: string;

  type: SNNModelType;
  status: ReleaseStatus;

  /** SPDX-ish identifier, or null when the project does not state one. */
  license: string | null;

  /**
   * Null means "no such resource exists", not "we didn't look".
   * The UI hides the link entirely rather than rendering a dead one.
   */
  github: string | null;
  paper: string | null;
  huggingface: string | null;
  homepage: string | null;

  description: string;
  characteristics: string[];
  pros: string[];
  cons: string[];

  /**
   * Editorial 1-10 ratings, not measurements. Rendered as "n/10" and
   * captioned as a subjective comparison aid.
   */
  bioPlausibility: number;
  energyEfficiency: number;

  defaultThreshold: number;
  defaultDecay: number;
  defaultLeak: number;
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
  synapticOps: number;
  flopsEquivalent: number;
  energyJoulesANN: number;
  energyJoulesSNN: number;
  energySavedPercent: number;
  averageFiringRate: number;
  latencyMs: number;
  /** Always true: these figures come from a formula, not from hardware. */
  simulated?: boolean;
}

export interface NeuronState {
  id: number;
  x: number;
  y: number;
  membranePotential: number;
  threshold: number;
  isFiring: boolean;
  type: "input" | "hidden" | "output";
}

export interface SynapseState {
  from: number;
  to: number;
  weight: number;
  lastSpikeTime: number;
  isStimulated: boolean;
}

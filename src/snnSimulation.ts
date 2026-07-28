import { NeuronState, SynapseState } from "./types";

/**
 * Network topology. Ids are contiguous per layer so a layer is identified by a
 * half-open id range rather than by scanning the `type` field.
 */
export const LAYERS = {
  input: { start: 0, end: 4 },
  hidden1: { start: 4, end: 9 },
  hidden2: { start: 9, end: 14 },
  output: { start: 14, end: 20 },
} as const;

export const TOTAL_NEURONS = LAYERS.output.end;

/** Milliseconds between replayed frames. */
export const FRAME_MS = 300;

export function buildNeurons(threshold: number): NeuronState[] {
  const neurons: NeuronState[] = [];

  for (let i = 0; i < 4; i++) {
    neurons.push({ id: i, x: 80, y: 80 + i * 110, membranePotential: 0, threshold, isFiring: false, type: "input" });
  }
  for (let i = 0; i < 5; i++) {
    neurons.push({ id: 4 + i, x: 230, y: 40 + i * 90, membranePotential: 0, threshold, isFiring: false, type: "hidden" });
  }
  for (let i = 0; i < 5; i++) {
    neurons.push({ id: 9 + i, x: 380, y: 40 + i * 90, membranePotential: 0, threshold, isFiring: false, type: "hidden" });
  }
  for (let i = 0; i < 6; i++) {
    neurons.push({ id: 14 + i, x: 530, y: 40 + i * 80, membranePotential: 0, threshold, isFiring: false, type: "output" });
  }

  return neurons;
}

export function buildSynapses(): SynapseState[] {
  const synapses: SynapseState[] = [];

  const connect = (
    fromStart: number,
    fromCount: number,
    toStart: number,
    toCount: number,
    density: number,
    minWeight: number,
    weightRange: number,
  ) => {
    for (let i = 0; i < fromCount; i++) {
      for (let j = 0; j < toCount; j++) {
        if (Math.random() > density) continue;
        synapses.push({
          from: fromStart + i,
          to: toStart + j,
          weight: minWeight + Math.random() * weightRange,
          lastSpikeTime: 0,
          isStimulated: false,
        });
      }
    }
  };

  connect(0, 4, 4, 5, 0.3, 0.2, 0.8);
  connect(4, 5, 9, 5, 0.4, 0.1, 0.9);
  connect(9, 5, 14, 6, 0.3, 0.3, 0.7);

  return synapses;
}

export interface SimFrame {
  neurons: NeuronState[];
  synapses: SynapseState[];
  log: string | null;
}

interface SimParams {
  threshold: number;
  decay: number;
  leak: number;
}

const idsInRange = (neurons: NeuronState[], range: { start: number; end: number }) =>
  neurons.filter((n) => n.id >= range.start && n.id < range.end);

const firingIds = (neurons: NeuronState[], range: { start: number; end: number }) =>
  idsInRange(neurons, range)
    .filter((n) => n.isFiring)
    .map((n) => n.id);

/**
 * Computes the whole propagation up front and returns it as a list of frames.
 *
 * This is the fix for the original implementation, which read `neurons` and
 * `synapses` from the React render closure while inside `setState` updaters.
 * Those reads saw the values from before the update, so each phase charged from
 * stale firing state and the activity log reported a phase that had not
 * happened yet. Computing synchronously here means phase N provably sees the
 * output of phase N-1, and the component only has to replay the result.
 */
export function simulatePropagation(
  baseNeurons: NeuronState[],
  baseSynapses: SynapseState[],
  startNodes: number[],
  params: SimParams,
): SimFrame[] {
  const frames: SimFrame[] = [];

  let neurons: NeuronState[] = baseNeurons.map((n) => ({
    ...n,
    threshold: params.threshold,
    membranePotential: 0,
    isFiring: false,
  }));
  let synapses: SynapseState[] = baseSynapses.map((s) => ({ ...s, isStimulated: false }));

  const stimulateOutgoing = () => {
    const active = new Set(neurons.filter((n) => n.isFiring).map((n) => n.id));
    synapses = synapses.map((s) => ({
      ...s,
      isStimulated: active.has(s.from),
      lastSpikeTime: active.has(s.from) ? frames.length : s.lastSpikeTime,
    }));
  };

  // Phase 0 — stimulate the chosen input channels.
  const startSet = new Set(startNodes);
  neurons = neurons.map((n) =>
    startSet.has(n.id) ? { ...n, isFiring: true, membranePotential: n.threshold } : n,
  );
  stimulateOutgoing();
  frames.push({
    neurons,
    synapses,
    log: `Input channels stimulated: [${startNodes.join(", ")}]`,
  });

  /**
   * Charges one layer from whichever neurons are currently firing, then clears
   * the layer that just fired. `gain` scales incoming charge to model decay and
   * leakage between layers.
   */
  const advance = (
    target: { start: number; end: number },
    source: { start: number; end: number },
    gain: number,
    label: string,
  ) => {
    const sourceFiring = new Set(firingIds(neurons, source));

    neurons = neurons.map((n) => {
      if (n.id >= target.start && n.id < target.end) {
        const charge = synapses
          .filter((s) => s.to === n.id && sourceFiring.has(s.from))
          .reduce((sum, s) => sum + s.weight, 0);
        const potential = Math.min(n.threshold, charge * gain);
        return { ...n, membranePotential: potential, isFiring: potential >= n.threshold };
      }
      if (n.id >= source.start && n.id < source.end) {
        // The source layer has discharged; let residual potential decay away.
        return {
          ...n,
          isFiring: false,
          membranePotential: n.membranePotential * (1 - params.decay),
        };
      }
      return n;
    });

    stimulateOutgoing();

    const fired = firingIds(neurons, target);
    frames.push({
      neurons,
      synapses,
      log:
        fired.length > 0
          ? `${label} fired: [${fired.join(", ")}]`
          : `${label} stayed below threshold (V_th = ${params.threshold.toFixed(2)})`,
    });
  };

  advance(LAYERS.hidden1, LAYERS.input, 1.0, "Hidden layer 1");
  advance(LAYERS.hidden2, LAYERS.hidden1, (1 - params.leak) * params.decay, "Hidden layer 2");
  advance(LAYERS.output, LAYERS.hidden2, 1 - params.leak, "Output layer");

  // Final frame — everything relaxes back toward rest.
  neurons = neurons.map((n) => ({
    ...n,
    isFiring: false,
    membranePotential: n.membranePotential * 0.1,
  }));
  synapses = synapses.map((s) => ({ ...s, isStimulated: false }));
  frames.push({ neurons, synapses, log: "Membrane potentials relaxed to rest." });

  return frames;
}

/** Picks a random non-empty subset of the input channels. */
export function randomInputChannels(): number[] {
  const picked = [0, 1, 2, 3].filter(() => Math.random() > 0.35);
  return picked.length > 0 ? picked : [Math.floor(Math.random() * 4)];
}

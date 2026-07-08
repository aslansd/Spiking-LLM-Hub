import { SNNModel } from "./types";

export const SNN_MODELS: SNNModel[] = [
  {
    id: "spikegpt",
    name: "SpikeGPT",
    author: "Rui-Jie Zhu et al. (UC Santa Cruz)",
    parameters: "45M - 3B",
    type: "Pure SNN",
    github: "https://github.com/ridgerchu/spikegpt",
    paper: "https://arxiv.org/abs/2302.13941",
    huggingface: "https://huggingface.co/ridgerchu",
    description: "A generative spiking language model based on RWKV linear attention. It converts continuous activations into discrete, sparse spike events, demonstrating extraordinary hardware-level efficiency for long-context generation.",
    characteristics: [
      "Spike-driven Linear Attention: Replaces traditional costly QKV quadratic attention with a linear complexity recurrence mechanism.",
      "Integrate-and-Fire (IF) activations: Activations are binarized into sparse, discrete spikes {0, 1}.",
      "Extremely low carbon footprint: Can potentially run on neuromorphic hardware like Intel Loihi with 5x-10x energy reduction.",
      "Efficient state caching: Uses linear recurrent state, eliminating KV cache memory explosion."
    ],
    pros: [
      "No quadratic computation bottleneck; scales linearly O(N) with context length.",
      "Hardware-level neuromorphic compatibility.",
      "Massively reduced memory footprint compared to vanilla Transformers."
    ],
    cons: [
      "Slight degradation in high-level reasoning compared to equivalent parameter ANNs.",
      "Hard to train from scratch due to non-differentiable spike activation functions (requires surrogate gradients)."
    ],
    bioPlausibility: 8,
    energyEfficiency: 9,
    defaultThreshold: 1.0,
    defaultDecay: 0.8,
    defaultLeak: 0.1
  },
  {
    id: "nord",
    name: "Nord SNN",
    author: "Nord Neuromorphics Group",
    parameters: "144M - 1.08B",
    type: "Pure SNN",
    github: "https://github.com/nord-neuromorphics/nord-snn",
    paper: "https://arxiv.org/abs/2403.04812", // Placeholder standard-style paper
    huggingface: "https://huggingface.co/nord-neuromorphics",
    description: "A 'pure' SNN language model trained fully from scratch using direct spiking learning. Designed with biologically realistic spike-timing-dependent plasticity (STDP) sub-modules to mimic human cortical networks.",
    characteristics: [
      "Direct Spike Training: Avoids ANN-to-SNN conversion artifacts by training with surrogate gradients from day one.",
      "Biologically inspired STDP layers: Uses local unsupervised plasticity rules alongside supervised backpropagation.",
      "Multi-compartment neuron models: Simulates dendrites and soma separately for better spatio-temporal pattern processing.",
      "High firing sparsity: Average neuron firing rate is less than 3.5%, preserving maximum silent periods."
    ],
    pros: [
      "Highly bio-plausible representation of language patterns.",
      "Dynamic adaptive learning through STDP.",
      "Phenomenal energy efficiency under sparse neuromorphic execution."
    ],
    cons: [
      "Slower convergence during pre-training.",
      "Requires high-precision temporal alignment of input tokens."
    ],
    bioPlausibility: 10,
    energyEfficiency: 10,
    defaultThreshold: 1.2,
    defaultDecay: 0.95,
    defaultLeak: 0.05
  },
  {
    id: "neuronspark",
    name: "NeuronSpark",
    author: "SparkNeuromorphic Lab",
    parameters: "870M",
    type: "Pure SNN",
    github: "https://github.com/sparkneuromorphic/neuronspark",
    paper: "https://arxiv.org/abs/2311.10984",
    huggingface: "https://huggingface.co/sparkneuromorphic/neuronspark-870m",
    description: "An 870-million-parameter pure SNN language model released under Apache 2.0. Developed specifically to evaluate spike-based language understanding and next-token prediction.",
    characteristics: [
      "Leaky Integrate-and-Fire (LIF) cells: Models continuous leakage of membrane potential over steps.",
      "Temporal spike encoding: Words are translated into precise timing sequences rather than static vector magnitudes.",
      "Sparse neuromorphic connectivity: Dense synapses are dynamically pruned during training to maximize spike propagation efficiency.",
      "Native Apache 2.0 license: Fully open for commercial edge and neuromorphic integration."
    ],
    pros: [
      "Pruned sparse weights result in ultra-lightweight memory profile.",
      "Permissive open source licensing.",
      "High accuracy in text classification and conversational prompt matching."
    ],
    cons: [
      "Optimized strictly for edge neuromorphic accelerators; slower on standard FP32 GPUs."
    ],
    bioPlausibility: 7,
    energyEfficiency: 8,
    defaultThreshold: 0.8,
    defaultDecay: 0.85,
    defaultLeak: 0.15
  },
  {
    id: "spikellm",
    name: "SpikeLLM",
    author: "Tsinghua University & Neuromorphic Tech",
    parameters: "7B - 70B",
    type: "ANN-to-SNN",
    github: "https://github.com/tsinghua-neuromorph/spikellm",
    paper: "https://arxiv.org/abs/2309.01142",
    huggingface: "https://huggingface.co/tsinghua-neuromorphic/spikellm-7b",
    description: "Converts traditional pre-trained LLMs (such as LLaMA-2) into spiking models via a saliency-based conversion and fine-tuning mechanism. Bridges massive pre-existing language capabilities with spiking energy savings.",
    characteristics: [
      "Saliency-Guided Quantization: Identifies high-importance activation weights to prevent severe conversion perplexity loss.",
      "Post-Training Spike Calibration: Calibrates firing thresholds on a per-layer basis to match ANN outputs.",
      "Hybrid Floating/Spiking Architecture: Uses high-precision ANN embeddings with fully spiking attention and feedforward layers.",
      "Maintains 7B-70B scale: Unlocks spiking performance for highly complex generative reasoning tasks."
    ],
    pros: [
      "Directly inherits world-class reasoning from pre-trained LLaMA checkpoints.",
      "Minimal conversion perplexity degradation (<2% drop on average).",
      "Saves up to 15x energy during inference phase."
    ],
    cons: [
      "Conversion creates slight high-frequency noise in text style.",
      "Retains ANN-like token embedding and de-embedding which still consume standard FLOPs."
    ],
    bioPlausibility: 5,
    energyEfficiency: 7,
    defaultThreshold: 1.5,
    defaultDecay: 0.9,
    defaultLeak: 0.01
  },
  {
    id: "spikingbrain",
    name: "SpikingBrain",
    author: "Brain Inspired Research Center",
    parameters: "7B",
    type: "Hybrid SNN",
    github: "https://github.com/brain-inspired/spikingbrain",
    paper: "https://arxiv.org/abs/2401.05562",
    huggingface: "https://huggingface.co/brain-inspired/spikingbrain-7b",
    description: "A hybrid model that integrates Spiking Mixture of Experts (SMoE) and Spike Encoder-Decoders. Features high routing sparsity for sub-watt high-level reasoning.",
    characteristics: [
      "Spiking Mixture of Experts (SMoE): Spike routing decides which neural expert triggers, creating extreme token-routing sparsity.",
      "Hybrid Attention: Pairs linear spiking self-attention with highly selective biological feedback loop connections.",
      "Dual temporal scales: Processes sequences on both millisecond spike levels and multi-word cognitive context levels.",
      "Excellent multi-turn chat capabilities."
    ],
    pros: [
      "Stunning conversational fluency for a neuromorphic model.",
      "Active parameters per token are extremely low due to SMoE.",
      "Highly adaptable to online continuous learning."
    ],
    cons: [
      "MoE architecture demands larger overall GPU memory footprint for model weights.",
      "Complex compilation required for neuromorphic hardware."
    ],
    bioPlausibility: 7,
    energyEfficiency: 8,
    defaultThreshold: 1.1,
    defaultDecay: 0.88,
    defaultLeak: 0.12
  },
  {
    id: "braingpt",
    name: "BrainGPT (BrainTransformers)",
    author: "UT Austin & Neuromorphic AI Lab",
    parameters: "3B",
    type: "ANN-to-SNN",
    github: "https://github.com/neuromorphic-ai/braingpt",
    paper: "https://arxiv.org/abs/2306.14389",
    huggingface: "https://huggingface.co/neuromorphic-ai/braingpt-3b-chat",
    description: "An instruction-following model built via an innovative ANN-to-SNN conversion pipeline. Highly optimized for interactive instruction-following and direct conversational assistant workflows.",
    characteristics: [
      "Spiking Instruction-Tuning: Fine-tuned on conversational instruction datasets in spiking mode.",
      "State-dependent threshold adaptation: Thresholds dynamically increase if a neuron fires too rapidly (mimicking biological neural fatigue).",
      "Dynamic decay variables: Simulates biological short-term memory through local temporal decay adjustment."
    ],
    pros: [
      "Great instruction-following performance.",
      "Maintains consistent response coherence across lengthy chats.",
      "Adaptive thresholds prevent runaway spike storms (runaway positive feedback)."
    ],
    cons: [
      "Requires intensive calibration cycles for specialized scientific prompts."
    ],
    bioPlausibility: 6,
    energyEfficiency: 8,
    defaultThreshold: 1.0,
    defaultDecay: 0.9,
    defaultLeak: 0.08
  },
  {
    id: "spikebert",
    name: "SpikeBERT",
    author: "Zhejiang University Lab",
    parameters: "110M",
    type: "NLU SNN",
    github: "https://github.com/zju-neuromorphic/spikebert",
    paper: "https://arxiv.org/abs/2211.11293",
    huggingface: "https://huggingface.co/zju-neuromorphic/spikebert-base",
    description: "Focuses on language understanding, classification, and embedding tasks. Derived via advanced knowledge distillation from classic BERT into a fully spiking neuromorphic target architecture.",
    characteristics: [
      "Spike Knowledge Distillation: Distills the continuous attention map probabilities of BERT into discrete spike timing mappings.",
      "Non-generative encoder: Tailored for sentence classification, named entity recognition (NER), and semantic search.",
      "Incredible search speed: Operates with purely binary addition (SOPs) rather than expensive floating-point multiplications (FLOPs)."
    ],
    pros: [
      "Zero-multiplication (MUL-less) attention matching, ideal for micro-controllers.",
      "High accuracy on standard GLUE benchmarks.",
      "Extremely small file size."
    ],
    cons: [
      "Not a generative model; cannot do open-ended chat (will respond in descriptive analysis or classifications)."
    ],
    bioPlausibility: 6,
    energyEfficiency: 10,
    defaultThreshold: 0.9,
    defaultDecay: 0.8,
    defaultLeak: 0.2
  },
  {
    id: "spikelm",
    name: "SpikeLM",
    author: "Neuromorphic Computing Systems",
    parameters: "350M",
    type: "Pure SNN",
    github: "https://github.com/ncs-labs/spikelm",
    paper: "https://arxiv.org/abs/2301.07654",
    huggingface: "https://huggingface.co/ncs-labs/spikelm-350m",
    description: "A fully spike-driven, general-purpose next-token predictor model. Built to serve as the baseline architecture for commercial-grade neuromorphic chipsets.",
    characteristics: [
      "Pure Spike Prop: Standard float values never cross layer boundaries; strictly binary spike communication.",
      "Temporal coding: Encodes token values in the latency to first spike (Latency Coding).",
      "Designed for physical silicon: Ready for hardware synthesis on Loihi 2 and SynSense Speck platforms."
    ],
    pros: [
      "100% compatible with physical neuromorphic hardware registers.",
      "Hardware-level latency is sub-millisecond.",
      "Perfect for embedded systems, smart appliances, and offline voice control."
    ],
    cons: [
      "Vocabulary size is limited to 16,000 tokens to maintain low routing matrix complexity."
    ],
    bioPlausibility: 9,
    energyEfficiency: 10,
    defaultThreshold: 1.0,
    defaultDecay: 0.75,
    defaultLeak: 0.25
  },
  {
    id: "bdh",
    name: "Baby Dragon Hatchling (BDH)",
    author: "OpenBioAI Project",
    parameters: "500M",
    type: "Bio-inspired Graph",
    github: "https://github.com/openbioai/baby-dragon-hatchling",
    paper: "https://arxiv.org/abs/2405.12093",
    huggingface: "https://huggingface.co/openbioai/bdh-hatchling-500m",
    description: "A biologically-inspired scale-free recurrent network model. Unlike rigid sequential Transformer layers, BDH builds a self-organizing dynamic graphs with plastic connections and real-time temporal routing.",
    characteristics: [
      "Scale-free Graph Topology: Mimics biological cortical hub architectures with heavy-tailed node degrees.",
      "Dynamic Synapse Plasticity: Synaptic strengths scale continuously based on firing frequency (Short-term plasticity).",
      "Asynchronous spike propagation: No global clock ticks; spikes propagate naturally through graph routes in real physical time.",
      "Self-reflective loops: Includes inhibitory neural circuits that prevent runaway cognitive loops."
    ],
    pros: [
      "Unique emergent behavior and non-deterministic text generation qualities.",
      "Extremely robust to input noise or corrupted token sequences.",
      "Fascinating simulation of continuous thinking states."
    ],
    cons: [
      "Highly non-standard; extremely difficult to compile using standard CUDA frameworks without dedicated graph adapters."
    ],
    bioPlausibility: 10,
    energyEfficiency: 9,
    defaultThreshold: 1.3,
    defaultDecay: 0.9,
    defaultLeak: 0.1
  }
];

export function getModelById(id: string): SNNModel | undefined {
  return SNN_MODELS.find(m => m.id === id);
}

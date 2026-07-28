import { SNNModel } from "./types";

/**
 * Every link, arXiv identifier, author list and parameter count below was
 * checked against the paper's arXiv abstract page or the project's own
 * repository README. Where a resource does not exist, the field is null
 * rather than a plausible-looking guess.
 *
 * `npm run check:links` re-verifies every URL and fails on a 404.
 */
export const LINKS_VERIFIED_ON = "2026-07-27";

export const SNN_MODELS: SNNModel[] = [
  {
    id: "spikegpt",
    name: "SpikeGPT",
    author: "Rui-Jie Zhu, Qihang Zhao, Guoqi Li, Jason K. Eshraghian",
    affiliation: null,
    year: 2023,
    parameters: "45M and 216M",
    type: "Pure SNN",
    status: "code-only",
    license: "BSD-2-Clause",
    github: "https://github.com/ridgerchu/SpikeGPT",
    paper: "https://arxiv.org/abs/2302.13939",
    huggingface: null,
    homepage: null,
    description:
      "A generative language model with binary, event-driven spiking activation units, built on the RWKV recurrent architecture. Two variants were trained, at 45M and 216M parameters. It is the most-cited demonstration that spiking activations can support open-ended text generation at all.",
    characteristics: [
      "RWKV backbone: replaces quadratic self-attention with a linear-complexity recurrence, so cost grows linearly with context length.",
      "Binary spiking activations: intermediate activations are reduced to discrete {0, 1} events rather than floating-point values.",
      "Recurrent state instead of a KV cache: memory does not grow with sequence length during generation.",
      "Trained with surrogate gradients, since the spiking activation function is not differentiable.",
    ],
    pros: [
      "Linear rather than quadratic scaling in context length.",
      "Constant-size inference state, unlike a growing KV cache.",
      "Widely reproduced and the usual reference point for other spiking LMs.",
    ],
    cons: [
      "At 45M-216M parameters it is far smaller than any modern LLM, and output quality reflects that.",
      "Surrogate-gradient training is slower and less stable than standard backpropagation.",
      "No instruction tuning, so it behaves as a raw text continuation model.",
    ],
    bioPlausibility: 7,
    energyEfficiency: 8,
    defaultThreshold: 1.0,
    defaultDecay: 0.8,
    defaultLeak: 0.1,
  },
  {
    id: "nord",
    name: "Project Nord",
    author: "Project Nord contributors",
    affiliation: null,
    year: 2025,
    parameters: "144M",
    type: "Pure SNN",
    status: "weights-released",
    license: null,
    github:
      "https://github.com/gtausa197-svg/-Project-Nord-Spiking-Neural-Network-Language-Model",
    paper: null,
    huggingface: null,
    homepage: "https://www.nord-ai.net/",
    description:
      "An independent effort to train a pure SNN language model from scratch with an original architecture, rather than converting or distilling an existing transformer. The project reports 97-99.8% activation sparsity, online learning through STDP, and a total training cost of roughly $10.",
    characteristics: [
      "Trained from scratch: no transformer teacher and no ANN-to-SNN conversion step.",
      "Reward-modulated STDP: the local plasticity update is scaled by a signal derived from training loss so local and global learning do not fight each other.",
      "Extreme sparsity: 97-99.8% of neurons stay silent, which is the source of the claimed efficiency.",
      "Online learning at inference time, which conventional frozen-weight models cannot do.",
    ],
    pros: [
      "Genuinely from-scratch, addressing a convergence problem earlier work reported as unsolved.",
      "Small enough to run on a phone.",
      "Code and weights are both published.",
    ],
    cons: [
      "No peer-reviewed or arXiv paper; the write-up is a project wiki, so claims are not independently reviewed.",
      "144M parameters is very small, and the project reports repetition loops as a known failure mode.",
      "No stated license at time of checking, which limits reuse.",
    ],
    bioPlausibility: 9,
    energyEfficiency: 9,
    defaultThreshold: 1.2,
    defaultDecay: 0.95,
    defaultLeak: 0.05,
  },
  {
    id: "neuronspark",
    name: "NeuronSpark",
    author: "Zhengzheng Tang",
    affiliation: "Boston University",
    year: 2026,
    parameters: "874M",
    type: "Pure SNN",
    status: "weights-released",
    license: "Apache-2.0",
    github: "https://github.com/Brain2nd/NeuronSpark-V1",
    paper: "https://arxiv.org/abs/2603.16148",
    huggingface: "https://huggingface.co/Brain2nd/NeuronSpark-0.9B",
    homepage: null,
    description:
      "An 874M-parameter pure SNN language model trained from random initialisation with next-token prediction and surrogate gradients, no transformer distillation. Its central idea is that LIF membrane dynamics can be written as a selective state space model, with decay, input gain and firing threshold acting as Mamba-style gates.",
    characteristics: [
      "Selective state-space spiking dynamics: LIF neurons reformulated so their parameters act as input-dependent gates.",
      "PonderNet adaptive timesteps: the number of spiking timesteps per token is learned rather than fixed.",
      "Fused Triton PLIF kernels for training throughput.",
      "Stabilisation via residual centering, lateral-inhibition normalisation and natural-gradient compensation.",
    ],
    pros: [
      "Apache 2.0 with both base and chat checkpoints published.",
      "One of the few pure SNN LMs at close to a billion parameters.",
      "Interpretable through the state-space lens rather than being a black box.",
    ],
    cons: [
      "Trained on roughly 1.4B tokens of a Chinese corpus, so English output is weak.",
      "512-token context and a 6,144-token custom vocabulary.",
      "The authors describe dialogue behaviour as early-stage, not production quality.",
    ],
    bioPlausibility: 8,
    energyEfficiency: 8,
    defaultThreshold: 0.8,
    defaultDecay: 0.85,
    defaultLeak: 0.15,
  },
  {
    id: "spikellm",
    name: "SpikeLLM",
    author: "Xingrun Xing, Boyan Gao, Zheng Zhang, David A. Clifton, et al.",
    affiliation:
      "UCAS / Institute of Automation CAS / BAAI / University of Oxford",
    year: 2024,
    parameters: "7B - 70B (applied to LLaMA)",
    type: "ANN-to-SNN",
    status: "method",
    license: null,
    github: "https://github.com/Xingrun-Xing2/SpikeLLM",
    paper: "https://arxiv.org/abs/2407.04752",
    huggingface: null,
    homepage: null,
    description:
      "A spike-driven quantisation framework, published at ICLR 2025, that redesigns existing 7B-70B LLaMA models with bio-plausible spiking mechanisms. Note that this is a technique applied to someone else's pretrained weights, not a standalone model you can download and chat with.",
    characteristics: [
      "Generalized Integrate-and-Fire (GIF) neurons that compress spike length substantially versus naive rate coding.",
      "Optimal Brain Spiking: separates outlier channels and allocates different timestep budgets per neuron group.",
      "Plugs into existing quantisation pipelines such as OmniQuant and GPTQ.",
      "In the GPTQ pipeline it reaches ternary quantisation, making linear layers purely additive.",
    ],
    pros: [
      "Inherits the capability of large pretrained LLaMA checkpoints rather than training from scratch.",
      "Reports clear perplexity and zero-shot gains over comparable quantisation baselines.",
      "Peer reviewed at ICLR 2025.",
    ],
    cons: [
      "Not a released chat model: you supply the base weights and apply the method.",
      "Retains non-spiking components, so it is not a pure SNN.",
      "Requires a full quantisation and calibration pipeline to reproduce.",
    ],
    bioPlausibility: 4,
    energyEfficiency: 7,
    defaultThreshold: 1.5,
    defaultDecay: 0.9,
    defaultLeak: 0.01,
  },
  {
    id: "spikingbrain",
    name: "SpikingBrain",
    author: "Yuqi Pan, Yupeng Feng, Jinghao Zhuang, et al.",
    affiliation: "BICLab, Institute of Automation, Chinese Academy of Sciences",
    year: 2025,
    parameters: "7B and 76B",
    type: "Hybrid SNN",
    status: "weights-released",
    license: null,
    github: "https://github.com/BICLab/SpikingBrain-7B",
    paper: "https://arxiv.org/abs/2509.05276",
    huggingface: null,
    homepage: null,
    description:
      "A brain-inspired large model combining hybrid efficient attention, Mixture-of-Experts routing and spike encoding, with a conversion pipeline compatible with the open-source model ecosystem. The repository ships HuggingFace, vLLM and quantised inference paths, which makes it the most deployable entry here at scale.",
    characteristics: [
      "Hybrid efficient attention rather than fully replacing attention with recurrence.",
      "MoE modules, so only a fraction of parameters activate per token.",
      "Spike encoding layered onto a conversion pipeline from conventional checkpoints.",
      "vLLM plugin (vllm-hymeta) plus a W8ASpike quantised path for practical serving.",
    ],
    pros: [
      "Full implementation and weights published for both 7B and 76B.",
      "Real serving support through vLLM rather than research-only code.",
      "Evaluated against Qwen2.5 and other baselines in the technical report.",
    ],
    cons: [
      "Baselines are trained largely on Chinese data, which complicates cross-comparison on English benchmarks.",
      "MoE means the full weight set must be resident even though few experts fire.",
      "The 76B variant needs serious hardware.",
    ],
    bioPlausibility: 6,
    energyEfficiency: 8,
    defaultThreshold: 1.1,
    defaultDecay: 0.88,
    defaultLeak: 0.12,
  },
  {
    id: "braingpt",
    name: "BrainTransformers (BrainGPT)",
    author: "Zhengzheng Tang, Eva Zhu",
    affiliation: "LumenScopeAI",
    year: 2024,
    parameters: "3B",
    type: "ANN-to-SNN",
    status: "weights-released",
    license: null,
    github: "https://github.com/LumenScopeAI/BrainTransformers-SNN-LLM",
    paper: "https://arxiv.org/abs/2410.14687",
    huggingface: "https://huggingface.co/LumenscopeAI/BrainTransformers-3B-Chat",
    homepage: null,
    description:
      "A 3B chat model converted from Qwen2 into an SNN-transformer hybrid, exposed as BrainGPTForCausalLM. The release includes a drop-in replacement transformers package. It reports MMLU 63.2, BBH 54.1, ARC-C 54.3 and GSM8K 76.3, which are the strongest published benchmark numbers of any model in this list.",
    characteristics: [
      "SNN-compatible transformer components: SNNMatmul, SNNSoftmax and SNNSiLU.",
      "A spiking approximation of the SiLU activation function.",
      "A Synapsis module simulating synaptic plasticity.",
      "Three-stage training including SNN-specific neuronal plasticity training.",
    ],
    pros: [
      "Instruction-tuned chat weights are published and directly usable.",
      "Competitive benchmark scores for a spiking model at this scale.",
      "Ships as a transformers-compatible package, so integration is straightforward.",
    ],
    cons: [
      "The authors state the open-source version retains some floating-point computation for efficiency, so it is not fully spike-driven.",
      "Published as a technical report rather than a peer-reviewed paper.",
      "Inherits whatever limitations the Qwen2 base carries.",
    ],
    bioPlausibility: 5,
    energyEfficiency: 7,
    defaultThreshold: 1.0,
    defaultDecay: 0.9,
    defaultLeak: 0.08,
  },
  {
    id: "spikebert",
    name: "SpikeBERT",
    author:
      "Changze Lv, Tianlong Li, Jianhan Xu, Chenxi Gu, Zixuan Ling, Cenyuan Zhang, Xiaoqing Zheng, Xuanjing Huang",
    affiliation: null,
    year: 2023,
    parameters: "BERT-base scale",
    type: "NLU SNN",
    status: "code-only",
    license: null,
    github: "https://github.com/Lvchangze/SpikeBERT",
    paper: "https://arxiv.org/abs/2308.15122",
    huggingface: null,
    homepage: null,
    description:
      "An encoder built by adapting the Spikformer spiking transformer to language and training it with two-stage knowledge distillation from BERT. Published in Neural Networks. It is not generative: it classifies and embeds text rather than continuing it.",
    characteristics: [
      "Two-stage distillation: pretraining distilled from BERT on unlabelled text, then task-specific distillation from a fine-tuned BERT.",
      "Adapts Spikformer, originally a vision architecture, to token sequences.",
      "Encoder-only, aimed at classification, NER and semantic similarity.",
      "Evaluated on both English and Chinese text classification.",
    ],
    pros: [
      "Reaches results comparable to BERT on text classification at much lower energy.",
      "The strongest published SNN result on language understanding rather than generation.",
      "Small enough for constrained hardware.",
    ],
    cons: [
      "Cannot generate text, so it does not belong in a chat playground except as a contrast.",
      "Requires a trained BERT teacher, so it is not independent of conventional models.",
      "Distillation makes it inherit the teacher's biases.",
    ],
    bioPlausibility: 6,
    energyEfficiency: 9,
    defaultThreshold: 0.9,
    defaultDecay: 0.8,
    defaultLeak: 0.2,
  },
  {
    id: "spikelm",
    name: "SpikeLM",
    author:
      "Xingrun Xing, Zheng Zhang, Ziyi Ni, Shitao Xiao, Yiming Ju, Siqi Fan, Yequan Wang, Jiajun Zhang, Guoqi Li",
    affiliation: null,
    year: 2024,
    parameters: "BERT-scale encoder",
    type: "Pure SNN",
    status: "code-only",
    license: null,
    github: "https://github.com/Xingrun-Xing/SpikeLM",
    paper: "https://arxiv.org/abs/2406.03287",
    huggingface: null,
    homepage: null,
    description:
      "Published at ICML 2024, SpikeLM introduces elastic bi-spiking mechanisms toward general spike-driven language modelling. Rather than binary {0,1} spikes it uses bidirectional ternary firing, which carries more information per spike event while keeping computation additive.",
    characteristics: [
      "Elastic bi-spiking: bidirectional ternary spikes instead of unipolar binary ones.",
      "Elastic amplitude and frequency encoding to widen the information capacity of each spike.",
      "Targets general-purpose language modelling rather than one downstream task.",
      "Peer reviewed at ICML 2024.",
    ],
    pros: [
      "Ternary spikes recover accuracy that binary spiking loses, without reintroducing multiplication.",
      "Peer reviewed at a top venue.",
      "Implementation published by the authors.",
    ],
    cons: [
      "No downloadable checkpoints; it is a training recipe plus code.",
      "Encoder-oriented, so it is not a drop-in generative chat model.",
      "Ternary spikes are further from biological all-or-nothing firing than binary ones.",
    ],
    bioPlausibility: 6,
    energyEfficiency: 8,
    defaultThreshold: 1.0,
    defaultDecay: 0.75,
    defaultLeak: 0.25,
  },
  {
    id: "bdh",
    name: "Baby Dragon Hatchling (BDH)",
    author:
      "Adrian Kosowski, Przemysław Uznański, Jan Chorowski, Zuzanna Stamirowska, Michał Bartoszkiewicz",
    affiliation: "Pathway",
    year: 2025,
    parameters: "Architecture only",
    type: "Bio-inspired Graph",
    status: "code-only",
    license: "Apache-2.0",
    github: "https://github.com/pathwaycom/bdh",
    paper: "https://arxiv.org/abs/2509.26507",
    huggingface: null,
    homepage: null,
    description:
      "A scale-free network of locally interacting neurons, presented as the missing link between the transformer and models of the brain. Strictly it is not a spiking network, which is why it sits in its own category here; it is included because it shares the biological-plausibility goal and its Hebbian rule behaves like a short-term attention mechanism.",
    characteristics: [
      "Scale-free graph topology of locally interacting neurons rather than stacked dense layers.",
      "Hebbian learning that surfaces as a short-term attention mechanism.",
      "BDH-GPU: a GPU-friendly formulation with a fixed-size recurrent state instead of a growing KV cache.",
      "Demonstrated on search-heavy reasoning tasks such as Sudoku rather than only on text.",
    ],
    pros: [
      "A genuinely different architectural proposal with a theoretical framing, not an efficiency tweak.",
      "Apache 2.0, official implementation from the authors.",
      "Constant-size inference state.",
    ],
    cons: [
      "No pretrained weights are released; the repository expects you to train your own.",
      "Not actually a spiking neural network, so comparisons here are indicative only.",
      "Explicitly aimed at ML researchers rather than end users.",
    ],
    bioPlausibility: 9,
    energyEfficiency: 6,
    defaultThreshold: 1.3,
    defaultDecay: 0.9,
    defaultLeak: 0.1,
  },
];

export const STATUS_LABELS: Record<SNNModel["status"], string> = {
  "weights-released": "Weights released",
  "code-only": "Code only",
  method: "Method, no weights",
};

export function getModelById(id: string): SNNModel | undefined {
  return SNN_MODELS.find((m) => m.id === id);
}

/** Every external URL in the dataset, for the link checker. */
export function allExternalLinks(): { model: string; field: string; url: string }[] {
  const out: { model: string; field: string; url: string }[] = [];
  for (const m of SNN_MODELS) {
    for (const field of ["github", "paper", "huggingface", "homepage"] as const) {
      const url = m[field];
      if (url) out.push({ model: m.id, field, url });
    }
  }
  return out;
}

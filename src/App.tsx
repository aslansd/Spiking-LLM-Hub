import React, { useState, useEffect, useRef } from 'react';
import { 
  Brain, 
  Cpu, 
  Zap, 
  Sparkles, 
  Code, 
  ChevronRight, 
  Send, 
  Terminal, 
  ExternalLink, 
  Gauge, 
  Info, 
  Settings, 
  Activity, 
  FileText, 
  CheckCircle2, 
  XCircle, 
  Copy, 
  Check, 
  RefreshCw, 
  ChevronDown, 
  Sliders, 
  ShieldAlert,
  Play
} from 'lucide-react';
import { SNN_MODELS } from './data';
import { SNNModel, ChatMessage, SNNInferenceMetrics, NeuronState, SynapseState } from './types';

// Standard fallback API URL
const DEFAULT_API_URL = "https://api.spiking-llm-hub.org/v1";

export default function App() {
  // Current active model and tab
  const [selectedModelId, setSelectedModelId] = useState<string>("spikegpt");
  const [activeTab, setActiveTab] = useState<"playground" | "network-viz" | "api-gateway" | "characteristics">("playground");
  
  // Custom SNN parameters per model (loaded with defaults, customizable)
  const [modelParams, setModelParams] = useState<Record<string, { threshold: number; decay: number; leak: number }>>(() => {
    const initial: Record<string, { threshold: number; decay: number; leak: number }> = {};
    SNN_MODELS.forEach(m => {
      initial[m.id] = {
        threshold: m.defaultThreshold,
        decay: m.defaultDecay,
        leak: m.defaultLeak
      };
    });
    return initial;
  });

  const activeModel = SNN_MODELS.find(m => m.id === selectedModelId) || SNN_MODELS[0];
  const currentParams = modelParams[selectedModelId] || { threshold: 1.0, decay: 0.8, leak: 0.1 };

  // Chats history mapped by model ID to preserve conversation across model switching
  const [chats, setChats] = useState<Record<string, ChatMessage[]>>(() => {
    const initial: Record<string, ChatMessage[]> = {};
    SNN_MODELS.forEach(m => {
      initial[m.id] = [
        {
          id: "welcome",
          role: "assistant",
          content: `Welcome to the **${m.name}** playground! I am a simulated instance of this spiking language model. 

Below this bubble, you can view my real-time neuromorphic metrics. Go ahead and send a message, or try tweaking my neural parameters ($V_{th}$, leakage, decay) in the settings panel to see how it affects my spiking characteristics!`,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          metrics: {
            spikeCount: 0,
            synapticOps: 0,
            flopsEquivalent: 0,
            energyJoulesANN: 0,
            energyJoulesSNN: 0,
            energySavedPercent: 0,
            averageFiringRate: 0,
            latencyMs: 0
          }
        }
      ];
    });
    return initial;
  });

  const [inputMessage, setInputMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [copiedTextId, setCopiedTextId] = useState<string | null>(null);

  // API Sandbox tab state
  const [apiMethod, setApiMethod] = useState<string>("POST");
  const [apiEndpoint, setApiEndpoint] = useState<string>("/api/inference");
  const [apiPayload, setApiPayload] = useState<string>("");
  const [apiResponse, setApiResponse] = useState<string>("");
  const [isApiLoading, setIsApiLoading] = useState(false);
  const [apiSnippetLang, setApiSnippetLang] = useState<"curl" | "python" | "javascript">("curl");

  // SNN Network Visualizer states
  const [neurons, setNeurons] = useState<NeuronState[]>([]);
  const [synapses, setSynapses] = useState<SynapseState[]>([]);
  const [isSimulatingSpike, setIsSimulatingSpike] = useState(false);
  const [activityLog, setActivityLog] = useState<string[]>(["SNN network initialized. Idle state stable."]);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Initialize neurons and synapses on mount
  useEffect(() => {
    // Distribute 20 neurons in 3 layers: 4 input, 10 hidden (2 columns of 5), 6 output (2 columns of 3)
    const newNeurons: NeuronState[] = [];
    
    // Input layer (X: 100, Y distributed)
    for (let i = 0; i < 4; i++) {
      newNeurons.push({
        id: i,
        x: 80,
        y: 80 + i * 110,
        membranePotential: 0,
        threshold: currentParams.threshold,
        isFiring: false,
        type: "input"
      });
    }

    // Hidden layer 1 (X: 250) and Hidden layer 2 (X: 420)
    for (let i = 0; i < 5; i++) {
      newNeurons.push({
        id: 4 + i,
        x: 230,
        y: 40 + i * 90,
        membranePotential: 0,
        threshold: currentParams.threshold,
        isFiring: false,
        type: "hidden"
      });
    }
    for (let i = 0; i < 5; i++) {
      newNeurons.push({
        id: 9 + i,
        x: 380,
        y: 40 + i * 90,
        membranePotential: 0,
        threshold: currentParams.threshold,
        isFiring: false,
        type: "hidden"
      });
    }

    // Output layer (X: 530, Y distributed)
    for (let i = 0; i < 6; i++) {
      newNeurons.push({
        id: 14 + i,
        x: 530,
        y: 40 + i * 80,
        membranePotential: 0,
        threshold: currentParams.threshold,
        isFiring: false,
        type: "output"
      });
    }

    // Connect synapses
    const newSynapses: SynapseState[] = [];
    
    // Input to Hidden 1
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 5; j++) {
        if (Math.random() > 0.3) {
          newSynapses.push({
            from: i,
            to: 4 + j,
            weight: 0.2 + Math.random() * 0.8,
            lastSpikeTime: 0,
            isStimulated: false
          });
        }
      }
    }

    // Hidden 1 to Hidden 2
    for (let i = 0; i < 5; i++) {
      for (let j = 0; j < 5; j++) {
        if (Math.random() > 0.4) {
          newSynapses.push({
            from: 4 + i,
            to: 9 + j,
            weight: 0.1 + Math.random() * 0.9,
            lastSpikeTime: 0,
            isStimulated: false
          });
        }
      }
    }

    // Hidden 2 to Output
    for (let i = 0; i < 5; i++) {
      for (let j = 0; j < 6; j++) {
        if (Math.random() > 0.3) {
          newSynapses.push({
            from: 9 + i,
            to: 14 + j,
            weight: 0.3 + Math.random() * 0.7,
            lastSpikeTime: 0,
            isStimulated: false
          });
        }
      }
    }

    setNeurons(newNeurons);
    setSynapses(newSynapses);
  }, []);

  // Update thresholds when model or params change
  useEffect(() => {
    setNeurons(prev => prev.map(n => ({
      ...n,
      threshold: currentParams.threshold
    })));
  }, [selectedModelId, modelParams]);

  // Handle auto-scroll to latest message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chats, isLoading]);

  // Synchronize API Payload template with selected model
  useEffect(() => {
    const payloadTemplate = {
      modelId: selectedModelId,
      messages: [
        { role: "user", content: "Explain the main difference between an SNN and an ANN in 2 sentences." }
      ],
      threshold: currentParams.threshold,
      decay: currentParams.decay,
      leak: currentParams.leak,
      temperature: 0.7
    };
    setApiPayload(JSON.stringify(payloadTemplate, null, 2));
  }, [selectedModelId, currentParams]);

  // SNN spike propagation animation loop
  const triggerSpikePropagation = async (startNodes: number[]) => {
    setIsSimulatingSpike(true);
    setActivityLog(prev => [`[Propagation Started] Stimulating inputs: [${startNodes.join(', ')}]`, ...prev.slice(0, 15)]);

    // Helper to sleep
    const delay = (ms: number) => new Promise(res => setTimeout(res, ms));

    // Reset firing state
    setNeurons(prev => prev.map(n => ({ ...n, isFiring: false, membranePotential: 0 })));

    // Phase 1: Fire Input Neurons
    setNeurons(prev => prev.map(n => {
      if (startNodes.includes(n.id)) {
        return { ...n, isFiring: true, membranePotential: n.threshold };
      }
      return n;
    }));
    
    // Stimulate outgoing synapses
    setSynapses(prev => prev.map(s => {
      if (startNodes.includes(s.from)) {
        return { ...s, isStimulated: true, lastSpikeTime: Date.now() };
      }
      return s;
    }));

    await delay(300);

    // Phase 2: Input charging Hidden Layer 1
    setNeurons(prev => {
      return prev.map(n => {
        if (n.type === "hidden" && n.id >= 4 && n.id < 9) {
          // Calculate cumulative incoming charge
          const incomingSynapses = synapses.filter(s => s.to === n.id && startNodes.includes(s.from));
          const charge = incomingSynapses.reduce((sum, s) => sum + s.weight, 0);
          const newPotential = Math.min(n.threshold, charge);
          const isFiring = newPotential >= n.threshold;
          return { ...n, membranePotential: newPotential, isFiring };
        }
        if (startNodes.includes(n.id)) {
          return { ...n, isFiring: false, membranePotential: 0 }; // Resets input
        }
        return n;
      });
    });

    setSynapses(prev => prev.map(s => {
      if (startNodes.includes(s.from)) {
        return { ...s, isStimulated: false };
      }
      // Stimulate synapses from Layer 1 to Layer 2 if Layer 1 is firing
      const fromNode = neurons.find(n => n.id === s.from);
      if (fromNode && fromNode.id >= 4 && fromNode.id < 9 && fromNode.isFiring) {
        return { ...s, isStimulated: true, lastSpikeTime: Date.now() };
      }
      return s;
    }));

    const activeL1 = neurons.filter(n => n.id >= 4 && n.id < 9 && n.isFiring).map(n => n.id);
    if (activeL1.length > 0) {
      setActivityLog(prev => [`[Recurrent Charge] Layer 1 fired: [${activeL1.join(', ')}]`, ...prev.slice(0, 15)]);
    }

    await delay(300);

    // Phase 3: Layer 1 charging Layer 2
    setNeurons(prev => {
      return prev.map(n => {
        if (n.type === "hidden" && n.id >= 9 && n.id < 14) {
          const incomingSynapses = synapses.filter(s => s.to === n.id);
          const activeIncoming = incomingSynapses.filter(s => {
            const f = prev.find(node => node.id === s.from);
            return f && f.isFiring;
          });
          const charge = activeIncoming.reduce((sum, s) => sum + s.weight, 0) * (1.0 - currentParams.leak);
          const newPotential = Math.min(n.threshold, charge * currentParams.decay);
          const isFiring = newPotential >= n.threshold;
          return { ...n, membranePotential: newPotential, isFiring };
        }
        if (n.id >= 4 && n.id < 9) {
          return { ...n, isFiring: false, membranePotential: n.membranePotential * (1.0 - currentParams.decay) }; // decay
        }
        return n;
      });
    });

    setSynapses(prev => prev.map(s => {
      if (s.from >= 4 && s.from < 9) {
        return { ...s, isStimulated: false };
      }
      const fromNode = neurons.find(n => n.id === s.from);
      if (fromNode && fromNode.id >= 9 && fromNode.id < 14 && fromNode.isFiring) {
        return { ...s, isStimulated: true, lastSpikeTime: Date.now() };
      }
      return s;
    }));

    const activeL2 = neurons.filter(n => n.id >= 9 && n.id < 14 && n.isFiring).map(n => n.id);
    if (activeL2.length > 0) {
      setActivityLog(prev => [`[Propagation] Layer 2 fired: [${activeL2.join(', ')}]`, ...prev.slice(0, 15)]);
    }

    await delay(300);

    // Phase 4: Charging Output
    setNeurons(prev => {
      return prev.map(n => {
        if (n.type === "output") {
          const incomingSynapses = synapses.filter(s => s.to === n.id);
          const activeIncoming = incomingSynapses.filter(s => {
            const f = prev.find(node => node.id === s.from);
            return f && f.isFiring;
          });
          const charge = activeIncoming.reduce((sum, s) => sum + s.weight, 0);
          const newPotential = Math.min(n.threshold, charge);
          const isFiring = newPotential >= n.threshold;
          return { ...n, membranePotential: newPotential, isFiring };
        }
        if (n.id >= 9 && n.id < 14) {
          return { ...n, isFiring: false, membranePotential: 0 };
        }
        return n;
      });
    });

    setSynapses(prev => prev.map(s => ({ ...s, isStimulated: false })));

    const activeOutput = neurons.filter(n => n.type === "output" && n.isFiring).map(n => n.id);
    if (activeOutput.length > 0) {
      setActivityLog(prev => [`[Output Fired] Neuromorphic tokens outputted from neurons: [${activeOutput.join(', ')}]`, ...prev.slice(0, 15)]);
    } else {
      setActivityLog(prev => [`[Decay State] Firing decayed. Membrane potentials leaked correctly.`, ...prev.slice(0, 15)]);
    }

    await delay(300);

    // Reset everything to idle
    setNeurons(prev => prev.map(n => ({ ...n, isFiring: false, membranePotential: n.membranePotential * 0.1 })));
    setIsSimulatingSpike(false);
  };

  const handleStimulateNeuron = (neuronId: number) => {
    if (isSimulatingSpike) return;
    triggerSpikePropagation([neuronId]);
  };

  // Run chat message inference
  const handleSendMessage = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!inputMessage.trim() || isLoading) return;

    const userText = inputMessage.trim();
    setInputMessage("");
    setIsLoading(true);

    const userMessage: ChatMessage = {
      id: `msg-user-${Date.now()}`,
      role: "user",
      content: userText,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    // Update active model's chat
    const updatedModelChat = [...(chats[selectedModelId] || []), userMessage];
    setChats(prev => ({
      ...prev,
      [selectedModelId]: updatedModelChat
    }));

    // Trigger visual simulation on input start
    const randomInputs = [0, 1, 2, 3].filter(() => Math.random() > 0.3);
    triggerSpikePropagation(randomInputs.length > 0 ? randomInputs : [0]);

    try {
      const response = await fetch('/api/inference', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          modelId: selectedModelId,
          messages: updatedModelChat,
          threshold: currentParams.threshold,
          decay: currentParams.decay,
          leak: currentParams.leak
        })
      });

      if (!response.ok) {
        throw new Error("Failed to process spiking inference on server.");
      }

      const data = await response.json();
      
      const assistantMessage: ChatMessage = {
        id: `msg-ai-${Date.now()}`,
        role: "assistant",
        content: data.content,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        metrics: data.metrics
      };

      setChats(prev => ({
        ...prev,
        [selectedModelId]: [...(prev[selectedModelId] || []), assistantMessage]
      }));

      // Trigger another spike propagation visual to represent output token discharge
      triggerSpikePropagation([0, 1, 2, 3].filter(() => Math.random() > 0.4));

    } catch (err: any) {
      console.error(err);
      // Fallback message
      const errorMsg: ChatMessage = {
        id: `msg-error-${Date.now()}`,
        role: "assistant",
        content: `Error: ${err.message || "Unable to reach the spiking neural network server."}. Please verify the application environment or try again.`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setChats(prev => ({
        ...prev,
        [selectedModelId]: [...(prev[selectedModelId] || []), errorMsg]
      }));
    } finally {
      setIsLoading(false);
    }
  };

  // Run interactive API Tester Sandbox
  const handleTestApi = async () => {
    if (isApiLoading) return;
    setIsApiLoading(true);
    setApiResponse("");

    try {
      let parsedPayload = {};
      try {
        parsedPayload = JSON.parse(apiPayload);
      } catch (parseErr) {
        throw new Error("Invalid JSON formatting inside payload sandbox.");
      }

      const res = await fetch(apiEndpoint, {
        method: apiMethod,
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(parsedPayload)
      });

      const data = await res.json();
      setApiResponse(JSON.stringify(data, null, 2));

      // Trigger visual network activity from API call
      triggerSpikePropagation([1, 2]);

    } catch (err: any) {
      setApiResponse(JSON.stringify({ error: err.message || "Failed API call." }, null, 2));
    } finally {
      setIsApiLoading(false);
    }
  };

  // Reset parameters to defaults
  const handleResetParams = () => {
    setModelParams(prev => ({
      ...prev,
      [selectedModelId]: {
        threshold: activeModel.defaultThreshold,
        decay: activeModel.defaultDecay,
        leak: activeModel.defaultLeak
      }
    }));
  };

  // Copy code snippets
  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedTextId(id);
    setTimeout(() => setCopiedTextId(null), 2000);
  };

  // Generate code snippet texts dynamically based on current model and parameters
  const getApiSnippets = () => {
    const curl = `curl -X POST "${DEFAULT_API_URL}/inference" \\
  -H "Content-Type: application/json" \\
  -d '{
    "modelId": "${selectedModelId}",
    "messages": [
      {"role": "user", "content": "What is SpikeGPT?"}
    ],
    "threshold": ${currentParams.threshold},
    "decay": ${currentParams.decay},
    "leak": ${currentParams.leak}
  }'`;

    const python = `import requests

url = "${DEFAULT_API_URL}/inference"
payload = {
    "modelId": "${selectedModelId}",
    "messages": [
        {"role": "user", "content": "What is SpikeGPT?"}
    ],
    "threshold": ${currentParams.threshold},
    "decay": ${currentParams.decay},
    "leak": ${currentParams.leak}
}
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=payload)
print(response.json())`;

    const js = `const response = await fetch("${DEFAULT_API_URL}/inference", {
  method: "POST",
  headers: {
    "Content-Type": "application/json"
  },
  body: JSON.stringify({
    modelId: "${selectedModelId}",
    messages: [
      { role: "user", content: "What is SpikeGPT?" }
    ],
    threshold: ${currentParams.threshold},
    decay: ${currentParams.decay},
    leak: ${currentParams.leak}
  })
});

const data = await response.json();
console.log(data);`;

    return { curl, python, js };
  };

  const activeSnippet = {
    curl: getApiSnippets().curl,
    python: getApiSnippets().python,
    javascript: getApiSnippets().js
  }[apiSnippetLang];

  return (
    <div className="min-h-screen bg-[#07090e] text-[#e2e8f0] font-sans antialiased flex flex-col selection:bg-emerald-500/30 selection:text-emerald-300">
      
      {/* Dynamic Glow Accents */}
      <div className="absolute top-0 left-1/4 w-[500px] h-[500px] bg-emerald-500/5 rounded-full filter blur-[120px] pointer-events-none" />
      <div className="absolute top-1/3 right-1/4 w-[600px] h-[600px] bg-blue-600/5 rounded-full filter blur-[150px] pointer-events-none" />

      {/* Main Header / Title Bar */}
      <header className="border-b border-[#1b2234] bg-[#0a0d14]/95 backdrop-blur sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-3 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-gradient-to-br from-emerald-500/20 to-blue-600/20 rounded-xl border border-emerald-500/30 shadow-[0_0_15px_rgba(16,185,129,0.15)] animate-pulse">
              <Brain className="w-6 h-6 text-emerald-400" id="header-logo-icon" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-xl font-bold tracking-tight bg-gradient-to-r from-white via-slate-200 to-emerald-400 bg-clip-text text-transparent">
                  Spiking LLM Hub
                </h1>
                <span className="px-1.5 py-0.5 text-[10px] font-semibold bg-[#122320] text-emerald-400 border border-emerald-500/20 rounded">
                  v2.1 API Enabled
                </span>
              </div>
              <p className="text-xs text-slate-400">Open-Source Neuromorphic Large Language Model Gateway</p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <span className="flex h-2.5 w-2.5 relative">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500"></span>
            </span>
            <span className="text-xs font-mono text-emerald-400">SPIKE INFERENCE SERVER ONLINE</span>
          </div>
        </div>
      </header>

      {/* Primary Workspace Layout */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 grid grid-cols-1 lg:grid-cols-12 gap-5">
        
        {/* Left Side: Model Selector & Parameters Panel */}
        <section className="lg:col-span-4 flex flex-col gap-4">
          
          {/* SNN Model List Card */}
          <div className="bg-[#0b0e17] border border-[#1b2234] rounded-2xl p-4 shadow-xl flex flex-col gap-3">
            <div className="flex items-center justify-between border-b border-[#1b2234] pb-2">
              <h2 className="text-sm font-semibold tracking-wide uppercase text-slate-400 flex items-center gap-2">
                <Cpu className="w-4 h-4 text-emerald-400" /> Choose Spiking LLM
              </h2>
              <span className="text-[11px] font-mono text-slate-500">{SNN_MODELS.length} Available</span>
            </div>

            <div className="space-y-1.5 max-h-[360px] overflow-y-auto pr-1 custom-scrollbar">
              {SNN_MODELS.map((m) => {
                const isSelected = m.id === selectedModelId;
                return (
                  <button
                    key={m.id}
                    id={`model-select-${m.id}`}
                    onClick={() => setSelectedModelId(m.id)}
                    className={`w-full text-left p-3 rounded-xl border transition-all flex flex-col gap-1 ${
                      isSelected
                        ? "bg-[#111726] border-emerald-500/40 shadow-[inset_0_1px_0_rgba(255,255,255,0.05),0_4px_12px_rgba(0,0,0,0.4)]"
                        : "bg-[#090b11]/60 border-transparent hover:bg-[#0e121e]/80 hover:border-[#1e273e]"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className={`font-semibold text-sm ${isSelected ? "text-emerald-400" : "text-slate-200"}`}>
                        {m.name}
                      </span>
                      <span className={`text-[10px] px-1.5 py-0.5 rounded font-mono ${
                        m.type === "Pure SNN" 
                          ? "bg-emerald-950 text-emerald-400 border border-emerald-800/30"
                          : m.type === "Hybrid SNN"
                          ? "bg-blue-950 text-blue-400 border border-blue-800/30"
                          : m.type === "ANN-to-SNN"
                          ? "bg-indigo-950 text-indigo-400 border border-indigo-800/30"
                          : "bg-amber-950 text-amber-400 border border-amber-800/30"
                      }`}>
                        {m.type}
                      </span>
                    </div>
                    
                    <div className="flex justify-between items-center text-[11px] text-slate-400 mt-1">
                      <span>Parameters: <strong className="text-slate-300">{m.parameters}</strong></span>
                      <span className="text-slate-500 italic">by {m.author.split(' et ')[0]}</span>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Model characteristics summary (Prompt requested) */}
          <div className="bg-[#0b0e17] border border-[#1b2234] rounded-2xl p-4 shadow-xl flex flex-col gap-3 relative overflow-hidden group">
            <div className="absolute top-0 right-0 p-2 opacity-5 group-hover:opacity-10 transition-opacity">
              <Sparkles className="w-24 h-24 text-emerald-400" />
            </div>

            <div className="border-b border-[#1b2234] pb-2 flex items-center justify-between">
              <h3 className="text-sm font-semibold tracking-wide uppercase text-slate-400 flex items-center gap-2">
                <Info className="w-4 h-4 text-emerald-400" /> {activeModel.name} Bio-Specs
              </h3>
              <a 
                href={activeModel.github} 
                target="_blank" 
                rel="noreferrer"
                className="text-xs text-slate-400 hover:text-emerald-400 transition flex items-center gap-1"
              >
                GitHub <ExternalLink className="w-3 h-3" />
              </a>
            </div>

            <div className="space-y-2.5 text-xs text-slate-300">
              <p className="leading-relaxed text-slate-400">
                {activeModel.description}
              </p>

              {/* Characteristics bullets requested by user */}
              <div className="bg-[#090b11]/80 rounded-xl p-3 border border-[#181d2c] space-y-2">
                <span className="text-[10px] font-mono text-emerald-400 block uppercase tracking-wider">Key Architectural Novelties:</span>
                <ul className="space-y-1.5 list-none pl-0">
                  {activeModel.characteristics.map((char, idx) => (
                    <li key={idx} className="flex gap-2 text-[11px] leading-relaxed">
                      <span className="text-emerald-400 font-bold">▪</span>
                      <span>{char}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Biological score scales */}
              <div className="grid grid-cols-2 gap-3 pt-1">
                <div className="bg-[#090b11]/60 p-2 rounded-xl border border-[#161c29] text-center">
                  <div className="text-[10px] text-slate-500 uppercase tracking-tight">Bio-Plausibility</div>
                  <div className="text-lg font-bold text-emerald-400 font-mono mt-0.5">{activeModel.bioPlausibility}/10</div>
                  <div className="w-full bg-[#181e2e] h-1 rounded-full mt-1 overflow-hidden">
                    <div 
                      className="bg-gradient-to-r from-emerald-600 to-emerald-400 h-full rounded-full"
                      style={{ width: `${activeModel.bioPlausibility * 10}%` }}
                    />
                  </div>
                </div>

                <div className="bg-[#090b11]/60 p-2 rounded-xl border border-[#161c29] text-center">
                  <div className="text-[10px] text-slate-500 uppercase tracking-tight">Energy Savings</div>
                  <div className="text-lg font-bold text-blue-400 font-mono mt-0.5">{activeModel.energyEfficiency}x</div>
                  <div className="w-full bg-[#181e2e] h-1 rounded-full mt-1 overflow-hidden">
                    <div 
                      className="bg-gradient-to-r from-blue-600 to-blue-400 h-full rounded-full"
                      style={{ width: `${activeModel.energyEfficiency * 10}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* SNN Neuromorphic Parameters Settings Card */}
          <div className="bg-[#0b0e17] border border-[#1b2234] rounded-2xl p-4 shadow-xl flex flex-col gap-3">
            <div className="flex items-center justify-between border-b border-[#1b2234] pb-2">
              <h2 className="text-sm font-semibold tracking-wide uppercase text-slate-400 flex items-center gap-2">
                <Sliders className="w-4 h-4 text-emerald-400" /> SNN Membrane Settings
              </h2>
              <button 
                onClick={handleResetParams}
                className="text-[11px] text-slate-500 hover:text-emerald-400 transition flex items-center gap-1 font-mono"
              >
                <RefreshCw className="w-3 h-3" /> RESET TO DEFAULT
              </button>
            </div>

            <div className="space-y-4 pt-1">
              {/* V_th Threshold */}
              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-300 font-mono flex items-center gap-1">
                    Membrane Threshold (V_th)
                  </span>
                  <span className="text-emerald-400 font-bold font-mono">{currentParams.threshold.toFixed(2)}</span>
                </div>
                <input 
                  type="range"
                  min="0.1"
                  max="2.5"
                  step="0.05"
                  value={currentParams.threshold}
                  onChange={(e) => {
                    const newVal = parseFloat(e.target.value);
                    setModelParams(prev => ({
                      ...prev,
                      [selectedModelId]: { ...prev[selectedModelId], threshold: newVal }
                    }));
                  }}
                  className="w-full h-1 bg-[#1b2234] rounded-lg appearance-none cursor-pointer accent-emerald-400"
                />
                <p className="text-[10px] text-slate-500">Higher values reduce spike emission rate and save extra processing energy, but might degrade high-level sequence context.</p>
              </div>

              {/* Decay constant */}
              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-300 font-mono flex items-center gap-1">
                    Temporal Decay Factor (tau)
                  </span>
                  <span className="text-blue-400 font-bold font-mono">{currentParams.decay.toFixed(2)}</span>
                </div>
                <input 
                  type="range"
                  min="0.4"
                  max="0.99"
                  step="0.01"
                  value={currentParams.decay}
                  onChange={(e) => {
                    const newVal = parseFloat(e.target.value);
                    setModelParams(prev => ({
                      ...prev,
                      [selectedModelId]: { ...prev[selectedModelId], decay: newVal }
                    }));
                  }}
                  className="w-full h-1 bg-[#1b2234] rounded-lg appearance-none cursor-pointer accent-blue-400"
                />
                <p className="text-[10px] text-slate-500">Determines how quickly the accumulated charge values decay. Lower values lead to faster memory loss between tokens.</p>
              </div>

              {/* Leakage constant */}
              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-300 font-mono flex items-center gap-1">
                    Leakage Constant (L)
                  </span>
                  <span className="text-indigo-400 font-bold font-mono">{currentParams.leak.toFixed(3)}</span>
                </div>
                <input 
                  type="range"
                  min="0.001"
                  max="0.5"
                  step="0.005"
                  value={currentParams.leak}
                  onChange={(e) => {
                    const newVal = parseFloat(e.target.value);
                    setModelParams(prev => ({
                      ...prev,
                      [selectedModelId]: { ...prev[selectedModelId], leak: newVal }
                    }));
                  }}
                  className="w-full h-1 bg-[#1b2234] rounded-lg appearance-none cursor-pointer accent-indigo-400"
                />
                <p className="text-[10px] text-slate-500">Constant electrical leakage of active neuron somatic potential. Keeps the spiking network stable and self-regulated.</p>
              </div>
            </div>
          </div>

        </section>

        {/* Right Side: Tabbed Interface for Playground, Network visualizer, API, Comparison */}
        <section className="lg:col-span-8 flex flex-col gap-4">
          
          {/* Tabs header */}
          <div className="bg-[#0b0e17] border border-[#1b2234] rounded-2xl p-1.5 flex gap-1">
            <button
              onClick={() => setActiveTab("playground")}
              className={`flex-1 py-2.5 rounded-xl text-xs font-semibold tracking-wide transition flex items-center justify-center gap-2 ${
                activeTab === "playground"
                  ? "bg-[#181d2d] text-white border border-[#2b3552] shadow"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <Sparkles className="w-4 h-4 text-emerald-400" /> Model Playground
            </button>

            <button
              onClick={() => setActiveTab("network-viz")}
              className={`flex-1 py-2.5 rounded-xl text-xs font-semibold tracking-wide transition flex items-center justify-center gap-2 ${
                activeTab === "network-viz"
                  ? "bg-[#181d2d] text-white border border-[#2b3552] shadow"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <Activity className="w-4 h-4 text-blue-400" /> Live Spikes Network
            </button>

            <button
              onClick={() => setActiveTab("api-gateway")}
              className={`flex-1 py-2.5 rounded-xl text-xs font-semibold tracking-wide transition flex items-center justify-center gap-2 ${
                activeTab === "api-gateway"
                  ? "bg-[#181d2d] text-white border border-[#2b3552] shadow"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <Code className="w-4 h-4 text-indigo-400" /> API Gateway
            </button>

            <button
              onClick={() => setActiveTab("characteristics")}
              className={`flex-1 py-2.5 rounded-xl text-xs font-semibold tracking-wide transition flex items-center justify-center gap-2 ${
                activeTab === "characteristics"
                  ? "bg-[#181d2d] text-white border border-[#2b3552] shadow"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <FileText className="w-4 h-4 text-amber-400" /> Characteristics Table
            </button>
          </div>

          {/* TAB 1: Playground / Interactive Chat with SNN simulated stats */}
          {activeTab === "playground" && (
            <div className="bg-[#0b0e17] border border-[#1b2234] rounded-3xl p-5 shadow-2xl flex flex-col h-[650px] relative">
              
              {/* Chat Header showing current model details */}
              <div className="flex items-center justify-between border-b border-[#1b2234] pb-3 mb-4">
                <div className="flex items-center gap-3">
                  <div className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse" />
                  <div>
                    <h3 className="font-bold text-slate-200 flex items-center gap-1.5">
                      {activeModel.name} <span className="text-xs text-slate-500 font-mono">({activeModel.parameters})</span>
                    </h3>
                    <p className="text-xs text-slate-400">Processing spikes synchronously via server-side gateway</p>
                  </div>
                </div>

                <div className="text-[11px] font-mono bg-[#111622] px-2.5 py-1 rounded border border-[#1d2639] text-slate-300">
                  V_th = {currentParams.threshold.toFixed(2)} | leak = {currentParams.leak.toFixed(3)}
                </div>
              </div>

              {/* Messages viewport */}
              <div className="flex-1 overflow-y-auto pr-1 space-y-4 custom-scrollbar">
                {(chats[selectedModelId] || []).map((msg) => {
                  const isUser = msg.role === "user";
                  return (
                    <div 
                      key={msg.id}
                      className={`flex flex-col gap-1.5 ${isUser ? "items-end" : "items-start"}`}
                    >
                      <div className={`text-[10px] text-slate-500 font-mono px-1 flex items-center gap-1`}>
                        <span>{isUser ? "You" : activeModel.name}</span>
                        <span>•</span>
                        <span>{msg.timestamp}</span>
                      </div>

                      <div className={`max-w-[85%] rounded-2xl px-4 py-3 leading-relaxed text-sm ${
                        isUser 
                          ? "bg-gradient-to-br from-emerald-600/90 to-teal-700/90 text-white rounded-tr-none shadow"
                          : "bg-[#111524]/90 border border-[#1d243c] text-slate-200 rounded-tl-none"
                      }`}>
                        <div className="whitespace-pre-wrap">{msg.content}</div>
                      </div>

                      {/* Render Neuromorphic Spiking Statistics if message has metrics */}
                      {msg.metrics && msg.metrics.spikeCount > 0 && (
                        <div className="w-[85%] bg-[#080b12] border border-[#1b2234] rounded-xl p-3 mt-1 text-xs text-slate-300 space-y-2">
                          <div className="flex items-center justify-between border-b border-[#141a29] pb-1.5">
                            <span className="text-emerald-400 font-mono font-bold flex items-center gap-1 text-[10px]">
                              <Zap className="w-3.5 h-3.5 fill-emerald-500/20" /> NEUROMORPHIC METRICS LOG
                            </span>
                            <span className="text-[10px] text-slate-500 font-mono">⚡ Ultra-Sparse Execution</span>
                          </div>

                          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-center">
                            <div className="bg-[#0e121e] p-1.5 rounded border border-[#1a2135]">
                              <div className="text-[10px] text-slate-500 font-mono">Spikes Count</div>
                              <div className="text-xs font-bold text-slate-200 font-mono mt-0.5">
                                {msg.metrics.spikeCount.toLocaleString()}
                              </div>
                            </div>

                            <div className="bg-[#0e121e] p-1.5 rounded border border-[#1a2135]">
                              <div className="text-[10px] text-slate-500 font-mono">Synaptic Ops (SOPs)</div>
                              <div className="text-xs font-bold text-emerald-400 font-mono mt-0.5">
                                {msg.metrics.synapticOps.toLocaleString()}
                              </div>
                            </div>

                            <div className="bg-[#0e121e] p-1.5 rounded border border-[#1a2135]">
                              <div className="text-[10px] text-slate-500 font-mono">Energy SNN</div>
                              <div className="text-xs font-bold text-blue-400 font-mono mt-0.5">
                                {msg.metrics.energyJoulesSNN.toExponential(3)} J
                              </div>
                            </div>

                            <div className="bg-[#0e121e] p-1.5 rounded border border-[#1a2135]">
                              <div className="text-[10px] text-slate-500 font-mono">Energy Savings</div>
                              <div className="text-xs font-bold text-emerald-400 font-mono mt-0.5">
                                {msg.metrics.energySavedPercent.toFixed(2)}%
                              </div>
                            </div>
                          </div>

                          {/* Beautiful energy comparison visualizer */}
                          <div className="bg-[#0a0d15] p-2 rounded-lg border border-[#121927]">
                            <div className="flex justify-between text-[10px] text-slate-400 mb-1">
                              <span>Standard ANN energy: {msg.metrics.energyJoulesANN.toExponential(2)} Joules (FLOPs-heavy)</span>
                              <span className="text-emerald-400 font-bold">-{msg.metrics.energySavedPercent.toFixed(1)}%</span>
                            </div>
                            <div className="w-full bg-[#1b2234] h-2 rounded overflow-hidden flex">
                              <div 
                                className="bg-gradient-to-r from-emerald-500 to-emerald-400 h-full"
                                style={{ width: `${100 - msg.metrics.energySavedPercent}%` }}
                              />
                              <div 
                                className="bg-[#121622] h-full"
                                style={{ width: `${msg.metrics.energySavedPercent}%` }}
                              />
                            </div>
                            <div className="flex justify-between text-[9px] text-slate-500 mt-1">
                              <span>SNN Active Potential (Spike sparse)</span>
                              <span>Average firing rate per token: {msg.metrics.averageFiringRate.toFixed(2)}%</span>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}

                {isLoading && (
                  <div className="flex flex-col gap-1.5 items-start">
                    <div className="text-[10px] text-slate-500 font-mono px-1">
                      {activeModel.name} is running spiking propagation...
                    </div>
                    <div className="bg-[#111524]/60 border border-[#1d243c] rounded-2xl rounded-tl-none px-4 py-3 text-sm flex items-center gap-2">
                      <div className="flex gap-1">
                        <span className="w-2 h-2 rounded-full bg-emerald-400 animate-bounce" style={{ animationDelay: '0ms' }} />
                        <span className="w-2 h-2 rounded-full bg-emerald-400 animate-bounce" style={{ animationDelay: '150ms' }} />
                        <span className="w-2 h-2 rounded-full bg-emerald-400 animate-bounce" style={{ animationDelay: '300ms' }} />
                      </div>
                      <span className="text-xs text-slate-400 font-mono">LIF integration on server...</span>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>

              {/* Chat Input form */}
              <form onSubmit={handleSendMessage} className="mt-4 flex gap-2">
                <input
                  type="text"
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  placeholder={`Send a prompt to test ${activeModel.name}... (e.g. "Draft an email explaining neural energy efficiency")`}
                  disabled={isLoading}
                  className="flex-1 bg-[#090c12] border border-[#1b2234] focus:border-emerald-500/50 rounded-xl px-4 py-3 text-sm text-slate-100 placeholder-slate-500 outline-none focus:ring-1 focus:ring-emerald-500/20 disabled:opacity-50 transition-all"
                />
                <button
                  type="submit"
                  disabled={isLoading || !inputMessage.trim()}
                  className="bg-gradient-to-r from-emerald-600 to-teal-700 hover:from-emerald-500 hover:to-teal-600 text-white font-semibold rounded-xl px-5 py-3 text-sm flex items-center gap-2 disabled:opacity-50 transition-all shadow-[0_0_15px_rgba(16,185,129,0.1)] active:scale-95"
                >
                  <span>Inference</span>
                  <Send className="w-4 h-4" />
                </button>
              </form>
            </div>
          )}

          {/* TAB 2: Live Spikes Network Visualizer */}
          {activeTab === "network-viz" && (
            <div className="bg-[#0b0e17] border border-[#1b2234] rounded-3xl p-5 shadow-2xl flex flex-col h-[650px]">
              
              <div className="flex items-center justify-between border-b border-[#1b2234] pb-3 mb-4">
                <div>
                  <h3 className="font-bold text-slate-200 flex items-center gap-2">
                    <Activity className="w-5 h-5 text-blue-400" /> Interactive SNN Spiking Canvas
                  </h3>
                  <p className="text-xs text-slate-400">Click input neurons to stimulate, or trigger a complete forward spike wave</p>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    disabled={isSimulatingSpike}
                    onClick={() => triggerSpikePropagation([0, 1, 2, 3].filter(() => Math.random() > 0.4))}
                    className="px-3 py-1.5 bg-[#12241e] border border-emerald-500/30 hover:border-emerald-500/50 text-emerald-400 hover:bg-[#152e25] text-xs font-mono rounded-lg transition-all flex items-center gap-1 disabled:opacity-40"
                  >
                    <Play className="w-3 h-3 fill-emerald-400/10" /> STIMULATE CHANNELS
                  </button>
                  <span className="text-[10px] bg-[#161a25] px-2 py-1 rounded text-slate-400 border border-[#232b3f] font-mono">
                    20 Neurons | {synapses.length} Plastic Synapses
                  </span>
                </div>
              </div>

              {/* Grid split: Canvas & Log output */}
              <div className="flex-1 grid grid-cols-1 md:grid-cols-12 gap-4 overflow-hidden">
                
                {/* SVG Network rendering */}
                <div className="md:col-span-8 bg-[#090b12] rounded-2xl border border-[#181f30] relative overflow-hidden flex items-center justify-center p-2">
                  
                  {/* Grid background representation */}
                  <div className="absolute inset-0 bg-[linear-gradient(to_right,#141724_1px,transparent_1px),linear-gradient(to_bottom,#141724_1px,transparent_1px)] bg-[size:24px_24px] opacity-20 pointer-events-none" />

                  <svg viewBox="0 0 600 500" className="w-full h-full max-h-[480px]">
                    {/* Render synapses connections first */}
                    {synapses.map((s, idx) => {
                      const fromNode = neurons.find(n => n.id === s.from);
                      const toNode = neurons.find(n => n.id === s.to);
                      if (!fromNode || !toNode) return null;

                      return (
                        <g key={`synapse-${idx}`}>
                          {/* Base synapse line */}
                          <line
                            x1={fromNode.x}
                            y1={fromNode.y}
                            x2={toNode.x}
                            y2={toNode.y}
                            stroke={s.isStimulated ? "#10b981" : "#1e293b"}
                            strokeWidth={s.isStimulated ? 2.5 : 1.2}
                            strokeOpacity={s.isStimulated ? 1.0 : 0.4}
                            className="transition-all duration-300"
                          />
                          {/* Moving spike pulse */}
                          {s.isStimulated && (
                            <circle r="4" fill="#34d399">
                              <animateMotion
                                dur="0.3s"
                                repeatCount="1"
                                path={`M ${fromNode.x} ${fromNode.y} L ${toNode.x} ${toNode.y}`}
                              />
                            </circle>
                          )}
                        </g>
                      );
                    })}

                    {/* Render Neurons on top */}
                    {neurons.map((n) => {
                      const potentialPercent = Math.min(100, (n.membranePotential / n.threshold) * 100);
                      const isInput = n.type === "input";
                      const isOutput = n.type === "output";
                      
                      let strokeColor = "#334155";
                      let fillColor = "#0f172a";
                      let shadowFilter = "";

                      if (n.isFiring) {
                        strokeColor = "#34d399";
                        fillColor = "#064e3b";
                        shadowFilter = "drop-shadow(0 0 8px rgba(52, 211, 153, 0.7))";
                      } else if (n.membranePotential > 0) {
                        strokeColor = isInput ? "#10b981" : "#3b82f6";
                        fillColor = "#111827";
                      }

                      return (
                        <g 
                          key={`neuron-${n.id}`} 
                          onClick={() => handleStimulateNeuron(n.id)}
                          className="cursor-pointer group"
                          style={{ filter: shadowFilter }}
                        >
                          {/* Click interaction ripple indicator */}
                          <circle
                            cx={n.x}
                            cy={n.y}
                            r="22"
                            fill="transparent"
                            stroke={n.isFiring ? "#10b981" : "transparent"}
                            strokeWidth="1"
                            className="opacity-20 group-hover:scale-125 transition-transform duration-300"
                          />

                          {/* Outer somatic boundary */}
                          <circle
                            cx={n.x}
                            cy={n.y}
                            r="16"
                            fill={fillColor}
                            stroke={strokeColor}
                            strokeWidth={n.isFiring ? 2.5 : 1.5}
                            className="transition-all duration-200"
                          />

                          {/* Inner membrane level filling visual */}
                          {potentialPercent > 0 && !n.isFiring && (
                            <circle
                              cx={n.x}
                              cy={n.y}
                              r="12"
                              fill="none"
                              stroke={isInput ? "#059669" : "#2563eb"}
                              strokeWidth="2.5"
                              strokeDasharray={`${(potentialPercent * 3.14 * 24) / 100} 100`}
                              className="transition-all duration-300 transform -rotate-90 origin-center"
                              style={{ transformOrigin: `${n.x}px ${n.y}px` }}
                            />
                          )}

                          {/* Numeric identity value or core label */}
                          <text
                            x={n.x}
                            y={n.y + 4}
                            textAnchor="middle"
                            fill={n.isFiring ? "#34d399" : "#94a3b8"}
                            fontSize="9"
                            fontFamily="monospace"
                            fontWeight="bold"
                            className="select-none pointer-events-none"
                          >
                            {isInput ? "IN" : isOutput ? "OUT" : `H${n.id - 3}`}
                          </text>

                          {/* Floating tooltip indicating active membrane potentials */}
                          <title>
                            {`Neuron ID: ${n.id}\nType: ${n.type}\nMembrane Potential: ${n.membranePotential.toFixed(2)}v\nThreshold: ${n.threshold.toFixed(2)}v`}
                          </title>
                        </g>
                      );
                    })}

                    {/* Decorative stage labels */}
                    <text x="80" y="25" fill="#64748b" fontSize="10" fontWeight="bold" textAnchor="middle" fontFamily="monospace">INPUT CHANNELS</text>
                    <text x="305" y="25" fill="#64748b" fontSize="10" fontWeight="bold" textAnchor="middle" fontFamily="monospace">HIDDEN INTER-SYNAPSE LAYERS</text>
                    <text x="530" y="25" fill="#64748b" fontSize="10" fontWeight="bold" textAnchor="middle" fontFamily="monospace">OUTPUT DISCHARGE</text>
                  </svg>
                </div>

                {/* Neuromorphic Log stream */}
                <div className="md:col-span-4 flex flex-col gap-3 h-full overflow-hidden">
                  <div className="bg-[#090b12] rounded-2xl border border-[#181f30] p-3 flex flex-col h-full">
                    <span className="text-[10px] font-mono text-blue-400 block border-b border-[#141b2a] pb-1.5 mb-2 uppercase tracking-widest font-bold">
                      SYSTEM MONITOR LOGS
                    </span>

                    <div className="flex-1 overflow-y-auto space-y-2 text-[11px] font-mono custom-scrollbar pr-1">
                      {activityLog.map((log, index) => (
                        <div key={index} className="leading-relaxed border-l-2 border-[#1c2437] pl-2 text-slate-400 hover:text-slate-200 transition-colors">
                          <span className="text-slate-600">[{new Date().toLocaleTimeString([], { hour12: false })}]</span> {log}
                        </div>
                      ))}
                    </div>

                    <div className="bg-[#0f1422] p-2 rounded-xl border border-[#1c2438] mt-2 space-y-1 text-[10px]">
                      <div className="flex justify-between">
                        <span className="text-slate-500">Threshold V_th:</span>
                        <span className="text-emerald-400 font-bold font-mono">{currentParams.threshold.toFixed(2)}v</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-500">Decay tau:</span>
                        <span className="text-blue-400 font-bold font-mono">{currentParams.decay.toFixed(2)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-500">Leak rate:</span>
                        <span className="text-indigo-400 font-bold font-mono">{currentParams.leak.toFixed(3)}</span>
                      </div>
                    </div>
                  </div>
                </div>

              </div>
            </div>
          )}

          {/* TAB 3: API Gateway & Sandbox Snippets (Supports the SciSciGPT feature set) */}
          {activeTab === "api-gateway" && (
            <div className="bg-[#0b0e17] border border-[#1b2234] rounded-3xl p-5 shadow-2xl flex flex-col h-[650px] overflow-hidden">
              
              <div className="flex items-center justify-between border-b border-[#1b2234] pb-3 mb-4 flex-shrink-0">
                <div>
                  <h3 className="font-bold text-slate-200 flex items-center gap-2">
                    <Terminal className="w-5 h-5 text-indigo-400" /> Spiking REST API Gateway
                  </h3>
                  <p className="text-xs text-slate-400">Integrate neuromorphic hardware endpoints into your pipelines</p>
                </div>

                <div className="flex items-center gap-2">
                  <span className="text-xs text-slate-500 font-mono">Compatibility:</span>
                  <span className="text-[11px] bg-emerald-950/80 text-emerald-400 border border-emerald-800/20 px-2 py-0.5 rounded font-mono font-bold">
                    OpenAI Chat Formats
                  </span>
                </div>
              </div>

              <div className="flex-1 grid grid-cols-1 md:grid-cols-2 gap-4 overflow-hidden">
                
                {/* Left Side: Snippets & Endpoint specs */}
                <div className="flex flex-col gap-3 overflow-y-auto pr-1 custom-scrollbar">
                  
                  {/* Endpoint specs */}
                  <div className="bg-[#090b12] rounded-2xl border border-[#181f30] p-4 space-y-3">
                    <span className="text-[10px] font-mono text-indigo-400 block uppercase tracking-widest font-bold">ACTIVE ROUTE</span>
                    
                    <div className="flex items-center gap-2">
                      <span className="bg-indigo-950 border border-indigo-800/30 text-indigo-400 px-2 py-1 rounded text-xs font-mono font-bold">
                        POST
                      </span>
                      <code className="text-xs text-slate-200 bg-[#121623] px-2 py-1 rounded flex-1 font-mono">
                        /api/inference
                      </code>
                    </div>

                    <p className="text-xs text-slate-400">
                      Submits conversational input histories to the simulated Leaky Integrate-and-Fire layers of the chosen open-source model.
                    </p>
                  </div>

                  {/* SDK Integration Code Blocks */}
                  <div className="bg-[#090b12] rounded-2xl border border-[#181f30] p-4 flex-1 flex flex-col">
                    <div className="flex justify-between items-center mb-3">
                      <span className="text-[10px] font-mono text-indigo-400 block uppercase tracking-widest font-bold">SDK SNIPPETS</span>
                      
                      <div className="flex bg-[#121622] rounded-lg p-0.5 border border-[#1c2438]">
                        {(["curl", "python", "javascript"] as const).map((lang) => (
                          <button
                            key={lang}
                            onClick={() => setApiSnippetLang(lang)}
                            className={`px-2 py-1 text-[10px] font-mono rounded-md transition ${
                              apiSnippetLang === lang 
                                ? "bg-[#1f283e] text-indigo-400 font-bold" 
                                : "text-slate-500 hover:text-slate-300"
                            }`}
                          >
                            {lang}
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* Code Container */}
                    <div className="relative bg-[#06080d] rounded-xl p-3 border border-[#121825] flex-1 font-mono text-xs text-slate-300 overflow-auto select-all max-h-[220px]">
                      <pre className="text-[11px] leading-relaxed whitespace-pre">{activeSnippet}</pre>
                      
                      <button
                        onClick={() => copyToClipboard(activeSnippet, "sdk-snippet")}
                        className="absolute top-2.5 right-2.5 p-1.5 bg-[#121825] hover:bg-[#1a2133] rounded-lg border border-[#232d46] transition-all text-slate-400 hover:text-slate-200"
                      >
                        {copiedTextId === "sdk-snippet" ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                      </button>
                    </div>
                  </div>

                </div>

                {/* Right Side: Sandbox Tester Console */}
                <div className="bg-[#090b12] rounded-2xl border border-[#181f30] p-4 flex flex-col h-full overflow-hidden">
                  <span className="text-[10px] font-mono text-indigo-400 block uppercase tracking-widest font-bold mb-3">SANDBOX API CONSOLE</span>
                  
                  {/* Console Payload edit */}
                  <div className="flex-1 flex flex-col gap-3 overflow-hidden">
                    <div className="flex-1 flex flex-col">
                      <div className="flex items-center justify-between text-[11px] text-slate-500 mb-1">
                        <span>Payload Request (JSON)</span>
                        <span className="font-mono text-slate-600">application/json</span>
                      </div>
                      <textarea
                        value={apiPayload}
                        onChange={(e) => setApiPayload(e.target.value)}
                        className="flex-1 bg-[#05060a] border border-[#1c2438] focus:border-indigo-500/50 rounded-xl p-3 font-mono text-[11px] text-indigo-300 outline-none resize-none"
                      />
                    </div>

                    {/* Run test button */}
                    <button
                      onClick={handleTestApi}
                      disabled={isApiLoading}
                      className="w-full py-2.5 bg-[#111322] border border-indigo-500/30 hover:border-indigo-500/50 hover:bg-[#14172a] text-indigo-400 font-mono text-xs rounded-xl flex items-center justify-center gap-2 transition active:scale-[0.98] disabled:opacity-50"
                    >
                      {isApiLoading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5 fill-indigo-400/10" />}
                      <span>SEND HTTP REQUEST TO SNN RECURSIVE LAYERS</span>
                    </button>

                    {/* Output Response block */}
                    <div className="flex-1 flex flex-col overflow-hidden">
                      <div className="flex items-center justify-between text-[11px] text-slate-500 mb-1">
                        <span>Response Body</span>
                        <span className={`font-mono text-xs ${apiResponse && !apiResponse.includes("error") ? "text-emerald-400" : "text-slate-600"}`}>
                          {apiResponse ? "HTTP 200 OK" : "Idle"}
                        </span>
                      </div>
                      <div className="flex-1 bg-[#05060a] border border-[#1c2438] rounded-xl p-3 font-mono text-[10px] text-slate-300 overflow-auto whitespace-pre">
                        {apiResponse || "// Push 'Send HTTP Request' to execute active network route."}
                      </div>
                    </div>
                  </div>

                </div>

              </div>
            </div>
          )}

          {/* TAB 4: In-depth characteristics & Comparisons */}
          {activeTab === "characteristics" && (
            <div className="bg-[#0b0e17] border border-[#1b2234] rounded-3xl p-5 shadow-2xl flex flex-col h-[650px] overflow-y-auto custom-scrollbar">
              
              <div className="border-b border-[#1b2234] pb-3 mb-5">
                <h3 className="font-bold text-slate-200 flex items-center gap-2">
                  <FileText className="w-5 h-5 text-amber-400" /> Comparative Spiking Model Matrix
                </h3>
                <p className="text-xs text-slate-400">Comprehensive characteristics comparison of top open-source SNN LLM frameworks</p>
              </div>

              {/* Matrix list layout */}
              <div className="space-y-4">
                {SNN_MODELS.map((m) => {
                  const isCurrent = m.id === selectedModelId;
                  return (
                    <div 
                      key={m.id}
                      className={`p-4 rounded-2xl border transition-all ${
                        isCurrent 
                          ? "bg-[#111727]/80 border-emerald-500/30 shadow-lg" 
                          : "bg-[#090b11]/50 border-[#1b2234]"
                      }`}
                    >
                      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2 pb-2.5 border-b border-[#182031] mb-3">
                        <div>
                          <span className="font-bold text-slate-200 text-sm flex items-center gap-1.5">
                            {m.name} 
                            {isCurrent && <span className="text-[10px] bg-[#12231e] text-emerald-400 border border-emerald-800/30 px-1.5 py-0.5 rounded font-mono font-bold">ACTIVE SELECTION</span>}
                          </span>
                          <span className="text-xs text-slate-500 font-mono">Weights: {m.parameters} | {m.author}</span>
                        </div>
                        <span className="px-2 py-0.5 bg-[#121622] rounded text-xs text-slate-400 font-mono border border-[#1e273f]">
                          {m.type}
                        </span>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-12 gap-4 text-xs">
                        {/* Summary description */}
                        <div className="md:col-span-5 text-slate-400 leading-relaxed">
                          <p>{m.description}</p>
                          <div className="mt-3 flex gap-2">
                            <span className="bg-[#141b2a] px-2 py-1 rounded text-slate-400 text-[10px] font-mono">Bio-Plausibility: {m.bioPlausibility}/10</span>
                            <span className="bg-[#141b2a] px-2 py-1 rounded text-emerald-400 text-[10px] font-mono">Efficiency: {m.energyEfficiency}x SOPs</span>
                          </div>
                        </div>

                        {/* Pros */}
                        <div className="md:col-span-4 bg-emerald-950/20 rounded-xl p-3 border border-emerald-900/10 flex flex-col gap-1.5">
                          <span className="text-[10px] font-mono text-emerald-400 font-bold uppercase tracking-wider flex items-center gap-1">
                            <CheckCircle2 className="w-3.5 h-3.5" /> Core Advantages
                          </span>
                          <ul className="space-y-1 text-[11px] text-slate-300 list-inside list-disc pl-1 leading-relaxed">
                            {m.pros.map((pro, index) => <li key={index}>{pro}</li>)}
                          </ul>
                        </div>

                        {/* Cons */}
                        <div className="md:col-span-3 bg-red-950/10 rounded-xl p-3 border border-red-900/10 flex flex-col gap-1.5">
                          <span className="text-[10px] font-mono text-red-400 font-bold uppercase tracking-wider flex items-center gap-1">
                            <XCircle className="w-3.5 h-3.5" /> Limitations
                          </span>
                          <ul className="space-y-1 text-[11px] text-slate-300 list-inside list-disc pl-1 leading-relaxed">
                            {m.cons.map((con, index) => <li key={index}>{con}</li>)}
                          </ul>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>

            </div>
          )}

        </section>

      </main>

      {/* Footer / Info Panel */}
      <footer className="border-t border-[#1b2234] bg-[#080a10] py-4 px-4 text-center mt-auto flex-shrink-0">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-slate-500 font-mono">
          <p>© 2026 Spiking LLM Hub. Inspired by SciSciGPT & open-source neuromorphic AI research communities.</p>
          <div className="flex gap-4">
            <a href="https://github.com/ridgerchu/spikegpt" target="_blank" rel="noreferrer" className="hover:text-emerald-400 transition">SpikeGPT Github</a>
            <span>•</span>
            <a href="https://arxiv.org/abs/2302.13941" target="_blank" rel="noreferrer" className="hover:text-emerald-400 transition">RWKV SNN paper</a>
          </div>
        </div>
      </footer>

    </div>
  );
}

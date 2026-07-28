import React, { useState, useEffect, useRef, useMemo } from 'react';
import { 
  Brain, 
  Cpu, 
  Zap, 
  Sparkles, 
  Code, 
  Send, 
  Terminal, 
  ExternalLink, 
  Info, 
  Activity, 
  FileText, 
  CheckCircle2, 
  XCircle, 
  Copy, 
  Check, 
  RefreshCw, 
  Sliders, 
  ShieldAlert,
  Square,
  Download,
  FileJson,
  Trash2,
  Play
} from 'lucide-react';
import { SNN_MODELS, STATUS_LABELS, LINKS_VERIFIED_ON } from './data';
import { Markdown } from './Markdown';
import { readSSE } from './sse';
import {
  loadChats,
  saveChats,
  loadParams,
  saveParams,
  clearStoredData,
  chatToMarkdown,
  chatToJSON,
  downloadTextFile,
  isStorageAvailable,
} from './storage';
import {
  buildNeurons,
  buildSynapses,
  simulatePropagation,
  randomInputChannels,
  FRAME_MS,
  TOTAL_NEURONS,
} from './snnSimulation';
import { SNNModel, ChatMessage, SNNInferenceMetrics, NeuronState, SynapseState } from './types';

// The gateway is served from the same origin as this app, so snippets and the
// sandbox always point at the deployment the user is actually looking at
// rather than a hardcoded domain that may not exist.
const API_ORIGIN = typeof window !== "undefined" ? window.location.origin : "";

// Keep in sync with MAX_MESSAGES on the server. Trimming client-side avoids a
// pointless 400 once a conversation gets long.
const MAX_HISTORY_MESSAGES = 40;

// The seeded greeting is UI copy, not model output. It must never be sent
// upstream: it would open the history with an assistant turn and skew replies.
const SEED_MESSAGE_ID = "welcome";

type TabId = "playground" | "network-viz" | "api-gateway" | "characteristics";

const TABS: { id: TabId; label: string; icon: typeof Sparkles; color: string }[] = [
  { id: "playground", label: "Model Playground", icon: Sparkles, color: "text-emerald-400" },
  { id: "network-viz", label: "Live Spikes Network", icon: Activity, color: "text-blue-400" },
  { id: "api-gateway", label: "API Gateway", icon: Code, color: "text-indigo-400" },
  { id: "characteristics", label: "Characteristics Table", icon: FileText, color: "text-amber-400" },
];

type ServerStatus = "checking" | "online" | "degraded" | "offline";

/** Timestamp is captured when the entry is created, not when it is rendered. */
interface LogEntry {
  id: string;
  time: string;
  text: string;
}

function seedMessage(model: SNNModel): ChatMessage {
  return {
    id: SEED_MESSAGE_ID,
    role: "assistant",
    content: `Welcome to the **${model.name}** playground.

This is a **simulator**: a general-purpose model answers in character as this architecture, and the metrics under each reply are calculated from a formula rather than measured on hardware.

Try adjusting the membrane threshold (\`V_th\`), decay or leak in the panel on the left and watch how the estimated firing rate and energy figures respond.`,
    timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
  };
}

let logSequence = 0;
function makeLogEntry(text: string): LogEntry {
  return {
    id: `log-${Date.now()}-${logSequence++}`,
    time: new Date().toLocaleTimeString([], { hour12: false }),
    text,
  };
}

/** Colour treatment for a model's release status. */
const STATUS_STYLES: Record<string, string> = {
  "weights-released": "bg-emerald-950/60 text-emerald-400 border-emerald-800/40",
  "code-only": "bg-blue-950/60 text-blue-400 border-blue-800/40",
  method: "bg-amber-950/50 text-amber-400 border-amber-800/40",
};

interface RequestFailure {
  message: string;
  retryAfterSeconds?: number;
}

/** Turn a failed Response into a message worth showing a human. */
async function describeFailure(response: Response): Promise<RequestFailure> {
  let serverMessage = "";
  try {
    const body = await response.json();
    if (body && typeof body.error === "string") serverMessage = body.error;
  } catch {
    /* body was not JSON; fall through to the status-based text below */
  }

  switch (response.status) {
    case 400:
      return { message: serverMessage || "The request was rejected as invalid." };
    case 401:
    case 403:
      return {
        message:
          serverMessage ||
          "This endpoint refused the request. The public API requires a key; the playground endpoint only accepts same-origin calls.",
      };
    case 413:
      return { message: serverMessage || "That message is too large. Try something shorter." };
    case 429: {
      const header = response.headers.get("Retry-After");
      const retryAfterSeconds = header ? Number(header) : undefined;
      return {
        message: serverMessage || "Rate limit reached.",
        retryAfterSeconds:
          Number.isFinite(retryAfterSeconds) && retryAfterSeconds! > 0
            ? retryAfterSeconds
            : 60,
      };
    }
    case 503:
      return {
        message:
          serverMessage || "The inference service is not available on this deployment.",
      };
    default:
      return {
        message:
          serverMessage || `The server responded with HTTP ${response.status}.`,
      };
  }
}

export default function App() {
  // Current active model and tab
  const [selectedModelId, setSelectedModelId] = useState<string>("spikegpt");
  const [activeTab, setActiveTab] = useState<TabId>("playground");
  const tabRefs = useRef<Partial<Record<TabId, HTMLButtonElement | null>>>({});
  
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
    // Saved values override defaults, but unknown models still get defaults.
    return { ...initial, ...(loadParams() || {}) };
  });

  const activeModel = SNN_MODELS.find(m => m.id === selectedModelId) || SNN_MODELS[0];
  const currentParams = modelParams[selectedModelId] || { threshold: 1.0, decay: 0.8, leak: 0.1 };


  // Chats history mapped by model ID to preserve conversation across model switching
  const [chats, setChats] = useState<Record<string, ChatMessage[]>>(() => {
    const stored = loadChats();
    const initial: Record<string, ChatMessage[]> = {};
    SNN_MODELS.forEach(m => {
      initial[m.id] = stored?.[m.id]?.length ? stored[m.id] : [seedMessage(m)];
    });
    return initial;
  });

  const [inputMessage, setInputMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [copiedTextId, setCopiedTextId] = useState<string | null>(null);

  // Text accumulated from the current SSE stream. Held separately from `chats`
  // so that a partial reply re-renders cheaply and is committed exactly once.
  const [streamingText, setStreamingText] = useState("");
  const [streamingForModel, setStreamingForModel] = useState<string | null>(null);

  // Surfaced when the server returns 429 so the user sees a real cooldown
  // instead of a generic failure bubble.
  const [cooldownSeconds, setCooldownSeconds] = useState(0);

  // Real health, polled from /healthz, rather than a hardcoded "ONLINE" badge.
  const [serverStatus, setServerStatus] = useState<ServerStatus>("checking");
  const [generationMode, setGenerationMode] = useState<"live" | "simulated" | null>(null);

  // API Sandbox tab state
  const [apiMethod, setApiMethod] = useState<string>("POST");
  const [apiEndpoint, setApiEndpoint] = useState<string>("/api/inference");
  const [apiPayload, setApiPayload] = useState<string>("");
  const [apiResponse, setApiResponse] = useState<string>("");
  const [apiStatus, setApiStatus] = useState<number | null>(null);
  const [isApiLoading, setIsApiLoading] = useState(false);
  const [apiSnippetLang, setApiSnippetLang] = useState<"curl" | "python" | "javascript">("curl");

  // SNN Network Visualizer states
  // Built once via lazy initialisers. The previous mount effect regenerated the
  // random synapse graph on every StrictMode double-invoke.
  const [neurons, setNeurons] = useState<NeuronState[]>(() =>
    buildNeurons(SNN_MODELS[0].defaultThreshold),
  );
  const [synapses, setSynapses] = useState<SynapseState[]>(() => buildSynapses());
  const [isSimulatingSpike, setIsSimulatingSpike] = useState(false);
  const [activityLog, setActivityLog] = useState<LogEntry[]>(() => [
    makeLogEntry("SNN network initialised. Idle state stable."),
  ]);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  // In-flight requests are cancelled on unmount and when the user switches
  // models, so a slow reply can never land in the wrong conversation or call
  // setState on an unmounted component.
  const inferenceAbortRef = useRef<AbortController | null>(null);
  const sandboxAbortRef = useRef<AbortController | null>(null);

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
  }, [chats, isLoading, streamingText]);

  // Poll real server health instead of asserting "ONLINE" unconditionally.
  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();

    const check = async () => {
      try {
        const res = await fetch('/healthz', { signal: controller.signal, cache: 'no-store' });
        if (cancelled) return;
        if (!res.ok) {
          setServerStatus("degraded");
          return;
        }
        const body = await res.json();
        if (cancelled) return;
        setServerStatus("online");
        setGenerationMode(body?.generation === 'live' ? 'live' : 'simulated');
      } catch {
        if (!cancelled) setServerStatus("offline");
      }
    };

    check();
    const interval = setInterval(check, 60_000);
    return () => {
      cancelled = true;
      controller.abort();
      clearInterval(interval);
    };
  }, []);

  // Count down the 429 cooldown so the input can re-enable itself.
  useEffect(() => {
    if (cooldownSeconds <= 0) return;
    const timer = setTimeout(() => setCooldownSeconds((s) => Math.max(0, s - 1)), 1000);
    return () => clearTimeout(timer);
  }, [cooldownSeconds]);

  // Cancel any in-flight inference when the user switches models.
  useEffect(() => {
    return () => {
      inferenceAbortRef.current?.abort();
      inferenceAbortRef.current = null;
    };
  }, [selectedModelId]);

  // Persist chats, debounced so a streaming reply does not write on every token.
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    saveTimerRef.current = setTimeout(() => saveChats(chats), 600);
    return () => {
      if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    };
  }, [chats]);

  useEffect(() => {
    saveParams(modelParams);
  }, [modelParams]);

  // Cancel everything on unmount.
  useEffect(() => {
    return () => {
      inferenceAbortRef.current?.abort();
      sandboxAbortRef.current?.abort();
    };
  }, []);

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

  // Refs mirror live state so the replayer can read current values without
  // being re-created on every render.
  const neuronsRef = useRef(neurons);
  const synapsesRef = useRef(synapses);
  const paramsRef = useRef(currentParams);
  useEffect(() => { neuronsRef.current = neurons; }, [neurons]);
  useEffect(() => { synapsesRef.current = synapses; }, [synapses]);
  useEffect(() => { paramsRef.current = currentParams; }, [currentParams]);

  // The SVG redraws on every animation frame; a linear scan per synapse made
  // that O(synapses x neurons). One map per neuron update instead.
  const neuronById = useMemo(() => {
    const map = new Map<number, NeuronState>();
    for (const n of neurons) map.set(n.id, n);
    return map;
  }, [neurons]);

  // Increments on every run so a superseded animation stops applying frames.
  const runTokenRef = useRef(0);
  const frameTimersRef = useRef<ReturnType<typeof setTimeout>[]>([]);

  const clearFrameTimers = () => {
    frameTimersRef.current.forEach(clearTimeout);
    frameTimersRef.current = [];
  };

  useEffect(() => clearFrameTimers, []);

  /**
   * Computes the propagation synchronously, then replays the resulting frames.
   * Starting a new run invalidates any run still in flight, so the overlapping
   * calls made when a message is sent and again when it completes no longer
   * interleave and corrupt each other's state.
   */
  const triggerSpikePropagation = (startNodes: number[]) => {
    if (startNodes.length === 0) return;

    const token = ++runTokenRef.current;
    clearFrameTimers();

    const frames = simulatePropagation(
      neuronsRef.current,
      synapsesRef.current,
      startNodes,
      paramsRef.current,
    );

    setIsSimulatingSpike(true);

    frames.forEach((frame, index) => {
      const timer = setTimeout(() => {
        if (runTokenRef.current !== token) return;

        setNeurons(frame.neurons);
        setSynapses(frame.synapses);
        if (frame.log) {
          setActivityLog(prev => [makeLogEntry(frame.log as string), ...prev].slice(0, 16));
        }
        if (index === frames.length - 1) {
          setIsSimulatingSpike(false);
        }
      }, index * FRAME_MS);

      frameTimersRef.current.push(timer);
    });
  };

  const handleStimulateNeuron = (neuronId: number) => {
    if (isSimulatingSpike) return;
    triggerSpikePropagation([neuronId]);
  };

  // Run chat message inference
  const handleSendMessage = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!inputMessage.trim() || isLoading || cooldownSeconds > 0) return;

    const userText = inputMessage.trim();
    const modelIdAtSend = selectedModelId;
    setInputMessage("");
    setIsLoading(true);

    const userMessage: ChatMessage = {
      id: `msg-user-${Date.now()}`,
      role: "user",
      content: userText,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    // Update active model's chat
    const updatedModelChat = [...(chats[modelIdAtSend] || []), userMessage];
    setChats(prev => ({
      ...prev,
      [modelIdAtSend]: updatedModelChat
    }));

    // Trigger visual simulation on input start
    triggerSpikePropagation(randomInputChannels());

    // Send only what the model needs: role and content, no seeded greeting, no
    // timestamps or metrics, and no more history than the server accepts.
    const outboundMessages = updatedModelChat
      .filter(m => m.id !== SEED_MESSAGE_ID)
      .slice(-MAX_HISTORY_MESSAGES)
      .map(m => ({ role: m.role, content: m.content }));

    inferenceAbortRef.current?.abort();
    const controller = new AbortController();
    inferenceAbortRef.current = controller;

    setStreamingText("");
    setStreamingForModel(modelIdAtSend);

    let accumulated = "";
    let metrics: SNNInferenceMetrics | undefined;

    const commit = (content: string, suffix = "") => {
      if (!content.trim() && !suffix) return;
      const assistantMessage: ChatMessage = {
        id: `msg-ai-${Date.now()}`,
        role: "assistant",
        content: content + suffix,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        metrics
      };
      setChats(prev => ({
        ...prev,
        [modelIdAtSend]: [...(prev[modelIdAtSend] || []), assistantMessage]
      }));
    };

    try {
      const response = await fetch('/api/inference', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify({
          modelId: modelIdAtSend,
          messages: outboundMessages,
          threshold: currentParams.threshold,
          decay: currentParams.decay,
          leak: currentParams.leak,
          stream: true
        })
      });

      if (!response.ok) {
        const failure = await describeFailure(response);
        if (failure.retryAfterSeconds) {
          setCooldownSeconds(failure.retryAfterSeconds);
        }
        const err = new Error(failure.message);
        (err as any).isRateLimit = Boolean(failure.retryAfterSeconds);
        throw err;
      }

      for await (const message of readSSE(response)) {
        if (message.event === 'delta' && typeof message.data?.text === 'string') {
          accumulated += message.data.text;
          setStreamingText(accumulated);
        } else if (message.event === 'metrics') {
          metrics = message.data as SNNInferenceMetrics;
        } else if (message.event === 'error') {
          throw new Error(message.data?.error || "Generation failed on the server.");
        }
      }

      commit(accumulated);

      // Output discharge visual, once the reply is actually complete.
      triggerSpikePropagation(randomInputChannels());

    } catch (err: any) {
      if (err?.name === 'AbortError') {
        // A deliberate stop. Keep whatever arrived rather than discarding it.
        commit(accumulated, accumulated ? "\n\n_[stopped]_" : "");
        return;
      }

      console.error(err);
      // Partial output is still worth keeping alongside the error note.
      if (accumulated) commit(accumulated, "\n\n_[interrupted]_");

      const errorMsg: ChatMessage = {
        id: `msg-error-${Date.now()}`,
        role: "assistant",
        content: err?.isRateLimit
          ? `${err.message} The playground limits how many prompts each visitor can send per minute so the shared inference budget stays available for everyone.`
          : `${err?.message || "Unable to reach the spiking inference server."} You can try again in a moment.`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setChats(prev => ({
        ...prev,
        [modelIdAtSend]: [...(prev[modelIdAtSend] || []), errorMsg]
      }));
    } finally {
      if (inferenceAbortRef.current === controller) {
        inferenceAbortRef.current = null;
      }
      setStreamingText("");
      setStreamingForModel(null);
      setIsLoading(false);
    }
  };

  // Cancel an in-flight generation, keeping the partial text.
  const handleStopGeneration = () => {
    inferenceAbortRef.current?.abort();
  };

  // Reset the active model's conversation back to its seed message.
  const handleClearChat = () => {
    inferenceAbortRef.current?.abort();
    setChats(prev => ({ ...prev, [selectedModelId]: [seedMessage(activeModel)] }));
  };

  const handleExport = (format: "markdown" | "json") => {
    const messages = (chats[selectedModelId] || []).filter(m => m.id !== SEED_MESSAGE_ID);
    if (messages.length === 0) return;

    const stamp = new Date().toISOString().slice(0, 10);
    if (format === "markdown") {
      downloadTextFile(
        `${selectedModelId}-transcript-${stamp}.md`,
        chatToMarkdown(activeModel, messages, currentParams),
        "text/markdown",
      );
    } else {
      downloadTextFile(
        `${selectedModelId}-transcript-${stamp}.json`,
        chatToJSON(activeModel, messages, currentParams),
        "application/json",
      );
    }
  };

  // Run interactive API Tester Sandbox
  const handleTestApi = async () => {
    if (isApiLoading) return;
    setIsApiLoading(true);
    setApiResponse("");
    setApiStatus(null);

    sandboxAbortRef.current?.abort();
    const controller = new AbortController();
    sandboxAbortRef.current = controller;

    try {
      let parsedPayload = {};
      try {
        parsedPayload = JSON.parse(apiPayload);
      } catch {
        throw new Error("Invalid JSON formatting inside payload sandbox.");
      }

      const res = await fetch(apiEndpoint, {
        method: apiMethod,
        headers: {
          'Content-Type': 'application/json'
        },
        signal: controller.signal,
        body: JSON.stringify(parsedPayload)
      });

      // Report the status the server actually returned rather than assuming 200.
      setApiStatus(res.status);

      const rawBody = await res.text();
      try {
        setApiResponse(JSON.stringify(JSON.parse(rawBody), null, 2));
      } catch {
        setApiResponse(rawBody || "// Empty response body");
      }

      if (res.status === 429) {
        const header = res.headers.get("Retry-After");
        const retryAfter = header ? Number(header) : 60;
        setCooldownSeconds(Number.isFinite(retryAfter) && retryAfter > 0 ? retryAfter : 60);
      }

      if (res.ok) {
        triggerSpikePropagation([1, 2]);
      }

    } catch (err: any) {
      if (err?.name === 'AbortError') return;
      setApiStatus(0);
      setApiResponse(JSON.stringify({ error: err?.message || "Failed API call." }, null, 2));
    } finally {
      if (sandboxAbortRef.current === controller) {
        sandboxAbortRef.current = null;
      }
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
  const copyToClipboard = async (text: string, id: string) => {
    try {
      // navigator.clipboard is undefined outside secure contexts.
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(text);
      } else {
        const scratch = document.createElement("textarea");
        scratch.value = text;
        scratch.setAttribute("readonly", "");
        scratch.style.position = "fixed";
        scratch.style.opacity = "0";
        document.body.appendChild(scratch);
        scratch.select();
        document.execCommand("copy");
        document.body.removeChild(scratch);
      }
      setCopiedTextId(id);
      setTimeout(() => setCopiedTextId(null), 2000);
    } catch (err) {
      console.error("Clipboard write failed:", err);
      setCopiedTextId(`${id}-failed`);
      setTimeout(() => setCopiedTextId(null), 2000);
    }
  };

  // Generate code snippet texts dynamically based on current model and parameters.
  // These target the public /v1 gateway on this deployment's own origin, so what
  // the user copies is exactly what they can run.
  const getApiSnippets = () => {
    const endpoint = `${API_ORIGIN}/v1/chat/completions`;

    const curl = `curl -X POST "${endpoint}" \\
  -H "Authorization: Bearer $SPIKING_HUB_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "${selectedModelId}",
    "messages": [
      {"role": "user", "content": "What is SpikeGPT?"}
    ],
    "threshold": ${currentParams.threshold},
    "decay": ${currentParams.decay},
    "leak": ${currentParams.leak}
  }'`;

    const python = `import os
import requests

url = "${endpoint}"
headers = {
    "Authorization": f"Bearer {os.environ['SPIKING_HUB_API_KEY']}",
    "Content-Type": "application/json",
}
payload = {
    "model": "${selectedModelId}",
    "messages": [
        {"role": "user", "content": "What is SpikeGPT?"}
    ],
    "threshold": ${currentParams.threshold},
    "decay": ${currentParams.decay},
    "leak": ${currentParams.leak},
}

response = requests.post(url, json=payload, headers=headers, timeout=60)
response.raise_for_status()
print(response.json())`;

    const js = `const response = await fetch("${endpoint}", {
  method: "POST",
  headers: {
    "Authorization": \`Bearer \${process.env.SPIKING_HUB_API_KEY}\`,
    "Content-Type": "application/json"
  },
  body: JSON.stringify({
    model: "${selectedModelId}",
    messages: [
      { role: "user", content: "What is SpikeGPT?" }
    ],
    threshold: ${currentParams.threshold},
    decay: ${currentParams.decay},
    leak: ${currentParams.leak}
  })
});

if (!response.ok) {
  throw new Error(\`Gateway returned \${response.status}\`);
}

const data = await response.json();
console.log(data);`;

    return { curl, python, javascript: js };
  };

  // Build once per render instead of three times.
  const activeSnippet = getApiSnippets()[apiSnippetLang];

  // Bring the selected model into view when the comparison tab opens, so the
  // highlighted card is not stranded below the fold.
  const matrixCardRefs = useRef<Record<string, HTMLDivElement | null>>({});
  useEffect(() => {
    if (activeTab !== "characteristics") return;
    const card = matrixCardRefs.current[selectedModelId];
    if (!card) return;
    const timer = setTimeout(() => {
      card.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 60);
    return () => clearTimeout(timer);
  }, [activeTab, selectedModelId]);

  // Roving focus: arrow keys move between tabs, Home/End jump to the ends.
  const handleTabKeyDown = (e: React.KeyboardEvent) => {
    const index = TABS.findIndex(t => t.id === activeTab);
    let next = index;
    if (e.key === "ArrowRight") next = (index + 1) % TABS.length;
    else if (e.key === "ArrowLeft") next = (index - 1 + TABS.length) % TABS.length;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = TABS.length - 1;
    else return;

    e.preventDefault();
    const target = TABS[next].id;
    setActiveTab(target);
    tabRefs.current[target]?.focus();
  };

  const storageAvailable = useMemo(() => isStorageAvailable(), []);
  const hasTranscript = (chats[selectedModelId] || []).some(m => m.id !== SEED_MESSAGE_ID);

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
            {(() => {
              const presentation = {
                checking: { dot: "bg-slate-500", text: "text-slate-400", label: "CHECKING SERVER…", ping: false },
                online: { dot: "bg-emerald-500", text: "text-emerald-400", label: generationMode === "simulated" ? "SERVER ONLINE · SIMULATED OUTPUT" : "SPIKE INFERENCE SERVER ONLINE", ping: true },
                degraded: { dot: "bg-amber-500", text: "text-amber-400", label: "SERVER DEGRADED", ping: false },
                offline: { dot: "bg-red-500", text: "text-red-400", label: "SERVER UNREACHABLE", ping: false },
              }[serverStatus];

              return (
                <>
                  <span className="flex h-2.5 w-2.5 relative" aria-hidden="true">
                    {presentation.ping && (
                      <span className={`animate-ping absolute inline-flex h-full w-full rounded-full ${presentation.dot} opacity-75`}></span>
                    )}
                    <span className={`relative inline-flex rounded-full h-2.5 w-2.5 ${presentation.dot}`}></span>
                  </span>
                  <span className={`text-xs font-mono ${presentation.text}`} role="status" aria-live="polite">
                    {presentation.label}
                  </span>
                </>
              );
            })()}
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

            <div role="radiogroup" aria-label="Choose a spiking language model" className="space-y-1.5 max-h-[360px] overflow-y-auto pr-1 custom-scrollbar">
              {SNN_MODELS.map((m) => {
                const isSelected = m.id === selectedModelId;
                return (
                  <button
                    key={m.id}
                    id={`model-select-${m.id}`}
                    type="button"
                    role="radio"
                    aria-checked={isSelected}
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
                      <span className="text-slate-500">{m.year}</span>
                    </div>

                    <div className="flex items-center gap-1.5 mt-0.5">
                      <span className={`text-[9px] px-1.5 py-0.5 rounded font-mono border ${STATUS_STYLES[m.status]}`}>
                        {STATUS_LABELS[m.status]}
                      </span>
                      {m.license && (
                        <span className="text-[9px] px-1.5 py-0.5 rounded font-mono bg-[#141b2a] text-slate-400 border border-[#1e273f]">
                          {m.license}
                        </span>
                      )}
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

            <div className="border-b border-[#1b2234] pb-2 flex items-center justify-between gap-2">
              <h3 className="text-sm font-semibold tracking-wide uppercase text-slate-400 flex items-center gap-2">
                <Info className="w-4 h-4 text-emerald-400" /> {activeModel.name} Bio-Specs
              </h3>
              <div className="flex items-center gap-2 shrink-0">
                {([
                  ["Code", activeModel.github],
                  ["Paper", activeModel.paper],
                  ["Weights", activeModel.huggingface],
                  ["Site", activeModel.homepage],
                ] as const)
                  .filter(([, href]) => Boolean(href))
                  .map(([label, href]) => (
                    <a
                      key={label}
                      href={href as string}
                      target="_blank"
                      rel="noreferrer noopener"
                      className="text-[11px] text-slate-400 hover:text-emerald-400 transition flex items-center gap-0.5"
                    >
                      {label} <ExternalLink className="w-3 h-3" />
                    </a>
                  ))}
              </div>
            </div>

            <div className="space-y-2.5 text-xs text-slate-300">
              <p className="text-[11px] text-slate-500">
                {activeModel.author}
                {activeModel.affiliation && <> · {activeModel.affiliation}</>}
              </p>

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

              {/* Editorial comparison ratings */}
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
                  <div className="text-[10px] text-slate-500 uppercase tracking-tight">Energy Efficiency</div>
                  <div className="text-lg font-bold text-blue-400 font-mono mt-0.5">{activeModel.energyEfficiency}/10</div>
                  <div className="w-full bg-[#181e2e] h-1 rounded-full mt-1 overflow-hidden">
                    <div 
                      className="bg-gradient-to-r from-blue-600 to-blue-400 h-full rounded-full"
                      style={{ width: `${activeModel.energyEfficiency * 10}%` }}
                    />
                  </div>
                </div>
              </div>

              {!storageAvailable && (
                <p className="text-[10px] text-amber-500/80 leading-relaxed">
                  Browser storage is unavailable, so this conversation will not be
                  saved when you reload.
                </p>
              )}

              <p className="text-[10px] text-slate-600 leading-relaxed">
                Both ratings are subjective editorial comparisons across the models listed
                here, not measurements. Links last verified {LINKS_VERIFIED_ON}.
              </p>
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
                  <label htmlFor="param-threshold" className="text-slate-300 font-mono flex items-center gap-1">
                    Membrane Threshold (V_th)
                  </label>
                  <span className="text-emerald-400 font-bold font-mono">{currentParams.threshold.toFixed(2)}</span>
                </div>
                <input 
                  type="range"
                  id="param-threshold"
                  aria-valuetext={`${currentParams.threshold.toFixed(2)} volts`}
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
                  <label htmlFor="param-decay" className="text-slate-300 font-mono flex items-center gap-1">
                    Temporal Decay Factor (tau)
                  </label>
                  <span className="text-blue-400 font-bold font-mono">{currentParams.decay.toFixed(2)}</span>
                </div>
                <input 
                  type="range"
                  id="param-decay"
                  aria-valuetext={`${currentParams.decay.toFixed(2)}`}
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
                  <label htmlFor="param-leak" className="text-slate-300 font-mono flex items-center gap-1">
                    Leakage Constant (L)
                  </label>
                  <span className="text-indigo-400 font-bold font-mono">{currentParams.leak.toFixed(3)}</span>
                </div>
                <input 
                  type="range"
                  id="param-leak"
                  aria-valuetext={`${currentParams.leak.toFixed(3)}`}
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
          <div
            role="tablist"
            aria-label="Workspace views"
            onKeyDown={handleTabKeyDown}
            className="bg-[#0b0e17] border border-[#1b2234] rounded-2xl p-1.5 flex gap-1 overflow-x-auto custom-scrollbar"
          >
            {TABS.map((tab) => {
              const isActive = activeTab === tab.id;
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  id={`tab-${tab.id}`}
                  role="tab"
                  type="button"
                  aria-selected={isActive}
                  aria-controls={`panel-${tab.id}`}
                  tabIndex={isActive ? 0 : -1}
                  ref={(el) => { tabRefs.current[tab.id] = el; }}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex-1 min-w-[52px] py-2.5 px-2 rounded-xl text-xs font-semibold tracking-wide transition flex items-center justify-center gap-2 whitespace-nowrap focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500/50 ${
                    isActive
                      ? "bg-[#181d2d] text-white border border-[#2b3552] shadow"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  <Icon className={`w-4 h-4 ${tab.color} shrink-0`} />
                  <span className="hidden md:inline">{tab.label}</span>
                </button>
              );
            })}
          </div>

          {/* TAB 1: Playground / Interactive Chat with SNN simulated stats */}
          {activeTab === "playground" && (
            <div role="tabpanel" id="panel-playground" aria-labelledby="tab-playground" tabIndex={0} className="bg-[#0b0e17] border border-[#1b2234] rounded-3xl p-5 shadow-2xl flex flex-col h-[min(650px,calc(100dvh-13rem))] min-h-[420px] relative">
              
              {/* Chat Header showing current model details */}
              <div className="flex items-center justify-between border-b border-[#1b2234] pb-3 mb-4">
                <div className="flex items-center gap-3">
                  <div className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse" />
                  <div>
                    <h3 className="font-bold text-slate-200 flex items-center gap-1.5">
                      {activeModel.name} <span className="text-xs text-slate-500 font-mono">({activeModel.parameters})</span>
                    </h3>
                    <p className="text-xs text-slate-400">Role-played response · simulated spiking metrics</p>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <div className="hidden sm:block text-[11px] font-mono bg-[#111622] px-2.5 py-1 rounded border border-[#1d2639] text-slate-300">
                    V_th = {currentParams.threshold.toFixed(2)} | leak = {currentParams.leak.toFixed(3)}
                  </div>

                  <button
                    type="button"
                    onClick={() => handleExport("markdown")}
                    disabled={!hasTranscript}
                    title="Download transcript as Markdown"
                    aria-label="Download transcript as Markdown"
                    className="p-1.5 rounded-lg border border-[#1d2639] bg-[#111622] text-slate-400 hover:text-emerald-400 hover:border-emerald-500/30 disabled:opacity-30 disabled:hover:text-slate-400 disabled:hover:border-[#1d2639] transition focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500/50"
                  >
                    <Download className="w-3.5 h-3.5" />
                  </button>

                  <button
                    type="button"
                    onClick={() => handleExport("json")}
                    disabled={!hasTranscript}
                    title="Download transcript as JSON"
                    aria-label="Download transcript as JSON"
                    className="p-1.5 rounded-lg border border-[#1d2639] bg-[#111622] text-slate-400 hover:text-emerald-400 hover:border-emerald-500/30 disabled:opacity-30 disabled:hover:text-slate-400 disabled:hover:border-[#1d2639] transition focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500/50"
                  >
                    <FileJson className="w-3.5 h-3.5" />
                  </button>

                  <button
                    type="button"
                    onClick={handleClearChat}
                    disabled={!hasTranscript}
                    title="Clear this conversation"
                    aria-label="Clear this conversation"
                    className="p-1.5 rounded-lg border border-[#1d2639] bg-[#111622] text-slate-400 hover:text-red-400 hover:border-red-500/30 disabled:opacity-30 disabled:hover:text-slate-400 disabled:hover:border-[#1d2639] transition focus:outline-none focus-visible:ring-2 focus-visible:ring-red-500/50"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>

              {/* What this playground actually does */}
              <div className="mb-3 flex items-start gap-2.5 bg-[#0e1220] border border-[#1c2438] rounded-xl px-3 py-2.5">
                <Info className="w-4 h-4 text-slate-400 shrink-0 mt-0.5" />
                <p className="text-[11px] text-slate-400 leading-relaxed">
                  <span className="text-slate-300 font-semibold">This is a simulator, not {activeModel.name}.</span>{" "}
                  A general-purpose LLM is prompted to answer in character as this architecture,
                  and the metrics below each reply are computed from a formula using published
                  per-operation energy figures. No spiking network runs here and nothing is
                  measured on neuromorphic hardware. Use the links in the panel on the left to
                  reach the real models.
                </p>
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
                        {isUser
                          ? <div className="whitespace-pre-wrap">{msg.content}</div>
                          : <Markdown>{msg.content}</Markdown>}
                      </div>

                      {/* Render Neuromorphic Spiking Statistics if message has metrics */}
                      {msg.metrics && msg.metrics.spikeCount > 0 && (
                        <div className="w-[85%] bg-[#080b12] border border-[#1b2234] rounded-xl p-3 mt-1 text-xs text-slate-300 space-y-2">
                          <div className="flex items-center justify-between border-b border-[#141a29] pb-1.5">
                            <span className="text-emerald-400 font-mono font-bold flex items-center gap-1 text-[10px]">
                              <Zap className="w-3.5 h-3.5 fill-emerald-500/20" /> ESTIMATED NEUROMORPHIC METRICS
                            </span>
                            <span className="text-[10px] text-slate-500 font-mono">Modelled, not measured</span>
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
                                style={{ width: `${Math.min(100, Math.max(0, 100 - msg.metrics.energySavedPercent))}%` }}
                              />
                              <div 
                                className="bg-[#121622] h-full"
                                style={{ width: `${Math.min(100, Math.max(0, msg.metrics.energySavedPercent))}%` }}
                              />
                            </div>
                            <div className="flex justify-between text-[9px] text-slate-500 mt-1">
                              <span>SNN Active Potential (Spike sparse)</span>
                              <span>Average firing rate per token: {msg.metrics.averageFiringRate.toFixed(2)}%</span>
                            </div>
                            <p className="text-[9px] text-slate-600 mt-1.5 leading-relaxed">
                              Assumes 1.4 pJ per synaptic operation and 1.5 pJ per floating-point
                              operation — the 45nm CMOS figures conventionally cited in the SNN
                              literature. Actual energy depends on hardware this app does not run on.
                            </p>
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}

                {/* Live stream: tokens appear here until the reply is committed */}
                {streamingForModel === selectedModelId && streamingText && (
                  <div className="flex flex-col gap-1.5 items-start">
                    <div className="text-[10px] text-slate-500 font-mono px-1 flex items-center gap-1">
                      <span>{activeModel.name}</span>
                      <span>•</span>
                      <span className="text-emerald-400">streaming</span>
                    </div>
                    <div className="max-w-[85%] rounded-2xl rounded-tl-none px-4 py-3 bg-[#111524]/90 border border-[#1d243c] text-slate-200">
                      <Markdown>{streamingText}</Markdown>
                      <span className="inline-block w-1.5 h-3.5 bg-emerald-400 align-text-bottom animate-pulse ml-0.5" />
                    </div>
                  </div>
                )}

                {/* Waiting for the first token */}
                {isLoading && !streamingText && (
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

              {/* Rate limit notice */}
              {cooldownSeconds > 0 && (
                <div
                  role="status"
                  aria-live="polite"
                  className="mt-4 flex items-start gap-2.5 bg-amber-950/30 border border-amber-700/30 rounded-xl px-3 py-2.5 text-xs text-amber-200"
                >
                  <ShieldAlert className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                  <div>
                    <span className="font-semibold">Rate limit reached.</span>{" "}
                    This is a shared demo, so each visitor gets a capped number of
                    prompts per minute. You can send another in{" "}
                    <span className="font-mono font-bold text-amber-300">{cooldownSeconds}s</span>.
                  </div>
                </div>
              )}

              {/* Chat Input form */}
              <form onSubmit={handleSendMessage} className="mt-4 flex gap-2">
                <input
                  type="text"
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  placeholder={
                    cooldownSeconds > 0
                      ? `Cooling down — ${cooldownSeconds}s remaining...`
                      : `Send a prompt to test ${activeModel.name}... (e.g. "Draft an email explaining neural energy efficiency")`
                  }
                  aria-label={`Prompt for ${activeModel.name}`}
                  maxLength={8000}
                  disabled={isLoading || cooldownSeconds > 0}
                  className="flex-1 bg-[#090c12] border border-[#1b2234] focus:border-emerald-500/50 rounded-xl px-4 py-3 text-sm text-slate-100 placeholder-slate-500 outline-none focus:ring-1 focus:ring-emerald-500/20 disabled:opacity-50 transition-all"
                />
                {isLoading ? (
                  <button
                    type="button"
                    onClick={handleStopGeneration}
                    aria-label="Stop generating"
                    className="bg-[#1a1f30] border border-[#2b3552] hover:border-red-500/40 hover:text-red-300 text-slate-300 font-semibold rounded-xl px-5 py-3 text-sm flex items-center gap-2 transition-all active:scale-95"
                  >
                    <span>Stop</span>
                    <Square className="w-3.5 h-3.5 fill-current" />
                  </button>
                ) : (
                  <button
                    type="submit"
                    disabled={cooldownSeconds > 0 || !inputMessage.trim()}
                    className="bg-gradient-to-r from-emerald-600 to-teal-700 hover:from-emerald-500 hover:to-teal-600 text-white font-semibold rounded-xl px-5 py-3 text-sm flex items-center gap-2 disabled:opacity-50 transition-all shadow-[0_0_15px_rgba(16,185,129,0.1)] active:scale-95"
                  >
                    <span>{cooldownSeconds > 0 ? `${cooldownSeconds}s` : "Inference"}</span>
                    <Send className="w-4 h-4" />
                  </button>
                )}
              </form>
            </div>
          )}

          {/* TAB 2: Live Spikes Network Visualizer */}
          {activeTab === "network-viz" && (
            <div role="tabpanel" id="panel-network-viz" aria-labelledby="tab-network-viz" tabIndex={0} className="bg-[#0b0e17] border border-[#1b2234] rounded-3xl p-5 shadow-2xl flex flex-col h-[min(650px,calc(100dvh-13rem))] min-h-[420px]">
              
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
                    onClick={() => triggerSpikePropagation(randomInputChannels())}
                    className="px-3 py-1.5 bg-[#12241e] border border-emerald-500/30 hover:border-emerald-500/50 text-emerald-400 hover:bg-[#152e25] text-xs font-mono rounded-lg transition-all flex items-center gap-1 disabled:opacity-40"
                  >
                    <Play className="w-3 h-3 fill-emerald-400/10" /> STIMULATE CHANNELS
                  </button>
                  <span className="text-[10px] bg-[#161a25] px-2 py-1 rounded text-slate-400 border border-[#232b3f] font-mono">
                    {TOTAL_NEURONS} Neurons | {synapses.length} Plastic Synapses
                  </span>
                </div>
              </div>

              {/* Grid split: Canvas & Log output */}
              <div className="flex-1 grid grid-cols-1 md:grid-cols-12 gap-4 overflow-hidden">
                
                {/* SVG Network rendering */}
                <div className="md:col-span-8 bg-[#090b12] rounded-2xl border border-[#181f30] relative overflow-hidden flex items-center justify-center p-2">
                  
                  {/* Grid background representation */}
                  <div className="absolute inset-0 bg-[linear-gradient(to_right,#141724_1px,transparent_1px),linear-gradient(to_bottom,#141724_1px,transparent_1px)] bg-[size:24px_24px] opacity-20 pointer-events-none" />

                  <svg
                    viewBox="0 0 600 500"
                    className="w-full h-full max-h-[480px]"
                    role="group"
                    aria-label={`Spiking network canvas: ${TOTAL_NEURONS} neurons across four layers, ${synapses.length} synapses. Activate any neuron to stimulate it.`}
                  >
                    {/* Render synapses connections first */}
                    {synapses.map((s) => {
                      const fromNode = neuronById.get(s.from);
                      const toNode = neuronById.get(s.to);
                      if (!fromNode || !toNode) return null;

                      return (
                        <g key={`synapse-${s.from}-${s.to}`}>
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
                          onKeyDown={(e) => {
                            if (e.key === "Enter" || e.key === " ") {
                              e.preventDefault();
                              handleStimulateNeuron(n.id);
                            }
                          }}
                          role="button"
                          tabIndex={0}
                          aria-label={`${n.type} neuron ${n.id}, membrane potential ${n.membranePotential.toFixed(2)} of ${n.threshold.toFixed(2)}. Activate to stimulate.`}
                          className="cursor-pointer group focus:outline-none focus-visible:[&>circle:nth-child(2)]:stroke-emerald-300"
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

                    <div role="log" aria-live="polite" aria-relevant="additions" className="flex-1 overflow-y-auto space-y-2 text-[11px] font-mono custom-scrollbar pr-1">
                      {activityLog.map((log) => (
                        <div key={log.id} className="leading-relaxed border-l-2 border-[#1c2437] pl-2 text-slate-400 hover:text-slate-200 transition-colors">
                          <span className="text-slate-600">[{log.time}]</span> {log.text}
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
            <div role="tabpanel" id="panel-api-gateway" aria-labelledby="tab-api-gateway" tabIndex={0} className="bg-[#0b0e17] border border-[#1b2234] rounded-3xl p-5 shadow-2xl flex flex-col h-[min(650px,calc(100dvh-13rem))] min-h-[420px] overflow-hidden">
              
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
                    <span className="text-[10px] font-mono text-indigo-400 block uppercase tracking-widest font-bold">ROUTES</span>

                    <div className="space-y-2">
                      <div className="flex items-center gap-2">
                        <span className="bg-indigo-950 border border-indigo-800/30 text-indigo-400 px-2 py-1 rounded text-xs font-mono font-bold">
                          POST
                        </span>
                        <code className="text-xs text-slate-200 bg-[#121623] px-2 py-1 rounded flex-1 font-mono">
                          /v1/chat/completions
                        </code>
                        <span className="text-[10px] font-mono bg-amber-950/40 text-amber-400 border border-amber-800/30 px-1.5 py-0.5 rounded">
                          KEY
                        </span>
                      </div>
                      <p className="text-xs text-slate-400">
                        The public gateway, in OpenAI chat-completions format. Requires{" "}
                        <code className="text-slate-300">Authorization: Bearer &lt;key&gt;</code>. Also accepts the
                        non-standard <code className="text-slate-300">threshold</code>,{" "}
                        <code className="text-slate-300">decay</code> and{" "}
                        <code className="text-slate-300">leak</code> fields.
                      </p>
                    </div>

                    <div className="space-y-2 pt-1 border-t border-[#141b2a]">
                      <div className="flex items-center gap-2 pt-2">
                        <span className="bg-slate-800 border border-slate-700/40 text-slate-400 px-2 py-1 rounded text-xs font-mono font-bold">
                          POST
                        </span>
                        <code className="text-xs text-slate-300 bg-[#121623] px-2 py-1 rounded flex-1 font-mono">
                          /api/inference
                        </code>
                        <span className="text-[10px] font-mono bg-slate-800/60 text-slate-400 border border-slate-700/40 px-1.5 py-0.5 rounded">
                          INTERNAL
                        </span>
                      </div>
                      <p className="text-xs text-slate-400">
                        Powers the playground on this page. Same-origin only and rate limited per
                        visitor, so it is not usable as a public API. Use{" "}
                        <code className="text-slate-300">/v1</code> for integrations.
                      </p>
                    </div>

                    <div className="flex items-start gap-2 bg-[#0e1220] border border-[#1c2438] rounded-xl p-2.5">
                      <ShieldAlert className="w-3.5 h-3.5 text-amber-400 shrink-0 mt-0.5" />
                      <p className="text-[11px] text-slate-400 leading-relaxed">
                        Never paste an API key into client-side code. Read it from an environment
                        variable and call the gateway from your own server.
                      </p>
                    </div>
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
                        aria-label="Copy code snippet"
                        className="absolute top-2.5 right-2.5 p-1.5 bg-[#121825] hover:bg-[#1a2133] rounded-lg border border-[#232d46] transition-all text-slate-400 hover:text-slate-200"
                      >
                        {copiedTextId === "sdk-snippet" ? (
                          <Check className="w-3.5 h-3.5 text-emerald-400" />
                        ) : copiedTextId === "sdk-snippet-failed" ? (
                          <XCircle className="w-3.5 h-3.5 text-red-400" />
                        ) : (
                          <Copy className="w-3.5 h-3.5" />
                        )}
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
                        <span
                          className={`font-mono text-xs ${
                            apiStatus === null
                              ? "text-slate-600"
                              : apiStatus >= 200 && apiStatus < 300
                                ? "text-emerald-400"
                                : apiStatus === 429
                                  ? "text-amber-400"
                                  : "text-red-400"
                          }`}
                        >
                          {apiStatus === null
                            ? "Idle"
                            : apiStatus === 0
                              ? "NETWORK ERROR"
                              : `HTTP ${apiStatus}`}
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
            <div role="tabpanel" id="panel-characteristics" aria-labelledby="tab-characteristics" tabIndex={0} className="bg-[#0b0e17] border border-[#1b2234] rounded-3xl p-5 shadow-2xl flex flex-col h-[min(650px,calc(100dvh-13rem))] min-h-[420px] overflow-y-auto custom-scrollbar">
              
              <div className="border-b border-[#1b2234] pb-3 mb-5">
                <h3 className="font-bold text-slate-200 flex items-center gap-2">
                  <FileText className="w-5 h-5 text-amber-400" /> Comparative Spiking Model Matrix
                </h3>
                <p className="text-xs text-slate-400">
                  All {SNN_MODELS.length} models, listed in full. Scroll to compare —{" "}
                  <span className="text-emerald-400">{activeModel.name}</span> is highlighted as your current selection.
                </p>
              </div>

              {/* Matrix list layout */}
              <div className="space-y-4">
                {SNN_MODELS.map((m) => {
                  const isCurrent = m.id === selectedModelId;
                  return (
                    <div 
                      key={m.id}
                      ref={(el) => { matrixCardRefs.current[m.id] = el; }}
                      className={`p-4 rounded-2xl border transition-all scroll-mt-4 ${
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
                          <span className="text-xs text-slate-500 font-mono">{m.parameters} · {m.year} · {m.author}</span>
                        </div>
                        <div className="flex items-center gap-1.5 shrink-0">
                          <span className={`px-1.5 py-0.5 rounded text-[10px] font-mono border ${STATUS_STYLES[m.status]}`}>
                            {STATUS_LABELS[m.status]}
                          </span>
                          <span className="px-2 py-0.5 bg-[#121622] rounded text-xs text-slate-400 font-mono border border-[#1e273f]">
                            {m.type}
                          </span>
                        </div>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-12 gap-4 text-xs">
                        {/* Summary description */}
                        <div className="md:col-span-5 text-slate-400 leading-relaxed">
                          <p>{m.description}</p>
                          <div className="mt-3 flex gap-2 flex-wrap">
                            <span className="bg-[#141b2a] px-2 py-1 rounded text-slate-400 text-[10px] font-mono">Bio-plausibility {m.bioPlausibility}/10</span>
                            <span className="bg-[#141b2a] px-2 py-1 rounded text-emerald-400 text-[10px] font-mono">Efficiency {m.energyEfficiency}/10</span>
                            {m.license && (
                              <span className="bg-[#141b2a] px-2 py-1 rounded text-slate-400 text-[10px] font-mono">{m.license}</span>
                            )}
                          </div>
                          <div className="mt-2 flex gap-3 flex-wrap">
                            {([
                              ["Code", m.github],
                              ["Paper", m.paper],
                              ["Weights", m.huggingface],
                              ["Site", m.homepage],
                            ] as const)
                              .filter(([, href]) => Boolean(href))
                              .map(([label, href]) => (
                                <a
                                  key={label}
                                  href={href as string}
                                  target="_blank"
                                  rel="noreferrer noopener"
                                  className="text-[10px] text-slate-400 hover:text-emerald-400 transition flex items-center gap-0.5 font-mono"
                                >
                                  {label} <ExternalLink className="w-2.5 h-2.5" />
                                </a>
                              ))}
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
          <p>© 2026 Spiking LLM Hub. An educational simulator for open-source neuromorphic language model research.</p>
          <div className="flex gap-4">
            <a href="https://github.com/ridgerchu/SpikeGPT" target="_blank" rel="noreferrer noopener" className="hover:text-emerald-400 transition">SpikeGPT code</a>
            <span>•</span>
            <a href="https://arxiv.org/abs/2302.13939" target="_blank" rel="noreferrer noopener" className="hover:text-emerald-400 transition">SpikeGPT paper</a>
          </div>
        </div>
      </footer>

    </div>
  );
}

import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import { GoogleGenAI } from '@google/genai';

// Load environment variables
dotenv.config();

const app = express();
app.use(express.json());

const PORT = 3000;

// Resolve __dirname safely across ESM and CJS without triggering ESBuild warnings
const currentDir = typeof __dirname !== 'undefined'
  ? __dirname
  : path.dirname(fileURLToPath(import.meta.url));

// Initialize Gemini SDK securely on the server
let aiClient: GoogleGenAI | null = null;
function getGeminiClient(): GoogleGenAI {
  if (!aiClient) {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      console.warn("WARNING: GEMINI_API_KEY environment variable is not set. Inference will use simulated fallback responses.");
    }
    aiClient = new GoogleGenAI({
      apiKey: apiKey || "MOCK_KEY",
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        }
      }
    });
  }
  return aiClient;
}

// Simulated SNN inference metric calculator based on model characteristics and input/output text
function calculateNeuromorphicMetrics(
  modelId: string,
  inputText: string,
  outputText: string,
  vTh: number,
  decay: number,
  leak: number
) {
  const combinedLength = inputText.length + outputText.length;
  const tokenCount = Math.max(1, Math.round(combinedLength / 4));
  
  // Base parameter scaling factors
  let paramFactor = 1.0;
  let baseSparsity = 0.05; // 5% average firing sparsity

  switch (modelId) {
    case "spikegpt":
      paramFactor = 0.5; // relatively smaller base size for default
      baseSparsity = 0.04;
      break;
    case "nord":
      paramFactor = 0.4;
      baseSparsity = 0.03; // highly sparse
      break;
    case "neuronspark":
      paramFactor = 0.87;
      baseSparsity = 0.045;
      break;
    case "spikellm":
      paramFactor = 7.0; // 7B scale
      baseSparsity = 0.08; // conversion SNNs are slightly less sparse
      break;
    case "spikingbrain":
      paramFactor = 1.2; // 7B scale with extreme MoE sparse routing (1.2B active)
      baseSparsity = 0.02; // extremely sparse routing
      break;
    case "braingpt":
      paramFactor = 3.0;
      baseSparsity = 0.06;
      break;
    case "spikebert":
      paramFactor = 0.11; // 110M
      baseSparsity = 0.05;
      break;
    case "spikelm":
      paramFactor = 0.35;
      baseSparsity = 0.035;
      break;
    case "bdh":
      paramFactor = 0.5;
      baseSparsity = 0.05;
      break;
  }

  // Adjust sparsity based on threshold, decay, leak
  // Higher threshold (vTh) -> fewer spikes -> lower firing rate and lower energy
  // Higher decay -> charges dissipate faster -> fewer spikes
  // Higher leak -> potential leaks away -> fewer spikes
  const thresholdMod = Math.max(0.2, vTh);
  const adjustedSparsity = Math.max(0.005, Math.min(0.25, baseSparsity * (1.0 / thresholdMod) * (1.1 - decay) * (1.1 - leak)));
  
  // Total virtual neurons in simulated active layers
  const activeNeurons = paramFactor * 1000000; // scaling representation
  
  // Calculate spikes
  const spikeCount = Math.round(tokenCount * activeNeurons * adjustedSparsity);
  
  // Synaptic Operations (SOPs)
  // Each active spike triggers synaptic evaluations
  const synapticOps = Math.round(spikeCount * 12); // Average fan-out per neuron block
  
  // FLOPs equivalent for a standard Transformer model of the same scale
  const flopsEquivalent = Math.round(tokenCount * activeNeurons * 24); 

  // Energy consumption (typical values: SOP = 1.4e-14 Joules, FLOP = 1.5e-12 Joules)
  const energySOP = 1.4e-14;
  const energyFLOP = 1.5e-12;
  
  const energyJoulesANN = flopsEquivalent * energyFLOP;
  // SNN includes active SOP energy + a tiny background leak energy
  const energyJoulesSNN = (synapticOps * energySOP) + (activeNeurons * tokenCount * leak * 1.0e-15);

  const energySavedPercent = Math.max(45, Math.min(99.8, ((energyJoulesANN - energyJoulesSNN) / energyJoulesANN) * 100));
  
  const averageFiringRate = Math.max(0.1, adjustedSparsity * 100); // % of active neurons firing per token step
  const latencyMs = Math.round(50 + (tokenCount * 3) + (Math.random() * 20));

  return {
    spikeCount,
    synapticOps,
    flopsEquivalent,
    energyJoulesANN,
    energyJoulesSNN,
    energySavedPercent,
    averageFiringRate,
    latencyMs
  };
}

// API endpoint for model chat/inference
app.post('/api/inference', async (req, res) => {
  try {
    const { modelId, messages, threshold, decay, leak } = req.body;
    
    if (!modelId) {
      return res.status(400).json({ error: "Missing modelId parameter" });
    }

    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({ error: "Missing or invalid messages array" });
    }

    const lastMessage = messages[messages.length - 1].content;
    const modelDetails = {
      spikegpt: { name: "SpikeGPT", type: "Pure SNN", desc: "based on RWKV linear attention" },
      nord: { name: "Nord SNN", type: "Pure SNN", desc: "trained fully from scratch with STDP" },
      neuronspark: { name: "NeuronSpark", type: "Pure SNN", desc: "870M-parameter pure SNN" },
      spikellm: { name: "SpikeLLM", type: "ANN-to-SNN", desc: "converted from traditional high-parameter LLMs" },
      spikingbrain: { name: "SpikingBrain", type: "Hybrid SNN", desc: "integrates hybrid attention and Mixture of Experts" },
      braingpt: { name: "BrainGPT", type: "ANN-to-SNN", desc: "instruction-following conversion model" },
      spikebert: { name: "SpikeBERT", type: "NLU SNN", desc: "Sentence understanding and classification" },
      spikelm: { name: "SpikeLM", type: "Pure SNN", desc: "general-purpose spike-driven baseline" },
      bdh: { name: "Baby Dragon Hatchling", type: "Bio-inspired Graph", desc: "scale-free recurrent network model" },
    }[modelId as string] || { name: "Custom SNN", type: "SNN", desc: "spiking language model" };

    let aiResponseText = "";
    
    // Check if API key is present, if so call Gemini, else fallback to mock simulation
    if (process.env.GEMINI_API_KEY) {
      try {
        const client = getGeminiClient();
        
        // Build the prompt context instructing Gemini to simulate the spiking model
        const systemInstruction = `You are simulating the spiking neural network language model: ${modelDetails.name} (${modelDetails.type}), which is ${modelDetails.desc}. 
Provide a helpful, precise, and scientifically accurate answer to the user's prompt. 
Answer in character, keeping the tone elegant, crisp, and technical when appropriate. 
If the model is 'SpikeBERT', keep the response geared towards text classification, embeddings, or sentence parsing, and explain that SpikeBERT is an NLU model. 
Otherwise, generate a natural generative chat completion. Keep the response to 150-250 words unless asked otherwise.`;

        // Format history for Gemini chat format
        const historyParts = messages.slice(0, -1).map(m => ({
          role: m.role === 'assistant' ? 'model' : 'user' as any,
          parts: [{ text: m.content }]
        }));

        // Call Gemini
        const response = await client.models.generateContent({
          model: 'gemini-3.5-flash',
          contents: [
            ...historyParts.map(hp => ({ role: hp.role, parts: hp.parts })),
            { role: 'user', parts: [{ text: lastMessage }] }
          ],
          config: {
            systemInstruction,
            temperature: req.body.temperature || 0.7,
          }
        });

        aiResponseText = response.text || "No response received from simulating model.";
      } catch (geminiError: any) {
        console.error("Gemini call error:", geminiError);
        aiResponseText = `[Simulated response due to model latency] Under spiking parameters (V_th: ${threshold || 1.0}, decay: ${decay || 0.8}, leak: ${leak || 0.1}), I have successfully integrated the input tokens. The query "${lastMessage.substring(0, 40)}..." triggered a series of sparse temporal spikes across my recurrent layers. Spiking models operate with remarkable energy savings by exchanging floating point calculations for simple binary additions. How else can I assist you with neuromorphic SNN computing?`;
      }
    } else {
      // High-quality mock simulation for standalone/preview modes without a key
      aiResponseText = `[Simulated ${modelDetails.name} Inference] Hello! I am running as ${modelDetails.name}, a ${modelDetails.type} model ${modelDetails.desc}.

When processing your prompt, my leaky integrate-and-fire (LIF) neurons accumulated charge values over time. When the membrane potential exceeded the threshold (V_th = ${threshold || 1.0}), they emitted discrete spikes, propagating the signal to downstream layers.

Because SNNs compute via additions (SOPs) rather than multiplications (FLOPs), I operated at maximum sparsity (approx ${((decay || 0.8) * 5).toFixed(2)}% active firing). This is a highly energy-efficient biological-like processing mode! Please configure a real GEMINI_API_KEY in the Secrets panel if you wish to see fully customized, intelligent responses.`;
    }

    // Compute spiking metrics
    const metrics = calculateNeuromorphicMetrics(
      modelId,
      lastMessage,
      aiResponseText,
      threshold || 1.0,
      decay || 0.8,
      leak || 0.1
    );

    res.json({
      content: aiResponseText,
      metrics
    });

  } catch (err: any) {
    console.error("Inference endpoint error:", err);
    res.status(500).json({ error: err.message || "Internal server error" });
  }
});

// OpenAI compatible completions endpoint
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature } = req.body;
    const modelId = model || "spikegpt";
    
    // Convert OpenAI messages to ChatMessage list
    const formattedMessages = (messages || []).map((m: any, idx: number) => ({
      id: `msg-${idx}`,
      role: m.role,
      content: m.content
    }));

    if (formattedMessages.length === 0) {
      return res.status(400).json({ error: "messages array is required" });
    }

    const lastMessage = formattedMessages[formattedMessages.length - 1].content;
    let aiResponseText = "";

    if (process.env.GEMINI_API_KEY) {
      const client = getGeminiClient();
      const response = await client.models.generateContent({
        model: 'gemini-3.5-flash',
        contents: lastMessage,
        config: {
          systemInstruction: `You are simulating the spiking language model ${modelId}. Provide a complete response as this SNN model.`,
          temperature: temperature || 0.7,
        }
      });
      aiResponseText = response.text || "No response received.";
    } else {
      aiResponseText = `[API Simulation] Successful inference run with spiking model: ${modelId}. To get fully customized responses, configure a GEMINI_API_KEY in the Secrets panel.`;
    }

    const metrics = calculateNeuromorphicMetrics(modelId, lastMessage, aiResponseText, 1.0, 0.8, 0.1);

    res.json({
      id: `chatcmpl-${Date.now()}`,
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model: modelId,
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content: aiResponseText
          },
          finish_reason: "stop"
        }
      ],
      usage: {
        prompt_tokens: Math.round(lastMessage.length / 4),
        completion_tokens: Math.round(aiResponseText.length / 4),
        total_tokens: Math.round((lastMessage.length + aiResponseText.length) / 4),
        spiking_metrics: metrics
      }
    });

  } catch (err: any) {
    res.status(500).json({ error: err.message || "Internal server error" });
  }
});

// Serve frontend application & start listening inside an async function to prevent top-level await errors in CJS
async function bootstrap() {
  if (process.env.NODE_ENV === 'production') {
    // Serve static assets from build directory using process.cwd()
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  } else {
    // In development, load Vite middleware dynamically to bind to Express
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Spiking LLM Hub server listening at http://0.0.0.0:${PORT}`);
  });
}

bootstrap().catch((err) => {
  console.error("Bootstrapping failed:", err);
});

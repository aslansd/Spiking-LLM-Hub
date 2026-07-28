import express, { type NextFunction, type Request, type Response } from 'express';
import path from 'path';
import crypto from 'crypto';
import dotenv from 'dotenv';
import { GoogleGenAI } from '@google/genai';

dotenv.config();

/* -------------------------------------------------------------------------- */
/* Configuration                                                              */
/* -------------------------------------------------------------------------- */

const IS_PRODUCTION = process.env.NODE_ENV === 'production';

// Cloud Run injects PORT (normally 8080). Never hardcode it.
const PORT = Number(process.env.PORT) || 8080;

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const GEMINI_MODEL = process.env.GEMINI_MODEL || 'gemini-3.5-flash';

// Comma-separated bearer tokens that may call the public /v1 gateway.
// If empty, /v1 is DISABLED rather than left open to the world.
const GATEWAY_API_KEYS = (process.env.GATEWAY_API_KEYS || '')
  .split(',')
  .map((k) => k.trim())
  .filter(Boolean);

// Origins allowed to call /v1 from a browser, and to call /api cross-origin.
const ALLOWED_ORIGINS = new Set(
  (process.env.ALLOWED_ORIGINS || '')
    .split(',')
    .map((o) => o.trim().replace(/\/$/, ''))
    .filter(Boolean),
);
if (process.env.APP_URL) {
  ALLOWED_ORIGINS.add(process.env.APP_URL.trim().replace(/\/$/, ''));
}
if (!IS_PRODUCTION) {
  ALLOWED_ORIGINS.add('http://localhost:8080');
  ALLOWED_ORIGINS.add('http://localhost:3000');
  ALLOWED_ORIGINS.add('http://localhost:5173');
}

const MAX_BODY_BYTES = process.env.MAX_BODY_BYTES || '128kb';
const MAX_MESSAGES = Number(process.env.MAX_MESSAGES) || 40;
const MAX_CHARS_PER_MESSAGE = Number(process.env.MAX_CHARS_PER_MESSAGE) || 8000;

// Rate limits (per instance; Cloud Run may run several, so treat as a floor).
const PLAYGROUND_WINDOW_MS = Number(process.env.PLAYGROUND_WINDOW_MS) || 60_000;
const PLAYGROUND_MAX = Number(process.env.PLAYGROUND_MAX) || 15;
const GATEWAY_WINDOW_MS = Number(process.env.GATEWAY_WINDOW_MS) || 60_000;
const GATEWAY_MAX = Number(process.env.GATEWAY_MAX) || 60;

// Hard ceiling on billable upstream calls per rolling 24h. Once exceeded the
// app keeps working but serves simulated responses instead of calling Gemini.
// This is the actual protection against a runaway bill.
const DAILY_UPSTREAM_BUDGET = Number(process.env.DAILY_UPSTREAM_BUDGET) || 2000;

const app = express();

// Cloud Run terminates TLS at the front end, so req.ip / req.protocol are only
// correct once Express is told to trust the forwarding proxy.
app.set('trust proxy', true);
app.disable('x-powered-by');

/* -------------------------------------------------------------------------- */
/* Gemini client                                                              */
/* -------------------------------------------------------------------------- */

let aiClient: GoogleGenAI | null = null;
function getGeminiClient(): GoogleGenAI | null {
  if (!GEMINI_API_KEY) return null;
  if (!aiClient) {
    aiClient = new GoogleGenAI({
      apiKey: GEMINI_API_KEY,
      httpOptions: { headers: { 'User-Agent': 'spiking-llm-hub' } },
    });
  }
  return aiClient;
}

// Rolling 24h budget counter.
const upstreamCalls: number[] = [];
function upstreamBudgetAvailable(): boolean {
  const cutoff = Date.now() - 86_400_000;
  while (upstreamCalls.length && upstreamCalls[0] < cutoff) upstreamCalls.shift();
  return upstreamCalls.length < DAILY_UPSTREAM_BUDGET;
}
function recordUpstreamCall(): void {
  upstreamCalls.push(Date.now());
}

/* -------------------------------------------------------------------------- */
/* Security middleware                                                        */
/* -------------------------------------------------------------------------- */

app.use((_req: Request, res: Response, next: NextFunction) => {
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-Frame-Options', 'DENY');
  res.setHeader('Referrer-Policy', 'strict-origin-when-cross-origin');
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Permissions-Policy', 'geolocation=(), microphone=(), camera=()');
  if (IS_PRODUCTION) {
    res.setHeader('Strict-Transport-Security', 'max-age=31536000; includeSubDomains');
    // 'unsafe-inline' for style is required because the UI uses React inline
    // style attributes throughout. Scripts stay locked to same-origin.
    res.setHeader(
      'Content-Security-Policy',
      [
        "default-src 'self'",
        "script-src 'self'",
        "style-src 'self' 'unsafe-inline'",
        "img-src 'self' data:",
        "font-src 'self' data:",
        "connect-src 'self'",
        "base-uri 'self'",
        "form-action 'self'",
        "frame-ancestors 'none'",
        'object-src \'none\'',
      ].join('; '),
    );
  }
  next();
});

function originAllowed(origin: string | undefined): boolean {
  if (!origin) return false;
  return ALLOWED_ORIGINS.has(origin.replace(/\/$/, ''));
}

/**
 * True when the browser's Origin matches the host this request arrived on.
 *
 * Browsers attach an Origin header to every non-GET/HEAD request, including
 * same-origin ones, so the playground's own POSTs carry one. Deriving the
 * comparison from the request means the app works on the Cloud Run URL, a
 * custom domain and localhost without any origin configuration at all —
 * ALLOWED_ORIGINS is then only needed for genuinely third-party callers.
 */
function isSameOrigin(req: Request): boolean {
  const origin = req.headers.origin;
  if (!origin) return true; // curl and server-to-server callers send none
  const host = req.headers.host;
  if (!host) return false;
  try {
    return new URL(origin).host === host;
  } catch {
    return false;
  }
}

// CORS for the public gateway only. /api stays same-origin.
app.use('/v1', (req: Request, res: Response, next: NextFunction) => {
  const origin = req.headers.origin;
  if (origin && (isSameOrigin(req) || originAllowed(origin))) {
    res.setHeader('Access-Control-Allow-Origin', origin as string);
    res.setHeader('Vary', 'Origin');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Max-Age', '600');
  }
  if (req.method === 'OPTIONS') {
    res.status(204).end();
    return;
  }
  next();
});

// Block cross-origin browser calls to the internal playground endpoint.
// Requests with no Origin header (curl, server-to-server) are still allowed,
// but they are rate limited and budget capped like everything else.
app.use('/api', (req: Request, res: Response, next: NextFunction) => {
  const origin = req.headers.origin;
  if (origin && !isSameOrigin(req) && !originAllowed(origin)) {
    res.status(403).json({ error: 'Cross-origin requests are not permitted on /api. Use the /v1 gateway with an API key.' });
    return;
  }
  next();
});

app.use(express.json({ limit: MAX_BODY_BYTES }));

// Reject malformed JSON with a clean 400 instead of a stack trace.
app.use((err: any, _req: Request, res: Response, next: NextFunction) => {
  if (err && (err.type === 'entity.parse.failed' || err instanceof SyntaxError)) {
    res.status(400).json({ error: 'Request body is not valid JSON.' });
    return;
  }
  if (err && err.type === 'entity.too.large') {
    res.status(413).json({ error: 'Request body too large.' });
    return;
  }
  next(err);
});

/* -------------------------------------------------------------------------- */
/* Rate limiting (in-memory fixed window)                                      */
/* -------------------------------------------------------------------------- */

interface Bucket {
  count: number;
  resetAt: number;
}
const buckets = new Map<string, Bucket>();

// Sweep expired buckets so an attacker cycling IPs cannot grow the map forever.
const sweepTimer = setInterval(() => {
  const now = Date.now();
  for (const [key, bucket] of buckets) {
    if (bucket.resetAt <= now) buckets.delete(key);
  }
}, 60_000);
sweepTimer.unref?.();

function rateLimit(opts: {
  windowMs: number;
  max: number;
  keyFn: (req: Request) => string;
  scope: string;
}) {
  return (req: Request, res: Response, next: NextFunction) => {
    const key = `${opts.scope}:${opts.keyFn(req)}`;
    const now = Date.now();
    let bucket = buckets.get(key);
    if (!bucket || bucket.resetAt <= now) {
      bucket = { count: 0, resetAt: now + opts.windowMs };
      buckets.set(key, bucket);
    }
    bucket.count += 1;

    const remaining = Math.max(0, opts.max - bucket.count);
    res.setHeader('RateLimit-Limit', String(opts.max));
    res.setHeader('RateLimit-Remaining', String(remaining));
    res.setHeader('RateLimit-Reset', String(Math.ceil((bucket.resetAt - now) / 1000)));

    if (bucket.count > opts.max) {
      res.setHeader('Retry-After', String(Math.ceil((bucket.resetAt - now) / 1000)));
      res.status(429).json({ error: 'Rate limit exceeded. Please slow down.' });
      return;
    }
    next();
  };
}

const clientIp = (req: Request): string => req.ip || 'unknown';

/* -------------------------------------------------------------------------- */
/* Gateway authentication                                                     */
/* -------------------------------------------------------------------------- */

function timingSafeEqual(a: string, b: string): boolean {
  const bufA = Buffer.from(a);
  const bufB = Buffer.from(b);
  if (bufA.length !== bufB.length) return false;
  return crypto.timingSafeEqual(bufA, bufB);
}

function extractBearer(req: Request): string | null {
  const header = req.headers.authorization;
  if (!header || !header.startsWith('Bearer ')) return null;
  return header.slice(7).trim() || null;
}

// Attach a stable, non-reversible id for the key so we can rate limit per key
// without ever writing the key itself to logs.
function keyFingerprint(key: string): string {
  return crypto.createHash('sha256').update(key).digest('hex').slice(0, 12);
}

function requireGatewayKey(req: Request, res: Response, next: NextFunction) {
  if (GATEWAY_API_KEYS.length === 0) {
    res.status(503).json({
      error:
        'The public API gateway is not enabled on this deployment. Set GATEWAY_API_KEYS to enable it.',
    });
    return;
  }
  const presented = extractBearer(req);
  if (!presented) {
    res.setHeader('WWW-Authenticate', 'Bearer');
    res.status(401).json({ error: 'Missing API key. Send it as: Authorization: Bearer <key>' });
    return;
  }
  const matched = GATEWAY_API_KEYS.some((k) => timingSafeEqual(k, presented));
  if (!matched) {
    res.status(401).json({ error: 'Invalid API key.' });
    return;
  }
  (req as any).apiKeyId = keyFingerprint(presented);
  next();
}

/* -------------------------------------------------------------------------- */
/* Validation helpers                                                         */
/* -------------------------------------------------------------------------- */

const KNOWN_MODELS: Record<string, { name: string; type: string; desc: string }> = {
  spikegpt: { name: 'SpikeGPT', type: 'Pure SNN', desc: 'a 45M-216M generative model built on the RWKV recurrent architecture' },
  nord: { name: 'Project Nord', type: 'Pure SNN', desc: 'a 144M pure SNN trained from scratch with reward-modulated STDP' },
  neuronspark: { name: 'NeuronSpark', type: 'Pure SNN', desc: 'an 874M pure SNN whose LIF dynamics are formulated as a selective state space model' },
  spikellm: { name: 'SpikeLLM', type: 'ANN-to-SNN', desc: 'a saliency-based spiking quantisation framework applied to 7B-70B LLaMA models' },
  spikingbrain: { name: 'SpikingBrain', type: 'Hybrid SNN', desc: 'a 7B/76B model combining hybrid attention, MoE routing and spike encoding' },
  braingpt: { name: 'BrainTransformers', type: 'ANN-to-SNN', desc: 'a 3B chat model converted from Qwen2 into an SNN-transformer hybrid' },
  spikebert: { name: 'SpikeBERT', type: 'NLU SNN', desc: 'a spiking encoder distilled from BERT for classification and embeddings, not generation' },
  spikelm: { name: 'SpikeLM', type: 'Pure SNN', desc: 'a spike-driven language model using elastic bidirectional ternary spiking' },
  bdh: { name: 'Baby Dragon Hatchling', type: 'Bio-inspired Graph', desc: 'a scale-free network of locally interacting neurons with Hebbian dynamics' },
};

interface CleanMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

class ValidationError extends Error {}

function clampNumber(value: unknown, min: number, max: number, fallback: number): number {
  const n = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(max, Math.max(min, n));
}

function validateMessages(raw: unknown): CleanMessage[] {
  if (!Array.isArray(raw) || raw.length === 0) {
    throw new ValidationError('`messages` must be a non-empty array.');
  }
  if (raw.length > MAX_MESSAGES) {
    throw new ValidationError(`\`messages\` may contain at most ${MAX_MESSAGES} entries.`);
  }
  return raw.map((m: any, i: number) => {
    if (!m || typeof m !== 'object') {
      throw new ValidationError(`messages[${i}] must be an object.`);
    }
    // Accept OpenAI's multimodal array form by flattening its text parts.
    let content = m.content;
    if (Array.isArray(content)) {
      content = content
        .filter((p: any) => p && typeof p.text === 'string')
        .map((p: any) => p.text)
        .join('\n');
    }
    if (typeof content !== 'string' || content.trim() === '') {
      throw new ValidationError(`messages[${i}].content must be a non-empty string.`);
    }
    if (content.length > MAX_CHARS_PER_MESSAGE) {
      throw new ValidationError(
        `messages[${i}].content exceeds ${MAX_CHARS_PER_MESSAGE} characters.`,
      );
    }
    const role = m.role === 'assistant' || m.role === 'system' ? m.role : 'user';
    return { role, content } as CleanMessage;
  });
}

function validateModelId(raw: unknown, fallback: string): string {
  const id = typeof raw === 'string' ? raw.toLowerCase().trim() : fallback;
  if (!Object.prototype.hasOwnProperty.call(KNOWN_MODELS, id)) {
    throw new ValidationError(
      `Unknown model "${id}". Valid ids: ${Object.keys(KNOWN_MODELS).join(', ')}.`,
    );
  }
  return id;
}

/**
 * Gemini requires `contents` to start with a user turn and to alternate.
 * The playground's seeded welcome message and any error bubbles violate that,
 * which was silently forcing every first request into the fallback path.
 */
function buildGeminiContents(messages: CleanMessage[]) {
  const turns = messages
    .filter((m) => m.role !== 'system')
    .map((m) => ({
      role: m.role === 'assistant' ? 'model' : 'user',
      parts: [{ text: m.content }],
    }));

  while (turns.length && turns[0].role === 'model') turns.shift();

  const merged: typeof turns = [];
  for (const turn of turns) {
    const prev = merged[merged.length - 1];
    if (prev && prev.role === turn.role) {
      prev.parts[0].text += `\n\n${turn.parts[0].text}`;
    } else {
      merged.push(turn);
    }
  }
  return merged;
}

/* -------------------------------------------------------------------------- */
/* Neuromorphic metric simulation (unchanged behaviour)                        */
/* -------------------------------------------------------------------------- */

function calculateNeuromorphicMetrics(
  modelId: string,
  inputText: string,
  outputText: string,
  vTh: number,
  decay: number,
  leak: number,
) {
  const combinedLength = inputText.length + outputText.length;
  const tokenCount = Math.max(1, Math.round(combinedLength / 4));

  const profiles: Record<string, { paramFactor: number; baseSparsity: number }> = {
    spikegpt: { paramFactor: 0.5, baseSparsity: 0.04 },
    nord: { paramFactor: 0.4, baseSparsity: 0.03 },
    neuronspark: { paramFactor: 0.87, baseSparsity: 0.045 },
    spikellm: { paramFactor: 7.0, baseSparsity: 0.08 },
    spikingbrain: { paramFactor: 1.2, baseSparsity: 0.02 },
    braingpt: { paramFactor: 3.0, baseSparsity: 0.06 },
    spikebert: { paramFactor: 0.11, baseSparsity: 0.05 },
    spikelm: { paramFactor: 0.35, baseSparsity: 0.035 },
    bdh: { paramFactor: 0.5, baseSparsity: 0.05 },
  };
  const { paramFactor, baseSparsity } = profiles[modelId] || {
    paramFactor: 1.0,
    baseSparsity: 0.05,
  };

  const thresholdMod = Math.max(0.2, vTh);
  const adjustedSparsity = Math.max(
    0.005,
    Math.min(0.25, baseSparsity * (1.0 / thresholdMod) * (1.1 - decay) * (1.1 - leak)),
  );

  const activeNeurons = paramFactor * 1_000_000;
  const spikeCount = Math.round(tokenCount * activeNeurons * adjustedSparsity);
  const synapticOps = Math.round(spikeCount * 12);
  const flopsEquivalent = Math.round(tokenCount * activeNeurons * 24);

  const energySOP = 1.4e-14;
  const energyFLOP = 1.5e-12;
  const energyJoulesANN = flopsEquivalent * energyFLOP;
  const energyJoulesSNN =
    synapticOps * energySOP + activeNeurons * tokenCount * leak * 1.0e-15;

  // No floor here: report whatever the model actually yields. The previous
  // Math.max(45, ...) guaranteed a headline saving even when the inputs said
  // otherwise, which quietly made every figure on the page untrustworthy.
  const energySavedPercent = Math.min(
    99.8,
    ((energyJoulesANN - energyJoulesSNN) / energyJoulesANN) * 100,
  );
  const averageFiringRate = Math.max(0.1, adjustedSparsity * 100);
  const latencyMs = Math.round(50 + tokenCount * 3 + Math.random() * 20);

  return {
    spikeCount,
    synapticOps,
    flopsEquivalent,
    energyJoulesANN,
    energyJoulesSNN,
    energySavedPercent,
    averageFiringRate,
    latencyMs,
    simulated: true as const,
  };
}

/* -------------------------------------------------------------------------- */
/* Text generation                                                            */
/* -------------------------------------------------------------------------- */

function simulatedReply(
  model: { name: string; type: string; desc: string },
  threshold: number,
  decay: number,
  reason: 'no-key' | 'budget' | 'upstream-error',
): string {
  const note =
    reason === 'no-key'
      ? 'This deployment has no GEMINI_API_KEY configured, so responses are canned.'
      : reason === 'budget'
        ? 'The daily upstream generation budget for this demo has been reached; responses are canned until it resets.'
        : 'The upstream generation service is temporarily unavailable, so this response is canned.';

  return `[Simulated ${model.name} inference] I am running as ${model.name}, a ${model.type} model ${model.desc}.

Processing your prompt, leaky integrate-and-fire (LIF) neurons accumulate charge across timesteps. Once membrane potential crosses the threshold (V_th = ${threshold.toFixed(2)}, decay = ${decay.toFixed(2)}), each neuron emits a discrete spike that propagates downstream. Because SNNs compute with accumulations (SOPs) rather than multiply-accumulates (FLOPs), the energy profile is very different from a dense transformer.

Note: ${note}`;
}

async function generateText(
  modelId: string,
  messages: CleanMessage[],
  temperature: number,
  systemInstruction: string,
  threshold: number,
  decay: number,
): Promise<{ text: string; upstream: boolean }> {
  const model = KNOWN_MODELS[modelId];
  const client = getGeminiClient();

  if (!client) {
    return { text: simulatedReply(model, threshold, decay, 'no-key'), upstream: false };
  }
  if (!upstreamBudgetAvailable()) {
    console.warn('Daily upstream budget exhausted; serving simulated response.');
    return { text: simulatedReply(model, threshold, decay, 'budget'), upstream: false };
  }

  try {
    recordUpstreamCall();
    const response = await client.models.generateContent({
      model: GEMINI_MODEL,
      contents: buildGeminiContents(messages),
      config: { systemInstruction, temperature },
    });
    const text = response.text?.trim();
    if (!text) {
      return { text: simulatedReply(model, threshold, decay, 'upstream-error'), upstream: true };
    }
    return { text, upstream: true };
  } catch (err) {
    // Log detail server-side only. Never echo upstream errors to the caller:
    // they can leak key state, quota info and internal endpoints.
    console.error('Upstream generation failed:', err);
    return { text: simulatedReply(model, threshold, decay, 'upstream-error'), upstream: false };
  }
}

function systemInstructionFor(modelId: string): string {
  const model = KNOWN_MODELS[modelId];
  return `You are role-playing as the spiking neural network language model ${model.name} (${model.type}), which is ${model.desc}.
Give a helpful, precise and scientifically accurate answer to the user's prompt.
Stay in character with an elegant, crisp, technical tone where appropriate.
If the model is SpikeBERT, orient the response toward text classification, embeddings or sentence parsing, and note that SpikeBERT is an NLU model rather than a generative one.
Otherwise produce a natural generative chat completion. Keep responses to 150-250 words unless asked otherwise.`;
}

/* -------------------------------------------------------------------------- */
/* Streaming                                                                  */
/* -------------------------------------------------------------------------- */

/**
 * Emits a fixed string in small pieces so that fallback responses travel down
 * the same streaming path as live ones. Without this the client would need two
 * separate code paths for "key configured" and "key missing".
 */
async function* chunkText(text: string, signal?: AbortSignal) {
  const pieces = text.match(/\S+\s*/g) || [text];
  for (const piece of pieces) {
    if (signal?.aborted) return;
    yield piece;
    await new Promise((resolve) => setTimeout(resolve, 12));
  }
}

async function* generateTextStream(
  modelId: string,
  messages: CleanMessage[],
  temperature: number,
  systemInstruction: string,
  threshold: number,
  decay: number,
  signal: AbortSignal,
): AsyncGenerator<string> {
  const model = KNOWN_MODELS[modelId];
  const client = getGeminiClient();

  if (!client) {
    yield* chunkText(simulatedReply(model, threshold, decay, 'no-key'), signal);
    return;
  }
  if (!upstreamBudgetAvailable()) {
    console.warn('Daily upstream budget exhausted; streaming simulated response.');
    yield* chunkText(simulatedReply(model, threshold, decay, 'budget'), signal);
    return;
  }

  let emittedAnything = false;

  try {
    recordUpstreamCall();
    const stream = await client.models.generateContentStream({
      model: GEMINI_MODEL,
      contents: buildGeminiContents(messages),
      config: { systemInstruction, temperature },
    });

    for await (const chunk of stream) {
      if (signal.aborted) return;
      const text = chunk.text;
      if (text) {
        emittedAnything = true;
        yield text;
      }
    }
  } catch (err) {
    console.error('Upstream streaming failed:', err);
    // Once bytes are on the wire we cannot retract them, so only substitute a
    // fallback if the failure happened before any content was produced.
    if (!emittedAnything) {
      yield* chunkText(simulatedReply(model, threshold, decay, 'upstream-error'), signal);
      return;
    }
    yield '\n\n[Generation was interrupted before completion.]';
    return;
  }

  if (!emittedAnything) {
    yield* chunkText(simulatedReply(model, threshold, decay, 'upstream-error'), signal);
  }
}

/**
 * Cloud Run buffers responses that look bufferable. Omitting Content-Length and
 * disabling downstream buffering is what makes tokens actually arrive live.
 */
function openSSE(res: Response): NodeJS.Timeout {
  res.writeHead(200, {
    'Content-Type': 'text/event-stream; charset=utf-8',
    'Cache-Control': 'no-cache, no-transform',
    Connection: 'keep-alive',
    'X-Accel-Buffering': 'no',
  });
  res.write(': open\n\n');

  // Idle proxies drop long-lived connections; a comment frame keeps them open
  // without being visible to the client parser.
  const heartbeat = setInterval(() => {
    res.write(': ping\n\n');
  }, 15_000);
  heartbeat.unref?.();
  return heartbeat;
}

function sseSend(res: Response, event: string, data: unknown): void {
  res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
}

function sseSendRaw(res: Response, payload: string): void {
  res.write(`data: ${payload}\n\n`);
}


/* -------------------------------------------------------------------------- */
/* Routes                                                                     */
/* -------------------------------------------------------------------------- */

// Liveness/readiness probe. Deliberately leaks no configuration detail.
app.get('/healthz', (_req: Request, res: Response) => {
  res.status(200).json({
    status: 'ok',
    uptimeSeconds: Math.round(process.uptime()),
    generation: GEMINI_API_KEY ? 'live' : 'simulated',
    gatewayEnabled: GATEWAY_API_KEYS.length > 0,
  });
});

const playgroundLimiter = rateLimit({
  windowMs: PLAYGROUND_WINDOW_MS,
  max: PLAYGROUND_MAX,
  keyFn: clientIp,
  scope: 'playground',
});

const gatewayLimiter = rateLimit({
  windowMs: GATEWAY_WINDOW_MS,
  max: GATEWAY_MAX,
  keyFn: (req) => (req as any).apiKeyId || clientIp(req),
  scope: 'gateway',
});

// Internal endpoint used by the browser playground. No API key (the browser has
// no safe place to keep one) but rate limited, origin locked and budget capped.
app.post('/api/inference', playgroundLimiter, async (req: Request, res: Response) => {
  let modelId: string;
  let messages: CleanMessage[];
  let threshold: number;
  let decay: number;
  let leak: number;
  let temperature: number;

  // Validate before opening a stream: once SSE headers are sent we can no
  // longer return a 400 status.
  try {
    modelId = validateModelId(req.body?.modelId, 'spikegpt');
    messages = validateMessages(req.body?.messages);
    threshold = clampNumber(req.body?.threshold, 0.1, 2.5, 1.0);
    decay = clampNumber(req.body?.decay, 0.4, 0.99, 0.8);
    leak = clampNumber(req.body?.leak, 0.001, 0.5, 0.1);
    temperature = clampNumber(req.body?.temperature, 0, 2, 0.7);
  } catch (err) {
    if (err instanceof ValidationError) {
      res.status(400).json({ error: err.message });
      return;
    }
    console.error('Inference validation error:', err);
    res.status(500).json({ error: 'Internal server error.' });
    return;
  }

  const lastMessage = messages[messages.length - 1].content;
  const wantsStream = req.body?.stream === true;

  if (!wantsStream) {
    try {
      const { text } = await generateText(
        modelId,
        messages,
        temperature,
        systemInstructionFor(modelId),
        threshold,
        decay,
      );
      const metrics = calculateNeuromorphicMetrics(
        modelId,
        lastMessage,
        text,
        threshold,
        decay,
        leak,
      );
      res.json({ content: text, metrics });
    } catch (err) {
      console.error('Inference endpoint error:', err);
      res.status(500).json({ error: 'Internal server error.' });
    }
    return;
  }

  // Abort the upstream call as soon as the browser goes away, so a user who
  // hits Stop or closes the tab stops costing tokens immediately.
  //
  // This MUST listen on `res`, not `req`. Since Node 16, `req` emits 'close'
  // as soon as the request body has been fully read, which happens in
  // express.json() before this handler even runs. Listening there aborts a
  // perfectly healthy connection after roughly one chunk.
  const controller = new AbortController();
  res.on('close', () => {
    if (!res.writableEnded) controller.abort();
  });

  const heartbeat = openSSE(res);
  let full = '';

  try {
    for await (const delta of generateTextStream(
      modelId,
      messages,
      temperature,
      systemInstructionFor(modelId),
      threshold,
      decay,
      controller.signal,
    )) {
      full += delta;
      sseSend(res, 'delta', { text: delta });
    }

    if (!controller.signal.aborted) {
      sseSend(
        res,
        'metrics',
        calculateNeuromorphicMetrics(modelId, lastMessage, full, threshold, decay, leak),
      );
      sseSend(res, 'done', { length: full.length });
    }
  } catch (err) {
    console.error('Streaming inference error:', err);
    if (!res.writableEnded) sseSend(res, 'error', { error: 'Generation failed.' });
  } finally {
    clearInterval(heartbeat);
    if (!res.writableEnded) res.end();
  }
});

// Public OpenAI-compatible gateway. Requires a bearer key.
app.get('/v1/models', requireGatewayKey, gatewayLimiter, (_req: Request, res: Response) => {
  res.json({
    object: 'list',
    data: Object.keys(KNOWN_MODELS).map((id) => ({
      id,
      object: 'model',
      created: 0,
      owned_by: 'spiking-llm-hub',
    })),
  });
});

app.post(
  '/v1/chat/completions',
  requireGatewayKey,
  gatewayLimiter,
  async (req: Request, res: Response) => {
    try {
      const modelId = validateModelId(req.body?.model, 'spikegpt');
      const messages = validateMessages(req.body?.messages);
      const temperature = clampNumber(req.body?.temperature, 0, 2, 0.7);

      // Non-standard extensions: the neuromorphic parameters the playground
      // exposes. Omitted by ordinary OpenAI clients, in which case the model's
      // defaults apply.
      const threshold = clampNumber(req.body?.threshold, 0.1, 2.5, 1.0);
      const decay = clampNumber(req.body?.decay, 0.4, 0.99, 0.8);
      const leak = clampNumber(req.body?.leak, 0.001, 0.5, 0.1);

      const systemFromRequest = messages
        .filter((m) => m.role === 'system')
        .map((m) => m.content)
        .join('\n');
      const systemInstruction = systemFromRequest
        ? `${systemInstructionFor(modelId)}\n\nAdditional instructions from the caller:\n${systemFromRequest}`
        : systemInstructionFor(modelId);

      const lastMessage = messages[messages.length - 1].content;

      if (req.body?.stream === true) {
        const completionId = `chatcmpl-${crypto.randomUUID()}`;
        const created = Math.floor(Date.now() / 1000);
        const controller = new AbortController();
        // See the note in /api/inference: 'close' on the response, never the
        // request, is what actually signals a client disconnect.
        res.on('close', () => {
          if (!res.writableEnded) controller.abort();
        });

        const chunk = (delta: Record<string, unknown>, finish: string | null) => ({
          id: completionId,
          object: 'chat.completion.chunk',
          created,
          model: modelId,
          choices: [{ index: 0, delta, finish_reason: finish }],
        });

        const heartbeat = openSSE(res);
        let full = '';

        try {
          sseSendRaw(res, JSON.stringify(chunk({ role: 'assistant' }, null)));

          for await (const delta of generateTextStream(
            modelId,
            messages,
            temperature,
            systemInstruction,
            threshold,
            decay,
            controller.signal,
          )) {
            full += delta;
            sseSendRaw(res, JSON.stringify(chunk({ content: delta }, null)));
          }

          if (!controller.signal.aborted) {
            sseSendRaw(res, JSON.stringify(chunk({}, 'stop')));

            // Non-standard trailing frame carrying usage and the simulated
            // spiking figures. Ordinary OpenAI clients stop at [DONE] and
            // ignore it; ours reads it.
            sseSendRaw(
              res,
              JSON.stringify({
                ...chunk({}, 'stop'),
                usage: {
                  prompt_tokens: Math.round(lastMessage.length / 4),
                  completion_tokens: Math.round(full.length / 4),
                  total_tokens: Math.round((lastMessage.length + full.length) / 4),
                  spiking_metrics: calculateNeuromorphicMetrics(
                    modelId,
                    lastMessage,
                    full,
                    threshold,
                    decay,
                    leak,
                  ),
                },
              }),
            );
            sseSendRaw(res, '[DONE]');
          }
        } catch (err) {
          console.error('Gateway streaming error:', err);
        } finally {
          clearInterval(heartbeat);
          if (!res.writableEnded) res.end();
        }
        return;
      }

      const { text } = await generateText(
        modelId,
        messages,
        temperature,
        systemInstruction,
        threshold,
        decay,
      );

      const metrics = calculateNeuromorphicMetrics(
        modelId,
        lastMessage,
        text,
        threshold,
        decay,
        leak,
      );

      res.json({
        id: `chatcmpl-${crypto.randomUUID()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: modelId,
        choices: [
          {
            index: 0,
            message: { role: 'assistant', content: text },
            finish_reason: 'stop',
          },
        ],
        usage: {
          prompt_tokens: Math.round(lastMessage.length / 4),
          completion_tokens: Math.round(text.length / 4),
          total_tokens: Math.round((lastMessage.length + text.length) / 4),
          spiking_metrics: metrics,
        },
      });
    } catch (err) {
      if (err instanceof ValidationError) {
        res.status(400).json({ error: err.message });
        return;
      }
      console.error('Gateway endpoint error:', err);
      res.status(500).json({ error: 'Internal server error.' });
    }
  },
);

// Unmatched API routes must return JSON, not the SPA shell.
app.use(['/api', '/v1'], (_req: Request, res: Response) => {
  res.status(404).json({ error: 'Not found.' });
});

/* -------------------------------------------------------------------------- */
/* Static assets + bootstrap                                                  */
/* -------------------------------------------------------------------------- */

async function bootstrap() {
  if (IS_PRODUCTION) {
    const distPath = path.join(process.cwd(), 'dist');

    // Hashed Vite bundles are immutable and safe to cache hard.
    app.use(
      '/assets',
      express.static(path.join(distPath, 'assets'), {
        immutable: true,
        maxAge: '1y',
      }),
    );
    app.use(express.static(distPath, { maxAge: '1h', index: false }));

    app.get('*', (_req: Request, res: Response) => {
      res.setHeader('Cache-Control', 'no-cache');
      res.sendFile(path.join(distPath, 'index.html'));
    });
  } else {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  }

  const server = app.listen(PORT, '0.0.0.0', () => {
    console.log(
      `Spiking LLM Hub listening on :${PORT} | env=${IS_PRODUCTION ? 'production' : 'development'} | generation=${GEMINI_API_KEY ? 'live' : 'simulated'} | gateway=${GATEWAY_API_KEYS.length > 0 ? 'enabled' : 'disabled'}`,
    );
    if (IS_PRODUCTION && GATEWAY_API_KEYS.length === 0) {
      console.warn('GATEWAY_API_KEYS is unset: /v1 will return 503. This is the safe default.');
    }
    if (IS_PRODUCTION && ALLOWED_ORIGINS.size === 0) {
      console.info(
        'No ALLOWED_ORIGINS/APP_URL set. Same-origin requests still work; only third-party browser origins are refused.',
      );
    }
  });

  // Cloud Run sends SIGTERM before reclaiming an instance. Draining cleanly
  // avoids dropping in-flight requests during a deploy.
  const shutdown = (signal: string) => () => {
    console.log(`${signal} received, draining connections...`);
    clearInterval(sweepTimer);
    server.close(() => process.exit(0));
    setTimeout(() => process.exit(0), 10_000).unref();
  };
  process.on('SIGTERM', shutdown('SIGTERM'));
  process.on('SIGINT', shutdown('SIGINT'));
}

bootstrap().catch((err) => {
  console.error('Bootstrapping failed:', err);
  process.exit(1);
});

export default app;

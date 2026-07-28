# Spiking LLM Hub

An interactive explorer for open-source **spiking neural network (SNN) language models**, with a
playground that visualises how membrane threshold, decay and leak affect a simulated
leaky integrate-and-fire network.

## What this is, and what it is not

**It is** a catalogue of nine real SNN language model projects with verified links to their
papers, code and weights, plus a teaching playground that models neuromorphic energy
characteristics.

**It is not** running any spiking neural network. Chat replies are produced by a
general-purpose LLM prompted to answer in character as the selected architecture, and the
metrics under each reply are calculated from a formula using published per-operation energy
figures — they are estimates, not hardware measurements. The UI states this in several
places, and exported transcripts carry the same note.

If you want the real models, use the Code / Paper / Weights links in the app. Four of the
nine have downloadable weights; the rest are code or method releases.

## Running locally

**Prerequisites:** Node.js 20 to 24. Verified against Node 24.

```bash
npm install
cp .env.example .env           # then fill in the values you need
npm run dev                    # http://localhost:8080
```

`.env` is the filename `dotenv` reads by default — `.env.local` will be ignored by the
server. It is already covered by `.gitignore`.

The app runs without a `GEMINI_API_KEY` — it serves clearly labelled simulated responses
instead, which is enough to develop the whole UI including streaming.

### Scripts

| Script | Purpose |
| --- | --- |
| `npm run dev` | Vite dev server via Express middleware |
| `npm run build` | Builds the client and bundles the server to `dist/` |
| `npm start` | Runs the production build (sets `NODE_ENV=production`) |
| `npm run lint` | TypeScript typecheck, no emit |
| `npm run check:links` | Verifies every URL in `src/data.ts` still resolves |

## Configuration

All variables are documented in `.env.example`. The ones that matter most:

| Variable | Default | Notes |
| --- | --- | --- |
| `PORT` | `8080` | Injected by Cloud Run. Never hardcode it. |
| `GEMINI_API_KEY` | unset | Without it the app serves simulated responses. Store in Secret Manager. |
| `GEMINI_MODEL` | `gemini-3.5-flash` | Any current Gemini model id. |
| `GATEWAY_API_KEYS` | empty | Comma-separated bearer tokens for `/v1`. **Empty disables `/v1` entirely (503).** |
| `ALLOWED_ORIGINS` | empty | Extra *third-party* browser origins. Same-origin calls always work without config. |
| `DAILY_UPSTREAM_BUDGET` | `2000` | Hard ceiling on billable calls per rolling 24h, per instance. |
| `PLAYGROUND_MAX` | `15` | Requests per minute per IP for the browser playground. |

## API

### `POST /api/inference` — internal

Powers the playground. Same-origin only and rate limited per visitor, so it is not usable as
a public API. Send `"stream": true` for Server-Sent Events.

### `POST /v1/chat/completions` — public gateway

OpenAI chat-completions format. Requires `Authorization: Bearer <key>` where the key is one of
`GATEWAY_API_KEYS`. Supports `stream: true`, and accepts the non-standard `threshold`, `decay`
and `leak` fields.

```bash
curl -X POST "$APP_URL/v1/chat/completions" \
  -H "Authorization: Bearer $SPIKING_HUB_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"spikegpt","messages":[{"role":"user","content":"What is an SNN?"}]}'
```

Also available: `GET /v1/models`, and `GET /healthz` for probes.

## Deploying to Cloud Run

This repo is wired to a Cloud Run service that redeploys on push.

**Before the first push of this version**, set these on the service, because the defaults are
deliberately closed:

1. **Container port must be 8080** (or set `PORT` to whatever the service expects).
2. **`GEMINI_API_KEY` should be a Secret Manager secret**, mounted as an environment variable —
   not a plain env var.
3. **`GATEWAY_API_KEYS`** — generate with `openssl rand -hex 32`. Leave it unset only if you
   want `/v1` disabled.
4. **`APP_URL`** — optional. Same-origin requests are detected from the request itself,
   so the playground works without it. Only set `APP_URL`/`ALLOWED_ORIGINS` if a *different*
   site needs to call this API from a browser.

```bash
gcloud run services update SERVICE_NAME \
  --region REGION \
  --port 8080 \
  --set-secrets GEMINI_API_KEY=gemini-api-key:latest \
  --set-env-vars "APP_URL=https://YOUR-SERVICE-URL,GATEWAY_API_KEYS=YOUR_KEY"
```

After deploying, check `https://YOUR-SERVICE-URL/healthz`. It reports whether generation is
`live` or `simulated` and whether the gateway is enabled, without leaking configuration.

To roll back, use Cloud Run revisions — every deploy creates one, and traffic can be pointed
back to the previous revision immediately.

## Adding a model

Models live in `src/data.ts`. Every entry must have real, checked links; set a field to `null`
rather than guessing. `npm run check:links` runs in CI on any PR touching that file and fails
on a dead URL.

Set `status` honestly:

- `weights-released` — downloadable checkpoints exist
- `code-only` — implementation published, no checkpoints
- `method` — a technique applied to someone else's base model

## Known gaps

- No spiking model is actually executed; see the disclaimer above.
- The metric model's per-architecture scaling factors are illustrative, not derived from
  published measurements.
- Rate limiting is in-memory and therefore per-instance; with multiple instances the effective
  limit is higher than configured.
- There is no automated test suite yet.

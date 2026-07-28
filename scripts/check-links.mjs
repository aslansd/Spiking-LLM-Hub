/**
 * Verifies every external URL in src/data.ts still resolves.
 *
 * Run locally with `npm run check:links`. CI runs it on a schedule rather than
 * on every push, so a third party taking their repo down does not block an
 * unrelated deploy.
 *
 * Reads data.ts as text and extracts URLs with a regex, so the script has no
 * build step and no dependencies.
 */

import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DATA_FILE = path.join(HERE, '..', 'src', 'data.ts');

const TIMEOUT_MS = 20_000;
const CONCURRENCY = 4;

// Some hosts reject HEAD or bot-like agents. Present as a normal browser and
// fall back to a ranged GET before declaring a link dead.
const HEADERS = {
  'User-Agent':
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36',
  Accept: 'text/html,application/xhtml+xml,*/*',
};

async function request(url, method) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    return await fetch(url, {
      method,
      redirect: 'follow',
      signal: controller.signal,
      headers: method === 'GET' ? { ...HEADERS, Range: 'bytes=0-2047' } : HEADERS,
    });
  } finally {
    clearTimeout(timer);
  }
}

async function checkUrl(url) {
  try {
    let res = await request(url, 'HEAD');
    if (res.status === 405 || res.status === 403 || res.status === 501) {
      res = await request(url, 'GET');
    }
    if (res.ok || res.status === 206) {
      return { url, ok: true, status: res.status };
    }
    return { url, ok: false, status: res.status, reason: `HTTP ${res.status}` };
  } catch (err) {
    return {
      url,
      ok: false,
      status: 0,
      reason: err?.name === 'AbortError' ? 'timed out' : err?.message || 'network error',
    };
  }
}

async function main() {
  const source = await readFile(DATA_FILE, 'utf8');

  const urls = [...new Set(source.match(/https?:\/\/[^\s"'`)]+/g) || [])].filter(
    (u) => !u.includes('example.com'),
  );

  if (urls.length === 0) {
    console.error('No URLs found in src/data.ts. Has the file moved?');
    process.exit(1);
  }

  console.log(`Checking ${urls.length} links from src/data.ts\n`);

  const results = [];
  const queue = [...urls];

  await Promise.all(
    Array.from({ length: Math.min(CONCURRENCY, queue.length) }, async () => {
      while (queue.length) {
        const url = queue.shift();
        const result = await checkUrl(url);
        results.push(result);
        console.log(`${result.ok ? 'ok  ' : 'DEAD'}  ${result.url}${result.ok ? '' : `  (${result.reason})`}`);
      }
    }),
  );

  const dead = results.filter((r) => !r.ok);

  console.log(`\n${results.length - dead.length}/${results.length} links reachable.`);

  if (dead.length > 0) {
    console.error('\nDead links found. Update src/data.ts, or set the field to null');
    console.error('if the resource no longer exists. Never substitute a guess.\n');
    for (const d of dead) console.error(`  ${d.url} — ${d.reason}`);
    process.exit(1);
  }
}

main().catch((err) => {
  console.error('Link check failed to run:', err);
  process.exit(1);
});

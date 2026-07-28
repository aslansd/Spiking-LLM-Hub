/**
 * Minimal Server-Sent Events reader.
 *
 * The browser's own EventSource cannot be used here because it only issues GET
 * requests and cannot send a JSON body or an Authorization header, so the
 * stream is read straight off the fetch response instead.
 */

export interface SSEMessage {
  event: string;
  data: any;
}

export async function* readSSE(response: Response): AsyncGenerator<SSEMessage> {
  if (!response.body) {
    throw new Error('This browser cannot read streaming responses.');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Frames are separated by a blank line. Tolerate CRLF from proxies.
      let boundary: number;
      while ((boundary = findBoundary(buffer)) !== -1) {
        const raw = buffer.slice(0, boundary).replace(/\r/g, '');
        buffer = buffer.slice(boundary).replace(/^(\r?\n){2}/, '');

        const parsed = parseFrame(raw);
        if (parsed) yield parsed;
      }
    }
  } finally {
    // Releasing the lock lets an aborted fetch tear the socket down promptly.
    try {
      reader.releaseLock();
    } catch {
      /* already released */
    }
  }
}

function findBoundary(buffer: string): number {
  const lf = buffer.indexOf('\n\n');
  const crlf = buffer.indexOf('\r\n\r\n');
  if (lf === -1) return crlf;
  if (crlf === -1) return lf;
  return Math.min(lf, crlf);
}

function parseFrame(raw: string): SSEMessage | null {
  // Lines beginning with ':' are comments, used here as keep-alive pings.
  if (!raw.trim() || raw.startsWith(':')) return null;

  let event = 'message';
  const dataLines: string[] = [];

  for (const line of raw.split('\n')) {
    if (line.startsWith('event:')) {
      event = line.slice(6).trim();
    } else if (line.startsWith('data:')) {
      dataLines.push(line.slice(5).replace(/^ /, ''));
    }
  }

  if (dataLines.length === 0) return null;

  const payload = dataLines.join('\n');

  // OpenAI-compatible streams terminate with this sentinel rather than JSON.
  if (payload === '[DONE]') return { event: 'done', data: {} };

  try {
    return { event, data: JSON.parse(payload) };
  } catch {
    return null;
  }
}

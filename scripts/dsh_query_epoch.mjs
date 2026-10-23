// Preserve real user Query identity across the DSH -> OpenAI wire conversion.
import { createHash } from 'node:crypto';

export const DEFAULT_QUERY_ROUTE = Object.freeze({
  provider: 'kvmem-main-dflash2-continuation',
  model: 'kvmem-dsh-dflash2',
  baseURL: 'http://127.0.0.1:18385/v1',
});

// A deployment passes this fourth argument explicitly. Missing fields never
// inherit the old route, so a misspelled configuration cannot silently bind it.
export function normalizeQueryRoute(route) {
  if (!route || typeof route !== 'object' || Array.isArray(route)) {
    throw new TypeError('KVMem Query route must be an object');
  }
  for (const field of ['provider', 'model', 'baseURL']) {
    const value = route[field];
    if (typeof value !== 'string' || !value || value !== value.trim() ||
        /[\u0000-\u001f\u007f]/.test(value)) {
      throw new TypeError('KVMem Query route requires a nonempty ' + field);
    }
  }
  let endpoint;
  try { endpoint = new URL(route.baseURL); } catch {
    throw new TypeError('KVMem Query baseURL must be an absolute HTTP(S) URL');
  }
  if (!/^https?:\/\//i.test(route.baseURL) || route.baseURL.includes('\\') ||
      !['http:', 'https:'].includes(endpoint.protocol) ||
      endpoint.username || endpoint.password || endpoint.search || endpoint.hash) {
    throw new TypeError('KVMem Query baseURL must be HTTP(S), without credentials, query, or fragment');
  }
  return Object.freeze({
    provider: route.provider,
    model: route.model,
    baseURL: route.baseURL.replace(/\/+$/, ''),
  });
}

const hash = text => createHash('sha256').update(text).digest('hex');
const textOf = content => typeof content === 'string' ? content :
  (content || []).filter(b => b.type === 'text').map(b => b.text).join('');

export function kvmemQueryEpochPayload(provider, model, options, route = DEFAULT_QUERY_ROUTE) {
  const target = normalizeQueryRoute(route);
  if (provider !== target.provider || model.id !== target.model ||
      model.baseUrl?.replace(/\/+$/, '') !== target.baseURL) return undefined;
  const users = [];
  let query = null;
  for (const message of options.messages) {
    if (message.role === 'assistant') continue;
    const text = textOf(message.content);
    const results = message.content.filter(b => b.type === 'tool-result');
    if (message.role === 'system' || text.length || !results.length) {
      if (message.role === 'user' && message.source?.kind === 'user') {
        if (typeof message.id !== 'string' || !message.id) throw new Error('User Query has no stable identity');
        query = { ordinal: users.length, id: message.id };
      }
      users.push(hash(text));
    }
  }
  const digest = hash(JSON.stringify(users));
  return payload => {
    const wire = payload.messages.filter(m => m.role === 'user').map(m => hash(textOf(m.content)));
    if (hash(JSON.stringify(wire)) !== digest) throw new Error('KVMem Query provenance differs from provider user messages');
    if (payload.prompt_cache_key !== String(options.sessionId)) throw new Error('KVMem Query session identity differs');
    payload.kvmem_query_provenance = {
      schema_version: 1, session_id: String(options.sessionId), user_messages_sha256: digest,
      query_user_ordinal: query?.ordinal ?? null, query_message_id: query?.id ?? null,
    };
  };
}

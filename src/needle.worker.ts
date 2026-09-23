// Hosts the Needle 3 WebAssembly engine off the main thread. The engine is a
// single global, non-thread-safe instance, so requests are handled one at a time.
import needleSource from './vendor/needle/needle.js?raw';
import needleWasmUrl from './vendor/needle/needle.wasm?url';

type NeedleModule = {
  ccall: (name: string, returnType: 'number' | 'string' | null, argTypes: string[], args: unknown[]) => unknown;
  _malloc: (size: number) => number;
  HEAPU8: Uint8Array;
  UTF8ToString: (pointer: number) => string;
};

export type NeedleWorkerRequest =
  | { id: number; type: 'load'; modelUrl: string; modelSha256: string; toolsJson: string }
  | { id: number; type: 'plan'; input: string; maxNewTokens: number; toolsJson: string };

export type NeedleWorkerResponse =
  | { id: number; type: 'progress'; message: string }
  | { id: number; type: 'result'; value: unknown }
  | { id: number; type: 'error'; message: string };

const MODEL_CACHE = 'needle-models-v1';
const OUTPUT_CAPACITY = 16_384;

let needle: NeedleModule | null = null;
let outputPointer = 0;
let activeTools = '';
let defaultTools = '';
let queue = Promise.resolve();

// Typed locally: the shared tsconfig uses DOM types, which conflict with the webworker lib.
const scope = self as unknown as {
  postMessage: (message: NeedleWorkerResponse) => void;
  onmessage: ((event: MessageEvent<NeedleWorkerRequest>) => void) | null;
};
const post = (message: NeedleWorkerResponse) => scope.postMessage(message);

function lastError(module: NeedleModule) {
  return String(module.ccall('needle_last_error', 'string', [], []) || 'Unknown Needle error.');
}

async function sha256(buffer: ArrayBuffer) {
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', buffer));
  return Array.from(digest, (byte) => byte.toString(16).padStart(2, '0')).join('');
}

async function download(url: string, onProgress: (message: string) => void) {
  const response = await fetch(url);
  if (!response.ok || !response.body) throw new Error(`Needle model download failed (${response.status}).`);
  const total = Number(response.headers.get('content-length')) || 0;
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.byteLength;
    if (total) onProgress(`Downloading Needle 3 weights… ${Math.round((received / total) * 100)}%`);
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes.buffer;
}

/** Returns verified model bytes, from the browser cache when a valid copy exists. */
async function modelBytes(url: string, expectedSha256: string, onProgress: (message: string) => void) {
  const cache = await caches.open(MODEL_CACHE).catch(() => null);
  const cached = await cache?.match(url);
  if (cached) {
    onProgress('Verifying cached Needle 3 weights…');
    const buffer = await cached.arrayBuffer();
    if (await sha256(buffer) === expectedSha256) return buffer;
    await cache?.delete(url);
  }
  const buffer = await download(url, onProgress);
  if (await sha256(buffer) !== expectedSha256) throw new Error('Needle model checksum did not match the pinned release.');
  await cache?.put(url, new Response(buffer.slice(0), { headers: { 'content-type': 'application/octet-stream' } })).catch(() => undefined);
  return buffer;
}

async function createNeedle(): Promise<NeedleModule> {
  // The vendored engine is an Emscripten UMD script. Evaluate it with a
  // CommonJS shim and hand it the WebAssembly bytes directly.
  const shim = { exports: {} as unknown };
  new Function('module', 'exports', needleSource)(shim, shim.exports);
  const factory = shim.exports as (options: Record<string, unknown>) => Promise<NeedleModule>;
  const wasmResponse = await fetch(needleWasmUrl);
  if (!wasmResponse.ok) throw new Error(`Needle engine download failed (${wasmResponse.status}).`);
  return factory({ wasmBinary: await wasmResponse.arrayBuffer(), print: () => undefined, printErr: (text: string) => console.warn(text) });
}

async function load(request: Extract<NeedleWorkerRequest, { type: 'load' }>) {
  const progress = (message: string) => post({ id: request.id, type: 'progress', message });
  progress('Starting the Needle 3 WebAssembly engine…');
  const module = needle ?? await createNeedle();
  const bytes = new Uint8Array(await modelBytes(request.modelUrl, request.modelSha256, progress));
  progress('Loading Needle 3 weights…');
  // The engine may keep referring to these bytes, so they are never freed.
  const pointer = module._malloc(bytes.byteLength);
  module.HEAPU8.set(bytes, pointer);
  if (Number(module.ccall('needle_load', 'number', ['number', 'number'], [pointer, BigInt(bytes.byteLength)])) < 0) {
    throw new Error(`Needle could not load its weights: ${lastError(module)}`);
  }
  const prefix = initTools(module, request.toolsJson);
  defaultTools = request.toolsJson;
  outputPointer ||= module._malloc(OUTPUT_CAPACITY);
  needle = module;
  return { prefixTokens: prefix };
}

/** Needle can be re-initialized with a different tool catalog without reloading weights. */
function initTools(module: NeedleModule, toolsJson: string) {
  const prefix = Number(module.ccall('needle_init', 'number', ['string', 'string', 'number'], ['', toolsJson, 0]));
  if (prefix < 0) throw new Error(`Needle could not accept the tool catalog: ${lastError(module)}`);
  activeTools = toolsJson;
  return prefix;
}

function plan(request: Extract<NeedleWorkerRequest, { type: 'plan' }>) {
  if (!needle) throw new Error('Needle 3 is not loaded.');
  if (request.toolsJson !== activeTools) initTools(needle, request.toolsJson);
  needle.ccall('needle_reset', null, [], []);
  const status = Number(needle.ccall('needle_complete', 'number', ['string', 'number', 'number', 'number'],
    [request.input, request.maxNewTokens, outputPointer, OUTPUT_CAPACITY]));
  if (status < 0) throw new Error(`Needle could not plan that request: ${lastError(needle)}`);
  return { output: needle.UTF8ToString(outputPointer) };
}

scope.onmessage = (event) => {
  const request = event.data;
  queue = queue.then(async () => {
    try {
      const value = request.type === 'load' ? await load(request) : plan(request);
      post({ id: request.id, type: 'result', value });
    } catch (error) {
      post({ id: request.id, type: 'error', message: error instanceof Error ? error.message : String(error) });
    }
    // Re-reading the full catalog takes seconds on CPU, so restore it right
    // after a one-off catalog while the user reads the result.
    if (needle && defaultTools && activeTools !== defaultTools) {
      try { initTools(needle, defaultTools); } catch (error) { console.warn('Needle could not restore its tool catalog', error); }
    }
  });
};

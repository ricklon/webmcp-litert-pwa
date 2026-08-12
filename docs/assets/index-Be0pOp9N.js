var Ga=Object.defineProperty;var Pa=(b,m,c)=>m in b?Ga(b,m,{enumerable:!0,configurable:!0,writable:!0,value:c}):b[m]=c;var wr=(b,m,c)=>Pa(b,typeof m!="symbol"?m+"":m,c);const za={add:`// Elementwise residual add: y[i] = a[i] + b[i].
struct Params { n: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  if (i >= p.n) { return; }
  y[i] = a[i] + b[i];
}
`,argmax:`// GPU argmax over the logits, writing one token id into a GPU buffer so the token never leaves the
// GPU (enables the deferred-sync decode loop). Single workgroup, WG threads strided-scan the N
// logits tracking (maxVal, maxIdx), then a shared-mem tree reduction. Tie-break = LOWEST index, to
// match the CPU argmax (strict > keeps the first max). No subgroup ops -> all devices.
override WG: u32 = 256u;
struct Params { N: u32, outIdx: u32, _0: u32, _1: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read_write> outTok: array<u32>;   // outTok[p.outIdx] = argmax

var<workgroup> sval: array<f32, 256>;
var<workgroup> sidx: array<u32, 256>;

@compute @workgroup_size(WG)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  var bv = -3.4e38;
  var bi = 0u;
  for (var i = tid; i < p.N; i = i + WG) {
    let v = logits[i];
    if (v > bv) { bv = v; bi = i; }      // strict > keeps the lowest index within this thread's stride
  }
  sval[tid] = bv; sidx[tid] = bi;
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s = s >> 1u) {
    if (tid < s) {
      let ov = sval[tid + s]; let oi = sidx[tid + s];
      if (ov > sval[tid] || (ov == sval[tid] && oi < sidx[tid])) { sval[tid] = ov; sidx[tid] = oi; }
    }
    workgroupBarrier();
  }
  if (tid == 0u) { outTok[p.outIdx] = sidx[0]; }
}
`,argmax_masked:`// Masked argmax: like argmax.wgsl but skips any id already chosen in a prior round, and writes BOTH
// the winning id and its logit value. Calling it K times (roundCount = 0..K-1, all in one compute
// pass so each round sees the prior rounds' writes) yields the exact top-K (id, logit) pairs in
// descending order = ONNX TopK over the (penalty-filtered) logits, which is what the transformers.js
// sampler consumes. Then only K pairs are read back (not the full vocab), and the CPU does
// temperature + softmax + multinomial. Single workgroup, no subgroup ops -> all devices. Tie-break =
// lowest index (strict >), matching argmax.wgsl / ORT TopK in practice.
override WG: u32 = 256u;
struct Params { N: u32, roundCount: u32, _0: u32, _1: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read_write> candIds: array<u32>;   // [K]; reads 0..roundCount-1, writes [roundCount]
@group(0) @binding(3) var<storage, read_write> candVals: array<f32>;  // [K]; writes [roundCount]

var<workgroup> sval: array<f32, 256>;
var<workgroup> sidx: array<u32, 256>;

@compute @workgroup_size(WG)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  var bv = -3.4e38;
  var bi = 0u;
  for (var i = tid; i < p.N; i = i + WG) {
    let v = logits[i];
    if (v > bv) {
      var skip = false;
      for (var r = 0u; r < p.roundCount; r = r + 1u) { if (candIds[r] == i) { skip = true; break; } }
      if (!skip) { bv = v; bi = i; }     // strict > keeps the lowest index within this thread's stride
    }
  }
  sval[tid] = bv; sidx[tid] = bi;
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s = s >> 1u) {
    if (tid < s) {
      let ov = sval[tid + s]; let oi = sidx[tid + s];
      if (ov > sval[tid] || (ov == sval[tid] && oi < sidx[tid])) { sval[tid] = ov; sidx[tid] = oi; }
    }
    workgroupBarrier();
  }
  if (tid == 0u) { candIds[p.roundCount] = sidx[0]; candVals[p.roundCount] = sval[0]; }
}
`,attention_online:`// Causal GQA attention with online (flash) softmax, head-dim up to the workgroup size (256) - the
// Qwen3.5 full-attention layers use head_dim 256, past the <=128 the register-array kernels assume.
// One workgroup per query (s,h); thread d owns output dim d. Streams keys j<=s keeping running
// max/sum/acc, so no O(S) score storage. Output gate + RoPE + QK-norm are applied separately.
override WGD: u32 = 256u;                  // threads == head_dim D
struct Params { S: u32, H: u32, KV: u32, D: u32, scale: f32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;    // [S, H, D]
@group(0) @binding(2) var<storage, read> k: array<f32>;    // [S, KV, D]
@group(0) @binding(3) var<storage, read> v: array<f32>;    // [S, KV, D]
@group(0) @binding(4) var<storage, read_write> outp: array<f32>; // [S, H, D]
var<workgroup> qsh: array<f32, 256>;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(WGD)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let qi = wg.x;                 // query flat index = s*H + h
  let s = qi / p.H;
  let h = qi % p.H;
  let hkv = h / (p.H / p.KV);    // GQA: which kv head
  let d = lid.x;
  let D = p.D;
  if (d < D) { qsh[d] = q[qi * D + d]; }
  workgroupBarrier();

  var m = -1e30;
  var l = 0.0;
  var acc = 0.0;
  for (var j = 0u; j <= s; j = j + 1u) {
    red[d] = select(0.0, qsh[d] * k[(j * p.KV + hkv) * D + d], d < D);
    workgroupBarrier();
    for (var st = WGD / 2u; st > 0u; st = st >> 1u) {
      if (d < st) { red[d] = red[d] + red[d + st]; }
      workgroupBarrier();
    }
    let score = red[0] * p.scale;
    let mn = max(m, score);
    let corr = exp(m - mn);
    let pj = exp(score - mn);
    l = l * corr + pj;
    if (d < D) { acc = acc * corr + pj * v[(j * p.KV + hkv) * D + d]; }
    m = mn;
    workgroupBarrier();               // before next j overwrites red
  }
  if (d < D) { outp[qi * D + d] = acc / l; }
}
`,attention_online_cache:`// Causal GQA attention (online/flash softmax, head_dim up to 256) reading K/V from the persistent
// f32 cache (Kc/Vc, layout [pos*KV + kv_head, D]) - the Qwen3.5 full-attention path for both prefill
// and decode. One workgroup per query (s,h); thread d owns output dim d. The query at absolute
// position posBase+s attends to cache positions 0 .. posBase+s (causal). Keys are cached already
// RoPE'd, so no read-time rotation.
override WGD: u32 = 256u;                  // threads == head_dim D
struct Params { S: u32, H: u32, KV: u32, D: u32, scale: f32, posBase: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;    // [S, H, D]
@group(0) @binding(2) var<storage, read> kc: array<f32>;   // cache [cap*KV, D]
@group(0) @binding(3) var<storage, read> vc: array<f32>;   // cache [cap*KV, D]
@group(0) @binding(4) var<storage, read_write> outp: array<f32>; // [S, H, D]
var<workgroup> qsh: array<f32, 256>;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(WGD)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let qi = wg.x;                 // query flat index = s*H + h
  let s = qi / p.H;
  let h = qi % p.H;
  let hkv = h / (p.H / p.KV);
  let d = lid.x;
  let D = p.D;
  if (d < D) { qsh[d] = q[qi * D + d]; }
  workgroupBarrier();

  var m = -1e30;
  var l = 0.0;
  var acc = 0.0;
  let last = p.posBase + s;      // inclusive: attend cache positions 0..last
  for (var j = 0u; j <= last; j = j + 1u) {
    red[d] = select(0.0, qsh[d] * kc[(j * p.KV + hkv) * D + d], d < D);
    workgroupBarrier();
    for (var st = WGD / 2u; st > 0u; st = st >> 1u) {
      if (d < st) { red[d] = red[d] + red[d + st]; }
      workgroupBarrier();
    }
    let score = red[0] * p.scale;
    let mn = max(m, score);
    let corr = exp(m - mn);
    let pj = exp(score - mn);
    l = l * corr + pj;
    if (d < D) { acc = acc * corr + pj * vc[(j * p.KV + hkv) * D + d]; }
    m = mn;
    workgroupBarrier();
  }
  if (d < D) { outp[qi * D + d] = acc / l; }
}
`,attention_online_cache_kv8:`// q8 variant of attention_online_cache: the Qwen3.5 full-attention path reading K/V from the packed
// snorm8 cache (kcQ/vcQ = 4 x snorm8 per u32 word, kcS/vcS = one f32 scale per 32-element block,
// llama.cpp q8_0-style, written by copy_kv8). Each element is dequantized with one unpack4x8snorm +
// block-scale multiply at read time; all online-softmax arithmetic stays f32, so this matches the f32
// attention_online_cache exactly except for the single snorm8 rounding of K/V at write (nothing
// compounds). Same structure: one workgroup per query (s,h), thread d owns output dim d. head_dim <=256.
override WGD: u32 = 256u;                  // threads == head_dim D
struct Params { S: u32, H: u32, KV: u32, D: u32, scale: f32, posBase: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;    // [S, H, D]
@group(0) @binding(2) var<storage, read> kcQ: array<u32>;  // cache [cap*KV, D/4] packed snorm8
@group(0) @binding(3) var<storage, read> kcS: array<f32>;  // cache [cap*KV, D/32] block scales
@group(0) @binding(4) var<storage, read> vcQ: array<u32>;  // cache [cap*KV, D/4]
@group(0) @binding(5) var<storage, read> vcS: array<f32>;  // cache [cap*KV, D/32]
@group(0) @binding(6) var<storage, read_write> outp: array<f32>; // [S, H, D]
var<workgroup> qsh: array<f32, 256>;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(WGD)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let qi = wg.x;                 // query flat index = s*H + h
  let s = qi / p.H;
  let h = qi % p.H;
  let hkv = h / (p.H / p.KV);
  let d = lid.x;
  let D = p.D;
  let W4 = D / 4u;
  let NB = D / 32u;
  if (d < D) { qsh[d] = q[qi * D + d]; }
  workgroupBarrier();

  var m = -1e30;
  var l = 0.0;
  var acc = 0.0;
  let last = p.posBase + s;      // inclusive: attend cache positions 0..last
  for (var j = 0u; j <= last; j = j + 1u) {
    let row = j * p.KV + hkv;
    var kval = 0.0;
    if (d < D) { kval = unpack4x8snorm(kcQ[row * W4 + (d >> 2u)])[d & 3u] * kcS[row * NB + (d >> 5u)]; }
    red[d] = select(0.0, qsh[d] * kval, d < D);
    workgroupBarrier();
    for (var st = WGD / 2u; st > 0u; st = st >> 1u) {
      if (d < st) { red[d] = red[d] + red[d + st]; }
      workgroupBarrier();
    }
    let score = red[0] * p.scale;
    let mn = max(m, score);
    let corr = exp(m - mn);
    let pj = exp(score - mn);
    l = l * corr + pj;
    if (d < D) {
      let vval = unpack4x8snorm(vcQ[row * W4 + (d >> 2u)])[d & 3u] * vcS[row * NB + (d >> 5u)];
      acc = acc * corr + pj * vval;
    }
    m = mn;
    workgroupBarrier();
  }
  if (d < D) { outp[qi * D + d] = acc / l; }
}
`,attention_sg:`// Causal GQA attention, subgroup-parallel: one subgroup (= one workgroup) per (query, head).
// Lanes split head_dim; flash-style online softmax over the cached positions; the per-position
// score (q.k) is reduced with subgroupAdd. Fixes the decode bottleneck where attention ran only
// H threads. SG = device subgroup size (16/32/64, so head_dim/SG <= 8). Reads K/V from the cache.
enable subgroups;
override SG: u32 = 32u;
struct Params { S: u32, H: u32, KV: u32, D: u32, posBase: u32, Ltot: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;        // [S, H, D]
@group(0) @binding(2) var<storage, read> Kc: array<f32>;       // [Ltot, KV, D]
@group(0) @binding(3) var<storage, read> Vc: array<f32>;       // [Ltot, KV, D]
@group(0) @binding(4) var<storage, read_write> out: array<f32>; // [S, H, D]

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let idx = wg.x;
  if (idx >= p.S * p.H) { return; }
  let h = idx % p.H;
  let qi = idx / p.H;
  let pos = p.posBase + qi;
  let kvh = h / (p.H / p.KV);
  let qb = (qi * p.H + h) * p.D;
  let inv = 1.0 / sqrt(f32(p.D));
  let dper = p.D / SG;                         // <= 8 for SG>=16, D=128

  var acc: array<f32, 8>;
  for (var t = 0u; t < dper; t = t + 1u) { acc[t] = 0.0; }
  var m = -1e30;
  var l = 0.0;
  for (var j = 0u; j <= pos; j = j + 1u) {
    let kb = (j * p.KV + kvh) * p.D;
    var part = 0.0;
    for (var t = 0u; t < dper; t = t + 1u) { let d = lane + t * SG; part = part + q[qb + d] * Kc[kb + d]; }
    let score = subgroupAdd(part) * inv;       // full q.k dot, broadcast to all lanes
    let mnew = max(m, score);
    let corr = exp(m - mnew);
    let w = exp(score - mnew);
    l = l * corr + w;
    for (var t = 0u; t < dper; t = t + 1u) { let d = lane + t * SG; acc[t] = acc[t] * corr + w * Vc[kb + d]; }
    m = mnew;
  }
  let ob = (qi * p.H + h) * p.D;
  for (var t = 0u; t < dper; t = t + 1u) { let d = lane + t * SG; out[ob + d] = acc[t] / l; }
}
`,attention_sg_kv16:`// attention_sg with an f16-STORAGE KV cache (kvCache: 'f16'). Keep in lockstep with
// attention_sg.wgsl: the ONLY difference is Kc/Vc are array<f16> and each cached value is
// widened to f32 at the read. All arithmetic (dot, softmax, accumulation) stays f32, so the
// precision loss is exactly one rounding of K/V at cache-write time, nothing compounding.
enable subgroups;
enable f16;
override SG: u32 = 32u;
struct Params { S: u32, H: u32, KV: u32, D: u32, posBase: u32, Ltot: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;        // [S, H, D]
@group(0) @binding(2) var<storage, read> Kc: array<f16>;       // [Ltot, KV, D]
@group(0) @binding(3) var<storage, read> Vc: array<f16>;       // [Ltot, KV, D]
@group(0) @binding(4) var<storage, read_write> out: array<f32>; // [S, H, D]

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let idx = wg.x;
  if (idx >= p.S * p.H) { return; }
  let h = idx % p.H;
  let qi = idx / p.H;
  let pos = p.posBase + qi;
  let kvh = h / (p.H / p.KV);
  let qb = (qi * p.H + h) * p.D;
  let inv = 1.0 / sqrt(f32(p.D));
  let dper = p.D / SG;

  var acc: array<f32, 8>;
  for (var t = 0u; t < dper; t = t + 1u) { acc[t] = 0.0; }
  var m = -1e30;
  var l = 0.0;
  for (var j = 0u; j <= pos; j = j + 1u) {
    let kb = (j * p.KV + kvh) * p.D;
    var part = 0.0;
    for (var t = 0u; t < dper; t = t + 1u) { let d = lane + t * SG; part = part + q[qb + d] * f32(Kc[kb + d]); }
    let score = subgroupAdd(part) * inv;
    let mnew = max(m, score);
    let corr = exp(m - mnew);
    let w = exp(score - mnew);
    l = l * corr + w;
    for (var t = 0u; t < dper; t = t + 1u) { let d = lane + t * SG; acc[t] = acc[t] * corr + w * f32(Vc[kb + d]); }
    m = mnew;
  }
  let ob = (qi * p.H + h) * p.D;
  for (var t = 0u; t < dper; t = t + 1u) { let d = lane + t * SG; out[ob + d] = acc[t] / l; }
}
`,attention_sg_kv16_roll:`// attention_sg_kv16 for the rolling-window / attention-sinks mode (see attention_sg_roll.wgsl
// for the rope-at-read scheme). Keep in lockstep with attention_sg_kv16.wgsl: the ONLY
// difference is the K rotation in the score loop; each cached f16 value is widened to f32 at
// the read and rotated with the same \`k*cos + rot*sin\` operand order as rmsnorm_rope_sg.
// The engine only selects this kernel when SG <= D/2 (partner dim stays in-lane).
enable subgroups;
enable f16;
override SG: u32 = 32u;
struct Params { S: u32, H: u32, KV: u32, D: u32, posBase: u32, Ltot: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;        // [S, H, D] (roped, cache-relative)
@group(0) @binding(2) var<storage, read> Kc: array<f16>;       // [Ltot, KV, D] UNROPED
@group(0) @binding(3) var<storage, read> Vc: array<f16>;       // [Ltot, KV, D]
@group(0) @binding(4) var<storage, read> cosT: array<f32>;     // [positions, D/2]
@group(0) @binding(5) var<storage, read> sinT: array<f32>;     // [positions, D/2]
@group(0) @binding(6) var<storage, read_write> out: array<f32>; // [S, H, D]

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let idx = wg.x;
  if (idx >= p.S * p.H) { return; }
  let h = idx % p.H;
  let qi = idx / p.H;
  let pos = p.posBase + qi;
  let kvh = h / (p.H / p.KV);
  let qb = (qi * p.H + h) * p.D;
  let inv = 1.0 / sqrt(f32(p.D));
  let dper = p.D / SG;
  let half = p.D / 2u;
  let hs = half / SG;                          // strides from a dim to its rotate partner

  var acc: array<f32, 8>;
  for (var t = 0u; t < dper; t = t + 1u) { acc[t] = 0.0; }
  var m = -1e30;
  var l = 0.0;
  for (var j = 0u; j <= pos; j = j + 1u) {
    let kb = (j * p.KV + kvh) * p.D;
    var kd: array<f32, 8>;
    for (var t = 0u; t < dper; t = t + 1u) { kd[t] = f32(Kc[kb + lane + t * SG]); }
    var part = 0.0;
    for (var t = 0u; t < dper; t = t + 1u) {
      let d = lane + t * SG;
      var rot: f32;
      if (d < half) { rot = -kd[t + hs]; } else { rot = kd[t - hs]; }
      let rb = j * half + (d % half);
      part = part + q[qb + d] * (kd[t] * cosT[rb] + rot * sinT[rb]);
    }
    let score = subgroupAdd(part) * inv;
    let mnew = max(m, score);
    let corr = exp(m - mnew);
    let w = exp(score - mnew);
    l = l * corr + w;
    for (var t = 0u; t < dper; t = t + 1u) { let d = lane + t * SG; acc[t] = acc[t] * corr + w * f32(Vc[kb + d]); }
    m = mnew;
  }
  let ob = (qi * p.H + h) * p.D;
  for (var t = 0u; t < dper; t = t + 1u) { let d = lane + t * SG; out[ob + d] = acc[t] / l; }
}
`,attention_sg_kv8:`// attention_sg with a q8 KV cache (kvCache: 'q8'). Keep in lockstep with attention_sg.wgsl: the
// ONLY difference is Kc/Vc are packed snorm8 words dequantized at the read with their per-block
// f32 scales (32-element blocks, q8_0-style; see copy_kv8.wgsl). All arithmetic (dot, softmax,
// accumulation) stays f32. Each lane owns whole packed words, so q is read in matching groups
// of 4.
enable subgroups;
override SG: u32 = 32u;
struct Params { S: u32, H: u32, KV: u32, D: u32, posBase: u32, Ltot: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;         // [S, H, D]
@group(0) @binding(2) var<storage, read> Kq: array<u32>;        // [Ltot, KV, D/4] packed snorm8
@group(0) @binding(3) var<storage, read> Vq: array<u32>;        // [Ltot, KV, D/4] packed snorm8
@group(0) @binding(4) var<storage, read> Ks: array<f32>;        // [Ltot, KV, D/32] block scales
@group(0) @binding(5) var<storage, read> Vs: array<f32>;        // [Ltot, KV, D/32] block scales
@group(0) @binding(6) var<storage, read_write> out: array<f32>; // [S, H, D]

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let idx = wg.x;
  if (idx >= p.S * p.H) { return; }
  let h = idx % p.H;
  let qi = idx / p.H;
  let pos = p.posBase + qi;
  let kvh = h / (p.H / p.KV);
  let qb = (qi * p.H + h) * p.D;
  let inv = 1.0 / sqrt(f32(p.D));
  let W4 = p.D / 4u;
  let B32 = p.D / 32u;

  var acc: array<vec4<f32>, 8>;      // words per lane: W4/SG <= 8 for SG >= 4
  for (var t = 0u; t < 8u; t = t + 1u) { acc[t] = vec4<f32>(0.0); }
  var m = -1e30;
  var l = 0.0;
  for (var j = 0u; j <= pos; j = j + 1u) {
    let rowQ = (j * p.KV + kvh) * W4;
    let rowS = (j * p.KV + kvh) * B32;
    var part = 0.0;
    for (var w = lane; w < W4; w = w + SG) {
      let kw = unpack4x8snorm(Kq[rowQ + w]) * Ks[rowS + (w >> 3u)];
      let qv = vec4<f32>(q[qb + w * 4u], q[qb + w * 4u + 1u], q[qb + w * 4u + 2u], q[qb + w * 4u + 3u]);
      part = part + dot(qv, kw);
    }
    let score = subgroupAdd(part) * inv;
    let mnew = max(m, score);
    let corr = exp(m - mnew);
    let wgt = exp(score - mnew);
    l = l * corr + wgt;
    var wi = 0u;
    for (var w = lane; w < W4; w = w + SG) {
      let vw = unpack4x8snorm(Vq[rowQ + w]) * Vs[rowS + (w >> 3u)];
      acc[wi] = acc[wi] * corr + wgt * vw;
      wi = wi + 1u;
    }
    m = mnew;
  }
  let ob = (qi * p.H + h) * p.D;
  var wi = 0u;
  for (var w = lane; w < W4; w = w + SG) {
    let o = acc[wi] / l;
    out[ob + w * 4u] = o.x;
    out[ob + w * 4u + 1u] = o.y;
    out[ob + w * 4u + 2u] = o.z;
    out[ob + w * 4u + 3u] = o.w;
    wi = wi + 1u;
  }
}
`,attention_sg_kv8_roll:`// attention_sg_kv8 for the rolling-window / attention-sinks mode (see attention_sg_roll.wgsl
// for the rope-at-read scheme). Keep in lockstep with attention_sg_kv8.wgsl: the ONLY
// difference is the K rotation in the score loop. The cache holds UNROPED quantized keys -
// the whole point: the packed bytes are immutable, so eviction never requantizes (llama.cpp's
// K-shift cannot do this on a quantized cache at all). A word's rotate partner is the word
// D/8 away (all 4 dims of a word share one half), dequantized from global with its own scale.
enable subgroups;
override SG: u32 = 32u;
struct Params { S: u32, H: u32, KV: u32, D: u32, posBase: u32, Ltot: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;        // [S, H, D] (roped, cache-relative)
@group(0) @binding(2) var<storage, read> Kq: array<u32>;       // [Ltot, KV, D/4] packed snorm8, UNROPED
@group(0) @binding(3) var<storage, read> Vq: array<u32>;       // [Ltot, KV, D/4] packed snorm8
@group(0) @binding(4) var<storage, read> Ks: array<f32>;       // [Ltot, KV, D/32] block scales
@group(0) @binding(5) var<storage, read> Vs: array<f32>;       // [Ltot, KV, D/32] block scales
@group(0) @binding(6) var<storage, read> cosT: array<f32>;     // [positions, D/2]
@group(0) @binding(7) var<storage, read> sinT: array<f32>;     // [positions, D/2]
@group(0) @binding(8) var<storage, read_write> out: array<f32>; // [S, H, D]

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let idx = wg.x;
  if (idx >= p.S * p.H) { return; }
  let h = idx % p.H;
  let qi = idx / p.H;
  let pos = p.posBase + qi;
  let kvh = h / (p.H / p.KV);
  let qb = (qi * p.H + h) * p.D;
  let inv = 1.0 / sqrt(f32(p.D));
  let W4 = p.D / 4u;
  let B32 = p.D / 32u;
  let half = p.D / 2u;
  let hw = half / 4u;                          // words from a word to its rotate partner

  var acc: array<vec4<f32>, 8>;      // words per lane: W4/SG <= 8 for SG >= 4
  for (var t = 0u; t < 8u; t = t + 1u) { acc[t] = vec4<f32>(0.0); }
  var m = -1e30;
  var l = 0.0;
  for (var j = 0u; j <= pos; j = j + 1u) {
    let rowQ = (j * p.KV + kvh) * W4;
    let rowS = (j * p.KV + kvh) * B32;
    var part = 0.0;
    for (var w = lane; w < W4; w = w + SG) {
      let kw = unpack4x8snorm(Kq[rowQ + w]) * Ks[rowS + (w >> 3u)];
      let wp = select(w - hw, w + hw, w < hw);
      let kp = unpack4x8snorm(Kq[rowQ + wp]) * Ks[rowS + (wp >> 3u)];
      let rot = select(kp, -kp, w < hw);
      let cb = j * half + select(w - hw, w, w < hw) * 4u;
      let cs = vec4<f32>(cosT[cb], cosT[cb + 1u], cosT[cb + 2u], cosT[cb + 3u]);
      let sn = vec4<f32>(sinT[cb], sinT[cb + 1u], sinT[cb + 2u], sinT[cb + 3u]);
      let qv = vec4<f32>(q[qb + w * 4u], q[qb + w * 4u + 1u], q[qb + w * 4u + 2u], q[qb + w * 4u + 3u]);
      part = part + dot(qv, kw * cs + rot * sn);
    }
    let score = subgroupAdd(part) * inv;
    let mnew = max(m, score);
    let corr = exp(m - mnew);
    let wgt = exp(score - mnew);
    l = l * corr + wgt;
    var wi = 0u;
    for (var w = lane; w < W4; w = w + SG) {
      let vw = unpack4x8snorm(Vq[rowQ + w]) * Vs[rowS + (w >> 3u)];
      acc[wi] = acc[wi] * corr + wgt * vw;
      wi = wi + 1u;
    }
    m = mnew;
  }
  let ob = (qi * p.H + h) * p.D;
  var wi = 0u;
  for (var w = lane; w < W4; w = w + SG) {
    let o = acc[wi] / l;
    out[ob + w * 4u] = o.x;
    out[ob + w * 4u + 1u] = o.y;
    out[ob + w * 4u + 2u] = o.z;
    out[ob + w * 4u + 3u] = o.w;
    wi = wi + 1u;
  }
}
`,attention_sg_roll:`// attention_sg for the rolling-window / attention-sinks mode (overflow: 'sinks'): the cache
// holds UNROPED keys, and each cached row j is rotated AT READ by its cache-relative position
// (StreamingLLM-style; the cache bytes are immutable, so eviction compaction never re-rotates
// or requantizes anything). Keep in lockstep with attention_sg.wgsl: the ONLY difference is
// the K rotation in the score loop, written as \`k*cos + rot*sin\` with the same operand order
// as rmsnorm_rope_sg so the f32 path stays bit-identical to the roped-at-write kernels until
// the first eviction. cosT/sinT are the aux rope tables, [positions, D/2].
// Lane math: d = lane + t*SG, partner d±D/2 is (D/2)/SG strides away IN THE SAME LANE (the
// engine only selects this kernel when SG <= D/2).
enable subgroups;
override SG: u32 = 32u;
struct Params { S: u32, H: u32, KV: u32, D: u32, posBase: u32, Ltot: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;        // [S, H, D] (roped, cache-relative)
@group(0) @binding(2) var<storage, read> Kc: array<f32>;       // [Ltot, KV, D] UNROPED
@group(0) @binding(3) var<storage, read> Vc: array<f32>;       // [Ltot, KV, D]
@group(0) @binding(4) var<storage, read> cosT: array<f32>;     // [positions, D/2]
@group(0) @binding(5) var<storage, read> sinT: array<f32>;     // [positions, D/2]
@group(0) @binding(6) var<storage, read_write> out: array<f32>; // [S, H, D]

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let idx = wg.x;
  if (idx >= p.S * p.H) { return; }
  let h = idx % p.H;
  let qi = idx / p.H;
  let pos = p.posBase + qi;
  let kvh = h / (p.H / p.KV);
  let qb = (qi * p.H + h) * p.D;
  let inv = 1.0 / sqrt(f32(p.D));
  let dper = p.D / SG;
  let half = p.D / 2u;
  let hs = half / SG;                          // strides from a dim to its rotate partner

  var acc: array<f32, 8>;
  for (var t = 0u; t < dper; t = t + 1u) { acc[t] = 0.0; }
  var m = -1e30;
  var l = 0.0;
  for (var j = 0u; j <= pos; j = j + 1u) {
    let kb = (j * p.KV + kvh) * p.D;
    var kd: array<f32, 8>;
    for (var t = 0u; t < dper; t = t + 1u) { kd[t] = Kc[kb + lane + t * SG]; }
    var part = 0.0;
    for (var t = 0u; t < dper; t = t + 1u) {
      let d = lane + t * SG;
      var rot: f32;
      if (d < half) { rot = -kd[t + hs]; } else { rot = kd[t - hs]; }
      let rb = j * half + (d % half);
      part = part + q[qb + d] * (kd[t] * cosT[rb] + rot * sinT[rb]);
    }
    let score = subgroupAdd(part) * inv;
    let mnew = max(m, score);
    let corr = exp(m - mnew);
    let w = exp(score - mnew);
    l = l * corr + w;
    for (var t = 0u; t < dper; t = t + 1u) { let d = lane + t * SG; acc[t] = acc[t] * corr + w * Vc[kb + d]; }
    m = mnew;
  }
  let ob = (qi * p.H + h) * p.D;
  for (var t = 0u; t < dper; t = t + 1u) { let d = lane + t * SG; out[ob + d] = acc[t] / l; }
}
`,attention_wg:`// Causal GQA attention, no-subgroup fallback: one workgroup per (query, head); threads split
// head_dim; flash-style online softmax over the cached positions; the per-position q.k score
// is tree-reduced via shared memory. Replaces attention_cache on this path: its single thread
// per (query, head) walked the WHOLE context serially, so fallback decode degraded linearly
// with conversation length and prefill attention was latency-bound. Mirrors attention_sg with
// subgroupAdd swapped for the shared-memory reduction. Fixed workgroup of 64: the per-thread
// accumulator covers head_dim <= 128 (enforced at manifest validation) in 2 strides.
struct Params { S: u32, H: u32, KV: u32, D: u32, posBase: u32, Ltot: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;        // [S, H, D]
@group(0) @binding(2) var<storage, read> Kc: array<f32>;       // [Ltot, KV, D]
@group(0) @binding(3) var<storage, read> Vc: array<f32>;       // [Ltot, KV, D]
@group(0) @binding(4) var<storage, read_write> out: array<f32>; // [S, H, D]
var<workgroup> red: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let idx = wg.x;                        // uniform across the workgroup -> early return is barrier-safe
  if (idx >= p.S * p.H) { return; }
  let tid = lid.x;
  let h = idx % p.H;
  let qi = idx / p.H;
  let pos = p.posBase + qi;
  let kvh = h / (p.H / p.KV);
  let qb = (qi * p.H + h) * p.D;
  let inv = 1.0 / sqrt(f32(p.D));

  var acc: array<f32, 2>;
  acc[0] = 0.0;
  acc[1] = 0.0;
  var m = -1e30;
  var l = 0.0;
  for (var j = 0u; j <= pos; j = j + 1u) {
    let kb = (j * p.KV + kvh) * p.D;
    var part = 0.0;
    for (var t = 0u; t < 2u; t = t + 1u) {
      let d = tid + t * 64u;
      if (d < p.D) { part = part + q[qb + d] * Kc[kb + d]; }
    }
    red[tid] = part;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s = s >> 1u) {
      if (tid < s) { red[tid] = red[tid] + red[tid + s]; }
      workgroupBarrier();
    }
    let score = red[0] * inv;            // full q.k dot, visible to all threads
    workgroupBarrier();                  // red[0] consumed before the next position overwrites it
    let mnew = max(m, score);
    let corr = exp(m - mnew);
    let w = exp(score - mnew);
    l = l * corr + w;
    for (var t = 0u; t < 2u; t = t + 1u) {
      let d = tid + t * 64u;
      if (d < p.D) { acc[t] = acc[t] * corr + w * Vc[kb + d]; }
    }
    m = mnew;
  }
  let ob = (qi * p.H + h) * p.D;
  for (var t = 0u; t < 2u; t = t + 1u) {
    let d = tid + t * 64u;
    if (d < p.D) { out[ob + d] = acc[t] / l; }
  }
}
`,attention_wg_kv16:`// attention_wg with an f16-STORAGE KV cache (kvCache: 'f16'). Keep in lockstep with
// attention_wg.wgsl: the ONLY difference is Kc/Vc are array<f16>, widened to f32 at the read.
enable f16;
struct Params { S: u32, H: u32, KV: u32, D: u32, posBase: u32, Ltot: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;        // [S, H, D]
@group(0) @binding(2) var<storage, read> Kc: array<f16>;       // [Ltot, KV, D]
@group(0) @binding(3) var<storage, read> Vc: array<f16>;       // [Ltot, KV, D]
@group(0) @binding(4) var<storage, read_write> out: array<f32>; // [S, H, D]
var<workgroup> red: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let idx = wg.x;                        // uniform across the workgroup -> early return is barrier-safe
  if (idx >= p.S * p.H) { return; }
  let tid = lid.x;
  let h = idx % p.H;
  let qi = idx / p.H;
  let pos = p.posBase + qi;
  let kvh = h / (p.H / p.KV);
  let qb = (qi * p.H + h) * p.D;
  let inv = 1.0 / sqrt(f32(p.D));

  var acc: array<f32, 2>;
  acc[0] = 0.0;
  acc[1] = 0.0;
  var m = -1e30;
  var l = 0.0;
  for (var j = 0u; j <= pos; j = j + 1u) {
    let kb = (j * p.KV + kvh) * p.D;
    var part = 0.0;
    for (var t = 0u; t < 2u; t = t + 1u) {
      let d = tid + t * 64u;
      if (d < p.D) { part = part + q[qb + d] * f32(Kc[kb + d]); }
    }
    red[tid] = part;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s = s >> 1u) {
      if (tid < s) { red[tid] = red[tid] + red[tid + s]; }
      workgroupBarrier();
    }
    let score = red[0] * inv;
    workgroupBarrier();
    let mnew = max(m, score);
    let corr = exp(m - mnew);
    let w = exp(score - mnew);
    l = l * corr + w;
    for (var t = 0u; t < 2u; t = t + 1u) {
      let d = tid + t * 64u;
      if (d < p.D) { acc[t] = acc[t] * corr + w * f32(Vc[kb + d]); }
    }
    m = mnew;
  }
  let ob = (qi * p.H + h) * p.D;
  for (var t = 0u; t < 2u; t = t + 1u) {
    let d = tid + t * 64u;
    if (d < p.D) { out[ob + d] = acc[t] / l; }
  }
}
`,attention_wg_kv16_roll:`// attention_wg_kv16 for the rolling-window / attention-sinks mode (see attention_sg_roll.wgsl
// for the rope-at-read scheme). Keep in lockstep with attention_wg_kv16.wgsl: the ONLY
// differences are the shared-memory K stage (kk, widened to f32) - the rotate partner d±D/2
// may live in another thread's stride - and the rotation in the score loop, written as
// \`k*cos + rot*sin\` with the same operand order as rmsnorm_rope_sg.
enable f16;
struct Params { S: u32, H: u32, KV: u32, D: u32, posBase: u32, Ltot: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;        // [S, H, D] (roped, cache-relative)
@group(0) @binding(2) var<storage, read> Kc: array<f16>;       // [Ltot, KV, D] UNROPED
@group(0) @binding(3) var<storage, read> Vc: array<f16>;       // [Ltot, KV, D]
@group(0) @binding(4) var<storage, read> cosT: array<f32>;     // [positions, D/2]
@group(0) @binding(5) var<storage, read> sinT: array<f32>;     // [positions, D/2]
@group(0) @binding(6) var<storage, read_write> out: array<f32>; // [S, H, D]
var<workgroup> red: array<f32, 64>;
var<workgroup> kk: array<f32, 128>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let idx = wg.x;                        // uniform across the workgroup -> early return is barrier-safe
  if (idx >= p.S * p.H) { return; }
  let tid = lid.x;
  let h = idx % p.H;
  let qi = idx / p.H;
  let pos = p.posBase + qi;
  let kvh = h / (p.H / p.KV);
  let qb = (qi * p.H + h) * p.D;
  let inv = 1.0 / sqrt(f32(p.D));
  let half = p.D / 2u;

  var acc: array<f32, 2>;
  acc[0] = 0.0;
  acc[1] = 0.0;
  var m = -1e30;
  var l = 0.0;
  for (var j = 0u; j <= pos; j = j + 1u) {
    let kb = (j * p.KV + kvh) * p.D;
    for (var t = 0u; t < 2u; t = t + 1u) {
      let d = tid + t * 64u;
      if (d < p.D) { kk[d] = f32(Kc[kb + d]); }
    }
    workgroupBarrier();
    var part = 0.0;
    for (var t = 0u; t < 2u; t = t + 1u) {
      let d = tid + t * 64u;
      if (d < p.D) {
        var rot: f32;
        if (d < half) { rot = -kk[d + half]; } else { rot = kk[d - half]; }
        let rb = j * half + (d % half);
        part = part + q[qb + d] * (kk[d] * cosT[rb] + rot * sinT[rb]);
      }
    }
    red[tid] = part;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s = s >> 1u) {
      if (tid < s) { red[tid] = red[tid] + red[tid + s]; }
      workgroupBarrier();
    }
    let score = red[0] * inv;
    workgroupBarrier();                  // red[0] + kk consumed before the next position overwrites them
    let mnew = max(m, score);
    let corr = exp(m - mnew);
    let w = exp(score - mnew);
    l = l * corr + w;
    for (var t = 0u; t < 2u; t = t + 1u) {
      let d = tid + t * 64u;
      if (d < p.D) { acc[t] = acc[t] * corr + w * f32(Vc[kb + d]); }
    }
    m = mnew;
  }
  let ob = (qi * p.H + h) * p.D;
  for (var t = 0u; t < 2u; t = t + 1u) {
    let d = tid + t * 64u;
    if (d < p.D) { out[ob + d] = acc[t] / l; }
  }
}
`,attention_wg_kv8:`// attention_wg with a q8 KV cache (kvCache: 'q8'): the no-subgroup fallback reader for the
// packed-snorm8 cache (see copy_kv8.wgsl for the write side). Keep in lockstep with
// attention_wg.wgsl: same online softmax, all arithmetic f32; each thread owns one packed word
// (D <= 128 -> at most 32 words, so threads 32..63 only carry zeros through the reduction).
struct Params { S: u32, H: u32, KV: u32, D: u32, posBase: u32, Ltot: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;         // [S, H, D]
@group(0) @binding(2) var<storage, read> Kq: array<u32>;        // [Ltot, KV, D/4] packed snorm8
@group(0) @binding(3) var<storage, read> Vq: array<u32>;        // [Ltot, KV, D/4] packed snorm8
@group(0) @binding(4) var<storage, read> Ks: array<f32>;        // [Ltot, KV, D/32] block scales
@group(0) @binding(5) var<storage, read> Vs: array<f32>;        // [Ltot, KV, D/32] block scales
@group(0) @binding(6) var<storage, read_write> out: array<f32>; // [S, H, D]
var<workgroup> red: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let idx = wg.x;                    // uniform across the workgroup -> early return is barrier-safe
  if (idx >= p.S * p.H) { return; }
  let t = lid.x;
  let h = idx % p.H;
  let qi = idx / p.H;
  let pos = p.posBase + qi;
  let kvh = h / (p.H / p.KV);
  let qb = (qi * p.H + h) * p.D;
  let inv = 1.0 / sqrt(f32(p.D));
  let W4 = p.D / 4u;

  var qv = vec4<f32>(0.0);
  if (t < W4) {
    qv = vec4<f32>(q[qb + t * 4u], q[qb + t * 4u + 1u], q[qb + t * 4u + 2u], q[qb + t * 4u + 3u]);
  }
  var acc = vec4<f32>(0.0);
  var m = -1e30;
  var l = 0.0;
  for (var j = 0u; j <= pos; j = j + 1u) {
    let rowQ = (j * p.KV + kvh) * W4;
    let rowS = (j * p.KV + kvh) * (p.D / 32u);
    var part = 0.0;
    if (t < W4) {
      let kw = unpack4x8snorm(Kq[rowQ + t]) * Ks[rowS + (t >> 3u)];
      part = dot(qv, kw);
    }
    red[t] = part;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s = s >> 1u) {
      if (t < s) { red[t] = red[t] + red[t + s]; }
      workgroupBarrier();
    }
    let score = red[0] * inv;
    workgroupBarrier();
    let mnew = max(m, score);
    let corr = exp(m - mnew);
    let wgt = exp(score - mnew);
    l = l * corr + wgt;
    if (t < W4) {
      let vw = unpack4x8snorm(Vq[rowQ + t]) * Vs[rowS + (t >> 3u)];
      acc = acc * corr + wgt * vw;
    }
    m = mnew;
  }
  if (t < W4) {
    let ob = (qi * p.H + h) * p.D;
    let o = acc / l;
    out[ob + t * 4u] = o.x;
    out[ob + t * 4u + 1u] = o.y;
    out[ob + t * 4u + 2u] = o.z;
    out[ob + t * 4u + 3u] = o.w;
  }
}
`,attention_wg_kv8_roll:`// attention_wg_kv8 for the rolling-window / attention-sinks mode: the no-subgroup fallback of
// attention_sg_kv8_roll (see there and attention_sg_roll.wgsl for the rope-at-read scheme).
// Keep in lockstep with attention_wg_kv8.wgsl: the ONLY difference is the K rotation in the
// score loop; each thread's word rotates against its partner word D/8 away, dequantized from
// global with its own scale.
struct Params { S: u32, H: u32, KV: u32, D: u32, posBase: u32, Ltot: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;        // [S, H, D] (roped, cache-relative)
@group(0) @binding(2) var<storage, read> Kq: array<u32>;       // [Ltot, KV, D/4] packed snorm8, UNROPED
@group(0) @binding(3) var<storage, read> Vq: array<u32>;       // [Ltot, KV, D/4] packed snorm8
@group(0) @binding(4) var<storage, read> Ks: array<f32>;       // [Ltot, KV, D/32] block scales
@group(0) @binding(5) var<storage, read> Vs: array<f32>;       // [Ltot, KV, D/32] block scales
@group(0) @binding(6) var<storage, read> cosT: array<f32>;     // [positions, D/2]
@group(0) @binding(7) var<storage, read> sinT: array<f32>;     // [positions, D/2]
@group(0) @binding(8) var<storage, read_write> out: array<f32>; // [S, H, D]
var<workgroup> red: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let idx = wg.x;                    // uniform across the workgroup -> early return is barrier-safe
  if (idx >= p.S * p.H) { return; }
  let t = lid.x;
  let h = idx % p.H;
  let qi = idx / p.H;
  let pos = p.posBase + qi;
  let kvh = h / (p.H / p.KV);
  let qb = (qi * p.H + h) * p.D;
  let inv = 1.0 / sqrt(f32(p.D));
  let W4 = p.D / 4u;
  let half = p.D / 2u;
  let hw = half / 4u;                          // words from a word to its rotate partner

  var qv = vec4<f32>(0.0);
  if (t < W4) {
    qv = vec4<f32>(q[qb + t * 4u], q[qb + t * 4u + 1u], q[qb + t * 4u + 2u], q[qb + t * 4u + 3u]);
  }
  var acc = vec4<f32>(0.0);
  var m = -1e30;
  var l = 0.0;
  for (var j = 0u; j <= pos; j = j + 1u) {
    let rowQ = (j * p.KV + kvh) * W4;
    let rowS = (j * p.KV + kvh) * (p.D / 32u);
    var part = 0.0;
    if (t < W4) {
      let kw = unpack4x8snorm(Kq[rowQ + t]) * Ks[rowS + (t >> 3u)];
      let wp = select(t - hw, t + hw, t < hw);
      let kp = unpack4x8snorm(Kq[rowQ + wp]) * Ks[rowS + (wp >> 3u)];
      let rot = select(kp, -kp, t < hw);
      let cb = j * half + select(t - hw, t, t < hw) * 4u;
      let cs = vec4<f32>(cosT[cb], cosT[cb + 1u], cosT[cb + 2u], cosT[cb + 3u]);
      let sn = vec4<f32>(sinT[cb], sinT[cb + 1u], sinT[cb + 2u], sinT[cb + 3u]);
      part = dot(qv, kw * cs + rot * sn);
    }
    red[t] = part;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s = s >> 1u) {
      if (t < s) { red[t] = red[t] + red[t + s]; }
      workgroupBarrier();
    }
    let score = red[0] * inv;
    workgroupBarrier();
    let mnew = max(m, score);
    let corr = exp(m - mnew);
    let wgt = exp(score - mnew);
    l = l * corr + wgt;
    if (t < W4) {
      let vw = unpack4x8snorm(Vq[rowQ + t]) * Vs[rowS + (t >> 3u)];
      acc = acc * corr + wgt * vw;
    }
    m = mnew;
  }
  if (t < W4) {
    let ob = (qi * p.H + h) * p.D;
    let o = acc / l;
    out[ob + t * 4u] = o.x;
    out[ob + t * 4u + 1u] = o.y;
    out[ob + t * 4u + 2u] = o.z;
    out[ob + t * 4u + 3u] = o.w;
  }
}
`,attention_wg_roll:`// attention_wg for the rolling-window / attention-sinks mode: no-subgroup fallback of
// attention_sg_roll (see there for the rope-at-read scheme). Keep in lockstep with
// attention_wg.wgsl: the ONLY differences are the shared-memory K stage (kk) - the rotate
// partner d±D/2 may live in another thread's stride - and the rotation in the score loop,
// written as \`k*cos + rot*sin\` with the same operand order as rmsnorm_rope_sg.
struct Params { S: u32, H: u32, KV: u32, D: u32, posBase: u32, Ltot: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;        // [S, H, D] (roped, cache-relative)
@group(0) @binding(2) var<storage, read> Kc: array<f32>;       // [Ltot, KV, D] UNROPED
@group(0) @binding(3) var<storage, read> Vc: array<f32>;       // [Ltot, KV, D]
@group(0) @binding(4) var<storage, read> cosT: array<f32>;     // [positions, D/2]
@group(0) @binding(5) var<storage, read> sinT: array<f32>;     // [positions, D/2]
@group(0) @binding(6) var<storage, read_write> out: array<f32>; // [S, H, D]
var<workgroup> red: array<f32, 64>;
var<workgroup> kk: array<f32, 128>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let idx = wg.x;                        // uniform across the workgroup -> early return is barrier-safe
  if (idx >= p.S * p.H) { return; }
  let tid = lid.x;
  let h = idx % p.H;
  let qi = idx / p.H;
  let pos = p.posBase + qi;
  let kvh = h / (p.H / p.KV);
  let qb = (qi * p.H + h) * p.D;
  let inv = 1.0 / sqrt(f32(p.D));
  let half = p.D / 2u;

  var acc: array<f32, 2>;
  acc[0] = 0.0;
  acc[1] = 0.0;
  var m = -1e30;
  var l = 0.0;
  for (var j = 0u; j <= pos; j = j + 1u) {
    let kb = (j * p.KV + kvh) * p.D;
    for (var t = 0u; t < 2u; t = t + 1u) {
      let d = tid + t * 64u;
      if (d < p.D) { kk[d] = Kc[kb + d]; }
    }
    workgroupBarrier();
    var part = 0.0;
    for (var t = 0u; t < 2u; t = t + 1u) {
      let d = tid + t * 64u;
      if (d < p.D) {
        var rot: f32;
        if (d < half) { rot = -kk[d + half]; } else { rot = kk[d - half]; }
        let rb = j * half + (d % half);
        part = part + q[qb + d] * (kk[d] * cosT[rb] + rot * sinT[rb]);
      }
    }
    red[tid] = part;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s = s >> 1u) {
      if (tid < s) { red[tid] = red[tid] + red[tid + s]; }
      workgroupBarrier();
    }
    let score = red[0] * inv;            // full q.k dot, visible to all threads
    workgroupBarrier();                  // red[0] + kk consumed before the next position overwrites them
    let mnew = max(m, score);
    let corr = exp(m - mnew);
    let w = exp(score - mnew);
    l = l * corr + w;
    for (var t = 0u; t < 2u; t = t + 1u) {
      let d = tid + t * 64u;
      if (d < p.D) { acc[t] = acc[t] * corr + w * Vc[kb + d]; }
    }
    m = mnew;
  }
  let ob = (qi * p.H + h) * p.D;
  for (var t = 0u; t < 2u; t = t + 1u) {
    let d = tid + t * 64u;
    if (d < p.D) { out[ob + d] = acc[t] / l; }
  }
}
`,conv1d_causal:`// Depthwise causal Conv1d (kernel width K) + SiLU, for the gated-DeltaNet q/k/v stream. x is
// [S, C] (C = conv_dim channels), weight is [C, K] (per-channel taps, the GGUF ssm_conv1d layout).
// Carries a persistent left-context so segmented prefill and token-by-token decode continue across
// calls: state_in / state_out hold the last K-1 inputs ([K-1, C]); loadState!=0 uses them (else the
// causal left pad is zero). Extended input ext = [state_in (K-1), x (S)]:
//   y[t,c] = silu( sum_{j<K} w[c,j] * ext[t+j, c] ),   state_out[i,c] = ext[S+i, c]  (i < K-1)
struct Params { S: u32, C: u32, K: u32, loadState: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;          // [S, C]
@group(0) @binding(2) var<storage, read> w: array<f32>;          // [C, K]
@group(0) @binding(3) var<storage, read> state_in: array<f32>;   // [K-1, C]
@group(0) @binding(4) var<storage, read_write> y: array<f32>;    // [S, C]
@group(0) @binding(5) var<storage, read_write> state_out: array<f32>; // [K-1, C]

fn ext(m: u32, c: u32) -> f32 {
  if (m + 1u < p.K) { return select(0.0, state_in[m * p.C + c], p.loadState != 0u); }
  return x[(m - (p.K - 1u)) * p.C + c];
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  let outN = p.S * p.C;
  if (i < outN) {
    let c = i % p.C;
    let t = i / p.C;
    var acc = 0.0;
    for (var j = 0u; j < p.K; j = j + 1u) { acc = acc + w[c * p.K + j] * ext(t + j, c); }
    y[i] = acc / (1.0 + exp(-acc));  // SiLU
  } else if (i < outN + (p.K - 1u) * p.C) {
    let si = i - outN;
    let sc = si % p.C;
    let sk = si / p.C;                 // 0 .. K-2
    state_out[sk * p.C + sc] = ext(p.S + sk, sc);
  }
}
`,copy:`// Copy src[0..n) into dst[dstOff..dstOff+n). Used to append K/V into the persistent cache.
struct Params { n: u32, dstOff: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  if (i >= p.n) { return; }
  dst[p.dstOff + i] = src[i];
}
`,copy_kv16:`// copy with an f16-STORAGE destination (kvCache: 'f16'): appends f32 K/V rows into the f16
// cache (one f32 -> f16 rounding per value). Keep in lockstep with copy.wgsl.
enable f16;
struct Params { n: u32, dstOff: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f16>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  if (i >= p.n) { return; }
  dst[p.dstOff + i] = f16(src[i]);
}
`,copy_kv8:`// q8 cache append (kvCache: 'q8'): quantize f32 K/V rows into the packed-snorm8 cache, one f32
// scale per 32-element block (llama.cpp q8_0-style). One 64-thread workgroup per row of D
// elements: thread t owns packed word t (4 consecutive values), the workgroup reduces per-block
// absolute maxima through shared memory, then packs with pack4x8snorm. Replaces copy/copy_kv16
// at every cache-append site under q8. All attention arithmetic stays f32; the precision loss is
// exactly one snorm8 rounding of K/V at write time, nothing compounding.
struct Params { rows: u32, D: u32, dstRow0: u32, _p: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;          // [rows, D]
@group(0) @binding(2) var<storage, read_write> dstQ: array<u32>;   // packed 4 x snorm8 per word
@group(0) @binding(3) var<storage, read_write> dstS: array<f32>;   // [.., D/32] block scales

var<workgroup> wabs: array<f32, 64>; // per-word abs max (D <= 256 -> at most 64 words)
var<workgroup> wblk: array<f32, 8>;  // per-block scale (D/32 <= 8 blocks)

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wg.x;                    // uniform across the workgroup -> early return is barrier-safe
  if (row >= p.rows) { return; }
  let t = lid.x;
  let W4 = p.D / 4u;
  let base = row * p.D;
  var v = vec4<f32>(0.0);
  if (t < W4) {
    v = vec4<f32>(src[base + t * 4u], src[base + t * 4u + 1u], src[base + t * 4u + 2u], src[base + t * 4u + 3u]);
    wabs[t] = max(max(abs(v.x), abs(v.y)), max(abs(v.z), abs(v.w)));
  }
  workgroupBarrier();
  if (t < p.D / 32u) {
    var m = 0.0;
    for (var i = 0u; i < 8u; i = i + 1u) { m = max(m, wabs[t * 8u + i]); }
    let s = max(m, 1e-30);           // an all-zero block packs zeros, never NaN
    wblk[t] = s;
    dstS[(p.dstRow0 + row) * (p.D / 32u) + t] = s;
  }
  workgroupBarrier();
  if (t < W4) {
    dstQ[(p.dstRow0 + row) * W4 + t] = pack4x8snorm(v / wblk[t >> 3u]);
  }
}
`,deltanet_gbeta:`// DeltaNet gate/decay compute: from the a (decay input) and b (beta input) projections,
//   g[s,h]    = a_neg[h] * softplus(a[s,h] + dt_bias[h])     (<= 0, log-space decay)
//   beta[s,h] = sigmoid(b[s,h])
// per value head h. a_neg is -exp(A_log): the PrismML GGUF stores this pre-computed in the ssm_a
// tensor (verified against the transformers A_log), so no exp() here. One invocation per (s,h);
// output is [g (S*H) ; beta (S*H)] concatenated (engine binds two sub-ranges). Matches qwen35_numpy.
struct Params { S: u32, H: u32, _p0: u32, _p1: u32 };  // H = num_value_heads
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;        // [S, H]
@group(0) @binding(2) var<storage, read> b: array<f32>;        // [S, H]
@group(0) @binding(3) var<storage, read> a_neg: array<f32>;    // [H] = -exp(A_log)
@group(0) @binding(4) var<storage, read> dt_bias: array<f32>;  // [H]
@group(0) @binding(5) var<storage, read_write> out: array<f32>;// [2*S*H]: g then beta

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  let n = p.S * p.H;
  if (i >= n) { return; }
  let h = i % p.H;
  let x = a[i] + dt_bias[h];
  let sp = max(x, 0.0) + log(1.0 + exp(-abs(x)));   // softplus (stable)
  out[i] = a_neg[h] * sp;                            // g  (a_neg already = -exp(A_log))
  out[n + i] = 1.0 / (1.0 + exp(-b[i]));             // beta
}
`,deltanet_norm_gate:`// Gated RMSNorm for the DeltaNet output: y = gamma * rmsnorm(core) * silu(z), normalized over the
// value head dim (one workgroup per head-vector row). Unlike the model's plain RMSNorm this uses
// the weight directly (not 1+weight), matching tools/qwen35_numpy (Qwen3NextRMSNormGated).
override WG: u32 = 128u;
struct Params { rows: u32, DV: u32, eps: f32, _pad: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> core: array<f32>;   // [rows, DV]
@group(0) @binding(2) var<storage, read> z: array<f32>;      // [rows, DV] gate
@group(0) @binding(3) var<storage, read> gamma: array<f32>;  // [DV]
@group(0) @binding(4) var<storage, read_write> y: array<f32>;// [rows, DV]
var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wg.x;
  if (row >= p.rows) { return; }
  let tid = lid.x;
  let base = row * p.DV;
  var s = 0.0;
  for (var i = tid; i < p.DV; i = i + WG) { let c = core[base + i]; s = s + c * c; }
  sdata[tid] = s;
  workgroupBarrier();
  for (var st = WG / 2u; st > 0u; st = st >> 1u) {
    if (tid < st) { sdata[tid] = sdata[tid] + sdata[tid + st]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(sdata[0] / f32(p.DV) + p.eps);
  for (var i = tid; i < p.DV; i = i + WG) {
    let zz = z[base + i];
    y[base + i] = gamma[i] * (core[base + i] * inv) * (zz / (1.0 + exp(-zz)));  // * silu(z)
  }
}
`,deltanet_recur:`// Gated DeltaNet recurrent scan (the sequential O(1)/token gated delta rule; bitgpu's decode path
// and a correctness reference for prefill). One workgroup per value head; thread \`dv\` owns value
// column dv of the per-head state S[dk,dv], held in registers across the token loop. Per token:
//   S *= exp(g);  kv = Kn·S;  delta = (v - kv)·beta;  S += Kn⊗delta;  out = Qn·S
// with Kn = l2norm(k), Qn = l2norm(q)/sqrt(dk) (matches tools/qwen35_numpy._delta_recurrent).
// GQA: value head h reads q/k from key head h%HK. GGUF/bitgpu store value heads grouped
// [rep, n_key_heads] (transposed from HF's [n_key_heads, rep]), so the shared key/query head is
// h%HK (a "tile"), NOT h/(H/HK) (a "repeat-interleave"). loadState!=0 continues from state_in
// (persistent decode/cross-segment state); state_out always carries the final state out.
override WGV: u32 = 128u;                 // threads per workgroup == head_v_dim (dv)
struct Params { S: u32, H: u32, DK: u32, DV: u32, HK: u32, betaOff: u32, loadState: u32, tOff: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;      // [S, HK, DK]
@group(0) @binding(2) var<storage, read> k: array<f32>;      // [S, HK, DK]
@group(0) @binding(3) var<storage, read> v: array<f32>;      // [S, H, DV]
@group(0) @binding(4) var<storage, read> g: array<f32>;      // [S, H]
@group(0) @binding(5) var<storage, read> beta: array<f32>;   // [S, H]
@group(0) @binding(6) var<storage, read> state_in: array<f32>;    // [H, DK, DV]
@group(0) @binding(7) var<storage, read_write> core: array<f32>;  // [S, H, DV]
@group(0) @binding(8) var<storage, read_write> state_out: array<f32>; // [H, DK, DV]
var<workgroup> ksh: array<f32, 128>;      // current token's raw k (>= DK)
var<workgroup> qsh: array<f32, 128>;      // current token's raw q

@compute @workgroup_size(WGV)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wg.x;                           // value head
  let hk = h % p.HK;                      // GQA: shared key/query head (GGUF [rep,n_key] tile order)
  let dv = lid.x;
  let DK = p.DK;
  let sbase = h * DK * p.DV + dv;         // state column S[:, dv] of head h, stride DV
  let scale = inverseSqrt(f32(DK));
  var s: array<f32, 128>;                 // state column S[:, dv], length DK
  for (var dk = 0u; dk < DK; dk = dk + 1u) { s[dk] = select(0.0, state_in[sbase + dk * p.DV], p.loadState != 0u); }

  for (var t = 0u; t < p.S; t = t + 1u) {
    let base = (t + p.tOff) * p.H + h;    // value-head row (v, g, beta, out); tOff = this chunk's
    let basek = (t + p.tOff) * p.HK + hk; // token offset when a long segment's scan is sub-chunked
    for (var i = lid.x; i < DK; i = i + WGV) { ksh[i] = k[basek * DK + i]; qsh[i] = q[basek * DK + i]; }
    workgroupBarrier();
    var sk = 0.0;
    var sq = 0.0;
    for (var dk = 0u; dk < DK; dk = dk + 1u) { sk = sk + ksh[dk] * ksh[dk]; sq = sq + qsh[dk] * qsh[dk]; }
    let ik = inverseSqrt(sk + 1e-6);              // l2norm(k)
    let iq = inverseSqrt(sq + 1e-6) * scale;      // l2norm(q) / sqrt(dk)
    if (dv < p.DV) {
      let gt = exp(g[base]);
      let bt = beta[p.betaOff + base];   // beta may share g's buffer (engine: gbeta = [g; beta])
      for (var dk = 0u; dk < DK; dk = dk + 1u) { s[dk] = s[dk] * gt; }   // decay
      var kv = 0.0;
      for (var dk = 0u; dk < DK; dk = dk + 1u) { kv = kv + s[dk] * ksh[dk] * ik; }
      let delta = (v[base * p.DV + dv] - kv) * bt;
      var o = 0.0;
      for (var dk = 0u; dk < DK; dk = dk + 1u) {
        s[dk] = s[dk] + ksh[dk] * ik * delta;      // S += Kn ⊗ delta
        o = o + s[dk] * qsh[dk] * iq;              // out = Qn · S (updated)
      }
      core[base * p.DV + dv] = o;
    }
    workgroupBarrier();
  }
  if (dv < p.DV) { for (var dk = 0u; dk < DK; dk = dk + 1u) { state_out[sbase + dk * p.DV] = s[dk]; } }
}
`,embed_gather:`// GPU embedding gather + 4-bit dequant: reads a token id from a GPU buffer and writes that token's
// embedding (H f32) directly into a GPU buffer, so the decode loop never round-trips the token id to
// the CPU. Faithful port of the CPU embedDequant (4-bit codes via the tgt4 LUT, per-128 zero-point,
// per-block scale). uint8 source arrays are read as u32 and byte-extracted (little-endian).
override WG: u32 = 256u;
struct Params { H: u32, srcIdx: u32, _0: u32, _1: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> tokenId: array<u32>;   // tokenId[p.srcIdx] = the token to embed
@group(0) @binding(2) var<storage, read> embWq: array<u32>;     // uint8 [vocab * H/8] packed
@group(0) @binding(3) var<storage, read> tgt4: array<u32>;      // uint8 [256*4] packed (1 src byte -> 4)
@group(0) @binding(4) var<storage, read> embScales: array<f32>;// [vocab * H/128]
@group(0) @binding(5) var<storage, read> embZp: array<u32>;    // uint8 [vocab * ceil(H/256)] packed
@group(0) @binding(6) var<storage, read_write> out: array<f32>;// [H]

@compute @workgroup_size(WG)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let id = tokenId[p.srcIdx];
  // Per-row strides derived from H: rowBytes source bytes, scaleRow f32 scales,
  // zpRow packed zero-point bytes (H=2048 -> 256/16/8, H=2560 -> 320/20/10).
  let rowBytes = p.H >> 3u;
  let scaleRow = p.H >> 7u;
  let zpRow = (scaleRow + 1u) >> 1u;
  for (var k = lid.x; k < p.H; k = k + WG) {
    let i = k >> 3u;
    let qd = (k >> 1u) & 3u;
    let c = k & 1u;
    let wqIdx = id * rowBytes + i;
    let e = (embWq[wqIdx >> 2u] >> (8u * (wqIdx & 3u))) & 0xffu;   // source byte 0..255
    let tIdx = 4u * e + qd;
    let t = (tgt4[tIdx >> 2u] >> (8u * (tIdx & 3u))) & 0xffu;       // expanded byte (2 codes)
    let code = (t >> (4u * c)) & 0xfu;
    let blk = k >> 7u;
    let zpIdx = id * zpRow + (blk >> 1u);
    let zpByte = (embZp[zpIdx >> 2u] >> (8u * (zpIdx & 3u))) & 0xffu;
    let zp = (zpByte >> (4u * (blk & 1u))) & 0xfu;
    out[k] = (f32(code) - f32(zp)) * embScales[id * scaleRow + blk];
  }
}
`,embed_gather_batch:`// Batched GPU embedding gather + 4-bit dequant for PROMPT tokens: one invocation per output
// element writes out[s*H + k] for tokenIds[s]. A prefill segment uploads S u32 token ids
// instead of S*H dequantized floats, so the CPU-side embedding tables are not needed at all
// (~50-100 MB RAM per model). Same per-row stride math and dequant as embed_gather.wgsl
// (H=2048 -> 256/16/8 strides); uint8 sources read as u32 and byte-extracted (little-endian).
struct Params { S: u32, H: u32, _0: u32, _1: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> tokenIds: array<u32>;   // [S]
@group(0) @binding(2) var<storage, read> embWq: array<u32>;      // uint8 [vocab * H/8] packed
@group(0) @binding(3) var<storage, read> tgt4: array<u32>;       // uint8 [256*4] packed (1 src byte -> 4)
@group(0) @binding(4) var<storage, read> embScales: array<f32>; // [vocab * H/128]
@group(0) @binding(5) var<storage, read> embZp: array<u32>;     // uint8 [vocab * ceil(H/256)] packed
@group(0) @binding(6) var<storage, read_write> out: array<f32>; // [S * H]

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let gi = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  if (gi >= p.S * p.H) { return; }
  let k = gi % p.H;
  let id = tokenIds[gi / p.H];
  let rowBytes = p.H >> 3u;
  let scaleRow = p.H >> 7u;
  let zpRow = (scaleRow + 1u) >> 1u;
  let i = k >> 3u;
  let qd = (k >> 1u) & 3u;
  let c = k & 1u;
  let wqIdx = id * rowBytes + i;
  let e = (embWq[wqIdx >> 2u] >> (8u * (wqIdx & 3u))) & 0xffu;   // source byte 0..255
  let tIdx = 4u * e + qd;
  let t = (tgt4[tIdx >> 2u] >> (8u * (tIdx & 3u))) & 0xffu;       // expanded byte (2 codes)
  let code = (t >> (4u * c)) & 0xfu;
  let blk = k >> 7u;
  let zpIdx = id * zpRow + (blk >> 1u);
  let zpByte = (embZp[zpIdx >> 2u] >> (8u * (zpIdx & 3u))) & 0xffu;
  let zp = (zpByte >> (4u * (blk & 1u))) & 0xfu;
  out[gi] = (f32(code) - f32(zp)) * embScales[id * scaleRow + blk];
}
`,gate_sigmoid:`// Output gate for Qwen3.5 gated attention: y = x * sigmoid(gate), elementwise. Applied to the
// attention output before o_proj (the gate is the second half of the doubled q_proj).
struct Params { n: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> gate: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  if (i >= p.n) { return; }
  y[i] = x[i] / (1.0 + exp(-gate[i]));
}
`,logsumexp:`// log-sum-exp over the (penalty-filtered) logits, the softmax normalizer that turns a raw logit
// into a true logprob on the CPU: logprob(id) = logit[id] - lse. Runs AFTER sampler_penalty and
// BEFORE the argmax_masked rounds (those mask their winners in place, which would corrupt the
// sum). Two-phase single-workgroup reduction: strided max, then strided sum of exp(x - max);
// entries at the -inf sentinel (banned ids) contribute nothing. Only one f32 is read back.
override WG: u32 = 256u;
struct Params { N: u32, _0: u32, _1: u32, _2: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read_write> outLse: array<f32>;   // outLse[0] = max + log(sum)

const NEG_SENTINEL: f32 = -3.0e38;   // below any real logit; banned entries sit at f32 -inf

var<workgroup> sval: array<f32, 256>;

@compute @workgroup_size(WG)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  var m = -3.4e38;
  for (var i = tid; i < p.N; i = i + WG) {
    let v = logits[i];
    if (v > NEG_SENTINEL && v > m) { m = v; }
  }
  sval[tid] = m;
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s = s >> 1u) {
    if (tid < s && sval[tid + s] > sval[tid]) { sval[tid] = sval[tid + s]; }
    workgroupBarrier();
  }
  let gmax = sval[0];
  workgroupBarrier();
  var acc = 0.0;
  for (var i = tid; i < p.N; i = i + WG) {
    let v = logits[i];
    if (v > NEG_SENTINEL) { acc = acc + exp(v - gmax); }
  }
  sval[tid] = acc;
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s = s >> 1u) {
    if (tid < s) { sval[tid] = sval[tid] + sval[tid + s]; }
    workgroupBarrier();
  }
  if (tid == 0u) { outLse[0] = gmax + log(sval[0]); }
}
`,matmul_binary_vec4:`// Binary matmul, vectorized: y[M,N] = x[M,K] @ W[N,K]^T, W = (+/-1) * per-block scale.
// One thread per output; the K loop processes a 32-bit sign word at a time and the
// activations as vec4 via dot() (4 weights per FMA instead of 1 scalar op). M-agnostic
// (works for prefill M=S and decode M=1). x is bound as vec4 (K must be a multiple of 4).
struct Params { M: u32, N: u32, K: u32, nb: u32, block: u32, _pad: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;   // [M, K/4]
@group(0) @binding(2) var<storage, read> signbits: array<u32>;  // [N, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [N, nb]
@group(0) @binding(4) var<storage, read_write> y: array<f32>;   // [M, N]

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  if (idx >= p.M * p.N) { return; }
  let m = idx / p.N;
  let n = idx % p.N;
  let xRow = m * (p.K / 4u);
  let wRow = n * (p.K / 32u);
  let sbase = n * p.nb;
  let wordsPerBlock = p.block / 32u;   // 4 for block=128

  var acc = 0.0;
  for (var b = 0u; b < p.nb; b = b + 1u) {
    var bsum = 0.0;
    for (var w = 0u; w < wordsPerBlock; w = w + 1u) {
      let word = signbits[wRow + b * wordsPerBlock + w];
      let xb = xRow + b * (p.block / 4u) + w * 8u;   // vec4 base for this word (32 weights = 8 vec4)
      for (var g = 0u; g < 8u; g = g + 1u) {
        let bits4 = (word >> (g * 4u)) & 0xfu;
        let sv = vec4<f32>(
          select(-1.0, 1.0, (bits4 & 1u) != 0u),
          select(-1.0, 1.0, (bits4 & 2u) != 0u),
          select(-1.0, 1.0, (bits4 & 4u) != 0u),
          select(-1.0, 1.0, (bits4 & 8u) != 0u),
        );
        bsum = bsum + dot(x[xb + g], sv);
      }
    }
    acc = acc + bsum * scales[sbase + b];
  }
  y[idx] = acc;
}
`,matmul_q2:`// 2-bit dequant matmul (lm_head): y[M,N] = x[M,K] @ W[N,K]^T, W[n,k] = (code - zp) * scale[n, k/block].
// codes are 2-bit, 4 per byte, packed into u32 words. Correctness-first (one thread per output, fp32).
struct Params { M: u32, N: u32, K: u32, nb: u32, block: u32, zp: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;       // [M, K]
@group(0) @binding(2) var<storage, read> codes: array<u32>;   // [N, K/4] bytes packed as u32
@group(0) @binding(3) var<storage, read> scales: array<f32>;  // [N, nb]
@group(0) @binding(4) var<storage, read_write> y: array<f32>; // [M, N]

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  if (idx >= p.M * p.N) { return; }
  let m = idx / p.N;
  let n = idx % p.N;
  let xbase = m * p.K;
  let cbyteBase = n * (p.K / 4u);   // byte offset of row n in the codes stream
  let sbase = n * p.nb;
  let zpf = f32(p.zp);

  var acc = 0.0;
  for (var b = 0u; b < p.nb; b = b + 1u) {
    var bsum = 0.0;
    let k0 = b * p.block;
    for (var j = 0u; j < p.block; j = j + 1u) {
      let k = k0 + j;
      let byteIdx = cbyteBase + (k >> 2u);
      let word = codes[byteIdx >> 2u];
      let byte = (word >> (8u * (byteIdx & 3u))) & 0xffu;
      let code = (byte >> (2u * (k & 3u))) & 3u;
      bsum = bsum + (f32(code) - zpf) * x[xbase + k];
    }
    acc = acc + bsum * scales[sbase + b];
  }
  y[idx] = acc;
}
`,matmul_q2_sg:`// Subgroup split-K GEMV for the 2-bit lm_head (M=1 decode). One subgroup per output column,
// lanes split K (vec4), reduce with subgroupAdd. value = (code - zp) * per-block scale.
// 2D dispatch since N (vocab) > 65535.
enable subgroups;
override SG: u32 = 32u;
struct Params { N: u32, K: u32, nb: u32, zp: u32, gridX: u32, _pad: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;   // [K/4]
@group(0) @binding(2) var<storage, read> codes: array<u32>;     // [N, K/4] bytes packed as u32
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [N, nb]
@group(0) @binding(4) var<storage, read_write> y: array<f32>;   // [N]

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let n = wg.y * p.gridX + wg.x;
  if (n >= p.N) { return; }
  let cbase = n * (p.K / 4u);     // byte offset of row n in the codes stream
  let sbase = n * p.nb;
  let zpf = f32(p.zp);
  let Kvec = p.K / 4u;

  var acc = 0.0;
  for (var gi = lane; gi < Kvec; gi = gi + SG) {
    let byteIdx = cbase + gi;
    let word = codes[byteIdx >> 2u];
    let byte = (word >> (8u * (byteIdx & 3u))) & 0xffu;
    let cv = vec4<f32>(f32(byte & 3u) - zpf, f32((byte >> 2u) & 3u) - zpf,
                       f32((byte >> 4u) & 3u) - zpf, f32((byte >> 6u) & 3u) - zpf);
    acc = acc + dot(x[gi], cv) * scales[sbase + (gi >> 5u)];   // block = (gi*4)/128 = gi/32
  }
  let total = subgroupAdd(acc);
  if (lane == 0u) { y[n] = total; }
}
`,matmul_q2_sm:`// Small-batch (M = 2..9) subgroup split-K GEMV for the 2-bit lm_head: the speculative-decode
// verify pass needs logits for every drafted row, and the scalar M-row kernel re-reads the
// ~77 MB code stream per output thread. Here each code word is loaded once per (column,
// k-chunk) and dotted with all M rows. Per row the loop stride and accumulation expression
// match matmul_q2_sg, so each row is bit-identical to the M=1 decode path.
enable subgroups;
override SG: u32 = 32u;
struct Params { N: u32, K: u32, nb: u32, zp: u32, gridX: u32, M: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;   // [M, K/4] row-major
@group(0) @binding(2) var<storage, read> codes: array<u32>;     // [N, K/4] bytes packed as u32
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [N, nb]
@group(0) @binding(4) var<storage, read_write> y: array<f32>;   // [M, N]

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let n = wg.y * p.gridX + wg.x;
  if (n >= p.N) { return; }
  let cbase = n * (p.K / 4u);
  let sbase = n * p.nb;
  let zpf = f32(p.zp);
  let Kvec = p.K / 4u;

  var acc: array<f32, 9>; // M <= 9
  for (var m = 0u; m < p.M; m = m + 1u) { acc[m] = 0.0; }
  for (var gi = lane; gi < Kvec; gi = gi + SG) {
    let byteIdx = cbase + gi;
    let word = codes[byteIdx >> 2u];
    let byte = (word >> (8u * (byteIdx & 3u))) & 0xffu;
    let cv = vec4<f32>(f32(byte & 3u) - zpf, f32((byte >> 2u) & 3u) - zpf,
                       f32((byte >> 4u) & 3u) - zpf, f32((byte >> 6u) & 3u) - zpf);
    let s = scales[sbase + (gi >> 5u)]; // block = (gi*4)/128 = gi/32
    for (var m = 0u; m < p.M; m = m + 1u) {
      acc[m] = acc[m] + dot(x[m * Kvec + gi], cv) * s;
    }
  }
  for (var m = 0u; m < p.M; m = m + 1u) {
    let total = subgroupAdd(acc[m]);
    if (lane == 0u) { y[m * p.N + n] = total; }
  }
}
`,matmul_q2_wg:`// No-subgroup fallback: 2-bit lm_head GEMV for decode (M=1), workgroup-shared-memory reduction.
// One workgroup per output column; WG threads split K and tree-reduce. value = (code - zp) * scale.
// 2D dispatch since N (vocab) > 65535. This is the v1 path's biggest cost (scalar was ~48ms/token).
override WG: u32 = 64u;
struct Params { N: u32, K: u32, nb: u32, zp: u32, gridX: u32, _pad: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;   // [K/4]
@group(0) @binding(2) var<storage, read> codes: array<u32>;     // [N, K/4] bytes packed as u32
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [N, nb]
@group(0) @binding(4) var<storage, read_write> y: array<f32>;   // [N]
var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let n = wg.y * p.gridX + wg.x;
  if (n >= p.N) { return; }
  let tid = lid.x;
  let cbase = n * (p.K / 4u);
  let sbase = n * p.nb;
  let zpf = f32(p.zp);
  let Kvec = p.K / 4u;
  var acc = 0.0;
  for (var gi = tid; gi < Kvec; gi = gi + WG) {
    let byteIdx = cbase + gi;
    let word = codes[byteIdx >> 2u];
    let byte = (word >> (8u * (byteIdx & 3u))) & 0xffu;
    let cv = vec4<f32>(f32(byte & 3u) - zpf, f32((byte >> 2u) & 3u) - zpf,
                       f32((byte >> 4u) & 3u) - zpf, f32((byte >> 6u) & 3u) - zpf);
    acc = acc + dot(x[gi], cv) * scales[sbase + (gi >> 5u)];
  }
  sdata[tid] = acc;
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s = s >> 1u) {
    if (tid < s) { sdata[tid] = sdata[tid] + sdata[tid + s]; }
    workgroupBarrier();
  }
  if (tid == 0u) { y[n] = sdata[0]; }
}
`,matmul_resid:`// Binary matmul with a fused residual add: y[M,N] = x[M,K] @ W[N,K]^T + resid[M,N].
// Folds the residual add into o_proj / down_proj so it's not a separate dispatch.
struct Params { M: u32, N: u32, K: u32, nb: u32, block: u32, _pad: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;   // [M, K/4]
@group(0) @binding(2) var<storage, read> signbits: array<u32>;  // [N, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [N, nb]
@group(0) @binding(4) var<storage, read> resid: array<f32>;     // [M, N]
@group(0) @binding(5) var<storage, read_write> y: array<f32>;   // [M, N]

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  if (idx >= p.M * p.N) { return; }
  let n = idx % p.N;
  let xRow = (idx / p.N) * (p.K / 4u);
  let wRow = n * (p.K / 32u);
  let sbase = n * p.nb;

  var acc = 0.0;
  for (var b = 0u; b < p.nb; b = b + 1u) {
    var bsum = 0.0;
    for (var w = 0u; w < 4u; w = w + 1u) {
      let word = signbits[wRow + b * 4u + w];
      let xb = xRow + b * 32u + w * 8u;
      for (var g = 0u; g < 8u; g = g + 1u) {
        let bits4 = (word >> (g * 4u)) & 0xfu;
        let sv = vec4<f32>(select(-1.0, 1.0, (bits4 & 1u) != 0u), select(-1.0, 1.0, (bits4 & 2u) != 0u),
                           select(-1.0, 1.0, (bits4 & 4u) != 0u), select(-1.0, 1.0, (bits4 & 8u) != 0u));
        bsum = bsum + dot(x[xb + g], sv);
      }
    }
    acc = acc + bsum * scales[sbase + b];
  }
  y[idx] = acc + resid[idx];
}
`,matmul_resid_mr_sg:`// Multi-row subgroup GEMV for decode (M=1) with fused residual. Same as matmul_resid_sg but each
// workgroup computes ROWS output columns at once: per K-step it issues ROWS independent weight
// loads before the dots, giving the memory system more in-flight requests (memory-level
// parallelism) to hide latency on the bandwidth-bound decode GEMV. One subgroup per workgroup;
// lanes split K; ROWS accumulators reduced with subgroupAdd. value = sign * per-block scale.
enable subgroups;
override SG: u32 = 32u;
override ROWS: u32 = 4u;
struct Params { N: u32, K: u32, nb: u32, gridX: u32, _p0: u32, _p1: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;   // [K/4]
@group(0) @binding(2) var<storage, read> signbits: array<u32>;  // [N, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [N, nb]
@group(0) @binding(4) var<storage, read> resid: array<f32>;     // [N]
@group(0) @binding(5) var<storage, read_write> y: array<f32>;   // [N]

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let rowBase = (wg.y * p.gridX + wg.x) * ROWS;
  let Kvec = p.K / 4u;
  let wStride = p.K / 32u;

  var acc: array<f32, 8>;                         // ROWS <= 8
  for (var r = 0u; r < ROWS; r = r + 1u) { acc[r] = 0.0; }
  for (var gi = lane; gi < Kvec; gi = gi + SG) {
    let k = gi * 4u;
    let xv = x[gi];
    let widx = k >> 5u;
    let sh = k & 31u;
    let sc = k / 128u;
    for (var r = 0u; r < ROWS; r = r + 1u) {
      let n = rowBase + r;
      if (n < p.N) {
        let w = (signbits[n * wStride + widx] >> sh) & 0xfu;
        let sv = vec4<f32>(select(-1.0, 1.0, (w & 1u) != 0u), select(-1.0, 1.0, (w & 2u) != 0u),
                           select(-1.0, 1.0, (w & 4u) != 0u), select(-1.0, 1.0, (w & 8u) != 0u));
        acc[r] = acc[r] + dot(xv, sv) * scales[n * p.nb + sc];
      }
    }
  }
  for (var r = 0u; r < ROWS; r = r + 1u) {
    let n = rowBase + r;
    let total = subgroupAdd(acc[r]);             // collective: called for every r by all lanes
    if (lane == 0u && n < p.N) { y[n] = total + resid[n]; }
  }
}
`,matmul_resid_mr_sg_af16:`// f16-activation variant of matmul_resid_mr_sg (multi-row decode GEMV + fused residual, M=1),
// used for down_proj (its input is the f16 SwiGLU intermediate). Reads f16 x, dots in f16,
// accumulates in f32; the residual add and the output stay f32. Weights unchanged.
enable subgroups;
enable f16;
override SG: u32 = 32u;
override ROWS: u32 = 4u;
struct Params { N: u32, K: u32, nb: u32, gridX: u32, _p0: u32, _p1: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f16>>;   // [K/4] f16 activations
@group(0) @binding(2) var<storage, read> signbits: array<u32>;  // [N, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [N, nb]
@group(0) @binding(4) var<storage, read> resid: array<f32>;     // [N]
@group(0) @binding(5) var<storage, read_write> y: array<f32>;   // [N]

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let rowBase = (wg.y * p.gridX + wg.x) * ROWS;
  let Kvec = p.K / 4u;
  let wStride = p.K / 32u;

  var acc: array<f32, 8>;                         // ROWS <= 8
  for (var r = 0u; r < ROWS; r = r + 1u) { acc[r] = 0.0; }
  for (var gi = lane; gi < Kvec; gi = gi + SG) {
    let k = gi * 4u;
    let xv = x[gi];
    let widx = k >> 5u;
    let sh = k & 31u;
    let sc = k / 128u;
    for (var r = 0u; r < ROWS; r = r + 1u) {
      let n = rowBase + r;
      if (n < p.N) {
        let w = (signbits[n * wStride + widx] >> sh) & 0xfu;
        let sv = vec4<f16>(select(-1.0h, 1.0h, (w & 1u) != 0u), select(-1.0h, 1.0h, (w & 2u) != 0u),
                           select(-1.0h, 1.0h, (w & 4u) != 0u), select(-1.0h, 1.0h, (w & 8u) != 0u));
        acc[r] = acc[r] + f32(dot(xv, sv)) * scales[n * p.nb + sc];
      }
    }
  }
  for (var r = 0u; r < ROWS; r = r + 1u) {
    let n = rowBase + r;
    let total = subgroupAdd(acc[r]);
    if (lane == 0u && n < p.N) { y[n] = total + resid[n]; }
  }
}
`,matmul_resid_sm:`// Small-batch (M = 2..9) subgroup split-K GEMV with fused residual add (o_proj / down_proj in
// the speculative-decode verify pass). One workgroup per output column; each weight word is
// loaded once and dotted with all M activation rows. Per row the loop stride and accumulation
// expression match the validated M=1 kernels, so results are row-wise bit-identical to them.
enable subgroups;
override SG: u32 = 32u;
struct Params { N: u32, K: u32, nb: u32, gridX: u32, M: u32, _pad: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;   // [M, K/4] row-major
@group(0) @binding(2) var<storage, read> signbits: array<u32>;  // [N, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [N, nb]
@group(0) @binding(4) var<storage, read> resid: array<f32>;     // [M, N]
@group(0) @binding(5) var<storage, read_write> y: array<f32>;   // [M, N]

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let n = wg.y * p.gridX + wg.x;
  if (n >= p.N) { return; }
  let wRow = n * (p.K / 32u);
  let sbase = n * p.nb;
  let Kvec = p.K / 4u;

  var acc: array<f32, 9>; // M <= 9
  for (var m = 0u; m < p.M; m = m + 1u) { acc[m] = 0.0; }
  for (var gi = lane; gi < Kvec; gi = gi + SG) {
    let k = gi * 4u;
    let word = signbits[wRow + (k >> 5u)];
    let bits4 = (word >> (k & 31u)) & 0xfu;
    let sv = vec4<f32>(select(-1.0, 1.0, (bits4 & 1u) != 0u), select(-1.0, 1.0, (bits4 & 2u) != 0u),
                       select(-1.0, 1.0, (bits4 & 4u) != 0u), select(-1.0, 1.0, (bits4 & 8u) != 0u));
    let s = scales[sbase + (k / 128u)];
    for (var m = 0u; m < p.M; m = m + 1u) {
      acc[m] = acc[m] + dot(x[m * Kvec + gi], sv) * s;
    }
  }
  for (var m = 0u; m < p.M; m = m + 1u) {
    let total = subgroupAdd(acc[m]);
    if (lane == 0u) { y[m * p.N + n] = total + resid[m * p.N + n]; }
  }
}
`,matmul_resid_tiled:`// Tiled register-blocked binary GEMM with fused residual, for PREFILL (M>1), vec4 K-accumulation:
//   y[M,N] = x[M,K] @ W[N,K]^T + resid[M,N],  W binary {-1,+1} sign-packed, per-128-block fp32 scale.
// 64x64 output tile per workgroup, 16x16 threads each computing a 4x4 register tile, BK=16 K-step.
// Activation + decoded/scaled weight tiles are staged in shared memory as vec4 (4 K per element);
// each inner step is a dot() of vec4s, and one weight load decodes a whole nibble (4 signs) at once.
// No subgroup ops -> all devices. Near-bit-exact (f32 accum; tiled K-order differs in last ULPs).
const BM: u32 = 64u;
const BN: u32 = 64u;
const BKV: u32 = 4u;          // BK / 4  (BK = 16)
struct Params { M: u32, N: u32, K: u32, nb: u32, _0: u32, _1: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;  // [M, K/4]
@group(0) @binding(2) var<storage, read> signbits: array<u32>; // [N, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;   // [N, nb]
@group(0) @binding(4) var<storage, read> resid: array<f32>;    // [M, N]
@group(0) @binding(5) var<storage, read_write> y: array<f32>;  // [M, N]

var<workgroup> xs: array<vec4<f32>, 256>;   // BM*BKV
var<workgroup> ws: array<vec4<f32>, 256>;   // BN*BKV

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let tileM = wg.y * BM;
  let tileN = wg.x * BN;
  let tr = (tid / 16u) * 4u;
  let tc = (tid % 16u) * 4u;
  let Kv = p.K / 4u;
  var acc: array<f32, 16>;
  for (var i = 0u; i < 16u; i = i + 1u) { acc[i] = 0.0; }

  let Ksteps = Kv / BKV;
  for (var ks = 0u; ks < Ksteps; ks = ks + 1u) {
    let k0v = ks * BKV;
    for (var e = tid; e < BM * BKV; e = e + 256u) {           // stage activation tile (vec4)
      let m = e / BKV; let kv = e % BKV; let gm = tileM + m;
      xs[e] = select(vec4<f32>(0.0), x[gm * Kv + (k0v + kv)], gm < p.M);
    }
    for (var e = tid; e < BN * BKV; e = e + 256u) {           // stage decoded+scaled weight tile (vec4)
      let n = e / BKV; let kv = e % BKV; let gn = tileN + n; let k = (k0v + kv) * 4u;
      var wv = vec4<f32>(0.0);
      if (gn < p.N) {
        let bits4 = (signbits[gn * (p.K / 32u) + (k >> 5u)] >> (k & 31u)) & 0xfu;
        let s = scales[gn * p.nb + (k / 128u)];
        wv = vec4<f32>(select(-s, s, (bits4 & 1u) != 0u), select(-s, s, (bits4 & 2u) != 0u),
                       select(-s, s, (bits4 & 4u) != 0u), select(-s, s, (bits4 & 8u) != 0u));
      }
      ws[e] = wv;
    }
    workgroupBarrier();
    for (var kv = 0u; kv < BKV; kv = kv + 1u) {
      var xr: array<vec4<f32>, 4>;
      for (var tm = 0u; tm < 4u; tm = tm + 1u) { xr[tm] = xs[(tr + tm) * BKV + kv]; }
      for (var tn = 0u; tn < 4u; tn = tn + 1u) {
        let w = ws[(tc + tn) * BKV + kv];
        for (var tm = 0u; tm < 4u; tm = tm + 1u) { acc[tm * 4u + tn] = acc[tm * 4u + tn] + dot(xr[tm], w); }
      }
    }
    workgroupBarrier();
  }

  for (var tm = 0u; tm < 4u; tm = tm + 1u) {
    let gm = tileM + tr + tm;
    if (gm < p.M) {
      for (var tn = 0u; tn < 4u; tn = tn + 1u) {
        let gn = tileN + tc + tn;
        if (gn < p.N) { let idx = gm * p.N + gn; y[idx] = acc[tm * 4u + tn] + resid[idx]; }
      }
    }
  }
}
`,matmul_resid_wg:`// No-subgroup fallback: split-K GEMV for decode (M=1) with fused residual, workgroup-shared-memory
// reduction. One workgroup per output column; WG threads split K and tree-reduce. Used for o_proj/down.
override WG: u32 = 64u;
struct Params { N: u32, K: u32, nb: u32, gridX: u32, _p0: u32, _p1: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;   // [K/4]
@group(0) @binding(2) var<storage, read> signbits: array<u32>;  // [N, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [N, nb]
@group(0) @binding(4) var<storage, read> resid: array<f32>;     // [N]
@group(0) @binding(5) var<storage, read_write> y: array<f32>;   // [N]
var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let n = wg.y * p.gridX + wg.x;
  if (n >= p.N) { return; }
  let tid = lid.x;
  let wRow = n * (p.K / 32u);
  let sbase = n * p.nb;
  let Kvec = p.K / 4u;
  var acc = 0.0;
  for (var gi = tid; gi < Kvec; gi = gi + WG) {
    let k = gi * 4u;
    let word = signbits[wRow + (k >> 5u)];
    let bits4 = (word >> (k & 31u)) & 0xfu;
    let sv = vec4<f32>(select(-1.0, 1.0, (bits4 & 1u) != 0u), select(-1.0, 1.0, (bits4 & 2u) != 0u),
                       select(-1.0, 1.0, (bits4 & 4u) != 0u), select(-1.0, 1.0, (bits4 & 8u) != 0u));
    acc = acc + dot(x[gi], sv) * scales[sbase + (k / 128u)];
  }
  sdata[tid] = acc;
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s = s >> 1u) {
    if (tid < s) { sdata[tid] = sdata[tid] + sdata[tid + s]; }
    workgroupBarrier();
  }
  if (tid == 0u) { y[n] = sdata[0] + resid[n]; }
}
`,matmul_split:`// Fused binary matmul writing to up to 3 output buffers (qkv or gate/up in one dispatch).
// Weights for the outputs are concatenated along N (rows N0 | N1 | N2). One thread per
// output column n routes its result to out0/out1/out2 by range. Vectorized like matmul_binary_vec4.
struct Params { M: u32, K: u32, nb: u32, N0: u32, N1: u32, N2: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;   // [M, K/4]
@group(0) @binding(2) var<storage, read> signbits: array<u32>;  // [N0+N1+N2, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [N0+N1+N2, nb]
@group(0) @binding(4) var<storage, read_write> out0: array<f32>;
@group(0) @binding(5) var<storage, read_write> out1: array<f32>;
@group(0) @binding(6) var<storage, read_write> out2: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let Ntot = p.N0 + p.N1 + p.N2;
  let idx = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  if (idx >= p.M * Ntot) { return; }
  let row = idx / Ntot;
  let n = idx % Ntot;
  let xRow = row * (p.K / 4u);
  let wRow = n * (p.K / 32u);
  let sbase = n * p.nb;

  var acc = 0.0;
  for (var b = 0u; b < p.nb; b = b + 1u) {
    var bsum = 0.0;
    for (var w = 0u; w < 4u; w = w + 1u) {
      let word = signbits[wRow + b * 4u + w];
      let xb = xRow + b * 32u + w * 8u;
      for (var g = 0u; g < 8u; g = g + 1u) {
        let bits4 = (word >> (g * 4u)) & 0xfu;
        let sv = vec4<f32>(select(-1.0, 1.0, (bits4 & 1u) != 0u), select(-1.0, 1.0, (bits4 & 2u) != 0u),
                           select(-1.0, 1.0, (bits4 & 4u) != 0u), select(-1.0, 1.0, (bits4 & 8u) != 0u));
        bsum = bsum + dot(x[xb + g], sv);
      }
    }
    acc = acc + bsum * scales[sbase + b];
  }

  if (n < p.N0) { out0[row * p.N0 + n] = acc; }
  else if (n < p.N0 + p.N1) { out1[row * p.N1 + (n - p.N0)] = acc; }
  else { out2[row * p.N2 + (n - p.N0 - p.N1)] = acc; }
}
`,matmul_split_sg:`// Subgroup split-K GEMV for decode (M=1), fused: one subgroup (= one workgroup) per output
// column; lanes split the K dimension and reduce with subgroupAdd (register-only, no barriers).
// Cuts each matmul's latency ~SG-fold vs one-thread-per-output (the real decode bottleneck:
// kernels run at full latency in the dependent chain). Routes to out0/out1/out2 by range (qkv / gate-up).
enable subgroups;
override SG: u32 = 32u;
struct Params { K: u32, nb: u32, N0: u32, N1: u32, N2: u32, gridX: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;   // [K/4]
@group(0) @binding(2) var<storage, read> signbits: array<u32>;  // [N0+N1+N2, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [N0+N1+N2, nb]
@group(0) @binding(4) var<storage, read_write> out0: array<f32>;
@group(0) @binding(5) var<storage, read_write> out1: array<f32>;
@group(0) @binding(6) var<storage, read_write> out2: array<f32>;

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let Ntot = p.N0 + p.N1 + p.N2;
  let n = wg.y * p.gridX + wg.x;
  if (n >= Ntot) { return; }
  let wRow = n * (p.K / 32u);
  let sbase = n * p.nb;
  let Kvec = p.K / 4u;

  var acc = 0.0;
  for (var gi = lane; gi < Kvec; gi = gi + SG) {
    let k = gi * 4u;
    let word = signbits[wRow + (k >> 5u)];
    let bits4 = (word >> (k & 31u)) & 0xfu;
    let sv = vec4<f32>(select(-1.0, 1.0, (bits4 & 1u) != 0u), select(-1.0, 1.0, (bits4 & 2u) != 0u),
                       select(-1.0, 1.0, (bits4 & 4u) != 0u), select(-1.0, 1.0, (bits4 & 8u) != 0u));
    acc = acc + dot(x[gi], sv) * scales[sbase + (k / 128u)];
  }
  let total = subgroupAdd(acc);
  if (lane == 0u) {
    if (n < p.N0) { out0[n] = total; }
    else if (n < p.N0 + p.N1) { out1[n - p.N0] = total; }
    else { out2[n - p.N0 - p.N1] = total; }
  }
}
`,matmul_split_sg_af16:`// f16-activation variant of matmul_split_sg (fused QKV decode GEMV, M=1). The activation x is
// read as f16 and the per-group dot runs in f16 (2x ALU rate on Apple/AMD/recent NVIDIA); the
// per-block accumulation stays f32 (dot promoted before x scale, acc in f32) so accuracy tracks
// f32 to ~f16 rounding. Weights (sign bits + f32 block scales) are unchanged. Outputs f32.
enable subgroups;
enable f16;
override SG: u32 = 32u;
struct Params { K: u32, nb: u32, N0: u32, N1: u32, N2: u32, gridX: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f16>>;   // [K/4] f16 activations
@group(0) @binding(2) var<storage, read> signbits: array<u32>;  // [N0+N1+N2, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [N0+N1+N2, nb]
@group(0) @binding(4) var<storage, read_write> out0: array<f32>;
@group(0) @binding(5) var<storage, read_write> out1: array<f32>;
@group(0) @binding(6) var<storage, read_write> out2: array<f32>;

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let Ntot = p.N0 + p.N1 + p.N2;
  let n = wg.y * p.gridX + wg.x;
  if (n >= Ntot) { return; }
  let wRow = n * (p.K / 32u);
  let sbase = n * p.nb;
  let Kvec = p.K / 4u;

  var acc = 0.0;
  for (var gi = lane; gi < Kvec; gi = gi + SG) {
    let k = gi * 4u;
    let word = signbits[wRow + (k >> 5u)];
    let bits4 = (word >> (k & 31u)) & 0xfu;
    let sv = vec4<f16>(select(-1.0h, 1.0h, (bits4 & 1u) != 0u), select(-1.0h, 1.0h, (bits4 & 2u) != 0u),
                       select(-1.0h, 1.0h, (bits4 & 4u) != 0u), select(-1.0h, 1.0h, (bits4 & 8u) != 0u));
    acc = acc + f32(dot(x[gi], sv)) * scales[sbase + (k / 128u)];
  }
  let total = subgroupAdd(acc);
  if (lane == 0u) {
    if (n < p.N0) { out0[n] = total; }
    else if (n < p.N0 + p.N1) { out1[n - p.N0] = total; }
    else { out2[n - p.N0 - p.N1] = total; }
  }
}
`,matmul_split_sm:`// Small-batch (M = 2..9) subgroup split-K GEMV, fused qkv / gate-up. The speculative-decode
// verify pass computes M drafted rows in one forward; the scalar prefill kernels re-read the
// weights per output thread, so a k-row pass cost ~k GEMVs. Here each weight word is loaded
// ONCE per (column, k-chunk) and dotted with all M activation rows (activations are ~8 KB/row,
// cache-resident). Per row the loop stride and accumulation expression are IDENTICAL to
// matmul_split_sg, so each row's partials - and therefore the subgroupAdd result - match the
// M=1 decode path bit-for-bit.
enable subgroups;
override SG: u32 = 32u;
struct Params { K: u32, nb: u32, N0: u32, N1: u32, N2: u32, gridX: u32, M: u32, _pad: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;   // [M, K/4] row-major
@group(0) @binding(2) var<storage, read> signbits: array<u32>;  // [N0+N1+N2, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [N0+N1+N2, nb]
@group(0) @binding(4) var<storage, read_write> out0: array<f32>; // [M, N0]
@group(0) @binding(5) var<storage, read_write> out1: array<f32>; // [M, N1]
@group(0) @binding(6) var<storage, read_write> out2: array<f32>; // [M, N2]

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let Ntot = p.N0 + p.N1 + p.N2;
  let n = wg.y * p.gridX + wg.x;
  if (n >= Ntot) { return; } // uniform per workgroup: the whole subgroup exits together
  let wRow = n * (p.K / 32u);
  let sbase = n * p.nb;
  let Kvec = p.K / 4u;

  var acc: array<f32, 9>; // M <= 9
  for (var m = 0u; m < p.M; m = m + 1u) { acc[m] = 0.0; }
  for (var gi = lane; gi < Kvec; gi = gi + SG) {
    let k = gi * 4u;
    let word = signbits[wRow + (k >> 5u)];
    let bits4 = (word >> (k & 31u)) & 0xfu;
    let sv = vec4<f32>(select(-1.0, 1.0, (bits4 & 1u) != 0u), select(-1.0, 1.0, (bits4 & 2u) != 0u),
                       select(-1.0, 1.0, (bits4 & 4u) != 0u), select(-1.0, 1.0, (bits4 & 8u) != 0u));
    let s = scales[sbase + (k / 128u)];
    for (var m = 0u; m < p.M; m = m + 1u) {
      acc[m] = acc[m] + dot(x[m * Kvec + gi], sv) * s;
    }
  }
  for (var m = 0u; m < p.M; m = m + 1u) { // p.M is uniform: collective calls stay uniform
    let total = subgroupAdd(acc[m]);
    if (lane == 0u) {
      if (n < p.N0) { out0[m * p.N0 + n] = total; }
      else if (n < p.N0 + p.N1) { out1[m * p.N1 + (n - p.N0)] = total; }
      else { out2[m * p.N2 + (n - p.N0 - p.N1)] = total; }
    }
  }
}
`,matmul_split_tiled:`// Tiled register-blocked binary GEMM to 3 outputs (qkv or gate/up), PREFILL (M>1), vec4 K-accum.
// Weights concatenated along N (N0|N1|N2); each output element routes individually to
// out0/out1/out2 by its global column, so N0/N1/N2 need no alignment. Same vec4 design as
// matmul_resid_tiled.
const BM: u32 = 64u;
const BN: u32 = 64u;
const BKV: u32 = 4u;          // BK / 4  (BK = 16)
struct Params { M: u32, K: u32, nb: u32, N0: u32, N1: u32, N2: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;  // [M, K/4]
@group(0) @binding(2) var<storage, read> signbits: array<u32>; // [N0+N1+N2, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;   // [N0+N1+N2, nb]
@group(0) @binding(4) var<storage, read_write> out0: array<f32>;
@group(0) @binding(5) var<storage, read_write> out1: array<f32>;
@group(0) @binding(6) var<storage, read_write> out2: array<f32>;

var<workgroup> xs: array<vec4<f32>, 256>;
var<workgroup> ws: array<vec4<f32>, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let Ntot = p.N0 + p.N1 + p.N2;
  let tid = lid.x;
  let tileM = wg.y * BM;
  let tileN = wg.x * BN;
  let tr = (tid / 16u) * 4u;
  let tc = (tid % 16u) * 4u;
  let Kv = p.K / 4u;
  var acc: array<f32, 16>;
  for (var i = 0u; i < 16u; i = i + 1u) { acc[i] = 0.0; }

  let Ksteps = Kv / BKV;
  for (var ks = 0u; ks < Ksteps; ks = ks + 1u) {
    let k0v = ks * BKV;
    for (var e = tid; e < BM * BKV; e = e + 256u) {
      let m = e / BKV; let kv = e % BKV; let gm = tileM + m;
      xs[e] = select(vec4<f32>(0.0), x[gm * Kv + (k0v + kv)], gm < p.M);
    }
    for (var e = tid; e < BN * BKV; e = e + 256u) {
      let n = e / BKV; let kv = e % BKV; let gn = tileN + n; let k = (k0v + kv) * 4u;
      var wv = vec4<f32>(0.0);
      if (gn < Ntot) {
        let bits4 = (signbits[gn * (p.K / 32u) + (k >> 5u)] >> (k & 31u)) & 0xfu;
        let s = scales[gn * p.nb + (k / 128u)];
        wv = vec4<f32>(select(-s, s, (bits4 & 1u) != 0u), select(-s, s, (bits4 & 2u) != 0u),
                       select(-s, s, (bits4 & 4u) != 0u), select(-s, s, (bits4 & 8u) != 0u));
      }
      ws[e] = wv;
    }
    workgroupBarrier();
    for (var kv = 0u; kv < BKV; kv = kv + 1u) {
      var xr: array<vec4<f32>, 4>;
      for (var tm = 0u; tm < 4u; tm = tm + 1u) { xr[tm] = xs[(tr + tm) * BKV + kv]; }
      for (var tn = 0u; tn < 4u; tn = tn + 1u) {
        let w = ws[(tc + tn) * BKV + kv];
        for (var tm = 0u; tm < 4u; tm = tm + 1u) { acc[tm * 4u + tn] = acc[tm * 4u + tn] + dot(xr[tm], w); }
      }
    }
    workgroupBarrier();
  }

  for (var tm = 0u; tm < 4u; tm = tm + 1u) {
    let gm = tileM + tr + tm;
    if (gm >= p.M) { continue; }
    for (var tn = 0u; tn < 4u; tn = tn + 1u) {
      let gn = tileN + tc + tn;
      if (gn >= Ntot) { continue; }
      let v = acc[tm * 4u + tn];
      if (gn < p.N0) { out0[gm * p.N0 + gn] = v; }
      else if (gn < p.N0 + p.N1) { out1[gm * p.N1 + (gn - p.N0)] = v; }
      else { out2[gm * p.N2 + (gn - p.N0 - p.N1)] = v; }
    }
  }
}
`,matmul_split_wg:`// No-subgroup fallback: split-K GEMV for decode (M=1), workgroup-shared-memory reduction instead
// of subgroupAdd. One workgroup per output column; WG threads split K and tree-reduce via shared
// memory + barriers. ~WG-fold faster than one-thread-per-output (the v1 path). Routes qkv / gate-up.
override WG: u32 = 64u;
struct Params { K: u32, nb: u32, N0: u32, N1: u32, N2: u32, gridX: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;   // [K/4]
@group(0) @binding(2) var<storage, read> signbits: array<u32>;  // [N0+N1+N2, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [N0+N1+N2, nb]
@group(0) @binding(4) var<storage, read_write> out0: array<f32>;
@group(0) @binding(5) var<storage, read_write> out1: array<f32>;
@group(0) @binding(6) var<storage, read_write> out2: array<f32>;
var<workgroup> sdata: array<f32, 256>;                          // >= max WG

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let Ntot = p.N0 + p.N1 + p.N2;
  let n = wg.y * p.gridX + wg.x;          // uniform across the workgroup -> early return is barrier-safe
  if (n >= Ntot) { return; }
  let tid = lid.x;
  let wRow = n * (p.K / 32u);
  let sbase = n * p.nb;
  let Kvec = p.K / 4u;
  var acc = 0.0;
  for (var gi = tid; gi < Kvec; gi = gi + WG) {
    let k = gi * 4u;
    let word = signbits[wRow + (k >> 5u)];
    let bits4 = (word >> (k & 31u)) & 0xfu;
    let sv = vec4<f32>(select(-1.0, 1.0, (bits4 & 1u) != 0u), select(-1.0, 1.0, (bits4 & 2u) != 0u),
                       select(-1.0, 1.0, (bits4 & 4u) != 0u), select(-1.0, 1.0, (bits4 & 8u) != 0u));
    acc = acc + dot(x[gi], sv) * scales[sbase + (k / 128u)];
  }
  sdata[tid] = acc;
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s = s >> 1u) {
    if (tid < s) { sdata[tid] = sdata[tid] + sdata[tid + s]; }
    workgroupBarrier();
  }
  if (tid == 0u) {
    let total = sdata[0];
    if (n < p.N0) { out0[n] = total; }
    else if (n < p.N0 + p.N1) { out1[n - p.N0] = total; }
    else { out2[n - p.N0 - p.N1] = total; }
  }
}
`,matmul_swiglu_mr_sg:`// Multi-row fused gate/up GEMV + SwiGLU for decode (M=1). Each workgroup computes ROWS
// intermediate indices; per K-step it issues 2*ROWS independent weight loads (gate row n and up
// row F+n for each of the ROWS) before the dots, giving the bandwidth-bound decode GEMV more
// in-flight memory requests. One subgroup per workgroup; lanes split K; reduced with subgroupAdd.
enable subgroups;
override SG: u32 = 32u;
override ROWS: u32 = 4u;
struct Params { K: u32, nb: u32, F: u32, gridX: u32, _p0: u32, _p1: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;   // [K/4]
@group(0) @binding(2) var<storage, read> signbits: array<u32>;  // [2F, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [2F, nb]
@group(0) @binding(4) var<storage, read_write> y: array<f32>;   // [F]

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let nBase = (wg.y * p.gridX + wg.x) * ROWS;
  let Kvec = p.K / 4u;
  let wStride = p.K / 32u;

  var g: array<f32, 8>;                            // ROWS <= 8
  var u: array<f32, 8>;
  for (var r = 0u; r < ROWS; r = r + 1u) { g[r] = 0.0; u[r] = 0.0; }
  for (var gi = lane; gi < Kvec; gi = gi + SG) {
    let k = gi * 4u;
    let xv = x[gi];
    let widx = k >> 5u;
    let sh = k & 31u;
    let sc = k / 128u;
    for (var r = 0u; r < ROWS; r = r + 1u) {
      let n = nBase + r;
      if (n < p.F) {
        let gw = (signbits[n * wStride + widx] >> sh) & 0xfu;
        let gv = vec4<f32>(select(-1.0, 1.0, (gw & 1u) != 0u), select(-1.0, 1.0, (gw & 2u) != 0u),
                           select(-1.0, 1.0, (gw & 4u) != 0u), select(-1.0, 1.0, (gw & 8u) != 0u));
        g[r] = g[r] + dot(xv, gv) * scales[n * p.nb + sc];
        let uw = (signbits[(p.F + n) * wStride + widx] >> sh) & 0xfu;
        let uv = vec4<f32>(select(-1.0, 1.0, (uw & 1u) != 0u), select(-1.0, 1.0, (uw & 2u) != 0u),
                           select(-1.0, 1.0, (uw & 4u) != 0u), select(-1.0, 1.0, (uw & 8u) != 0u));
        u[r] = u[r] + dot(xv, uv) * scales[(p.F + n) * p.nb + sc];
      }
    }
  }
  for (var r = 0u; r < ROWS; r = r + 1u) {
    let n = nBase + r;
    let gt = subgroupAdd(g[r]);
    let ut = subgroupAdd(u[r]);
    if (lane == 0u && n < p.F) { y[n] = (gt / (1.0 + exp(-gt))) * ut; }
  }
}
`,matmul_swiglu_mr_sg_af16:`// f16-activation variant of matmul_swiglu_mr_sg (fused gate/up GEMV + SwiGLU, M=1). Reads the
// f16 activation x, dots in f16, accumulates each of gate/up in f32, applies SwiGLU in f32, and
// writes the intermediate as f16 (the input side of the f16 down_proj). Weights unchanged.
enable subgroups;
enable f16;
override SG: u32 = 32u;
override ROWS: u32 = 4u;
struct Params { K: u32, nb: u32, F: u32, gridX: u32, _p0: u32, _p1: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f16>>;   // [K/4] f16 activations
@group(0) @binding(2) var<storage, read> signbits: array<u32>;  // [2F, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [2F, nb]
@group(0) @binding(4) var<storage, read_write> y: array<f16>;   // [F] f16 intermediate

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let nBase = (wg.y * p.gridX + wg.x) * ROWS;
  let Kvec = p.K / 4u;
  let wStride = p.K / 32u;

  var g: array<f32, 8>;                            // ROWS <= 8
  var u: array<f32, 8>;
  for (var r = 0u; r < ROWS; r = r + 1u) { g[r] = 0.0; u[r] = 0.0; }
  for (var gi = lane; gi < Kvec; gi = gi + SG) {
    let k = gi * 4u;
    let xv = x[gi];
    let widx = k >> 5u;
    let sh = k & 31u;
    let sc = k / 128u;
    for (var r = 0u; r < ROWS; r = r + 1u) {
      let n = nBase + r;
      if (n < p.F) {
        let gw = (signbits[n * wStride + widx] >> sh) & 0xfu;
        let gv = vec4<f16>(select(-1.0h, 1.0h, (gw & 1u) != 0u), select(-1.0h, 1.0h, (gw & 2u) != 0u),
                           select(-1.0h, 1.0h, (gw & 4u) != 0u), select(-1.0h, 1.0h, (gw & 8u) != 0u));
        g[r] = g[r] + f32(dot(xv, gv)) * scales[n * p.nb + sc];
        let uw = (signbits[(p.F + n) * wStride + widx] >> sh) & 0xfu;
        let uv = vec4<f16>(select(-1.0h, 1.0h, (uw & 1u) != 0u), select(-1.0h, 1.0h, (uw & 2u) != 0u),
                           select(-1.0h, 1.0h, (uw & 4u) != 0u), select(-1.0h, 1.0h, (uw & 8u) != 0u));
        u[r] = u[r] + f32(dot(xv, uv)) * scales[(p.F + n) * p.nb + sc];
      }
    }
  }
  for (var r = 0u; r < ROWS; r = r + 1u) {
    let n = nBase + r;
    let gt = subgroupAdd(g[r]);
    let ut = subgroupAdd(u[r]);
    if (lane == 0u && n < p.F) { y[n] = f16((gt / (1.0 + exp(-gt))) * ut); }
  }
}
`,rmsnorm_rope_sg:`// Fused per-head RMSNorm + RoPE for decode (S=1). One subgroup (= one workgroup) per head row;
// lanes split head_dim, reduce sum-of-squares with subgroupAdd, then apply rope. rotate_half
// pairs (d, d+-D/2): with SG>=32 and D=128 a lane owns d in {lane, lane+32, lane+64, lane+96},
// so every (d, d+-64) pair is held by the same lane (no cross-lane reads for the rotate).
// outOff/outStride let the K result write straight into the KV cache at its position.
enable subgroups;
override SG: u32 = 32u;
struct Params { R: u32, D: u32, eps: f32, outOff: u32, outStride: u32, _p: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;        // [R, D]
@group(0) @binding(2) var<storage, read> gamma: array<f32>;    // [D]
@group(0) @binding(3) var<storage, read> cos: array<f32>;      // [D]
@group(0) @binding(4) var<storage, read> sin: array<f32>;      // [D]
@group(0) @binding(5) var<storage, read_write> y: array<f32>;  // [outOff + R*outStride]

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let row = wg.x;
  if (row >= p.R) { return; }
  let base = row * p.D;
  var s = 0.0;
  for (var i = lane; i < p.D; i = i + SG) { let v = x[base + i]; s = s + v * v; }
  let inv = inverseSqrt(subgroupAdd(s) / f32(p.D) + p.eps);
  let half = p.D / 2u;
  let ob = p.outOff + row * p.outStride;
  for (var i = lane; i < p.D; i = i + SG) {
    let nd = x[base + i] * inv * gamma[i];
    var pd: u32; var sgn: f32;
    if (i < half) { pd = i + half; sgn = -1.0; } else { pd = i - half; sgn = 1.0; }
    let rot = sgn * (x[base + pd] * inv * gamma[pd]);
    y[ob + i] = nd * cos[i] + rot * sin[i];
  }
}
`,rmsnorm_rope_sg_kv16:`// rmsnorm_rope_sg writing into an f16-STORAGE KV cache (kvCache: 'f16'): used ONLY for the K
// projection on the fused decode path, where the normed+roped K is written straight into the
// cache. Keep in lockstep with rmsnorm_rope_sg.wgsl: the ONLY difference is y is array<f16>
// (one f32 -> f16 rounding at the write). The q call keeps the f32 kernel.
enable subgroups;
enable f16;
override SG: u32 = 32u;
struct Params { R: u32, D: u32, eps: f32, outOff: u32, outStride: u32, _pad: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;        // [R, D]
@group(0) @binding(2) var<storage, read> gamma: array<f32>;    // [D]
@group(0) @binding(3) var<storage, read> cos: array<f32>;      // [D]
@group(0) @binding(4) var<storage, read> sin: array<f32>;      // [D]
@group(0) @binding(5) var<storage, read_write> y: array<f16>;  // [outOff + R*outStride]

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let row = wg.x;
  if (row >= p.R) { return; }
  let base = row * p.D;
  var s = 0.0;
  for (var i = lane; i < p.D; i = i + SG) { let v = x[base + i]; s = s + v * v; }
  let inv = inverseSqrt(subgroupAdd(s) / f32(p.D) + p.eps);
  let half = p.D / 2u;
  let ob = p.outOff + row * p.outStride;
  for (var i = lane; i < p.D; i = i + SG) {
    let nd = x[base + i] * inv * gamma[i];
    var pd: u32; var sgn: f32;
    if (i < half) { pd = i + half; sgn = -1.0; } else { pd = i - half; sgn = 1.0; }
    let rot = sgn * (x[base + pd] * inv * gamma[pd]);
    y[ob + i] = f16(nd * cos[i] + rot * sin[i]);
  }
}
`,rmsnorm_rope_sg_kv8:`// rmsnorm_rope_sg writing into the q8 cache (kvCache: 'q8'): used ONLY for the K projection on
// the fused decode path, where the normed+roped K quantizes straight into the cache. Keep the
// math in lockstep with rmsnorm_rope_sg.wgsl; the write side mirrors copy_kv8.wgsl (packed
// snorm8 words + one f32 scale per 32-element block). The q call keeps the f32 kernel.
enable subgroups;
override SG: u32 = 32u;
struct Params { R: u32, D: u32, eps: f32, outRow0: u32, _p0: u32, _p1: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;            // [R, D]
@group(0) @binding(2) var<storage, read> gamma: array<f32>;        // [D]
@group(0) @binding(3) var<storage, read> cos: array<f32>;          // [D]
@group(0) @binding(4) var<storage, read> sin: array<f32>;          // [D]
@group(0) @binding(5) var<storage, read_write> dstQ: array<u32>;   // packed 4 x snorm8 per word
@group(0) @binding(6) var<storage, read_write> dstS: array<f32>;   // [.., D/32] block scales

var<workgroup> wabs: array<f32, 32>; // per-word abs max (D <= 128 -> at most 32 words)
var<workgroup> wblk: array<f32, 4>;  // per-block scale (D/32 <= 4 blocks)

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let row = wg.x;                    // uniform: the barrier pattern below stays safe
  if (row >= p.R) { return; }
  let base = row * p.D;
  var s = 0.0;
  for (var i = lane; i < p.D; i = i + SG) { let v = x[base + i]; s = s + v * v; }
  let inv = inverseSqrt(subgroupAdd(s) / f32(p.D) + p.eps);
  let half = p.D / 2u;
  let W4 = p.D / 4u;

  var vals: array<vec4<f32>, 8>;     // words per lane: W4/SG <= 8 for SG >= 4
  var wi = 0u;
  for (var w = lane; w < W4; w = w + SG) {
    var vv = vec4<f32>(0.0);
    for (var e = 0u; e < 4u; e = e + 1u) {
      let i = w * 4u + e;
      let nd = x[base + i] * inv * gamma[i];
      var pd: u32; var sgn: f32;
      if (i < half) { pd = i + half; sgn = -1.0; } else { pd = i - half; sgn = 1.0; }
      let rot = sgn * (x[base + pd] * inv * gamma[pd]);
      vv[e] = nd * cos[i] + rot * sin[i];
    }
    vals[wi] = vv;
    wi = wi + 1u;
    wabs[w] = max(max(abs(vv.x), abs(vv.y)), max(abs(vv.z), abs(vv.w)));
  }
  workgroupBarrier();
  if (lane < p.D / 32u) {
    var m = 0.0;
    for (var i = 0u; i < 8u; i = i + 1u) { m = max(m, wabs[lane * 8u + i]); }
    let sc = max(m, 1e-30);
    wblk[lane] = sc;
    dstS[(p.outRow0 + row) * (p.D / 32u) + lane] = sc;
  }
  workgroupBarrier();
  wi = 0u;
  for (var w = lane; w < W4; w = w + SG) {
    dstQ[(p.outRow0 + row) * W4 + w] = pack4x8snorm(vals[wi] / wblk[w >> 3u]);
    wi = wi + 1u;
  }
}
`,rmsnorm_sg:`// RMSNorm, subgroup-parallel: one subgroup (= one workgroup) per row; lanes split D and
// reduce the sum-of-squares with subgroupAdd (register-only, no barriers/shared memory).
// Fixes the decode bottleneck where R=1 ran on a single thread. SG is set from the device's
// subgroup size at pipeline creation; requires workgroup_size == subgroup size.
enable subgroups;
override SG: u32 = 32u;
struct Params { R: u32, D: u32, eps: f32, _pad: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let row = wg.x;
  if (row >= p.R) { return; }
  let base = row * p.D;
  var s = 0.0;
  for (var i = lane; i < p.D; i = i + SG) { let v = x[base + i]; s = s + v * v; }
  let total = subgroupAdd(s);                 // sum across the subgroup, broadcast to all lanes
  let inv = inverseSqrt(total / f32(p.D) + p.eps);
  for (var i = lane; i < p.D; i = i + SG) { y[base + i] = x[base + i] * inv * gamma[i]; }
}
`,rmsnorm_sg_af16:`// RMSNorm (subgroup) that writes the normalized activation as f16 - the input side of the
// f16-activation decode matmuls (activation: 'f16'). Reads the f32 residual stream; the
// sum-of-squares reduction stays f32 (accuracy); only the stored output is rounded to f16.
// Identical reduction to rmsnorm_sg, so it is bit-comparable up to the final f16 rounding.
enable subgroups;
enable f16;
override SG: u32 = 32u;
struct Params { R: u32, D: u32, eps: f32, _pad: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f16>;

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let row = wg.x;
  if (row >= p.R) { return; }
  let base = row * p.D;
  var s = 0.0;
  for (var i = lane; i < p.D; i = i + SG) { let v = x[base + i]; s = s + v * v; }
  let total = subgroupAdd(s);
  let inv = inverseSqrt(total / f32(p.D) + p.eps);
  for (var i = lane; i < p.D; i = i + SG) { y[base + i] = f16(x[base + i] * inv * gamma[i]); }
}
`,rmsnorm_wg:`// RMSNorm, no-subgroup fallback: one workgroup per row; threads split D and tree-reduce the
// sum of squares via shared memory. Replaces the one-thread-per-row kernel on this path: at
// decode (R=1) that kernel walked 2xD elements serially on a single thread, latency-bound,
// and it ran twice per layer - the dominant cost of the whole fallback decode step.
// Mirrors rmsnorm_sg exactly, with subgroupAdd swapped for the shared-memory reduction.
override WG: u32 = 64u;
struct Params { R: u32, D: u32, eps: f32, _pad: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;       // [R, D]
@group(0) @binding(2) var<storage, read> gamma: array<f32>;   // [D]
@group(0) @binding(3) var<storage, read_write> y: array<f32>; // [R, D]
var<workgroup> sdata: array<f32, 256>;                        // >= max WG

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wg.x;                        // uniform across the workgroup -> early return is barrier-safe
  if (row >= p.R) { return; }
  let tid = lid.x;
  let base = row * p.D;
  var s = 0.0;
  for (var i = tid; i < p.D; i = i + WG) { let v = x[base + i]; s = s + v * v; }
  sdata[tid] = s;
  workgroupBarrier();
  for (var st = WG / 2u; st > 0u; st = st >> 1u) {
    if (tid < st) { sdata[tid] = sdata[tid] + sdata[tid + st]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(sdata[0] / f32(p.D) + p.eps);
  for (var i = tid; i < p.D; i = i + WG) { y[base + i] = x[base + i] * inv * gamma[i]; }
}
`,rope:`// RoPE (rotate_half) with precomputed full cos/sin [S, D]. x is [S, H, D]. One invocation per element.
struct Params { S: u32, H: u32, D: u32, _pad: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;       // [S, H, D]
@group(0) @binding(2) var<storage, read> cos: array<f32>;     // [S, D]
@group(0) @binding(3) var<storage, read> sin: array<f32>;     // [S, D]
@group(0) @binding(4) var<storage, read_write> y: array<f32>; // [S, H, D]

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  if (idx >= p.S * p.H * p.D) { return; }
  let d = idx % p.D;
  let sh = idx / p.D;
  let s = sh / p.H;
  let half = p.D / 2u;
  let row = sh * p.D;  // s*H*D + h*D
  var rot: f32;
  if (d < half) {
    rot = -x[row + d + half];
  } else {
    rot = x[row + d - half];
  }
  y[idx] = x[idx] * cos[s * p.D + d] + rot * sin[s * p.D + d];
}
`,rope_partial:`// Partial RoPE: rotate only the first ROT dims of each head (rotate_half within [0,ROT)); the
// remaining head_dim-ROT dims pass through unrotated. cos/sin are [S, ROT]. x/y are [S, H, D].
// Matches tools/qwen35_numpy._rope_partial (Qwen3.5 full-attention layers, partial_rotary_factor).
struct Params { S: u32, H: u32, D: u32, ROT: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;        // [S, H, D]
@group(0) @binding(2) var<storage, read> cosb: array<f32>;     // [S, ROT]
@group(0) @binding(3) var<storage, read> sinb: array<f32>;     // [S, ROT]
@group(0) @binding(4) var<storage, read_write> y: array<f32>;  // [S, H, D]

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  if (idx >= p.S * p.H * p.D) { return; }
  let d = idx % p.D;
  if (d >= p.ROT) { y[idx] = x[idx]; return; }   // passthrough tail
  let sh = idx / p.D;
  let s = sh / p.H;
  let half = p.ROT / 2u;
  var rot: f32;
  if (d < half) { rot = -x[idx + half]; } else { rot = x[idx - half]; }
  y[idx] = x[idx] * cosb[s * p.ROT + d] + rot * sinb[s * p.ROT + d];
}
`,sampler_penalty:`// GPU logits pre-filter for sampling: applies repetition_penalty + presence_penalty, then
// no_repeat_ngram bans, in place on the full vocab logit buffer, so only a tiny top-K candidate set
// has to be read back (not all ~151k logits). rep_penalty matches transformers.js over the DEDUPED
// prompt+generated id set (logit<0 ? *penalty : /penalty); presence_penalty then SUBTRACTS a flat
// amount from every seen token (the additive anti-repetition knob the Qwen3.5 family recommends,
// applied after the multiplicative rep_penalty like vLLM); then ngram-banned next-tokens go to
// -Infinity. Both id lists are computed on the CPU each step (exact, since at syncN=1 the full
// history is known) and uploaded. presence is 0 unless requested, so \`v*penalty - 0.0 == v*penalty\`
// keeps the rep-penalty-only path bit-identical. Temperature is NOT applied here: top-k is invariant
// under the monotonic divide, so temperature is applied on the CPU to just the K candidate values
// before softmax (bit-identical, one less pass). Single workgroup, no subgroup ops -> all devices.
// The storageBarrier guarantees every penalty write lands before any ban write, so a token that is
// both repeated and ngram-banned ends at -inf (ban wins, matching the reference order penalties -> ngram).
override WG: u32 = 256u;
// negInf carries the -Infinity bit pattern (0xff800000) from the host: bitcasting it at RUNTIME yields
// -inf, whereas bitcast<f32>(0xff800000u) is a const-expression evaluating to inf, which is a WGSL
// shader-creation error. (Runtime inf is fine; only const/override inf/nan is rejected.)
struct Params { affectedLen: u32, banLen: u32, penalty: f32, negInf: u32, presence: f32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> affectedIds: array<u32>;   // deduped prompt+generated ids
@group(0) @binding(2) var<storage, read> banIds: array<u32>;        // ngram-banned next-token ids
@group(0) @binding(3) var<storage, read_write> logits: array<f32>;  // [vocab], modified in place

@compute @workgroup_size(WG)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  for (var i = tid; i < p.affectedLen; i = i + WG) {
    let t = affectedIds[i];
    let v = logits[t];
    let rp = select(v / p.penalty, v * p.penalty, v < 0.0);   // repetition_penalty (multiplicative)
    logits[t] = rp - p.presence;                              // presence_penalty (subtractive; 0 = no-op)
  }
  storageBarrier();                                  // all penalty writes before any ban write
  for (var i = tid; i < p.banLen; i = i + WG) {
    logits[banIds[i]] = bitcast<f32>(p.negInf);      // -Infinity (runtime bitcast)
  }
}
`,sampler_sigma:`// Mean/variance statistics of the (penalty-filtered) logits for the top-n-sigma warper
// (arXiv 2411.07641): the CPU keeps candidates with logit >= max - n * sigma, where sigma is the
// standard deviation of the FULL logit vector (the paper's statistic - a top-K-only estimate is
// biased). Runs AFTER sampler_penalty and BEFORE the argmax_masked rounds (those mask winners in
// place, which would corrupt the moments). Banned entries (-inf sentinel) are excluded; numerical
// stability comes from centering on the global max before accumulating (logits are O(10), so
// sum-of-squares around the max stays well inside f32). Three f32s are read back:
// out = [sum(x - max), sum((x - max)^2), count] -> CPU: var = q/c - (s/c)^2.
override WG: u32 = 256u;
struct Params { N: u32, _0: u32, _1: u32, _2: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read_write> outStats: array<f32>; // [sum, sumsq, count] centered on max

const NEG_SENTINEL: f32 = -3.0e38; // below any real logit; banned entries sit at f32 -inf

var<workgroup> sa: array<f32, 256>;
var<workgroup> sb: array<f32, 256>;
var<workgroup> sc: array<f32, 256>;

@compute @workgroup_size(WG)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  var m = -3.4e38;
  for (var i = tid; i < p.N; i = i + WG) {
    let v = logits[i];
    if (v > NEG_SENTINEL && v > m) { m = v; }
  }
  sa[tid] = m;
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s = s >> 1u) {
    if (tid < s && sa[tid + s] > sa[tid]) { sa[tid] = sa[tid + s]; }
    workgroupBarrier();
  }
  let gmax = sa[0];
  workgroupBarrier();
  var acc = 0.0;
  var accq = 0.0;
  var cnt = 0.0;
  for (var i = tid; i < p.N; i = i + WG) {
    let v = logits[i];
    if (v > NEG_SENTINEL) {
      let d = v - gmax;
      acc = acc + d;
      accq = accq + d * d;
      cnt = cnt + 1.0;
    }
  }
  sa[tid] = acc;
  sb[tid] = accq;
  sc[tid] = cnt;
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s = s >> 1u) {
    if (tid < s) {
      sa[tid] = sa[tid] + sa[tid + s];
      sb[tid] = sb[tid] + sb[tid + s];
      sc[tid] = sc[tid] + sc[tid + s];
    }
    workgroupBarrier();
  }
  if (tid == 0u) {
    outStats[0] = sa[0];
    outStats[1] = sb[0];
    outStats[2] = sc[0];
  }
}
`,slice_cols:`// Extract a contiguous column range [off, off+w) from each row of a [rows, stride] buffer into a
// packed [rows, w] buffer. Splits the DeltaNet conv output (q|k|v concatenated per token) into the
// separate q/k/v activation buffers the scan reads.
struct Params { rows: u32, w: u32, stride: u32, off: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;      // [rows, stride]
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;// [rows, w]

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  if (i >= p.rows * p.w) { return; }
  let r = i / p.w;
  let c = i % p.w;
  dst[i] = src[r * p.stride + p.off + c];
}
`,split_head:`// De-interleave a per-head doubled projection [S, H, 2*Dh] into [S, H, Dh], taking the half at
// \`off\` (0 = query, Dh = gate). The Qwen3.5 gated-attention q_proj packs query and output-gate
// interleaved per head; this pulls one out into a packed buffer.
struct Params { S: u32, H: u32, Dh: u32, off: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;      // [S, H, 2*Dh]
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;// [S, H, Dh]

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  if (i >= p.S * p.H * p.Dh) { return; }
  let d = i % p.Dh;
  let sh = i / p.Dh;          // s*H + h
  dst[i] = src[sh * (2u * p.Dh) + p.off + d];
}
`,swiglu:`// SwiGLU gate: y[i] = silu(gate[i]) * up[i], silu(g) = g * sigmoid(g). One invocation per element.
struct Params { n: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> gate: array<f32>;
@group(0) @binding(2) var<storage, read> up: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  if (i >= p.n) { return; }
  let g = gate[i];
  y[i] = (g / (1.0 + exp(-g))) * up[i];
}
`};var $t=class extends Error{constructor(m){super(m);wr(this,"name","WebGPUUnavailableError")}},Tr=class extends Error{constructor(m){super(m);wr(this,"name","GpuOutOfMemoryError")}};function Va(b,m,c){if(c<=0)return[];for(let _=Math.min(m,b.length-1);_>=2;_--){const K=b.length-_;e:for(let oe=K-1;oe>=0;oe--){for(let E=0;E<_;E++)if(b[oe+E]!==b[K+E])continue e;const D=oe+_;return b.slice(D,Math.min(D+c,b.length))}}return[]}function Ra(b,m,c){return m<=0?!1:b/m>=(c?1.5:2)}var It=class{constructor(b){wr(this,"mt",new Uint32Array(624));wr(this,"idx",625);this.seed(b)}seed(b){if(b==null){const D=new Uint32Array(1);crypto.getRandomValues(D),b=D[0]}const m=this.mt,c=(D,E)=>Math.imul(D,E)>>>0,_=[];for(let D=b||0;D>0;D=Math.floor(D/4294967296))_.push(D&4294967295);_.length||_.push(0),m[0]=19650218;for(let D=1;D<624;++D)m[D]=c(1812433253,m[D-1]^m[D-1]>>>30)+D>>>0;let K=1,oe=0;for(let D=Math.max(624,_.length);D>0;--D,++K,++oe)K>=624&&(m[0]=m[623],K=1),oe>=_.length&&(oe=0),m[K]=(m[K]^c(m[K-1]^m[K-1]>>>30,1664525))+_[oe]+oe>>>0;for(let D=623;D>0;--D,++K)K>=624&&(m[0]=m[623],K=1),m[K]=(m[K]^c(m[K-1]^m[K-1]>>>30,1566083941))-K>>>0;m[0]=2147483648,this.idx=624}int32(){const b=this.mt;if(this.idx>=624){for(let c=0;c<624;++c){const _=b[c]&2147483648|b[(c+1)%624]&2147483647;b[c]=(b[(c+397)%624]^_>>>1^(_&1?2567483615:0))>>>0}this.idx=0}let m=b[this.idx++];return m^=m>>>11,m^=m<<7&2636928640,m^=m<<15&4022730752,m^=m>>>18,m>>>0}random(){return((this.int32()>>>5)*67108864+(this.int32()>>>6))/9007199254740992}};function ut(b){return Uint32Array.from(new Set(b))}function lt(b,m){if(b.length+1<m)return[];const c=new Map;for(let K=0;K<b.length+1-m;++K){const oe=[];for(let we=0;we<m;++we)oe.push(b[K+we]);const D=JSON.stringify(oe.slice(0,m-1)),E=c.get(D)??[];E.push(oe[m-1]),c.set(D,E)}const _=b.slice(b.length+1-m,b.length);return c.get(JSON.stringify(_))??[]}function ct(b,m,c,_){const K=c.length,oe=_.range>0?Math.max(0,K-_.range):0,D=_.allowedLength+32,E=Array.from(m),we=Array.from(b);if(K>oe&&_.multiplier>0)for(let d=0;d<we.length;d++){const S=we[d];if(_.breakers.has(S))continue;let I=0;for(let ie=oe;ie<K;ie++){if(c[ie]!==S)continue;let ce=0;for(;ce<D&&ie-1-ce>=oe&&!_.breakers.has(c[ie-1-ce])&&!_.breakers.has(c[K-1-ce])&&c[ie-1-ce]===c[K-1-ce];)ce++;if(ce>I&&(I=ce),I>=D)break}I>=_.allowedLength&&(E[d]-=_.multiplier*Math.pow(_.base,I-_.allowedLength))}const fe=Array.from(we.keys()).sort((d,S)=>E[S]-E[d]||d-S);return{ids:fe.map(d=>we[d]),vals:fe.map(d=>E[d])}}function Ha(b){let m=b[0];for(let K=1;K<b.length;++K)b[K]>m&&(m=b[K]);const c=Array.from(b,K=>Math.exp(K-m));let _=0;for(const K of c)_+=K;return c.map(K=>K/_)}function Ur(b,m,c,_,K=1,oe=0){const D=m.length,E=new Float32Array(D);for(let I=0;I<D;++I)E[I]=m[I]/c;const we=Ha(E);let fe=D;if(oe>0){const I=oe*we[0];let ie=1;for(;ie<D&&we[ie]>=I;)ie++;ie<fe&&(fe=ie)}if(K<1){let I=0,ie=0;for(;ie<D&&(I+=we[ie],ie++,!(I>=K)););ie<fe&&(fe=ie)}fe<1&&(fe=1);let d=0;for(let I=0;I<fe;++I)d+=we[I];let S=_.random()*d;for(let I=0;I<fe;++I)if(S-=we[I],S<0)return b[I];return b[fe-1]}const Ea={FLOAT:Float32Array,UINT8:Uint8Array,FLOAT16:Uint16Array},Aa=["matmul_split","matmul_resid","matmul_q2","rope","swiglu","copy"],Wa=2048,dt=256,Rn=16,Qt=new ArrayBuffer(64),Ot=new DataView(Qt),La=new Uint8Array(Qt);function Ft(b){for(let m=0;m<b.length;m++){const c=b[m];c[0]==="f"?Ot.setFloat32(m*4,c[1],!0):Ot.setUint32(m*4,c[1]>>>0,!0)}return La.subarray(0,Math.ceil(b.length/4)*16)}const ja=(b,m)=>{for(let c=0;c<m.length;c++)if(b[c]!==m[c])return!1;return!0};function Ta(b,m,c=b.head_dim){const _=c/2,K=b.rope,oe=K.rope_theta,D=K.rope_type==="yarn"?K.factor??1:1,E=new Float64Array(_),we=K.original_max_position_embeddings??0,fe=D===1?0:Math.max(0,Math.floor(c*Math.log(we/(64*Math.PI))/(2*Math.log(oe)))),d=D===1?0:Math.min(_-1,Math.ceil(c*Math.log(we/(2*Math.PI))/(2*Math.log(oe))));for(let ce=0;ce<_;ce++){const tn=oe**(2*ce/c);if(D===1){E[ce]=1/tn;continue}const Se=Math.min(1,Math.max(0,(ce-fe)/(d-fe)));E[ce]=1/(D*tn)*Se+1/tn*(1-Se)}const S=D===1?1:Math.fround(.1*Math.log(D)+1),I=new Float32Array(m*_),ie=new Float32Array(m*_);for(let ce=0;ce<m;ce++)for(let tn=0;tn<_;tn++){const Se=ce*E[tn];I[ce*_+tn]=Math.fround(Math.cos(Se)*S),ie[ce*_+tn]=Math.fround(Math.sin(Se)*S)}return[I,ie]}async function $a(b){var c;const m={};try{return await Ua(b,m)}catch(_){throw(c=m.device)==null||c.destroy(),_}}async function Ua(b,m){var Pt,zt,Vt,Rt,Ht,Et,At,Wt,Lt,jt;const c=typeof b=="string"?{modelUrl:b}:b,_=c.modelUrl?c.modelUrl.replace(/\/$/,""):null;if(!_&&!c.manifestUrl&&!c.manifest)throw new Error("createEngine: provide modelUrl, manifestUrl, or an in-memory manifest");if(c.manifest&&!_&&!c.dataUrl)throw new Error("createEngine: an in-memory manifest needs dataUrl (or modelUrl) for the weights file");const K=c.powerPreference??"high-performance",oe=c.fetchJson??(async e=>{const n=await fetch(e);if(!n.ok)throw new Error(`bitgpu: fetch ${e} failed: HTTP ${n.status}`);if((n.headers.get("content-type")??"").includes("text/html"))throw new Error(`bitgpu: ${e} returned HTML, not JSON (a SPA fallback is probably serving index.html for missing model files)`);return n.json()}),D=c.fetchArrayBuffer??(async e=>{var u;const n=await fetch(e);if(!n.ok)throw new Error(`bitgpu: fetch ${e} failed: HTTP ${n.status}`);const t=Number(n.headers.get("content-length")??0);if(!n.body||!t)return n.arrayBuffer();const r=n.body.getReader(),a=[];let o=0;for(;;){const{done:p,value:g}=await r.read();if(p)break;a.push(g),o+=g.byteLength,(u=c.onProgress)==null||u.call(c,{phase:"weights",loaded:o,total:t})}const l=new Uint8Array(o);let i=0;for(const p of a)l.set(p,i),i+=p.byteLength;return l.buffer});if(typeof navigator>"u"||!navigator.gpu)throw new $t("WebGPU is not available (no navigator.gpu). Use a WebGPU-capable browser over a secure context.");(Pt=c.onProgress)==null||Pt.call(c,{phase:"manifest"});const E=c.manifest??await oe(c.manifestUrl??`${_}/manifest.json`);(zt=c.onProgress)==null||zt.call(c,{phase:"weights"});const we=c.dataUrl??`${_}/${E.data_file}`;let fe;c.aux?fe=c.aux instanceof Uint8Array?new Uint8Array(c.aux).buffer:c.aux:fe=await D(c.auxUrl??`${_}/${E.aux_file}`);const d=E.arch,S=E.tensors,I=`layers.${d.layers}.final_norm_layernorm`;if(d.act!=="silu")throw new Error(`bitgpu: unsupported activation '${d.act}' (kernels implement silu/SwiGLU)`);if(d.head_dim>(d.hybrid?256:128))throw new Error(`bitgpu: unsupported head_dim ${d.head_dim}`);const ie=((Vt=d.hybrid)==null?void 0:Vt.rotary_dim)??d.head_dim;if(d.heads%d.kv_heads!==0)throw new Error(`bitgpu: heads ${d.heads} not divisible by kv_heads ${d.kv_heads} (GQA kernels assume an integer group size)`);if(!S[I])throw new Error(`bitgpu: manifest is missing the final norm tensor '${I}'`);if(E.version!==void 0&&E.version!==1&&E.version!==2)throw new Error(`bitgpu: unsupported manifest version ${E.version} (this engine reads versions 1 and 2)`);if(!S.cos_cache!=!S.sin_cache)throw new Error("bitgpu: manifest has only one of cos_cache/sin_cache");if(!S.cos_cache&&!(d.rope&&d.rope.rope_theta))throw new Error("bitgpu: manifest has neither baked cos_cache/sin_cache RoPE tensors nor arch.rope parameters");for(const[e,n]of Object.entries(S)){if(n.block!==void 0&&n.block!==128)throw new Error(`bitgpu: tensor ${e} has block ${n.block} (kernels assume 128)`);if(n.container===void 0)continue;if(n.container!=="q1_0")throw new Error(`bitgpu: tensor ${e} has unknown container '${n.container}'`);const t=n.N??n.rows,r=n.K??n.cols;if(!t||!r||r%128!==0)throw new Error(`bitgpu: tensor ${e}: q1_0 container needs N/K (or rows/cols) with K a multiple of 128`);const a=n.weight;if(!a||a.src!=="data"||a.len!==t*(r/128)*18)throw new Error(`bitgpu: tensor ${e}: q1_0 region is ${a==null?void 0:a.len} bytes in '${a==null?void 0:a.src}', expected ${t*(r/128)*18} in the data file`);n.q1_0=a,n.weight={dtype:"UINT8",src:a.src,off:a.off,len:t*(r/8)},n.scales={dtype:"FLOAT",src:a.src,off:a.off,len:t*(r/128)*4},n.zp=void 0}const ce=e=>{if(e.src!=="aux")throw new Error("bitgpu: internal - readRef reads aux-file refs; data-file tensors stream through routes");if(e.off+e.len>fe.byteLength)throw new Error(`bitgpu: tensor range ${e.off}+${e.len} exceeds the aux file (${fe.byteLength} bytes); the download is truncated or the manifest does not match it`);const n=Ea[e.dtype];if(n===Uint8Array)return new Uint8Array(fe,e.off,e.len);const t=n.BYTES_PER_ELEMENT;return e.off%t===0?new n(fe,e.off,e.len/t):new n(fe.slice(e.off,e.off+e.len))},tn=e=>ce(e),Se=await navigator.gpu.requestAdapter({powerPreference:K});if(!Se)throw new $t("No suitable WebGPU adapter was found.");const Xt=Se.features.has("subgroups"),In=Se.info??{},Ae=In.subgroupMaxSize??32,Yt=In.subgroupMinSize??Ae,Zt=c.forceNoSubgroups??!1,Jt=Math.min(256,Math.max(32,1<<Math.round(Math.log2(c.noSubgroupWorkgroupSize??64)))),ea=c.prefillTiling==="never",na=c.prefillTiling==="always",pt=e=>na||!ea&&e>=64,vr=Math.max(1,c.syncSteps??4),L=Math.max(1,c.maxSeqLen??Wa),Ne=Xt&&Yt===Ae&&(Ae===16||Ae===32||Ae===64)&&d.head_dim%Ae===0&&!Zt,an=c.kvCache==="f16"&&!d.hybrid&&Se.features.has("shader-f16"),ve=c.kvCache==="q8",ke=c.overflow==="sinks",Qe=ke?Math.max(1,Math.floor(c.sinkTokens??4)):0;if(ke&&d.hybrid)throw new Error("bitgpu: overflow 'sinks' is not yet supported for the qwen3_5 hybrid backbone (the full-attention read path has no sink/roll K-rotation, and windowing only the full layers while the linear layers keep full-history state is unvalidated)");if(ke&&L<Qe+64)throw new Error(`bitgpu: overflow 'sinks' needs maxSeqLen >= sinkTokens + 64 (got ${L} with ${Qe} sinks)`);const Dn=an?2:ve?1:4,nr=c.activation==="f16"&&Ne&&Se.features.has("shader-f16"),kr=[];Ne&&kr.push("subgroups"),(an||nr)&&kr.push("shader-f16");const yr=Se.features.has("timestamp-query");yr&&kr.push("timestamp-query");const ra=134217728,ta=268435456;let nn=0,ft=0;const Hn=e=>{const n=e+3&-4;nn=Math.max(nn,n),ft+=n};for(const e of Object.values(S))e.kind==="q2"?(Hn(e.weight.len*2),Hn(e.scales.len)):e.kind==="f32"&&e.weight&&Hn(e.weight.len);const gt=(e,n)=>e.reduce((t,r)=>t+S[r][n].len,0);for(let e=0;e<d.layers;e++){let n;d.hybrid?n=[...d.hybrid.layer_types[e]==="full"?["attn.q_proj","attn.k_proj","attn.v_proj","attn.o_proj"]:["linear.in_qkv","linear.z","linear.a","linear.b","linear.out_proj"],"mlp.gate_proj","mlp.up_proj","mlp.down_proj"].map(t=>[`layers.${e}.${t}`]):n=[[`layers.${e}.attn.q_proj`,`layers.${e}.attn.k_proj`,`layers.${e}.attn.v_proj`],[`layers.${e}.mlp.gate_proj`,`layers.${e}.mlp.up_proj`],[`layers.${e}.attn.o_proj`],[`layers.${e}.mlp.down_proj`]];for(const t of n)Hn(gt(t,"weight")),Hn(gt(t,"scales"))}const Cr=((Rt=S.embed_tokens.zp)==null?void 0:Rt.len)??S.embed_tokens.rows*(S.embed_tokens.cols/128)/2;for(const e of[S.embed_tokens.weight,S.embed_tokens.scales,E.luts.tgt4])Hn(e.len);if(Hn(Cr),ve&&d.head_dim%32!==0)throw new Error(`bitgpu: kvCache 'q8' needs head_dim divisible by 32 (got ${d.head_dim}); use 'f16' or 'f32' for this model`);const aa=L*d.kv_heads*d.head_dim*Dn;nn=Math.max(nn,aa,dt*Math.max(d.heads*d.head_dim,d.intermediate)*4,32*d.vocab*4);const _r={};if(nn>ra){if(nn>Se.limits.maxStorageBufferBindingSize)throw new Tr(`this model needs a ${Math.ceil(nn/1048576)} MiB storage binding but the adapter's maxStorageBufferBindingSize is ${Math.floor(Se.limits.maxStorageBufferBindingSize/1048576)} MiB`);_r.maxStorageBufferBindingSize=nn}if(nn>ta){if(nn>Se.limits.maxBufferSize)throw new Tr(`this model needs a ${Math.ceil(nn/1048576)} MiB buffer but the adapter's maxBufferSize is ${Math.floor(Se.limits.maxBufferSize/1048576)} MiB`);_r.maxBufferSize=nn}const s=await Se.requestDevice({requiredFeatures:kr,requiredLimits:Object.keys(_r).length?_r:void 0});m.device=s;const oa=s.lost.then(e=>{var t;const n={reason:String(e.reason??"unknown"),message:e.message};return n.reason!=="destroyed"&&((t=c.onDeviceLost)==null||t.call(c,n)),n});s.addEventListener("uncapturederror",e=>{console.error(`[bitgpu] uncaptured WebGPU error: ${e.error.message}`)}),(Ht=c.onProgress)==null||Ht.call(c,{phase:"pipelines"});const xr={},ia=async(e,n)=>{const t=za[e];if(t===void 0)throw new Error(`shader not found: ${e}`);const r=s.createShaderModule({code:t,label:e}),a=(await r.getCompilationInfo()).messages.find(o=>o.type==="error");if(a)throw new Error(`WGSL compile error in ${e} (L${a.lineNum}:${a.linePos}): ${a.message}`);xr[e]=await s.createComputePipelineAsync({layout:"auto",compute:{module:r,entryPoint:"main",constants:n}})},Kr=4,ye=[...Aa.map(e=>[e]),["matmul_split_tiled"],["matmul_resid_tiled"],["argmax"],["embed_gather"],["embed_gather_batch"],["sampler_penalty"],["argmax_masked"],["logsumexp"],["sampler_sigma"]];if(Ne){for(const e of["rmsnorm_sg","attention_sg","matmul_split_sg","matmul_q2_sg","rmsnorm_rope_sg"])ye.push([e,{SG:Ae}]);for(const e of["matmul_split_sm","matmul_resid_sm","matmul_q2_sm"])ye.push([e,{SG:Ae}]);for(const e of["matmul_resid_mr_sg","matmul_swiglu_mr_sg"])ye.push([e,{SG:Ae,ROWS:Kr}])}else{for(const e of["matmul_split_wg","matmul_resid_wg","matmul_q2_wg","rmsnorm_wg"])ye.push([e,{WG:Jt}]);ye.push(["attention_wg"])}if(an)if(ye.push(["copy_kv16"]),Ne)for(const e of["attention_sg_kv16","rmsnorm_rope_sg_kv16"])ye.push([e,{SG:Ae}]);else ye.push(["attention_wg_kv16"]);if(ve)if(ye.push(["copy_kv8"]),Ne)for(const e of["attention_sg_kv8","rmsnorm_rope_sg_kv8"])ye.push([e,{SG:Ae}]);else ye.push(["attention_wg_kv8"]);if(nr){for(const e of["rmsnorm_sg_af16","matmul_split_sg_af16"])ye.push([e,{SG:Ae}]);for(const e of["matmul_swiglu_mr_sg_af16","matmul_resid_mr_sg_af16"])ye.push([e,{SG:Ae,ROWS:Kr}])}const Nr=Ne&&Ae<=d.head_dim/2;if(ke){const e=an?"attention_sg_kv16_roll":ve?"attention_sg_kv8_roll":"attention_sg_roll",n=an?"attention_wg_kv16_roll":ve?"attention_wg_kv8_roll":"attention_wg_roll";Nr?ye.push([e,{SG:Ae}]):ye.push([n])}const mt=ke?an?Nr?"attention_sg_kv16_roll":"attention_wg_kv16_roll":ve?Nr?"attention_sg_kv8_roll":"attention_wg_kv8_roll":Nr?"attention_sg_roll":"attention_wg_roll":an?Ne?"attention_sg_kv16":"attention_wg_kv16":ve?Ne?"attention_sg_kv8":"attention_wg_kv8":Ne?"attention_sg":"attention_wg",sa=an?"rmsnorm_rope_sg_kv16":"rmsnorm_rope_sg",ua=an?"copy_kv16":"copy";if(d.hybrid){for(const e of["conv1d_causal","deltanet_gbeta","rope_partial","slice_cols","split_head","gate_sigmoid"])ye.push([e]);ye.push(["deltanet_recur",{WGV:d.hybrid.linear_head_dim}]),ye.push(["deltanet_norm_gate",{WG:64}]),ye.push(["attention_online",{WGD:d.head_dim}]),ye.push([ve?"attention_online_cache_kv8":"attention_online_cache",{WGD:d.head_dim}])}await Promise.all(ye.map(([e,n])=>ia(e,n)));const k=GPUBufferUsage.STORAGE,M=GPUBufferUsage.COPY_DST,z=GPUBufferUsage.COPY_SRC,ht=GPUBufferUsage.UNIFORM;let A=null;const Ie=()=>{if(A){for(const e of A)e.destroy();A=[],hn=null}};let hn=null,Mn=null;const bt=e=>{if(!hn)return;const n=hn.get(e.size);n?n.push(e):hn.set(e.size,[e])},qr=(e,n=k|M)=>{const t=s.createBuffer({size:e.byteLength,usage:n});return s.queue.writeBuffer(t,0,e),A==null||A.push(t),t},Dr={};let on=null,En=0,Mr=0,$r=0,wt=0;const Bn=(e,n=0,t=0)=>{on=e?Dr[e]??(Dr[e]={buf:[],disp:[]}):null,$r=n,wt=t,En=0,Mr=0},rr=()=>{for(const e of Object.values(Dr))for(const n of e.disp)n.bg=null,n.last=null},v=e=>{var r;if(!on){const a=(r=hn==null?void 0:hn.get(e*4))==null?void 0:r.pop();if(a)return Mn==null||Mn.push(a),a;const o=s.createBuffer({size:e*4,usage:k|z|M});return A==null||A.push(o),Mn==null||Mn.push(o),o}const n=$r>0?e/$r*wt:e;let t=on.buf[En];return(!t||t.size!==n*4)&&(t=s.createBuffer({size:n*4,usage:k|z|M}),on.buf[En]=t),En++,t},tr=e=>{const n=e*2;if(!on){const r=s.createBuffer({size:n,usage:k|z|M});return A==null||A.push(r),r}let t=on.buf[En];return(!t||t.size!==n)&&(t=s.createBuffer({size:n,usage:k|z|M}),on.buf[En]=t),En++,t},bn=s.createBuffer({size:16,usage:k}),xn=s.createBuffer({size:16,usage:k});let qe=[],De=0;const la=()=>{qe=[],De=0};if(E.luts.tgt2.src!=="aux")throw new Error("bitgpu: luts.tgt2 must live in the aux file (the streaming loader needs it before the weights arrive)");const Ir=tn(E.luts.tgt2),vt=new Uint8Array(256);for(let e=0;e<256;e++){let n=0;for(let t=0;t<8;t++)n|=((Ir[2*e+(t>>2)]>>2*(t&3)&3)>>1&1)<<t;vt[e]=n}const Br=[],Kn=e=>s.createBuffer({size:e+3&-4,usage:k|M}),Nn=(e,n)=>{let t=0,r=new Uint8Array(0);return{push(a){let o=a;r.length&&(o=new Uint8Array(r.length+a.length),o.set(r),o.set(a,r.length));const l=o.length&-4;l&&s.queue.writeBuffer(e,n+t,o,0,l),r=o.subarray(l).slice(),t+=l},finish(){if(!r.length)return;const a=new Uint8Array(4);a.set(r),s.queue.writeBuffer(e,n+t,a),t+=4,r=new Uint8Array(0)}}},Sr=(e,n,t)=>{e.src==="aux"?(n(new Uint8Array(ce(e).buffer,e.off,e.len)),t()):Br.push({off:e.off,len:e.len,push:n,finish:t})},Gr=(e,n,t=0)=>{const r=Nn(n,t);Sr(e,r.push,r.finish)},kt=e=>n=>{const t=new Uint8Array(n.length);for(let r=0;r<n.length;r++)t[r]=vt[n[r]];e(t)},yt=e=>n=>{const t=new Uint8Array(n.length*2);for(let r=0;r<n.length;r++)t[2*r]=Ir[2*n[r]],t[2*r+1]=Ir[2*n[r]+1];e(t)},ca=(e,n,t=0)=>{const r=Nn(n,t);Sr(e,kt(r.push),r.finish)},da=(e,n)=>{const t=Nn(n,0);Sr(e,yt(t.push),t.finish)},pa=e=>{const n=e&32768?-1:1,t=e>>10&31,r=e&1023;return t===0?n*r*2**-24:t===31?r?NaN:n*(1/0):n*(1024+r)*2**(t-25)},Or=(e,n,t,r,a)=>{let o=0,l=0;const i=u=>{const p=new Uint8Array(u.length),g=new Float32Array((u.length>>4)+2);let h=0,N=0;for(let G=0;G<u.length;G++)o===0?(l=u[G],o=1):o===1?(g[N++]=pa(l|u[G]<<8),o=2):(p[h++]=u[G],o=o===17?0:o+1);h&&n(p.subarray(0,h)),N&&r(new Uint8Array(g.buffer,0,N*4))};Br.push({off:e.off,len:e.len,push:i,finish:()=>{t(),a()}})},Fr=e=>{const n=new Uint8Array(e.len);let t=0;return Sr(e,r=>{n.set(r,t),t+=r.length},()=>{}),n};s.pushErrorScope("validation"),s.pushErrorScope("out-of-memory");const re={},Qr=[];for(const[e,n]of Object.entries(S))if(n.kind==="q2"){const t=Kn(n.weight.len*2),r=Kn(n.scales.len);if(n.q1_0){const o=Nn(t,0),l=Nn(r,0);Or(n.q1_0,yt(o.push),o.finish,l.push,l.finish)}else da(n.weight,t),Gr(n.scales,r);const a={N:n.N,K:n.K,nb:n.K/128,zp:2,codes:t,scales:r};if(n.zp){const o=Fr(n.zp);Qr.push(()=>{const l=o[0];for(let u=1;u<o.length;u++)if(o[u]!==l)throw new Error(`bitgpu: tensor ${e} has non-uniform 2-bit zero-points (the q2 kernels assume one zp for the whole tensor)`);const i=l&3;if(l!==i*85)throw new Error(`bitgpu: tensor ${e} has non-uniform 2-bit zero-points within a byte (the q2 kernels assume one zp for the whole tensor)`);a.zp=i})}re[e]=a}else if(n.kind==="f32"&&n.weight){const t=Kn(n.weight.len);Gr(n.weight,t),re[e]={buf:t}}const Pr=e=>{const n=Kn(e.reduce((o,l)=>o+l.weight.len,0)),t=Kn(e.reduce((o,l)=>o+l.scales.len,0));let r=0,a=0;for(const o of e){if(o.q1_0){const l=Nn(n,r),i=Nn(t,a);Or(o.q1_0,kt(l.push),l.finish,i.push,i.finish)}else ca(o.weight,n,r),Gr(o.scales,t,a);r+=o.weight.len,a+=o.scales.len}return{sign:n,scales:t}};if(E.arch.hybrid){const e=n=>{const t=S[n];re[n]={N:t.N,K:t.K,nb:t.K/128,N0:t.N,N1:0,N2:0,...Pr([t])}};for(let n=0;n<d.layers;n++){for(const t of["mlp.gate_proj","mlp.up_proj","mlp.down_proj"])e(`layers.${n}.${t}`);if(E.arch.hybrid.layer_types[n]==="full")for(const t of["attn.q_proj","attn.k_proj","attn.v_proj","attn.o_proj"])e(`layers.${n}.${t}`);else for(const t of["linear.in_qkv","linear.z","linear.a","linear.b","linear.out_proj"])e(`layers.${n}.${t}`)}}else for(let e=0;e<d.layers;e++){const n=S[`layers.${e}.attn.q_proj`],t=S[`layers.${e}.attn.k_proj`],r=S[`layers.${e}.attn.v_proj`];re[`layers.${e}.attn.qkv`]={K:n.K,nb:n.K/128,N0:n.N,N1:t.N,N2:r.N,...Pr([n,t,r])};const a=S[`layers.${e}.mlp.gate_proj`],o=S[`layers.${e}.mlp.up_proj`];re[`layers.${e}.mlp.gateup`]={K:a.K,nb:a.K/128,N0:a.N,N1:o.N,N2:0,...Pr([a,o])};for(const l of[`layers.${e}.attn.o_proj`,`layers.${e}.mlp.down_proj`]){const i=S[l];re[l]={N:i.N,K:i.K,nb:i.K/128,...Pr([i])}}}const zr=e=>{const n=Kn(e.len);return Gr(e,n),n};let An,Wn,Ln;if(S.embed_tokens.q1_0){An=Kn(S.embed_tokens.weight.len),Wn=Kn(S.embed_tokens.scales.len);const e=Nn(An,0),n=Nn(Wn,0);Or(S.embed_tokens.q1_0,e.push,e.finish,n.push,n.finish),Ln=Kn(Cr),s.queue.writeBuffer(Ln,0,new Uint8Array(Cr+3&-4).fill(136))}else An=zr(S.embed_tokens.weight),Wn=zr(S.embed_tokens.scales),Ln=zr(S.embed_tokens.zp);const Vr=zr(E.luts.tgt4);let On,ar;if(S.cos_cache){const e=Fr(S.cos_cache),n=Fr(S.sin_cache);On=new Float32Array(e.buffer),ar=new Float32Array(n.buffer);const t=On.length/(ie/2);if(L>t)throw new Error(`bitgpu: maxSeqLen ${L} exceeds the model's baked RoPE cache (${t} positions); lower maxSeqLen or re-export with a longer cache`)}else{const e=d.max_positions??40960;if(L>e)throw new Error(`bitgpu: maxSeqLen ${L} exceeds the model's max_positions (${e})`);[On,ar]=Ta(d,L,ie)}Br.sort((e,n)=>e.off-n.off||e.len-n.len);const wn=[];for(const e of Br){const n=wn[wn.length-1];if(n&&n.off===e.off&&n.len===e.len){const t=n.push,r=n.finish;n.push=a=>{t(a),e.push(a)},n.finish=()=>{r(),e.finish()}}else{if(n&&e.off<n.off+n.len)throw new Error("bitgpu: partially overlapping data-file tensor ranges (unsupported by the streaming loader)");wn.push(e)}}const Rr=wn.length?wn[wn.length-1].off+wn[wn.length-1].len:0,fa=(c.fetchStream?await c.fetchStream(we):c.fetchArrayBuffer?new Response(await c.fetchArrayBuffer(we)).body:await(async()=>{const e=await fetch(we);if(!e.ok)throw new Error(`bitgpu: fetch ${we} failed: HTTP ${e.status}`);return e.body??new Response(await e.arrayBuffer()).body})()).getReader();let sn=0,Hr=0;for(;;){const{done:e,value:n}=await fa.read();if(e)break;let t=0;for(;t<n.byteLength&&Hr<wn.length;){const r=wn[Hr];if(sn>=r.off+r.len){Hr++;continue}if(sn<r.off){const o=Math.min(r.off-sn,n.byteLength-t);sn+=o,t+=o;continue}const a=Math.min(r.off+r.len-sn,n.byteLength-t);r.push(n.subarray(t,t+a)),sn+=a,t+=a,sn===r.off+r.len&&(r.finish(),Hr++)}sn+=n.byteLength-t,(Et=c.onProgress)==null||Et.call(c,{phase:"weights",loaded:Math.min(sn,Rr),total:Rr})}if(sn<Rr)throw new Error(`bitgpu: the data file ended at ${sn} bytes but tensors extend to ${Rr}; the download is truncated or the manifest does not match it`);for(const e of Qr)e();Qr.length=0,fe=null;function or(e,n){const t=qr(new Uint32Array(n)),r=v(n.length*w),a=e.beginComputePass();return Re(a,"embed_gather_batch",[["u",n.length],["u",w],["u",0],["u",0]],[t,An,Vr,Wn,Ln],r,n.length*w),a.end(),r}function ga(e,n){const t=ie,r=t/2,a=new Float32Array(n*t),o=new Float32Array(n*t);for(let u=0;u<n;u++)for(let p=0;p<t;p++)a[u*t+p]=On[(e+u)*r+p%r],o[u*t+p]=ar[(e+u)*r+p%r];const l=v(n*t),i=v(n*t);return s.queue.writeBuffer(l,0,a),s.queue.writeBuffer(i,0,o),{cos:l,sin:i}}const y=d.kv_heads,f=d.head_dim,w=d.hidden,j=d.heads,je=d.intermediate,_t=((At=d.hybrid)==null?void 0:At.linear_key_heads)??0,Xe=((Wt=d.hybrid)==null?void 0:Wt.linear_value_heads)??0,jn=((Lt=d.hybrid)==null?void 0:Lt.linear_head_dim)??0,Xr=((jt=d.hybrid)==null?void 0:jt.conv_kernel)??0,vn=_t*jn,Sn=Xe*jn,kn=vn*2+Sn,ma=1.25*(1<<30),ha=(()=>{if(!d.hybrid)return 0;const e=3*w+3*je,n=2*kn+2*vn+4*Sn+4*Xe+w,t=8*j*f+4*y*f+w;return 3*(e+Math.max(n,t))*4})(),xt=d.hybrid?Math.max(Rn,Math.min(dt,Math.floor(ma/ha/Rn)*Rn)):dt;let un=Math.min(L,512);const Ve=[],Te=[],Ye=[],rn=[],ir=e=>e*y*(f/32)*4,ln=[];for(let e=0;e<d.layers;e++)(!d.hybrid||d.hybrid.layer_types[e]==="full")&&ln.push(e);const sr=[];for(let e=0;e<d.layers;e++)d.hybrid&&d.hybrid.layer_types[e]==="linear"&&sr.push(e);const Fn=d.hybrid?Xe*jn*jn*4:0,Qn=d.hybrid?(Xr-1)*kn*4:0;for(const e of ln)Ve[e]=s.createBuffer({size:un*y*f*Dn,usage:k|z|M}),Te[e]=s.createBuffer({size:un*y*f*Dn,usage:k|z|M}),ve&&(Ye[e]=s.createBuffer({size:ir(un),usage:k|z|M}),rn[e]=s.createBuffer({size:ir(un),usage:k|z|M}));const ur=[],lr=[];let cr=0;if(d.hybrid)for(const e of sr)ur[e]=[s.createBuffer({size:Fn,usage:k|z|M}),s.createBuffer({size:Fn,usage:k|z|M})],lr[e]=[s.createBuffer({size:Qn,usage:k|z|M}),s.createBuffer({size:Qn,usage:k|z|M})];let Yr=null,Zr=null,Jr=null,et=null;if(ke){const e=f/2;Yr=s.createBuffer({size:L*e*4,usage:k|M}),Zr=s.createBuffer({size:L*e*4,usage:k|M}),s.queue.writeBuffer(Yr,0,On.buffer,On.byteOffset,L*e*4),s.queue.writeBuffer(Zr,0,ar.buffer,ar.byteOffset,L*e*4),Jr=s.createBuffer({size:f*4,usage:k|M}),et=s.createBuffer({size:f*4,usage:k|M}),s.queue.writeBuffer(Jr,0,new Float32Array(f).fill(1)),s.queue.writeBuffer(et,0,new Float32Array(f))}const Kt=await s.popErrorScope(),Nt=await s.popErrorScope();if(Kt)throw new Tr(`GPU allocation failed while loading weights (~${Math.round(ft/1048576)} MB VRAM needed): ${Kt.message}`);if(Nt)throw new Error(`bitgpu: WebGPU validation error while loading weights: ${Nt.message}`);async function yn(e){if(e=Math.min(e,L),e<=un)return;const n=Math.min(L,Math.max(e,un*2)),t=un*y*f*Dn,r=ir(un);s.pushErrorScope("out-of-memory");const a=s.createCommandEncoder(),o=[];for(const i of ln){const u=s.createBuffer({size:n*y*f*Dn,usage:k|z|M}),p=s.createBuffer({size:n*y*f*Dn,usage:k|z|M});a.copyBufferToBuffer(Ve[i],0,u,0,t),a.copyBufferToBuffer(Te[i],0,p,0,t);const g=[Ve[i],Te[i]];if(Ve[i]=u,Te[i]=p,ve){const h=s.createBuffer({size:ir(n),usage:k|z|M}),N=s.createBuffer({size:ir(n),usage:k|z|M});a.copyBufferToBuffer(Ye[i],0,h,0,r),a.copyBufferToBuffer(rn[i],0,N,0,r),g.push(Ye[i],rn[i]),Ye[i]=h,rn[i]=N}o[i]=g}s.queue.submit([a.finish()]),await s.queue.onSubmittedWorkDone();const l=await s.popErrorScope();if(l){for(const i of ln)Ve[i].destroy(),Te[i].destroy(),Ve[i]=o[i][0],Te[i]=o[i][1],ve&&(Ye[i].destroy(),rn[i].destroy(),Ye[i]=o[i][2],rn[i]=o[i][3]);throw rr(),new Tr(`KV cache growth to ${n} positions failed: ${l.message}`)}for(const i of ln)for(const u of o[i])u.destroy();un=n,rr()}let Gn=null;function qt(e,n){const t=e-Qe,r=Math.min(t,Math.max(n,Math.ceil((L-Qe)/4))),a=t-r;if(a<=0)return Qe;const o=y*f*Dn,l=ve?y*(f/32)*4:0;(!Gn||Gn.size<a*o)&&(Gn==null||Gn.destroy(),Gn=s.createBuffer({size:a*o,usage:z|M}));const i=s.createCommandEncoder(),u=[];for(const p of ln)u.push([Ve[p],o],[Te[p],o]),ve&&u.push([Ye[p],l],[rn[p],l]);for(const[p,g]of u)i.copyBufferToBuffer(p,(Qe+r)*g,Gn,0,a*g),i.copyBufferToBuffer(Gn,0,p,Qe*g,a*g);return s.queue.submit([i.finish()]),Qe+a}const nt=(e,n)=>ke&&e+n>L?qt(e,e+n-L):e;async function Ze(e,n){const t=s.createBuffer({size:n*4,usage:GPUBufferUsage.MAP_READ|M}),r=s.createCommandEncoder();r.copyBufferToBuffer(e,0,t,0,n*4),s.queue.submit([r.finish()]),await t.mapAsync(GPUMapMode.READ);const a=new Float32Array(t.getMappedRange().slice(0));return t.unmap(),t.destroy(),a}async function dr(e,n){const t=s.createBuffer({size:n*4,usage:GPUBufferUsage.MAP_READ|M}),r=s.createCommandEncoder();r.copyBufferToBuffer(e,0,t,0,n*4),s.queue.submit([r.finish()]),await t.mapAsync(GPUMapMode.READ);const a=new Uint32Array(t.getMappedRange().slice(0));return t.unmap(),t.destroy(),a}let Er=null,rt=!1,ba=null,wa=null,va=null;const tt=()=>ba??(ba=s.createQuerySet({type:"timestamp",count:2})),Dt=()=>wa??(wa=s.createBuffer({size:16,usage:GPUBufferUsage.QUERY_RESOLVE|z})),Ar=()=>va??(va=s.createBuffer({size:16,usage:GPUBufferUsage.MAP_READ|M})),pr=e=>Er===null||Er.has(e)||e==="embed_gather_batch";let Xn=!1,Yn=0,Tn=null;const Zn=(e,n,t)=>{e===0&&Tn&&(Tn[n]=t)};function Ue(e,n,t,r,a){if(e.setPipeline(xr[n]),on){let o=on.disp[Mr];o||(o={uni:s.createBuffer({size:64,usage:ht|M}),bg:null,last:null,bufs:null},on.disp[Mr]=o);const l=Ft(t);if((!o.last||!ja(o.last,l))&&(s.queue.writeBuffer(o.uni,0,l),o.last=l.slice()),o.bg&&o.bufs){const i=r.length+a.length;if(o.bufs.length!==i)o.bg=null;else{for(let u=0;u<r.length;u++)if(o.bufs[u]!==r[u]){o.bg=null;break}if(o.bg){for(let u=0;u<a.length;u++)if(o.bufs[r.length+u]!==a[u]){o.bg=null;break}}}}if(!o.bg){const i=[{binding:0,resource:{buffer:o.uni}}];r.forEach((u,p)=>i.push({binding:p+1,resource:{buffer:u}})),a.forEach((u,p)=>i.push({binding:1+r.length+p,resource:{buffer:u}})),o.bg=s.createBindGroup({layout:xr[n].getBindGroupLayout(0),entries:i}),o.bufs=[...r,...a]}e.setBindGroup(0,o.bg),Mr++}else{const o=[{binding:0,resource:{buffer:qr(Ft(t),ht|M)}}];r.forEach((l,i)=>o.push({binding:i+1,resource:{buffer:l}})),a.forEach((l,i)=>o.push({binding:1+r.length+i,resource:{buffer:l}})),e.setBindGroup(0,s.createBindGroup({layout:xr[n].getBindGroupLayout(0),entries:o}))}}const Mt=e=>{const n=Math.ceil(e/65535);return[Math.ceil(e/n),n]};function at(e,n,t,r,a,o){if(Ue(e,n,t,r,a),!pr(n))return void e.dispatchWorkgroups(1);const[l,i]=Mt(Math.ceil(o/64));e.dispatchWorkgroups(l,i,1)}const Re=(e,n,t,r,a,o)=>at(e,n,t,r,[a],o);function We(e,n,t,r,a,o){Ue(e,n,t,r,[a]),e.dispatchWorkgroups(pr(n)?o:1)}function Je(e,n,t,r,a,o,l){Ue(e,n,t,r,a);const i=pr(n);e.dispatchWorkgroups(i?o:1,i?l:1,1)}const _n=(e,n,t,r,a,o,l=!1)=>l?We(e,"rmsnorm_sg_af16",[["u",r],["u",a],["f",d.rms_eps],["u",0]],[n,re[t].buf],o,r):Ne?We(e,"rmsnorm_sg",[["u",r],["u",a],["f",d.rms_eps],["u",0]],[n,re[t].buf],o,r):We(e,"rmsnorm_wg",[["u",r],["u",a],["f",d.rms_eps],["u",0]],[n,re[t].buf],o,r);function cn(e,n,t,r,a,o=!1){const l=n.N0+n.N1+n.N2;if(Ne&&r===1){const i=Math.min(l,65535);Je(e,o?"matmul_split_sg_af16":"matmul_split_sg",[["u",n.K],["u",n.nb],["u",n.N0],["u",n.N1],["u",n.N2],["u",i]],[t,n.sign,n.scales],a,i,Math.ceil(l/i))}else if(r===1){const i=Math.min(l,65535);Je(e,"matmul_split_wg",[["u",n.K],["u",n.nb],["u",n.N0],["u",n.N1],["u",n.N2],["u",i]],[t,n.sign,n.scales],a,i,Math.ceil(l/i))}else if(Ne&&r===Yn){const i=Math.min(l,65535);Je(e,"matmul_split_sm",[["u",n.K],["u",n.nb],["u",n.N0],["u",n.N1],["u",n.N2],["u",i],["u",r]],[t,n.sign,n.scales],a,i,Math.ceil(l/i))}else pt(r)?Je(e,"matmul_split_tiled",[["u",r],["u",n.K],["u",n.nb],["u",n.N0],["u",n.N1],["u",n.N2]],[t,n.sign,n.scales],a,Math.ceil(l/64),Math.ceil(r/64)):at(e,"matmul_split",[["u",r],["u",n.K],["u",n.nb],["u",n.N0],["u",n.N1],["u",n.N2]],[t,n.sign,n.scales],a,r*l)}function Un(e,n,t,r,a,o,l=!1){if(Ne&&a===1){const i=Math.ceil(n.N/Kr),u=Math.min(i,65535);Je(e,l?"matmul_resid_mr_sg_af16":"matmul_resid_mr_sg",[["u",n.N],["u",n.K],["u",n.nb],["u",u],["u",0],["u",0]],[t,n.sign,n.scales,r],[o],u,Math.ceil(i/u))}else if(a===1){const i=Math.min(n.N,65535);Je(e,"matmul_resid_wg",[["u",n.N],["u",n.K],["u",n.nb],["u",i],["u",0],["u",0]],[t,n.sign,n.scales,r],[o],i,Math.ceil(n.N/i))}else if(Ne&&a===Yn){const i=Math.min(n.N,65535);Je(e,"matmul_resid_sm",[["u",n.N],["u",n.K],["u",n.nb],["u",i],["u",a],["u",0]],[t,n.sign,n.scales,r],[o],i,Math.ceil(n.N/i))}else pt(a)?Je(e,"matmul_resid_tiled",[["u",a],["u",n.N],["u",n.K],["u",n.nb],["u",0],["u",0]],[t,n.sign,n.scales,r],[o],Math.ceil(n.N/64),Math.ceil(a/64)):at(e,"matmul_resid",[["u",a],["u",n.N],["u",n.K],["u",n.nb],["u",128],["u",0]],[t,n.sign,n.scales,r],[o],a*n.N)}function fr(e,n,t,r,a,o){const l=t===0?Ve[r]:Te[r];ve?(Ue(e,"copy_kv8",[["u",a],["u",f],["u",o],["u",0]],[n],[l,t===0?Ye[r]:rn[r]]),e.dispatchWorkgroups(pr("copy_kv8")?a:1)):Re(e,ua,[["u",a*f],["u",o*f],["u",0],["u",0]],[n],l,a*f)}const Bt=(e,n)=>{const t=ve?[e,Ve[n],Te[n],Ye[n],rn[n]]:[e,Ve[n],Te[n]];return ke&&t.push(Yr,Zr),t};function ka(e,n,t,r,a,o,l){const i=E.arch.hybrid,u=q=>re[`layers.${n}.${q}`],p=a>0?1:0,g=cr,h=nr&&r===1&&!Xn,N=h?tr(r*w):v(r*w);_n(e,t,`layers.${n}.input_layernorm`,r,w,N,h);let G;if(i.layer_types[n]==="linear"){const q=v(r*kn);cn(e,u("linear.in_qkv"),N,r,[q,bn,xn],h);const x=v(r*kn),se=(Xr-1)*kn;Ue(e,"conv1d_causal",[["u",r],["u",kn],["u",Xr],["u",p]],[q,u("linear.conv1d").buf,lr[n][g]],[x,lr[n][g^1]]);{const[F,_e]=Mt(Math.ceil((r*kn+se)/64));e.dispatchWorkgroups(F,_e,1)}const he=v(r*vn),X=v(r*vn),J=v(r*Sn);Re(e,"slice_cols",[["u",r],["u",vn],["u",kn],["u",0]],[x],he,r*vn),Re(e,"slice_cols",[["u",r],["u",vn],["u",kn],["u",vn]],[x],X,r*vn),Re(e,"slice_cols",[["u",r],["u",Sn],["u",kn],["u",2*vn]],[x],J,r*Sn);const ee=v(r*Xe),C=v(r*Xe);cn(e,u("linear.a"),N,r,[ee,bn,xn],h),cn(e,u("linear.b"),N,r,[C,bn,xn],h);const ue=v(2*r*Xe);Re(e,"deltanet_gbeta",[["u",r],["u",Xe],["u",0],["u",0]],[ee,C,u("linear.A_log").buf,u("linear.dt_bias").buf],ue,r*Xe);const te=v(r*Sn);{const F=[];Math.ceil(r/Rn)%2===0&&F.push(Rn/2,Rn/2);for(let U=r-F.reduce((Be,Me)=>Be+Me,0);U>0;U-=Rn)F.push(Math.min(U,Rn));let _e=0;for(let U=0;U<F.length;U++){const Be=ur[n][g^U&1],Me=ur[n][g^(U&1^1)];Ue(e,"deltanet_recur",[["u",F[U]],["u",Xe],["u",jn],["u",jn],["u",_t],["u",r*Xe],["u",U===0?p:1],["u",_e]],[he,X,J,ue,ue,Be],[te,Me]),e.dispatchWorkgroups(Xe),_e+=F[U]}}const Y=v(r*Sn);cn(e,u("linear.z"),N,r,[Y,bn,xn],h);const ne=v(r*Sn);We(e,"deltanet_norm_gate",[["u",r*Xe],["u",jn],["f",d.rms_eps],["u",0]],[te,Y,u("linear.norm").buf],ne,r*Xe),G=v(r*w),Un(e,u("linear.out_proj"),ne,t,r,G)}else{const q=v(r*j*f*2);cn(e,u("attn.q_proj"),N,r,[q,bn,xn],h);const x=v(r*j*f),se=v(r*j*f);Re(e,"split_head",[["u",r],["u",j],["u",f],["u",0]],[q],x,r*j*f),Re(e,"split_head",[["u",r],["u",j],["u",f],["u",f]],[q],se,r*j*f);const he=v(r*j*f),X=v(r*j*f);_n(e,x,`layers.${n}.attn.q_norm`,r*j,f,he),Re(e,"rope_partial",[["u",r],["u",j],["u",f],["u",ie]],[he,o,l],X,r*j*f);const J=v(r*y*f);cn(e,u("attn.k_proj"),N,r,[J,bn,xn],h);const ee=v(r*y*f),C=v(r*y*f);_n(e,J,`layers.${n}.attn.k_norm`,r*y,f,ee),Re(e,"rope_partial",[["u",r],["u",y],["u",f],["u",ie]],[ee,o,l],C,r*y*f);const ue=v(r*y*f);cn(e,u("attn.v_proj"),N,r,[ue,bn,xn],h),fr(e,C,0,n,r*y,a*y),fr(e,ue,1,n,r*y,a*y);const te=v(r*j*f),Y=[["u",r],["u",j],["u",y],["u",f],["f",1/Math.sqrt(f)],["u",a],["u",0],["u",0]];ve?We(e,"attention_online_cache_kv8",Y,[X,Ve[n],Ye[n],Te[n],rn[n]],te,r*j):We(e,"attention_online_cache",Y,[X,Ve[n],Te[n]],te,r*j);const ne=v(r*j*f);Re(e,"gate_sigmoid",[["u",r*j*f],["u",0],["u",0],["u",0]],[te,se],ne,r*j*f),G=v(r*w),Un(e,u("attn.o_proj"),ne,t,r,G)}const V=h?tr(r*w):v(r*w);_n(e,G,`layers.${n}.post_attention_layernorm`,r,w,V,h);const O=v(r*je),de=v(r*je);cn(e,u("mlp.gate_proj"),V,r,[O,bn,xn],h),cn(e,u("mlp.up_proj"),V,r,[de,bn,xn],h);const P=v(r*je);Re(e,"swiglu",[["u",r*je],["u",0],["u",0],["u",0]],[O,de],P,r*je);const T=v(r*w);return Un(e,u("mlp.down_proj"),P,G,r,T),T}function ya(e,n,t,r,a,o,l){const i=a+r;if(E.arch.hybrid)return ka(e,n,t,r,a,o,l);const u=nr&&r===1&&!Xn,p=u?tr(w):v(r*w);_n(e,t,`layers.${n}.input_layernorm`,r,w,p,u);const g=re[`layers.${n}.attn.qkv`];if(Ne&&r===1&&!Xn){const te=v(j*f),Y=v(y*f),ne=v(y*f),F=g.N0+g.N1+g.N2,_e=Math.min(F,65535);Je(e,u?"matmul_split_sg_af16":"matmul_split_sg",[["u",g.K],["u",g.nb],["u",g.N0],["u",g.N1],["u",g.N2],["u",_e]],[p,g.sign,g.scales],[te,Y,ne],_e,Math.ceil(F/_e)),fr(e,ne,1,n,y,a*y);const U=v(j*f);We(e,"rmsnorm_rope_sg",[["u",j],["u",f],["f",d.rms_eps],["u",0],["u",f],["u",0]],[te,re[`layers.${n}.attn.q_norm`].buf,o,l],U,j);const Be=ke?Jr:o,Me=ke?et:l;ve?(Ue(e,"rmsnorm_rope_sg_kv8",[["u",y],["u",f],["f",d.rms_eps],["u",a*y],["u",0],["u",0]],[Y,re[`layers.${n}.attn.k_norm`].buf,Be,Me],[Ve[n],Ye[n]]),e.dispatchWorkgroups(pr("rmsnorm_rope_sg_kv8")?y:1)):We(e,sa,[["u",y],["u",f],["f",d.rms_eps],["u",a*y*f],["u",f],["u",0]],[Y,re[`layers.${n}.attn.k_norm`].buf,Be,Me],Ve[n],y),Zn(n,"qr",U);const fn=v(j*f);We(e,mt,[["u",1],["u",j],["u",y],["u",f],["u",a],["u",i]],Bt(U,n),fn,j),Zn(n,"att",fn);const ae=re[`layers.${n}.attn.o_proj`],R=v(w);Un(e,ae,fn,t,1,R);const pe=u?tr(w):v(w);_n(e,R,`layers.${n}.post_attention_layernorm`,1,w,pe,u);const He=re[`layers.${n}.mlp.gateup`],xe=u?tr(je):v(je),Ee=Math.ceil(je/Kr),Ge=Math.min(Ee,65535);Je(e,u?"matmul_swiglu_mr_sg_af16":"matmul_swiglu_mr_sg",[["u",He.K],["u",He.nb],["u",je],["u",Ge],["u",0],["u",0]],[pe,He.sign,He.scales],[xe],Ge,Math.ceil(Ee/Ge)),Zn(n,"sw",xe);const mr=re[`layers.${n}.mlp.down_proj`],gn=v(w);return Un(e,mr,xe,R,1,gn,u),gn}const h=v(r*j*f),N=v(r*y*f),G=v(r*y*f);cn(e,g,p,r,[h,N,G]);const V=v(r*j*f),O=v(r*y*f);_n(e,h,`layers.${n}.attn.q_norm`,r*j,f,V),_n(e,N,`layers.${n}.attn.k_norm`,r*y,f,O);const de=v(r*j*f),P=v(r*y*f);Re(e,"rope",[["u",r],["u",j],["u",f],["u",0]],[V,o,l],de,r*j*f),ke||Re(e,"rope",[["u",r],["u",y],["u",f],["u",0]],[O,o,l],P,r*y*f),fr(e,ke?O:P,0,n,r*y,a*y),fr(e,G,1,n,r*y,a*y),Zn(n,"qr",de);const T=v(r*j*f);We(e,mt,[["u",r],["u",j],["u",y],["u",f],["u",a],["u",i]],Bt(de,n),T,r*j),Zn(n,"att",T);const q=re[`layers.${n}.attn.o_proj`],x=v(r*w);Un(e,q,T,t,r,x);const se=v(r*w);_n(e,x,`layers.${n}.post_attention_layernorm`,r,w,se);const he=re[`layers.${n}.mlp.gateup`],X=v(r*je),J=v(r*je);cn(e,he,se,r,[X,J,bn]);const ee=v(r*je);Re(e,"swiglu",[["u",r*je],["u",0],["u",0],["u",0]],[X,J],ee,r*je),Zn(n,"sw",ee);const C=re[`layers.${n}.mlp.down_proj`],ue=v(r*w);return Un(e,C,ee,x,r,ue),ue}function qn(e,n,t,r){const a=re.lm_head;if(Ne&&t===1){const o=Math.min(a.N,65535);Je(e,"matmul_q2_sg",[["u",a.N],["u",a.K],["u",a.nb],["u",a.zp],["u",o],["u",0]],[n,a.codes,a.scales],[r],o,Math.ceil(a.N/o))}else if(t===1){const o=Math.min(a.N,65535);Je(e,"matmul_q2_wg",[["u",a.N],["u",a.K],["u",a.nb],["u",a.zp],["u",o],["u",0]],[n,a.codes,a.scales],[r],o,Math.ceil(a.N/o))}else if(Ne&&t===Yn){const o=Math.min(a.N,65535);Je(e,"matmul_q2_sm",[["u",a.N],["u",a.K],["u",a.nb],["u",a.zp],["u",o],["u",t]],[n,a.codes,a.scales],[r],o,Math.ceil(a.N/o))}else Re(e,"matmul_q2",[["u",t],["u",a.N],["u",a.K],["u",a.nb],["u",128],["u",a.zp]],[n,a.codes,a.scales],r,t*a.N)}function Pn(e,n,t,r){const{cos:a,sin:o}=ga(r,t),l=e.beginComputePass(),i=!on&&!Tn;let u=null,p=n;i&&(hn=hn??new Map);for(let h=0;h<d.layers;h++){const N=[];i&&(Mn=N);const G=p;if(p=ya(l,h,G,t,r,a,o),h===0&&(u=p),i){Mn=null;for(const V of N)V!==p&&bt(V);G!==n&&G!==u&&bt(G)}}d.hybrid&&(cr^=1);const g=v(t*w);return _n(l,p,I,t,w,g),l.end(),hn=null,{fn:g,layer0:u}}async function _a(e){const n=e.length;if(n===0)throw new Error("forward: no tokens to process");if(n>L)throw new Error(`forward: sequence length ${n} exceeds maxSeqLen ${L}`);qe=[],De=0,await yn(n);const t=re.lm_head.N,r=new Float32Array(n*w),a=new Float32Array(n*w),o=new Float32Array(n*w),l=new Float32Array(n*t);A=[];const i=Math.min((globalThis.__SEG??0)||xt,32);try{for(let u=0;u<n;u+=i){const p=e.slice(u,u+i),g=s.createCommandEncoder(),h=or(g,p),{fn:N,layer0:G}=Pn(g,h,p.length,u),V=s.createBuffer({size:p.length*t*4,usage:k|z});A.push(V);const O=g.beginComputePass();qn(O,N,p.length,V),O.end(),s.queue.submit([g.finish()]),await s.queue.onSubmittedWorkDone(),r.set(await Ze(h,p.length*w),u*w),a.set(await Ze(G,p.length*w),u*w),o.set(await Ze(N,p.length*w),u*w),l.set(await Ze(V,p.length*t),u*t),Ie()}return{embed:r,layer0:a,finalnorm:o,logits:l,vocab:t,sequenceLength:n}}finally{Ie(),A=null}}async function gr(e,n,t){let r=null,a=0;const o=(globalThis.__SEG??0)||xt;for(let l=0;l<e.length;l+=o){if(l>0&&(t!=null&&t.aborted))return qe=[],De=0,null;const i=e.slice(l,l+o);s.pushErrorScope("out-of-memory");const u=s.createCommandEncoder();if(r=Pn(u,or(u,i),i.length,n+l).fn,a=i.length-1,s.queue.submit([u.finish()]),await s.popErrorScope())throw new Error(`bitgpu: GPU out of memory during prefill (segment of ${i.length} tokens at position ${n+l}) - the output would have been silently corrupted. Lower maxSeqLen, use kvCache: 'q8', or free GPU memory.`);l+o<e.length&&(await s.queue.onSubmittedWorkDone(),Ie())}return{fn:r,lastRow:a}}async function ot(e,n,t,r=null,a=vr,o){var g,h,N;await yn(n+e.length+t),Er=r;const l=re.lm_head.N,i=s.createBuffer({size:Math.max(1,t)*4,usage:k|z}),u=s.createBuffer({size:w*4,usage:k|z|M}),p=s.createBuffer({size:l*4,usage:k|z});A=[];try{const G=performance.now(),V=await gr(e,n,o==null?void 0:o.signal);if(!V)return{prefillMs:performance.now()-G,decodeMs:0,tokPerSec:0,tokens:[],firstArgmax:-1,recMs:0,gpuMs:0,rbMs:0};const O=s.createCommandEncoder(),de=v(w);O.copyBufferToBuffer(V.fn,V.lastRow*w*4,de,0,w*4);let P=O.beginComputePass();qn(P,de,1,p),P.end(),P=O.beginComputePass(),We(P,"argmax",[["u",l],["u",0],["u",0],["u",0]],[p],i,1),P.end(),s.queue.submit([O.finish()]),await s.queue.onSubmittedWorkDone();const T=(await dr(i,1))[0];Ie();const q=performance.now()-G,x=[];let se=0,he=0,X=0,J=0;const ee=rt&&yr,C=performance.now();let ue=1;const te=o!=null&&o.stopTokens?new Set(o.stopTokens):null;let Y=(te==null?void 0:te.has(T))??!1;Y||(x.push(T),(g=o==null?void 0:o.onToken)==null||g.call(o,T)),rr();let ne=n+e.length;for(;ue<t&&!Y&&!((h=o==null?void 0:o.signal)!=null&&h.aborted);){const U=Math.min(a,t-ue);ne=nt(ne,U),Bn("decode");let Be=performance.now();const Me=s.createCommandEncoder();for(let R=0;R<U;R++){const pe=ue+R,He=ne+R;let xe=Me.beginComputePass(ee&&R===0?{timestampWrites:{querySet:tt(),beginningOfPassWriteIndex:0}}:void 0);We(xe,"embed_gather",[["u",w],["u",pe-1],["u",0],["u",0]],[i,An,Vr,Wn,Ln],u,1),xe.end();const Ee=Pn(Me,u,1,He),Ge=v(w);Me.copyBufferToBuffer(Ee.fn,0,Ge,0,w*4),xe=Me.beginComputePass(ee&&R===U-1?{timestampWrites:{querySet:tt(),endOfPassWriteIndex:1}}:void 0),qn(xe,Ge,1,p),We(xe,"argmax",[["u",l],["u",pe],["u",0],["u",0]],[p],i,1),xe.end()}if(ee&&(Me.resolveQuerySet(tt(),0,2,Dt(),0),Me.copyBufferToBuffer(Dt(),0,Ar(),0,16)),s.queue.submit([Me.finish()]),se+=performance.now()-Be,Be=performance.now(),await s.queue.onSubmittedWorkDone(),he+=performance.now()-Be,ee){await Ar().mapAsync(GPUMapMode.READ);const R=new BigUint64Array(Ar().getMappedRange());J+=Number(R[1]-R[0]),Ar().unmap()}Be=performance.now();const fn=await dr(i,ue+U);X+=performance.now()-Be;let ae=U;for(let R=0;R<U;R++){const pe=fn[ue+R];if(te!=null&&te.has(pe)){Y=!0,ae=R;break}x.push(pe),(N=o==null?void 0:o.onToken)==null||N.call(o,pe)}ue+=U,ne+=ae}De=ne;const F=performance.now()-C,_e=Math.max(1,x.length-1);return{prefillMs:q,decodeMs:F,tokPerSec:_e/(F/1e3),tokens:x,firstArgmax:T,recMs:se/_e,gpuMs:he/_e,rbMs:X/_e,tsMs:ee?J/1e6/_e:0}}finally{Bn(null),Er=null,Ie(),A=null,i.destroy(),u.destroy(),p.destroy()}}async function St(e,n,t,r,a,o){await yn(n+e.length+t);const l=r.temperature!=null&&r.temperature>0&&r.temperature!==1,i=re.lm_head.N,u=Math.max(1,Math.min(r.topK??20,i)),p=Math.max(0,Math.min(Math.floor(r.logprobs??0),32,i)),g=Math.max(u,p),h=r.temperature??1,N=r.repetitionPenalty??1,G=r.presencePenalty??0,V=r.topP??1,O=r.minP??0,de=r.noRepeatNgramSize??0,P=(r.dryMultiplier??0)>0?{multiplier:r.dryMultiplier,base:r.dryBase??1.75,allowedLength:r.dryAllowedLength??2,range:r.dryRange??0,breakers:new Set(r.dryBreakers??[])}:null,T=r.topNSigma??0,q=r.stopTokens?new Set(r.stopTokens):null,x=r.onToken,se=r.signal,he=o??new It(r.seed),X=s.createBuffer({size:Math.max(1,t)*4,usage:k|z|M}),J=s.createBuffer({size:i*4,usage:k|z}),ee=s.createBuffer({size:g*4,usage:k|z}),C=s.createBuffer({size:g*4,usage:k|z}),ue=s.createBuffer({size:4,usage:k|z}),te=s.createBuffer({size:12,usage:k|z}),Y=g*8+(p?4:0);let ne=s.createBuffer({size:L*4,usage:k|M}),F=s.createBuffer({size:L*4,usage:k|M});const _e=(B,H)=>H<=B.size?B:(B.destroy(),s.createBuffer({size:1<<32-Math.clz32(H-1),usage:k|M})),U=s.createBuffer({size:Y+(T>0?12:0),usage:GPUBufferUsage.MAP_READ|M}),Be=s.createBuffer({size:w*4,usage:k|z|M}),Me=B=>{const H=N!==1||G!==0?ut(B):new Uint32Array(0);H.length&&s.queue.writeBuffer(ne=_e(ne,H.byteLength),0,H);const W=de>0?lt(B,de):[];return W.length&&s.queue.writeBuffer(F=_e(F,W.length*4),0,Uint32Array.from(W)),{affLen:H.length,banLen:W.length}},fn=(B,H,W)=>{Ue(B,"sampler_penalty",[["u",H],["u",W],["f",N],["u",4286578688],["f",G]],[ne,F],[J]),B.dispatchWorkgroups(1),p&&(Ue(B,"logsumexp",[["u",i],["u",0],["u",0],["u",0]],[J],[ue]),B.dispatchWorkgroups(1)),T>0&&(Ue(B,"sampler_sigma",[["u",i],["u",0],["u",0],["u",0]],[J],[te]),B.dispatchWorkgroups(1));for(let Q=0;Q<g;Q++)Ue(B,"argmax_masked",[["u",i],["u",Q],["u",0],["u",0]],[J],[ee,C]),B.dispatchWorkgroups(1)},ae=B=>{B.copyBufferToBuffer(ee,0,U,0,g*4),B.copyBufferToBuffer(C,0,U,g*4,g*4),p&&B.copyBufferToBuffer(ue,0,U,g*8,4),T>0&&B.copyBufferToBuffer(te,0,U,Y,12)};let R=0;const pe=async()=>{await U.mapAsync(GPUMapMode.READ);const B=U.getMappedRange(),H=new Uint32Array(B.slice(0,g*4)),W=new Float32Array(B.slice(g*4,g*8)),Q=p?new Float32Array(B.slice(g*8,g*8+4))[0]:0;if(T>0){const[Z,le,ge]=new Float32Array(B.slice(Y,Y+12));R=ge>0?Math.sqrt(Math.max(0,le/ge-(Z/ge)**2)):0}return U.unmap(),{ci:H,cv:W,lse:Q}},He=(B,H)=>{if(!(T>0)||H.length===0)return null;const W=H[0]-T*R;let Q=1;for(;Q<H.length&&H[Q]>=W;)Q++;return{ids:Array.prototype.slice.call(B,0,Q),vals:Array.prototype.slice.call(H,0,Q)}};let xe=0;const Ee=p?[]:null,Ge=(B,H,W)=>{if(!Ee)return;const Q=[];for(let Z=0;Z<p;Z++)Q.push({id:B[Z],logprob:H[Z]-W});Ee.push({logprob:xe-W,top:Q})},mr=(B,H)=>{const W=B.subarray(0,u),Q=H.subarray(0,u);let Z=W,le=Q;const ge=He(W,Q);if(ge&&(Z=ge.ids,le=ge.vals),P){const Le=ct(Z,le,a,P);Z=Le.ids,le=Le.vals}const me=l?Ur(Z,le,h,he,V,O):Z[0];return xe=Q[W.indexOf(me)],me},gn=r.candidateFilter,Cn=async(B,H)=>{if(!gn)return mr(B,H);const W=B.subarray(0,u),Q=H.subarray(0,u),Z=new Set(gn(W,Q));{const le=[],ge=[];for(let $=0;$<W.length;$++)Z.has(W[$])&&(le.push(W[$]),ge.push(Q[$]));if(le.length===0)return hr(B,H);let me=le,Le=ge;const Ce=He(le,ge);if(Ce&&(me=Ce.ids,Le=Ce.vals),P){const $=ct(me,Le,a,P);me=$.ids,Le=$.vals}const Oe=l?Ur(me,Le,h,he,V,O):me[0];return xe=ge[le.indexOf(Oe)],Oe}},hr=async(B,H)=>{const W=await Ze(J,i);for(let $=0;$<B.length;$++)W[B[$]]=H[$];const Q=Array.from(W.keys()).sort(($,Pe)=>W[Pe]-W[$]||$-Pe),Z=[],le=[],ge=512;for(let $=0;$<Q.length&&Z.length<u&&!(W[Q[$]]===-1/0&&Z.length>0);$+=ge){const Pe=Q.slice($,$+ge),Ke=new Set(gn(Uint32Array.from(Pe),Float32Array.from(Pe.map(en=>W[en]))));for(const en of Pe)if(Ke.has(en)&&(Z.push(en),le.push(W[en]),Z.length>=u))break;if(le[0]===-1/0){Z.length=1,le.length=1;break}}if(Z.length===0)throw new Error("bitgpu: candidateFilter permitted no token in the entire vocabulary");let me=Z,Le=le;const Ce=He(Z,le);if(Ce&&(me=Ce.ids,Le=Ce.vals),P){const $=ct(me,Le,a,P);me=$.ids,Le=$.vals}const Oe=l?Ur(me,Le,h,he,V,O):me[0];return xe=le[Z.indexOf(Oe)],Oe};A=[];try{const B=performance.now(),H=await gr(e,n,se);if(!H)return{prefillMs:performance.now()-B,decodeMs:0,tokPerSec:0,tokens:[],firstArgmax:-1,recMs:0,gpuMs:0,rbMs:0,rng:he};const W=s.createCommandEncoder(),Q=v(w);W.copyBufferToBuffer(H.fn,H.lastRow*w*4,Q,0,w*4);const Z=Me(a);let le=W.beginComputePass();qn(le,Q,1,J),fn(le,Z.affLen,Z.banLen),le.end(),ae(W),s.queue.submit([W.finish()]);const ge=await pe(),me=await Cn(ge.ci,ge.cv);Ie();const Le=performance.now()-B,Ce=[];let Oe=(q==null?void 0:q.has(me))??!1;Oe||(Ce.push(me),a.push(me),Ge(ge.ci,ge.cv,ge.lse),x==null||x(me),s.queue.writeBuffer(X,0,new Uint32Array([me])));let $=0,Pe=0,Ke=0;const en=performance.now();let mn=1;rr();let Vn=n+e.length;for(;mn<t&&!Oe&&!(se!=null&&se.aborted);){Bn("decode"),Vn=nt(Vn,1);const be=mn,ze=Vn;let $e=performance.now();const{affLen:$n,banLen:jr}=Me(a),Jn=s.createCommandEncoder();let er=Jn.beginComputePass();We(er,"embed_gather",[["u",w],["u",be-1],["u",0],["u",0]],[X,An,Vr,Wn,Ln],Be,1),er.end();const Ba=Pn(Jn,Be,1,ze),Tt=v(w);Jn.copyBufferToBuffer(Ba.fn,0,Tt,0,w*4),er=Jn.beginComputePass(),qn(er,Tt,1,J),fn(er,$n,jr),er.end(),ae(Jn),s.queue.submit([Jn.finish()]),$+=performance.now()-$e,$e=performance.now();const{ci:Ut,cv:Ct,lse:Sa}=await pe();Pe+=performance.now()-$e,$e=performance.now();const br=await Cn(Ut,Ct);if(Ke+=performance.now()-$e,mn+=1,q!=null&&q.has(br)){Oe=!0;break}Ce.push(br),a.push(br),Ge(Ut,Ct,Sa),x==null||x(br),s.queue.writeBuffer(X,be*4,new Uint32Array([br])),Vn+=1}De=Vn;const Lr=performance.now()-en,Fe=Math.max(1,Ce.length-1);return{prefillMs:Le,decodeMs:Lr,tokPerSec:Fe/(Lr/1e3),tokens:Ce,firstArgmax:me,recMs:$/Fe,gpuMs:Pe/Fe,rbMs:Ke/Fe,rng:he,...Ee?{lp:Ee}:{}}}finally{Bn(null),Ie(),A=null;for(const B of[X,J,ee,C,ue,te,ne,F,U,Be])B.destroy()}}async function it(e,n,t,r,a,o){await yn(n+e.length+t);const l=r.temperature!=null&&r.temperature>0&&r.temperature!==1,i=re.lm_head.N,u=Math.max(1,Math.min(r.topK??20,i)),p=r.temperature??1,g=r.repetitionPenalty??1,h=r.presencePenalty??0,N=r.topP??1,G=r.minP??0,V=r.noRepeatNgramSize??0,O=typeof r.promptLookup=="object"&&r.promptLookup!==null?r.promptLookup:{},de=Math.max(2,O.ngramSize??3),P=Math.max(1,Math.min(O.maxDraft??8,31)),T=r.stopTokens?new Set(r.stopTokens):null,q=r.onToken,x=r.signal,se=o??new It(r.seed),he=l||g!==1||h!==0||V>0,X=s.createBuffer({size:i*4,usage:k|z|M}),J=s.createBuffer({size:(P+1)*i*4,usage:k|z}),ee=s.createBuffer({size:(P+1)*4,usage:k|z}),C=s.createBuffer({size:u*4,usage:k|z}),ue=s.createBuffer({size:u*4,usage:k|z});let te=s.createBuffer({size:(L+P+1)*4,usage:k|M}),Y=s.createBuffer({size:(L+P+1)*4,usage:k|M});const ne=(ae,R)=>R<=ae.size?ae:(ae.destroy(),s.createBuffer({size:1<<32-Math.clz32(R-1),usage:k|M})),F=s.createBuffer({size:(P+1)*u*8,usage:GPUBufferUsage.MAP_READ|M}),_e=s.createBuffer({size:(P+1)*4,usage:k|M}),U=s.createBuffer({size:(P+1)*w*4,usage:k|M}),Be=ae=>{const R=g!==1||h!==0?ut(ae):new Uint32Array(0);R.length&&s.queue.writeBuffer(te=ne(te,R.byteLength),0,R);const pe=V>0?lt(ae,V):[];return pe.length&&s.queue.writeBuffer(Y=ne(Y,pe.length*4),0,Uint32Array.from(pe)),{affLen:R.length,banLen:pe.length}},Me=(ae,R,pe)=>{Ue(ae,"sampler_penalty",[["u",R],["u",pe],["f",g],["u",4286578688],["f",h]],[te,Y],[X]),ae.dispatchWorkgroups(1);for(let He=0;He<u;He++)Ue(ae,"argmax_masked",[["u",i],["u",He],["u",0],["u",0]],[X],[C,ue]),ae.dispatchWorkgroups(1)},fn=(ae,R)=>l?Ur(new Uint32Array(ae,R*u*8,u),new Float32Array(ae,R*u*8+u*4,u),p,se,N,G):new Uint32Array(ae,R*u*8,1)[0];A=[];try{const ae=performance.now(),R=await gr(e,n,x);if(!R)return{prefillMs:performance.now()-ae,decodeMs:0,tokPerSec:0,tokens:[],firstArgmax:-1,recMs:0,gpuMs:0,rbMs:0,spec:{steps:0,drafted:0,accepted:0},rng:se};const pe=s.createCommandEncoder(),He=v(w);pe.copyBufferToBuffer(R.fn,R.lastRow*w*4,He,0,w*4);const xe=he?Be(a):null;let Ee=pe.beginComputePass();qn(Ee,He,1,X),xe?Me(Ee,xe.affLen,xe.banLen):We(Ee,"argmax",[["u",i],["u",0],["u",0],["u",0]],[X],ee,1),Ee.end(),xe&&(pe.copyBufferToBuffer(C,0,F,0,u*4),pe.copyBufferToBuffer(ue,0,F,u*4,u*4)),s.queue.submit([pe.finish()]),await s.queue.onSubmittedWorkDone();let Ge;if(xe){await F.mapAsync(GPUMapMode.READ);const $=F.getMappedRange().slice(0);F.unmap(),Ge=fn($,0)}else Ge=(await dr(ee,1))[0];Ie();const mr=performance.now()-ae,gn=[];let Cn=(T==null?void 0:T.has(Ge))??!1;Cn||(gn.push(Ge),a.push(Ge),q==null||q(Ge)),rr();let hr=1,B=Ge,H=n+e.length,W=0,Q=0,Z=0,le=0,ge=0,me=0;const Le=performance.now();for(;hr<t&&!Cn&&!(x!=null&&x.aborted);){H=nt(H,P+1);const $=Math.min(P,t-hr-1,L-1-H),Pe=$>0?Va(a,de,$):[],Ke=Pe.length+1;await yn(H+Ke);let en=performance.now();Ke===1?Bn("pld1"):Ne&&Ke<=9?Bn("pldm",Ke,9):Bn(null),s.queue.writeBuffer(_e,0,new Uint32Array([B,...Pe])),Yn=Ne&&Ke>=2&&Ke<=9?Ke:0;const mn=s.createCommandEncoder(),Vn=mn.beginComputePass();Re(Vn,"embed_gather_batch",[["u",Ke],["u",w],["u",0],["u",0]],[_e,An,Vr,Wn,Ln],U,Ke*w),Vn.end();const Lr=Pn(mn,U,Ke,H);if(Ee=mn.beginComputePass(),qn(Ee,Lr.fn,Ke,J),Ee.end(),Yn=0,he){s.queue.submit([mn.finish()]);for(let be=0;be<Ke;be++){const{affLen:ze,banLen:$e}=Be(be===0?a:[...a,...Pe.slice(0,be)]),$n=s.createCommandEncoder();$n.copyBufferToBuffer(J,be*i*4,X,0,i*4);const jr=$n.beginComputePass();Me(jr,ze,$e),jr.end(),$n.copyBufferToBuffer(C,0,F,be*u*8,u*4),$n.copyBufferToBuffer(ue,0,F,be*u*8+u*4,u*4),s.queue.submit([$n.finish()])}}else{for(let be=0;be<Ke;be++){mn.copyBufferToBuffer(J,be*i*4,X,0,i*4);const ze=mn.beginComputePass();We(ze,"argmax",[["u",i],["u",be],["u",0],["u",0]],[X],ee,1),ze.end()}s.queue.submit([mn.finish()])}le+=performance.now()-en,en=performance.now(),await s.queue.onSubmittedWorkDone(),ge+=performance.now()-en,en=performance.now();const Fe=[];if(he){await F.mapAsync(GPUMapMode.READ);const be=F.getMappedRange().slice(0);F.unmap();for(let ze=0;ze<Ke;ze++){const $e=fn(be,ze);if(T!=null&&T.has($e)){Cn=!0;break}if(Fe.push($e),ze<Pe.length&&$e!==Pe[ze])break}}else{const be=await dr(ee,Ke);for(let ze=0;ze<Ke;ze++){const $e=be[ze];if(T!=null&&T.has($e)){Cn=!0;break}if(Fe.push($e),ze<Pe.length&&$e!==Pe[ze])break}}me+=performance.now()-en,W++,Q+=Pe.length,Z+=Math.max(0,Fe.length-1);for(const be of Fe)gn.push(be),a.push(be),q==null||q(be);if(hr+=Fe.length,Ie(),Fe.length===0)break;H+=Fe.length,B=Fe[Fe.length-1]}De=H;const Ce=performance.now()-Le,Oe=Math.max(1,gn.length-1);return{prefillMs:mr,decodeMs:Ce,tokPerSec:Oe/(Ce/1e3),tokens:gn,firstArgmax:Ge,recMs:le/Oe,gpuMs:ge/Oe,rbMs:me/Oe,spec:{steps:W,drafted:Q,accepted:Z},rng:se}}finally{Yn=0,Bn(null),Ie(),A=null;for(const ae of[X,J,ee,C,ue,te,Y,F,U,_e])ae.destroy()}}async function xa(e){qe=[],De=0,await yn(e.length+1),A=[];const n=s.createCommandEncoder();Pn(n,or(n,e),e.length,0),s.queue.submit([n.finish()]),await s.queue.onSubmittedWorkDone();const t=e.length,r=e[e.length-1],a=async o=>{Xn=o,Tn={};const l=s.createCommandEncoder(),i=Pn(l,or(l,[r]),1,t),u=s.createBuffer({size:re.lm_head.N*4,usage:k|z});A==null||A.push(u);const p=l.beginComputePass();qn(p,i.fn,1,u),p.end(),s.queue.submit([l.finish()]),await s.queue.onSubmittedWorkDone();const g={};for(const[N,G]of Object.entries(Tn))g[N]=await Ze(G,G.size/4);const h=t*y*f;if(!an&&!ve&&ln.length){const N=ln[0];g.kc=(await Ze(Ve[N],un*y*f)).slice(h,h+y*f),g.vc=(await Ze(Te[N],un*y*f)).slice(h,h+y*f)}return g.fn=await Ze(i.fn,w),g.logits=await Ze(u,re.lm_head.N),Xn=!1,Tn=null,g};try{return{fast:await a(!1),slow:await a(!0)}}finally{Xn=!1,Tn=null,Ie(),A=null}}async function Ka(e,n){qe=[],De=0,A=[];const t=re.lm_head.N,r=Math.max(1,Math.min(n.topK??20,t)),a=n.repetitionPenalty??1,o=n.presencePenalty??0,l=n.noRepeatNgramSize??0,i=s.createBuffer({size:t*4,usage:k|z}),u=s.createBuffer({size:r*4,usage:k|z}),p=s.createBuffer({size:r*4,usage:k|z});A==null||A.push(i,u,p);const g=a!==1||o!==0?ut(e):new Uint32Array(0),h=l>0?lt(e,l):[],N=qr(g.length?g:new Uint32Array(1),k|M),G=qr(h.length?Uint32Array.from(h):new Uint32Array(1),k|M),V=s.createCommandEncoder(),{fn:O}=Pn(V,or(V,e),e.length,0),de=s.createBuffer({size:w*4,usage:k|z|M});A==null||A.push(de),V.copyBufferToBuffer(O,(e.length-1)*w*4,de,0,w*4);let P=V.beginComputePass();qn(P,de,1,i),P.end(),s.queue.submit([V.finish()]),await s.queue.onSubmittedWorkDone();const T=await Ze(i,t),q=s.createCommandEncoder();P=q.beginComputePass(),Ue(P,"sampler_penalty",[["u",g.length],["u",h.length],["f",a],["u",4286578688],["f",o]],[N,G],[i]),P.dispatchWorkgroups(1);for(let x=0;x<r;x++)Ue(P,"argmax_masked",[["u",t],["u",x],["u",0],["u",0]],[i],[u,p]),P.dispatchWorkgroups(1);P.end(),s.queue.submit([q.finish()]),await s.queue.onSubmittedWorkDone();try{return{base:T,penalized:await Ze(i,t),candIds:await dr(u,r),candVals:await Ze(p,r)}}finally{Ie(),A=null}}const Wr={useSubgroups:Ne,subgroupSize:Ae,kvCache:an?"f16":ve?"q8":"f32",activation:nr?"f16":"f32",overflow:ke?"sinks":"error",maxSeqLen:L,adapter:{vendor:In.vendor,architecture:In.architecture,device:In.device,description:In.description},limits:{maxStorageBufferBindingSize:Number(s.limits.maxStorageBufferBindingSize),maxComputeWorkgroupStorageSize:s.limits.maxComputeWorkgroupStorageSize},timestampQuery:yr};async function Na(e,n={}){var g,h,N,G,V,O,de,P;const t=n.temperature!=null&&n.temperature>0&&n.temperature!==1,r=(n.repetitionPenalty??1)!==1||(n.noRepeatNgramSize??0)>0||(n.presencePenalty??0)!==0||(n.dryMultiplier??0)>0;if(((n.dryMultiplier??0)>0||(n.topNSigma??0)>0)&&n.promptLookup&&n.promptLookup!=="auto")throw new Error("bitgpu: dryMultiplier/topNSigma are not supported with promptLookup (they need per-position statistics; auto simply disables lookup)");if(d.hybrid&&n.promptLookup&&n.promptLookup!=="auto")throw new Error("bitgpu: promptLookup is not supported on the qwen3_5 hybrid backbone (rejected drafts would corrupt the linear-attention recurrent state); use promptLookup: 'auto' or omit it");const a=(n.reuseCache??!1)&&qe.length>0;if((g=n.signal)!=null&&g.aborted)return{tokens:[],prefillMs:0,decodeMs:0,tokensPerSecond:0,timing:{recordMs:0,gpuMs:0,readbackMs:0}};let o=a?ke?De:qe.length-1:0;const l=a?[qe[qe.length-1],...e]:e,i=a?qe:[...e];if(l.length===0)throw new Error("generate: no tokens to process");if(ke&&o+l.length+1>L){if(Qe+l.length+1>L)throw new Error(`generate: prompt length ${l.length} exceeds the rolling window (maxSeqLen ${L} minus ${Qe} sinks); trim the prompt`);await yn(Math.min(L,o+l.length)),o=qt(o,o+l.length+1-L),De=o}const u=L-o-l.length;if(u<1)throw new Error(`generate: prompt length ${o+l.length} exceeds maxSeqLen ${L}; trim history or raise maxSeqLen`);const p=ke?n.maxTokens??256:Math.min(n.maxTokens??256,u);a?i.push(...e):qe=i;try{if(p<1){await yn(o+l.length),A=[];try{const x=performance.now();return l.length>1&&(await gr(l.slice(0,-1),o,n.signal),await s.queue.onSubmittedWorkDone()),De=o+l.length-1,{tokens:[],prefillMs:performance.now()-x,decodeMs:0,tokensPerSecond:0,timing:{recordMs:0,gpuMs:0,readbackMs:0}}}finally{Ie(),A=null}}const T=!!n.candidateFilter||(n.logprobs??0)>0;let q;if(!T&&!d.hybrid&&(n.dryMultiplier??0)===0&&(n.topNSigma??0)===0&&n.promptLookup==="auto"&&p>24){const x=await it(l,o,24,n,i),se=x.tokens.length;if(se<24)q=x;else{const he=Ra(se,((h=x.spec)==null?void 0:h.steps)??0,t||r),X=[x.tokens[se-1]],J=o+l.length+se-1,ee=p-se;let C;he?C=await it(X,J,ee,n,i,x.rng):t||r?C=await St(X,J,ee,n,i,x.rng):(C=await ot(X,J,ee,null,vr,{stopTokens:n.stopTokens,onToken:n.onToken,signal:n.signal}),i.push(...C.tokens));const ue=se+C.tokens.length,te=x.decodeMs+C.prefillMs+C.decodeMs,Y=Math.max(1,se-1),ne=Math.max(0,C.tokens.length);q={prefillMs:x.prefillMs,decodeMs:te,tokPerSec:Math.max(1,ue-1)/(te/1e3),tokens:[...x.tokens,...C.tokens],firstArgmax:x.firstArgmax,recMs:(x.recMs*Y+C.recMs*ne)/(Y+ne),gpuMs:(x.gpuMs*Y+C.gpuMs*ne)/(Y+ne),rbMs:(x.rbMs*Y+C.rbMs*ne)/(Y+ne),spec:{steps:(((N=x.spec)==null?void 0:N.steps)??0)+(((G=C.spec)==null?void 0:G.steps)??0),drafted:(((V=x.spec)==null?void 0:V.drafted)??0)+(((O=C.spec)==null?void 0:O.drafted)??0),accepted:(((de=x.spec)==null?void 0:de.accepted)??0)+(((P=C.spec)==null?void 0:P.accepted)??0),bailed:!he}}}}else!T&&!d.hybrid&&(n.dryMultiplier??0)===0&&(n.topNSigma??0)===0&&n.promptLookup?q=await it(l,o,p,n,i):t||r||T?q=await St(l,o,p,n,i):(q=await ot(l,o,p,null,vr,{stopTokens:n.stopTokens,onToken:n.onToken,signal:n.signal}),i.push(...q.tokens));return{tokens:q.tokens,prefillMs:q.prefillMs,decodeMs:q.decodeMs,tokensPerSecond:q.tokPerSec,timing:{recordMs:q.recMs,gpuMs:q.gpuMs,readbackMs:q.rbMs},...q.spec?{speculation:q.spec}:{},...q.lp?{logprobs:q.lp}:{}}}catch(T){throw qe=[],De=0,T}}async function qa(e){if(e.length===0)throw new Error("prefill: no tokens to process");if(e.length>L)throw new Error(`prefill: sequence length ${e.length} exceeds maxSeqLen ${L}`);qe=[],De=0,await yn(e.length),A=[];try{const n=performance.now();return e.length>1&&(await gr(e.slice(0,-1),0),await s.queue.onSubmittedWorkDone()),qe=[...e],De=e.length-1,{prefillMs:performance.now()-n}}finally{Ie(),A=null}}const dn=y*f*Dn,pn=ve?y*(f/32)*4:0,st=e=>(d.hybrid?ln.length:d.layers)*2*e*(dn+pn)+sr.length*(Fn+Qn);async function Da(e){if(qe.length===0)return null;const n=ke?De:qe.length-1,t=Math.max(0,Math.min(Math.floor((e==null?void 0:e.from)??0),n));if(t>0&&ke)throw new Error("saveCache: delta snapshots ({ from }) are not supported under overflow 'sinks'");if(t>0&&d.hybrid)throw new Error("saveCache: delta snapshots ({ from }) are not supported for the qwen3_5 hybrid backbone");const r=n-t,a=st(r),o=new ArrayBuffer(a);if(a>0){const l=Math.max(4,Math.min((globalThis.__RBCAP??0)||134217728,s.limits.maxBufferSize)&-4),i=[];if(r>0)for(const h of ln)i.push({src:Ve[h],off:t*dn,size:r*dn}),i.push({src:Te[h],off:t*dn,size:r*dn}),ve&&(i.push({src:Ye[h],off:t*pn,size:r*pn}),i.push({src:rn[h],off:t*pn,size:r*pn}));for(const h of sr)i.push({src:ur[h][cr],off:0,size:Fn}),i.push({src:lr[h][cr],off:0,size:Qn});const u=s.createBuffer({size:Math.min(a,l),usage:GPUBufferUsage.MAP_READ|M});let p=0,g=0;for(let h=0;h<a;){const N=Math.min(l,a-h),G=s.createCommandEncoder();for(let V=0;V<N;){const O=i[p],de=Math.min(O.size-g,N-V);G.copyBufferToBuffer(O.src,O.off+g,u,V,de),V+=de,g+=de,g===O.size&&(p++,g=0)}s.queue.submit([G.finish()]),await u.mapAsync(GPUMapMode.READ,0,N),new Uint8Array(o,h,N).set(new Uint8Array(u.getMappedRange(0,N))),u.unmap(),h+=N}u.destroy()}return{version:ke?2:1,kvCache:Wr.kvCache,model:{layers:d.layers,kvHeads:y,headDim:f,hidden:w,vocab:d.vocab},ids:[...qe],...t>0?{base:t}:{},...ke?{roll:{sinkTokens:Qe,cacheLen:De}}:{},data:o}}async function Ma(e){var l,i;if(!e||e.version!==1&&e.version!==2)throw new Error(`restoreCache: unsupported snapshot version ${e==null?void 0:e.version}`);if(e.version===2!==ke)throw new Error(e.version===2?"restoreCache: snapshot was saved under overflow 'sinks' (unroped keys); this engine runs overflow 'error'":"restoreCache: snapshot was saved under overflow 'error' (roped keys); this engine runs overflow 'sinks'");if(e.version===2&&((l=e.roll)==null?void 0:l.sinkTokens)!==Qe)throw new Error(`restoreCache: snapshot uses ${(i=e.roll)==null?void 0:i.sinkTokens} sink tokens but this engine uses ${Qe}`);if(e.kvCache!==Wr.kvCache)throw new Error(`restoreCache: snapshot was saved under kvCache '${e.kvCache}' but this engine runs '${Wr.kvCache}' - snapshots do not convert across modes`);const n=e.model;if(!n||n.layers!==d.layers||n.kvHeads!==y||n.headDim!==f||n.hidden!==w||n.vocab!==d.vocab)throw new Error("restoreCache: snapshot is from a different model (architecture mismatch)");if(!Array.isArray(e.ids)||e.ids.length===0)throw new Error("restoreCache: snapshot holds no tokens");const t=e.version===2?e.roll.cacheLen:e.ids.length-1,r=Math.max(0,Math.floor(e.base??0));if(r>0&&d.hybrid)throw new Error("restoreCache: delta snapshots are not supported for the qwen3_5 hybrid backbone");const a=t-r;if(t+(e.version===2?0:1)>L)throw new Error(`restoreCache: snapshot needs ${t+(e.version===2?0:1)} cache slots but maxSeqLen is ${L}`);if(e.data.byteLength!==st(a))throw new Error(`restoreCache: snapshot data is ${e.data.byteLength} bytes, expected ${st(a)}`);if(r>0){if(De!==r)throw new Error(`restoreCache: delta snapshot expects the cache at position ${r} (prewarm the shared prefix first); it is at ${De}`);for(let u=0;u<r;u++)if(qe[u]!==e.ids[u])throw new Error(`restoreCache: delta snapshot prefix does not match the current prewarm (token ${u})`)}await yn(t);let o=0;if(a>0)for(const u of ln)s.queue.writeBuffer(Ve[u],r*dn,e.data,o,a*dn),o+=a*dn,s.queue.writeBuffer(Te[u],r*dn,e.data,o,a*dn),o+=a*dn,ve&&(s.queue.writeBuffer(Ye[u],r*pn,e.data,o,a*pn),o+=a*pn,s.queue.writeBuffer(rn[u],r*pn,e.data,o,a*pn),o+=a*pn);for(const u of sr)s.queue.writeBuffer(ur[u][0],0,e.data,o,Fn),o+=Fn,s.queue.writeBuffer(lr[u][0],0,e.data,o,Qn),o+=Qn;d.hybrid&&(cr=0),qe=[...e.ids],De=t}let Gt=Promise.resolve();const zn=e=>(...n)=>{const t=Gt.then(()=>e(...n),()=>e(...n));return Gt=t.catch(()=>{}),t};return{generate:zn(Na),prefill:zn(qa),forward:zn(_a),saveCache:zn(Da),restoreCache:zn(Ma),resetCache:la,capabilities:Wr,lost:oa,dispose:()=>s.destroy(),device:s,adapter:Se,profileDecode:zn(async(e,n,t=null,r=vr)=>{qe=[],De=0,rt=yr;try{return await ot(e,0,n,t,r)}finally{rt=!1}}),debugDecode:zn(xa),debugSampler:zn(Ka)}}export{Tr as GpuOutOfMemoryError,$t as WebGPUUnavailableError,$a as createEngine};

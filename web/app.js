// app.js - In-browser bird classifier.
//
// Flow:  photo -> resize shorter side to 256 -> centre crop 224 -> normalise
//        -> ONNX model -> { probs [1,200], cam [1,200,h,w] }
//        -> top-5 list + heatmap overlay of the top-1 class.
// The preprocessing copies the PyTorch eval transform in data.py exactly,
// so the browser sees the same input the model was tested on.

const $ = (id) => document.getElementById(id);
let session = null;   // ONNX Runtime inference session
let meta = null;      // meta.json: classes, preprocessing, examples, metrics
let backend = "";     // "webgpu" or "wasm"
let last = null;      // last result, so the heatmap slider can redraw

// ---------------------------------------------------------------- loading
async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = +res.headers.get("Content-Length") || 0;
  if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onProgress(got / total);
  }
  const out = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  return out;
}

async function init() {
  try {
    meta = await (await fetch("meta.json")).json();
    renderStats();
    renderExamples();
    $("howModel").textContent = meta.model;

    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
    const bytes = await fetchWithProgress("model.onnx", (f) => {
      $("loadBar").style.width = `${Math.round(f * 100)}%`;
      $("statusText").textContent = `Downloading model… ${Math.round(f * meta.file_mb)} / ${meta.file_mb} MB`;
    });
    $("statusText").textContent = "Starting the network…";

    // Prefer WebGPU (fast, uses the graphics card). Fall back to WebAssembly (CPU, works everywhere).
    for (const ep of ["webgpu", "wasm"]) {
      if (ep === "webgpu" && !navigator.gpu) continue;
      try {
        session = await ort.InferenceSession.create(bytes, { executionProviders: [ep] });
        backend = ep;
        break;
      } catch (e) { console.warn(`${ep} failed, trying next backend`, e); }
    }
    if (!session) throw new Error("No ONNX Runtime backend could start.");
    $("status").classList.add("done");
  } catch (e) {
    $("statusText").textContent = `Could not load the model: ${e.message}`;
    console.error(e);
  }
}

function renderStats() {
  const m = meta.test_metrics || {};
  const pct = (v) => (v == null ? "–" : `${(v * 100).toFixed(1)}%`);
  $("stats").innerHTML = [
    ["Top-1 (test)", pct(m.top1)],
    ["Avg per class", pct(m.avg_class_acc)],
    ["Top-5", pct(m.top5)],
  ].map(([k, v]) => `<div class="stat"><b>${v}</b><span>${k}</span></div>`).join("");
}

function renderExamples() {
  $("examples").innerHTML = "";
  meta.examples.forEach((ex) => {
    const b = document.createElement("button");
    b.title = "Try this test image";
    b.innerHTML = `<img src="${ex.file}" alt="Test image" loading="lazy">`;
    b.onclick = () => loadImage(ex.file, ex.label);
    $("examples").appendChild(b);
  });
}

// ---------------------------------------------------------------- input
function loadImage(src, trueLabel = null) {
  const img = new Image();
  img.onload = () => classify(img, trueLabel);
  img.onerror = () => alert("This file could not be opened as an image.");
  img.src = src;
}
function loadFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  loadImage(URL.createObjectURL(file));
}

$("file").addEventListener("change", (e) => loadFile(e.target.files[0]));
const drop = $("drop");
drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") $("file").click(); });
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => loadFile(e.dataTransfer.files[0]));
window.addEventListener("paste", (e) => {
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
  if (item) loadFile(item.getAsFile());
});

// ---------------------------------------------------------------- preprocessing
// Same as torchvision: Resize(256) (shorter side) -> CenterCrop(224) -> ToTensor -> Normalize.
function preprocess(img) {
  const S = meta.img_size, R = meta.resize;
  const scale = R / Math.min(img.naturalWidth, img.naturalHeight);
  const w = Math.round(img.naturalWidth * scale), h = Math.round(img.naturalHeight * scale);
  const crop = document.createElement("canvas");
  crop.width = S; crop.height = S;
  const ctx = crop.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  // Draw the resized image shifted so that its centre S x S square lands on the canvas.
  ctx.drawImage(img, -Math.round((w - S) / 2), -Math.round((h - S) / 2), w, h);

  const px = ctx.getImageData(0, 0, S, S).data;      // RGBA, row-major, 0..255
  const data = new Float32Array(3 * S * S);           // CHW layout for the CNN
  for (let i = 0; i < S * S; i++) {
    for (let c = 0; c < 3; c++) {
      data[c * S * S + i] = (px[i * 4 + c] / 255 - meta.mean[c]) / meta.std[c];
    }
  }
  return { tensor: new ort.Tensor("float32", data, [1, 3, S, S]), crop };
}

// ---------------------------------------------------------------- inference
async function classify(img, trueLabel) {
  if (!session) { alert("The model is still loading. Please wait a moment."); return; }
  const { tensor, crop } = preprocess(img);
  const t0 = performance.now();
  const out = await session.run({ input: tensor });
  const ms = performance.now() - t0;

  const probs = out.probs.data;                          // Float32Array(200)
  const [, , ch, cw] = out.cam.dims;
  const order = [...probs.keys()].sort((a, b) => probs[b] - probs[a]).slice(0, 5);
  const top = order[0];
  // Heatmap of the top-1 class: slice its h x w block out of [1, 200, h, w].
  const cam = out.cam.data.slice(top * ch * cw, (top + 1) * ch * cw);

  last = { crop, cam, ch, cw };
  showResult(order, probs, trueLabel, ms);
  drawView();
}

// ---------------------------------------------------------------- output
function showResult(order, probs, trueLabel, ms) {
  const name = (i) => meta.classes[i];
  const top = order[0];
  $("empty").hidden = true;
  $("result").hidden = false;
  $("previewBox").hidden = false;

  $("top1Name").textContent = name(top);
  $("top1Conf").textContent = `${(probs[top] * 100).toFixed(1)}% confidence`;
  $("top1Ref").src = `refs/${top}.jpg`;
  $("top1Ref").alt = `Reference photo: ${name(top)}`;
  $("lowConf").hidden = probs[top] >= 0.4;

  if (trueLabel == null) {
    $("truth").innerHTML = "";
  } else if (trueLabel === top) {
    $("truth").innerHTML = `<span class="ok">✓ Correct</span> <span class="muted">(test image)</span>`;
  } else {
    $("truth").innerHTML = `<span class="bad">✗ True species: ${name(trueLabel)}</span>`;
  }

  $("top5").innerHTML = order.map((i) => `
    <li class="${i === trueLabel ? "truth" : ""}">
      <img src="refs/${i}.jpg" alt="">
      <div>
        <div class="row"><span>${name(i)}</span><span>${(probs[i] * 100).toFixed(1)}%</span></div>
        <div class="track"><div class="fill" style="width:${(probs[i] * 100).toFixed(1)}%"></div></div>
      </div>
    </li>`).join("");

  $("timing").textContent = `Inference ${ms.toFixed(0)} ms on ${backend === "webgpu" ? "WebGPU (graphics card)" : "WebAssembly (CPU)"} · nothing was uploaded.`;
}

// Jet-like colour map: 0 = blue (ignored) ... 1 = red (most important).
function colormap(t) {
  const r = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 3)));
  const g = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 2)));
  const b = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 1)));
  return [r * 255, g * 255, b * 255];
}

function drawView() {
  if (!last) return;
  const view = $("view");
  const ctx = view.getContext("2d");
  ctx.drawImage(last.crop, 0, 0, view.width, view.height);
  if (!$("camOn").checked) return;

  // Normalise the h x w CAM to 0..1 (keep only positive evidence, like Grad-CAM's ReLU).
  const { cam, ch, cw } = last;
  const pos = Array.from(cam, (v) => Math.max(0, v));
  const max = Math.max(...pos) || 1;
  const small = document.createElement("canvas");
  small.width = cw; small.height = ch;
  const sctx = small.getContext("2d");
  const id = sctx.createImageData(cw, ch);
  pos.forEach((v, i) => {
    const [r, g, b] = colormap(v / max);
    id.data.set([r, g, b, 255], i * 4);
  });
  sctx.putImageData(id, 0, 0);

  // Upsample the tiny map with smoothing and blend it over the photo.
  ctx.globalAlpha = $("camAlpha").value / 100;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(small, 0, 0, view.width, view.height);
  ctx.globalAlpha = 1;
}
$("camOn").addEventListener("change", drawView);
$("camAlpha").addEventListener("input", drawView);

init();

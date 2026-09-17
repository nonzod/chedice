"use strict";

// ---- State ----
let selectedFile = null;
let pollTimer = null;
let activeJobId = null;
let currentJob = null;

const SPEAKER_COLORS = [
  "#6c8bff", "#3ddc97", "#ff9f6b", "#ff6bd6",
  "#f5d76e", "#6be1ff", "#b06bff", "#ff6b6b",
];

// ---- Elements ----
const $ = (id) => document.getElementById(id);
const dropzone = $("dropzone");
const fileInput = $("fileInput");
const fileName = $("fileName");
const startBtn = $("startBtn");
const empty = $("empty");
const jobView = $("jobView");
const jobTitle = $("jobTitle");
const jobMeta = $("jobMeta");
const downloads = $("downloads");
const progressWrap = $("progressWrap");
const progressBar = $("progressBar");
const stageLabel = $("stageLabel");
const transcriptEl = $("transcript");
const historyList = $("historyList");
const speakerEditor = $("speakerEditor");
const speakerFields = $("speakerFields");
const saveNamesBtn = $("saveNamesBtn");
const ytUrl = $("ytUrl");
const ytBtn = $("ytBtn");
const categoryInput = $("category");
const categoryList = $("categoryList");
const categoryEditor = $("categoryEditor");
const jobCategory = $("jobCategory");
const saveCategoryBtn = $("saveCategoryBtn");

// ---- File selection ----
function pickFile(file) {
  if (!file) return;
  selectedFile = file;
  fileName.textContent = `${file.name} · ${formatSize(file.size)}`;
  startBtn.disabled = false;
}

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
});
fileInput.addEventListener("change", () => pickFile(fileInput.files[0]));

["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  })
);
dropzone.addEventListener("drop", (e) => pickFile(e.dataTransfer.files[0]));

// ---- Start a job ----
startBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  startBtn.disabled = true;
  startBtn.textContent = "Caricamento…";

  const form = new FormData();
  form.append("file", selectedFile);
  form.append("language", $("language").value);
  form.append("num_speakers", $("numSpeakers").value);
  form.append("category", categoryInput.value.trim());

  try {
    const res = await fetch("/api/jobs", { method: "POST", body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Errore nel caricamento");
    }
    const job = await res.json();
    resetUploadForm();
    showJob(job);
    startPolling(job.id);
    loadHistory();
    loadCategories();
  } catch (err) {
    alert(err.message);
  } finally {
    startBtn.textContent = "Avvia trascrizione";
  }
});

function resetUploadForm() {
  selectedFile = null;
  fileInput.value = "";
  fileName.textContent = "";
  startBtn.disabled = true;
}

// ---- Start a YouTube job ----
ytBtn.addEventListener("click", async () => {
  const url = ytUrl.value.trim();
  if (!url) { ytUrl.focus(); return; }
  ytBtn.disabled = true;
  const original = ytBtn.textContent;
  ytBtn.textContent = "Invio…";

  try {
    const res = await fetch("/api/jobs/youtube", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        url,
        language: $("language").value,
        num_speakers: $("numSpeakers").value,
        category: categoryInput.value.trim(),
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Errore nell'avvio del job YouTube");
    }
    const job = await res.json();
    ytUrl.value = "";
    showJob(job);
    startPolling(job.id);
    loadHistory();
    loadCategories();
  } catch (err) {
    alert(err.message);
  } finally {
    ytBtn.disabled = false;
    ytBtn.textContent = original;
  }
});

ytUrl.addEventListener("keydown", (e) => {
  if (e.key === "Enter") { e.preventDefault(); ytBtn.click(); }
});

// ---- Polling ----
function startPolling(jobId) {
  activeJobId = jobId;
  clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    const res = await fetch(`/api/jobs/${jobId}`);
    if (!res.ok) { clearInterval(pollTimer); return; }
    const job = await res.json();
    if (job.id !== activeJobId) return;
    showJob(job);
    if (job.status === "completed" || job.status === "failed") {
      clearInterval(pollTimer);
      loadHistory();
    }
  }, 1500);
}

// ---- Rendering ----
function showJob(job) {
  activeJobId = job.id;
  currentJob = job;
  empty.classList.add("hidden");
  jobView.classList.remove("hidden");

  jobTitle.textContent = job.filename;
  jobMeta.innerHTML = "";
  jobMeta.appendChild(statusChip(job));
  if (job.source_url) jobMeta.appendChild(chip("▶️ YouTube"));
  if (job.detected_language) jobMeta.appendChild(chip(`🌐 ${job.detected_language.toUpperCase()}`));
  if (job.speaker_count) jobMeta.appendChild(chip(`👥 ${job.speaker_count} parlanti`));
  if (job.duration) jobMeta.appendChild(chip(`⏱️ ${formatDuration(job.duration)}`));

  categoryEditor.classList.remove("hidden");
  // Don't clobber the field while the user is typing (showJob runs on each poll).
  if (document.activeElement !== jobCategory) jobCategory.value = job.category || "";

  const running = job.status === "queued" || job.status === "processing";
  progressWrap.classList.toggle("hidden", !running);
  progressBar.style.width = `${job.progress || 0}%`;
  stageLabel.textContent = running ? `${job.stage} · ${Math.round(job.progress)}%` : "";

  const done = job.status === "completed";
  downloads.classList.toggle("hidden", !done);
  if (done) {
    // The ?v param busts the browser cache when speaker names change.
    const v = encodeURIComponent(JSON.stringify(job.speaker_names || {}));
    downloads.querySelectorAll(".dl").forEach((a) => {
      a.href = `/api/jobs/${job.id}/download/${a.dataset.fmt}?v=${v}`;
    });
  }

  if (done) buildSpeakerEditor(job);
  else speakerEditor.classList.add("hidden");

  if (job.status === "failed") {
    transcriptEl.innerHTML = `<p class="stage-label">❌ ${escapeHtml(job.error || "Errore")}</p>`;
  } else if (job.segments && job.segments.length) {
    renderTranscript(job.segments);
  } else {
    transcriptEl.innerHTML = "";
  }
}

function displayName(speaker) {
  if (!speaker) return null;
  const names = currentJob && currentJob.speaker_names;
  return (names && names[speaker]) || speaker;
}

function buildSpeakerEditor(job) {
  const canonical = [];
  for (const seg of job.segments) {
    if (seg.speaker && !canonical.includes(seg.speaker)) canonical.push(seg.speaker);
  }
  if (!canonical.length) {
    speakerEditor.classList.add("hidden");
    return;
  }
  speakerFields.innerHTML = "";
  canonical.forEach((sp, i) => {
    const color = SPEAKER_COLORS[i % SPEAKER_COLORS.length];
    const value = (job.speaker_names || {})[sp] || "";
    const row = document.createElement("div");
    row.className = "se-row";
    const dot = document.createElement("span");
    dot.className = "dot";
    dot.style.background = color;
    const input = document.createElement("input");
    input.type = "text";
    input.dataset.canon = sp;
    input.placeholder = sp;
    input.value = value;
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") saveNamesBtn.click();
    });
    row.append(dot, input);
    speakerFields.appendChild(row);
  });
  speakerEditor.classList.remove("hidden");
}

saveNamesBtn.addEventListener("click", async () => {
  if (!currentJob) return;
  const map = {};
  speakerFields.querySelectorAll("input").forEach((inp) => {
    const val = inp.value.trim();
    if (val) map[inp.dataset.canon] = val;
  });
  saveNamesBtn.disabled = true;
  saveNamesBtn.textContent = "Salvataggio…";
  try {
    const res = await fetch(`/api/jobs/${currentJob.id}/speakers`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(map),
    });
    if (!res.ok) throw new Error("Errore nel salvataggio dei nomi");
    showJob(await res.json());
  } catch (err) {
    alert(err.message);
  } finally {
    saveNamesBtn.disabled = false;
    saveNamesBtn.textContent = "Salva nomi";
  }
});

saveCategoryBtn.addEventListener("click", async () => {
  if (!currentJob) return;
  saveCategoryBtn.disabled = true;
  saveCategoryBtn.textContent = "Salvataggio…";
  try {
    const res = await fetch(`/api/jobs/${currentJob.id}/category`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ category: jobCategory.value.trim() }),
    });
    if (!res.ok) throw new Error("Errore nel salvataggio della categoria");
    showJob(await res.json());
    loadHistory();
    loadCategories();
  } catch (err) {
    alert(err.message);
  } finally {
    saveCategoryBtn.disabled = false;
    saveCategoryBtn.textContent = "Salva";
  }
});

jobCategory.addEventListener("keydown", (e) => {
  if (e.key === "Enter") { e.preventDefault(); saveCategoryBtn.click(); }
});

function renderTranscript(segments) {
  const speakers = [...new Set(segments.map((s) => s.speaker).filter(Boolean))];
  const colorOf = (sp) =>
    sp ? SPEAKER_COLORS[speakers.indexOf(sp) % SPEAKER_COLORS.length] : "#6c8bff";

  // Merge consecutive segments from the same speaker into a single turn.
  const turns = [];
  for (const seg of segments) {
    const last = turns[turns.length - 1];
    if (last && last.speaker === seg.speaker) {
      last.text += " " + seg.text;
      last.end = seg.end;
    } else {
      turns.push({ speaker: seg.speaker, text: seg.text, start: seg.start, end: seg.end });
    }
  }

  transcriptEl.innerHTML = "";
  for (const turn of turns) {
    const color = colorOf(turn.speaker);
    const name = displayName(turn.speaker);
    const row = document.createElement("div");
    row.className = "turn";
    row.innerHTML = `
      <div class="avatar" style="background:${color}">${initialsOf(name)}</div>
      <div class="bubble">
        <div class="who" style="color:${color}">
          ${name ? escapeHtml(name) : "Trascrizione"}
          <span class="time">${formatTime(turn.start)}</span>
        </div>
        <div class="text">${escapeHtml(turn.text)}</div>
      </div>`;
    transcriptEl.appendChild(row);
  }
}

function initialsOf(name) {
  if (!name) return "•";
  const m = name.match(/^SPEAKER\s+(\d+)$/i);
  if (m) return m[1];
  const words = name.trim().split(/\s+/);
  return (words[0][0] + (words[1] ? words[1][0] : "")).toUpperCase();
}

// ---- History ----
async function loadHistory() {
  const res = await fetch("/api/jobs");
  if (!res.ok) return;
  const jobs = await res.json();
  historyList.innerHTML = "";
  for (const job of jobs) {
    const li = document.createElement("li");
    li.className = "history-item";
    li.innerHTML = `
      <div style="display:flex;align-items:center;gap:10px;min-width:0;flex:1">
        <span class="dot ${job.status}"></span>
        <div style="min-width:0">
          <div class="hi-name">${escapeHtml(job.filename)}</div>
          <div class="hi-status">${job.category ? "🏷️ " + escapeHtml(job.category) + " · " : ""}${statusLabel(job)}</div>
        </div>
      </div>
      <button class="hi-del" title="Elimina">✕</button>`;
    li.querySelector(".hi-name").addEventListener("click", () => openJob(job.id));
    li.querySelector(".dot").addEventListener("click", () => openJob(job.id));
    li.querySelector(".hi-status").addEventListener("click", () => openJob(job.id));
    li.querySelector(".hi-del").addEventListener("click", async (e) => {
      e.stopPropagation();
      await fetch(`/api/jobs/${job.id}`, { method: "DELETE" });
      if (activeJobId === job.id) {
        jobView.classList.add("hidden");
        empty.classList.remove("hidden");
        clearInterval(pollTimer);
      }
      loadHistory();
    });
    historyList.appendChild(li);
  }
}

// ---- Categories ----
async function loadCategories() {
  const res = await fetch("/api/categories");
  if (!res.ok) return;
  const categories = await res.json();
  categoryList.innerHTML = "";
  for (const name of categories) {
    const opt = document.createElement("option");
    opt.value = name;
    categoryList.appendChild(opt);
  }
}

async function openJob(jobId) {
  const res = await fetch(`/api/jobs/${jobId}`);
  if (!res.ok) return;
  const job = await res.json();
  showJob(job);
  if (job.status === "queued" || job.status === "processing") startPolling(jobId);
}

// ---- Helpers ----
function statusChip(job) {
  const map = {
    queued: ["In coda", ""],
    processing: ["In elaborazione", ""],
    completed: ["Completato", "ok"],
    failed: ["Errore", "err"],
  };
  const [label, cls] = map[job.status] || [job.status, ""];
  return chip(label, cls);
}
function statusLabel(job) {
  if (job.status === "processing") return `${job.stage} · ${Math.round(job.progress)}%`;
  return { queued: "In coda", completed: "Completato", failed: "Errore" }[job.status] || job.status;
}
function chip(text, cls = "") {
  const el = document.createElement("span");
  el.className = `chip ${cls}`.trim();
  el.textContent = text;
  return el;
}
function formatSize(bytes) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
}
function formatDuration(sec) {
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}
function formatTime(sec) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}
function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// ---- Init ----
loadHistory();
loadCategories();

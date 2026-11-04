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
const visualPanel = $("visualPanel");
const describeScenesBtn = $("describeScenesBtn");
const visionConfigHint = $("visionConfigHint");
const visionOpenSettings = $("visionOpenSettings");
const visualStatus = $("visualStatus");
const visualDescription = $("visualDescription");
const visualDetail = $("visualDetail");
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
// Settings modal
const settingsBtn = $("settingsBtn");
const settingsModal = $("settingsModal");
const settingsClose = $("settingsClose");
const settingsSave = $("settingsSave");
const settingsStatus = $("settingsStatus");
const llmEnabled = $("llmEnabled");
const llmFallback = $("llmFallback");
const llmTemperature = $("llmTemperature");
const visionEnabled = $("visionEnabled");
const visionFallback = $("visionFallback");
const visionTemperature = $("visionTemperature");
const visionSettingsSave = $("visionSettingsSave");
const visionSettingsStatus = $("visionSettingsStatus");
// AI panel
const aiPanel = $("aiPanel");
const aiHistory = $("aiHistory");
const aiPrompt = $("aiPrompt");
const aiSendBtn = $("aiSendBtn");
const aiConfigHint = $("aiConfigHint");
const aiOpenSettings = $("aiOpenSettings");
// Whisper model / cancel / resume
const whisperModel = $("whisperModel");
const retryModel = $("retryModel");
const cancelJobBtn = $("cancelJobBtn");
const resumeBox = $("resumeBox");
const resumeText = $("resumeText");
const retryJobBtn = $("retryJobBtn");
const jobWarning = $("jobWarning");
// Speaker fixes
const speakerFix = $("speakerFix");
const mergeSource = $("mergeSource");
const mergeTarget = $("mergeTarget");
const mergeBtn = $("mergeBtn");
const rediarizeCount = $("rediarizeCount");
const rediarizeBtn = $("rediarizeBtn");
// Inline transcript editing
const transcriptToolbar = $("transcriptToolbar");
const editTranscriptBtn = $("editTranscriptBtn");
const saveEditBtn = $("saveEditBtn");
const cancelEditBtn = $("cancelEditBtn");
const editStatus = $("editStatus");
// Views: transcription / archive
const navTranscribe = $("navTranscribe");
const navArchive = $("navArchive");
const transcribeView = $("transcribeView");
const archiveView = $("archiveView");
const archiveCrumbs = $("archiveCrumbs");
const archiveSummary = $("archiveSummary");
const archiveSearch = $("archiveSearch");
const archiveGrid = $("archiveGrid");
const archiveEmpty = $("archiveEmpty");
// Details modal
const detailsModal = $("detailsModal");
const detailsTitle = $("detailsTitle");
const detailsBody = $("detailsBody");
const detailsClose = $("detailsClose");
const detailsOpenBtn = $("detailsOpenBtn");
const jobDetailsBtn = $("jobDetailsBtn");
// Retention settings
const retentionMedia = $("retentionMedia");
const retentionJobs = $("retentionJobs");
const retentionUsage = $("retentionUsage");
const retentionStatus = $("retentionStatus");
const retentionSave = $("retentionSave");
const cleanupNowBtn = $("cleanupNowBtn");

let editMode = false;
let defaultModel = "";

// Whether an LLM is configured & enabled (drives the AI panel hint).
let llmReady = false;
let visionReady = false;

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
  form.append("whisper_model", whisperModel.value);

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
        whisper_model: whisperModel.value,
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
    const sceneRunning = job.visual_status === "queued" || job.visual_status === "processing";
    if (!isRunning(job) && !sceneRunning) {
      clearInterval(pollTimer);
      loadHistory();
    }
  }, 1500);
}

// ---- Rendering ----
function isRunning(job) {
  return job.status === "queued" || job.status === "processing";
}

function showJob(job) {
  if (editMode && (!currentJob || currentJob.id !== job.id)) exitEditMode();
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
  if (job.whisper_model) jobMeta.appendChild(chip(`🧠 ${job.whisper_model}`));

  categoryEditor.classList.remove("hidden");
  // Don't clobber the field while the user is typing (showJob runs on each poll).
  if (document.activeElement !== jobCategory) jobCategory.value = job.category || "";

  const running = isRunning(job);
  progressWrap.classList.toggle("hidden", !running);
  progressBar.style.width = `${job.progress || 0}%`;
  stageLabel.textContent = running ? `${job.stage} · ${Math.round(job.progress)}%` : "";
  const cancelling = job.stage === "Annullamento…";
  cancelJobBtn.disabled = cancelling;
  cancelJobBtn.textContent = cancelling ? "Annullamento…" : "✕ Annulla";

  // Cancelled / failed jobs can be resumed, optionally with another model.
  const resumable = job.status === "cancelled" || job.status === "failed";
  resumeBox.classList.toggle("hidden", !resumable);
  if (resumable) {
    resumeText.textContent = job.status === "cancelled"
      ? "⏸️ Trascrizione annullata."
      : "Trascrizione non riuscita.";
    if (document.activeElement !== retryModel) retryModel.value = job.requested_model || "";
  }

  // A completed job may carry a non-fatal error (e.g. failed speaker re-recognition).
  const warning = job.status === "completed" && job.error;
  jobWarning.classList.toggle("hidden", !warning);
  jobWarning.textContent = warning ? `⚠️ ${job.error}` : "";

  const done = job.status === "completed";
  downloads.classList.toggle("hidden", !done);
  if (done) {
    // The ?v param busts the browser cache when speaker names or text change.
    const v = encodeURIComponent(JSON.stringify(job.speaker_names || {}) + (job.segments || []).length);
    downloads.querySelectorAll(".dl").forEach((a) => {
      a.href = `/api/jobs/${job.id}/download/${a.dataset.fmt}?v=${v}`;
    });
  }

  if (!editMode) {
    if (done) buildSpeakerEditor(job);
    else speakerEditor.classList.add("hidden");
  }

  visualPanel.classList.toggle("hidden", !done || !job.has_video);
  if (done && job.has_video) {
    visualDetail.value = job.visual_detail || "medium";
    const sceneRunning = job.visual_status === "queued" || job.visual_status === "processing";
    describeScenesBtn.disabled = sceneRunning;
    describeScenesBtn.textContent = sceneRunning
      ? "Analisi in corso…"
      : job.visual_status === "completed" ? "Rigenera analisi"
        : job.visual_status === "failed" ? "Riprova analisi" : "Analizza scene";
    visualStatus.textContent = job.visual_status === "failed" ? (job.visual_error || "Analisi non riuscita") : "";
    visualStatus.classList.toggle("error", job.visual_status === "failed");
    visualDescription.textContent = job.visual_description || "";
    visualDescription.classList.toggle("hidden", !job.visual_description);
    visionConfigHint.classList.toggle("hidden", visionReady);
  }

  const hasText = !!(job.segments && job.segments.length);
  transcriptToolbar.classList.toggle("hidden", !done || !hasText);
  if (!editMode) {
    if (job.status === "failed") {
      transcriptEl.innerHTML = `<p class="stage-label">❌ ${escapeHtml(job.error || "Errore")}</p>`;
    } else if (hasText) {
      renderTranscript(job.segments);
    } else {
      transcriptEl.innerHTML = "";
    }
  }

  // AI panel: available only once a transcript exists.
  if (done && hasText) {
    aiPanel.classList.remove("hidden");
    renderAiHistory(job.ai_messages || []);
    aiConfigHint.classList.toggle("hidden", llmReady);
  } else {
    aiPanel.classList.add("hidden");
  }
}

// ---- Cancel / resume ----
cancelJobBtn.addEventListener("click", async () => {
  if (!currentJob) return;
  cancelJobBtn.disabled = true;
  try {
    const res = await fetch(`/api/jobs/${currentJob.id}/cancel`, { method: "POST" });
    const job = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(job.detail || "Impossibile annullare il job");
    showJob(job);
    loadHistory();
  } catch (err) {
    alert(err.message);
    cancelJobBtn.disabled = false;
  }
});

retryJobBtn.addEventListener("click", async () => {
  if (!currentJob) return;
  retryJobBtn.disabled = true;
  try {
    const res = await fetch(`/api/jobs/${currentJob.id}/retry`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ whisper_model: retryModel.value }),
    });
    const job = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(job.detail || "Impossibile riprendere il job");
    showJob(job);
    startPolling(job.id);
    loadHistory();
  } catch (err) {
    alert(err.message);
  } finally {
    retryJobBtn.disabled = false;
  }
});

// ---- Whisper models ----
async function loadModels() {
  const res = await fetch("/api/models").catch(() => null);
  if (!res || !res.ok) return;
  const { default: def, choices } = await res.json();
  defaultModel = def;
  for (const select of [whisperModel, retryModel]) {
    const keep = select.value;
    select.innerHTML = "";
    select.appendChild(new Option(`Predefinito (${def})`, ""));
    for (const name of choices) {
      if (name !== def) select.appendChild(new Option(name, name));
    }
    select.value = keep;
    if (select.selectedIndex < 0) select.value = "";
  }
}

describeScenesBtn.addEventListener("click", async () => {
  if (!currentJob) return;
  const jobId = currentJob.id;
  describeScenesBtn.disabled = true;
  describeScenesBtn.textContent = "Analisi in corso…";
  visualStatus.textContent = "Estrazione dei fotogrammi e analisi locale…";
  visualStatus.classList.remove("error");
  try {
    const res = await fetch(`/api/jobs/${jobId}/describe-scenes`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ detail: visualDetail.value }),
    });
    const job = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(job.detail || "Errore durante l’analisi delle scene");
    if (activeJobId === jobId) {
      showJob(job);
      if (job.visual_status === "queued" || job.visual_status === "processing") startPolling(jobId);
    }
  } catch (err) {
    visualStatus.textContent = `❌ ${err.message}`;
    visualStatus.classList.add("error");
  } finally {
    if (currentJob && currentJob.id === jobId) {
      const sceneRunning = currentJob.visual_status === "queued" || currentJob.visual_status === "processing";
      describeScenesBtn.disabled = sceneRunning;
      describeScenesBtn.textContent = sceneRunning
        ? "Analisi in corso…"
        : currentJob.visual_status === "failed" ? "Riprova analisi"
          : currentJob.visual_description ? "Rigenera analisi" : "Analizza scene";
    } else {
      describeScenesBtn.disabled = false;
    }
  }
});

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
  stopSample(); // drop any player tied to the rows we're about to rebuild
  speakerFields.innerHTML = "";
  canonical.forEach((sp, i) => {
    const color = SPEAKER_COLORS[i % SPEAKER_COLORS.length];
    const value = (job.speaker_names || {})[sp] || "";
    const row = document.createElement("div");
    row.className = "se-row";
    const play = document.createElement("button");
    play.type = "button";
    play.className = "se-play";
    play.textContent = "▶";
    play.title = "Ascolta un campione di questa voce";
    play.addEventListener("click", () => playSample(play, job.id, sp));
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
    row.append(play, dot, input);
    speakerFields.appendChild(row);
  });
  speakerEditor.classList.remove("hidden");
  fillMergeSelects(canonical);
}

// ---- Fix misrecognized voices ----
function speakersOf(job) {
  const list = [];
  for (const seg of job.segments || []) {
    if (seg.speaker && !list.includes(seg.speaker)) list.push(seg.speaker);
  }
  return list;
}

function speakerOptionLabel(sp) {
  const name = displayName(sp);
  return name !== sp ? `${name} (${sp})` : sp;
}

function fillMergeSelects(canonical) {
  const keepSource = mergeSource.value;
  const keepTarget = mergeTarget.value;
  for (const select of [mergeSource, mergeTarget]) {
    select.innerHTML = "";
    for (const sp of canonical) select.appendChild(new Option(speakerOptionLabel(sp), sp));
  }
  if (canonical.includes(keepSource)) mergeSource.value = keepSource;
  if (canonical.includes(keepTarget)) mergeTarget.value = keepTarget;
  else if (canonical.length > 1) mergeTarget.value = canonical[1];
  mergeBtn.closest(".fix-row").classList.toggle("hidden", canonical.length < 2);
  const busy = currentJob && currentJob.visual_status &&
    ["queued", "processing"].includes(currentJob.visual_status);
  rediarizeBtn.disabled = !!busy;
  rediarizeBtn.title = busy ? "Attendi la fine dell'analisi delle scene" : "";
}

mergeBtn.addEventListener("click", async () => {
  if (!currentJob) return;
  const source = mergeSource.value;
  const target = mergeTarget.value;
  if (!source || !target || source === target) {
    alert("Scegli due voci diverse da unire.");
    return;
  }
  mergeBtn.disabled = true;
  try {
    const res = await fetch(`/api/jobs/${currentJob.id}/speakers/merge`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source, target }),
    });
    const job = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(job.detail || "Errore nell'unione delle voci");
    showJob(job);
  } catch (err) {
    alert(err.message);
  } finally {
    mergeBtn.disabled = false;
  }
});

rediarizeBtn.addEventListener("click", async () => {
  if (!currentJob) return;
  rediarizeBtn.disabled = true;
  try {
    const res = await fetch(`/api/jobs/${currentJob.id}/rediarize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ num_speakers: parseInt(rediarizeCount.value, 10) || null }),
    });
    const job = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(job.detail || "Errore nel ricalcolo dei parlanti");
    speakerFix.open = false;
    showJob(job);
    startPolling(job.id);
    loadHistory();
  } catch (err) {
    alert(err.message);
  } finally {
    rediarizeBtn.disabled = false;
  }
});

// ---- Inline transcript editing ----
function enterEditMode() {
  if (!currentJob || !currentJob.segments || !currentJob.segments.length) return;
  editMode = true;
  editTranscriptBtn.classList.add("hidden");
  saveEditBtn.classList.remove("hidden");
  cancelEditBtn.classList.remove("hidden");
  editStatus.textContent = "Modifica il testo o il parlante di ogni frase. Testo vuoto = frase eliminata.";
  renderEditableTranscript(currentJob);
}

function exitEditMode() {
  editMode = false;
  editTranscriptBtn.classList.remove("hidden");
  saveEditBtn.classList.add("hidden");
  cancelEditBtn.classList.add("hidden");
  editStatus.textContent = "";
}

function renderEditableTranscript(job) {
  const speakers = speakersOf(job);
  const maxN = speakers.reduce((m, sp) => {
    const match = sp.match(/^SPEAKER (\d+)$/);
    return match ? Math.max(m, parseInt(match[1], 10)) : m;
  }, 0);
  const newLabel = `SPEAKER ${maxN + 1}`;

  transcriptEl.innerHTML = "";
  job.segments.forEach((seg, index) => {
    const row = document.createElement("div");
    row.className = "edit-row";
    row.dataset.index = index;

    const time = document.createElement("span");
    time.className = "time";
    time.textContent = formatTime(seg.start);

    const select = document.createElement("select");
    select.className = "edit-speaker";
    if (!seg.speaker) select.appendChild(new Option("—", ""));
    for (const sp of speakers) select.appendChild(new Option(speakerOptionLabel(sp), sp));
    if (speakers.length) select.appendChild(new Option(`➕ Nuovo parlante (${newLabel})`, newLabel));
    select.value = seg.speaker || "";
    select.disabled = !speakers.length;
    const paint = () => {
      const i = speakers.indexOf(select.value);
      select.style.borderColor = i >= 0 ? SPEAKER_COLORS[i % SPEAKER_COLORS.length] : "";
    };
    select.addEventListener("change", paint);
    paint();

    const text = document.createElement("textarea");
    text.className = "edit-text";
    text.rows = 1;
    text.value = seg.text;
    const grow = () => { text.style.height = "auto"; text.style.height = `${text.scrollHeight}px`; };
    text.addEventListener("input", grow);

    row.append(time, select, text);
    transcriptEl.appendChild(row);
    requestAnimationFrame(grow);
  });
}

editTranscriptBtn.addEventListener("click", enterEditMode);
cancelEditBtn.addEventListener("click", () => {
  exitEditMode();
  if (currentJob) showJob(currentJob);
});

saveEditBtn.addEventListener("click", async () => {
  if (!currentJob) return;
  const edits = [];
  transcriptEl.querySelectorAll(".edit-row").forEach((row) => {
    const index = parseInt(row.dataset.index, 10);
    const seg = currentJob.segments[index];
    const text = row.querySelector(".edit-text").value.replace(/\s+/g, " ").trim();
    const speaker = row.querySelector(".edit-speaker").value;
    const edit = { index };
    if (text !== seg.text) edit.text = text;
    if (speaker && speaker !== seg.speaker) edit.speaker = speaker;
    if (Object.keys(edit).length > 1) edits.push(edit);
  });
  if (!edits.length) {
    exitEditMode();
    showJob(currentJob);
    return;
  }
  saveEditBtn.disabled = true;
  editStatus.textContent = "Salvataggio…";
  try {
    const res = await fetch(`/api/jobs/${currentJob.id}/segments`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ edits }),
    });
    const job = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(job.detail || "Errore nel salvataggio delle modifiche");
    exitEditMode();
    showJob(job);
  } catch (err) {
    editStatus.textContent = `❌ ${err.message}`;
  } finally {
    saveEditBtn.disabled = false;
  }
});

// ---- Voice sample playback ----
let sampleAudio = null;
let sampleButton = null;

function stopSample() {
  if (sampleAudio) {
    sampleAudio.pause();
    sampleAudio = null;
  }
  if (sampleButton) {
    sampleButton.classList.remove("playing");
    sampleButton.textContent = "▶";
    sampleButton = null;
  }
}

function playSample(btn, jobId, speaker) {
  if (sampleButton === btn) { // clicking the playing voice again stops it
    stopSample();
    return;
  }
  stopSample();
  btn.classList.remove("error");
  const audio = new Audio(`/api/jobs/${jobId}/sample/${encodeURIComponent(speaker)}`);
  sampleAudio = audio;
  sampleButton = btn;
  btn.classList.add("playing");
  btn.textContent = "⏸";
  audio.addEventListener("ended", stopSample);
  audio.addEventListener("error", () => {
    stopSample();
    btn.classList.add("error");
    btn.title = "Campione non disponibile";
  });
  audio.play().catch(() => {}); // playback errors surface via the 'error' event
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
      <button class="hi-info" title="Dettagli">ℹ️</button>
      <button class="hi-del" title="Elimina">✕</button>`;
    li.addEventListener("click", () => openJob(job.id));
    li.querySelector(".hi-info").addEventListener("click", (e) => {
      e.stopPropagation();
      openDetails(job.id);
    });
    li.querySelector(".hi-del").addEventListener("click", async (e) => {
      e.stopPropagation();
      if (!confirm(`Eliminare «${job.filename}» e i relativi file?`)) return;
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
    cancelled: ["Annullato", ""],
  };
  let [label, cls] = map[job.status] || [job.status, ""];
  if (job.task === "diarize" && isRunning(job)) label = "Ricalcolo parlanti";
  return chip(label, cls);
}
function statusLabel(job) {
  if (job.status === "processing") return `${job.stage} · ${Math.round(job.progress)}%`;
  return {
    queued: "In coda", completed: "Completato", failed: "Errore", cancelled: "Annullato",
  }[job.status] || job.status;
}
function formatDate(ts) {
  return new Date(ts * 1000).toLocaleString("it-IT", {
    day: "2-digit", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit",
  });
}
function formatLongDuration(sec) {
  if (!sec) return "—";
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.round(sec % 60);
  return h ? `${h}h ${String(m).padStart(2, "0")}m` : `${m}:${String(s).padStart(2, "0")}`;
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

// ---- AI: ask about the transcript ----
function renderAiHistory(messages) {
  aiHistory.innerHTML = "";
  for (const m of messages) {
    const item = document.createElement("div");
    item.className = "ai-item";
    item.innerHTML = `
      <div class="ai-q">🙋 ${escapeHtml(m.prompt)}</div>
      <div class="ai-a">${escapeHtml(m.answer)}</div>
      ${m.provider ? `<div class="ai-via">via ${escapeHtml(m.provider)}</div>` : ""}`;
    aiHistory.appendChild(item);
  }
  aiHistory.scrollTop = aiHistory.scrollHeight;
}

async function sendAiPrompt(prompt) {
  if (!currentJob || !prompt.trim()) return;
  const jobId = currentJob.id;
  aiSendBtn.disabled = true;
  aiPrompt.disabled = true;
  const original = aiSendBtn.textContent;
  aiSendBtn.textContent = "Sto pensando…";

  // Placeholder that we fill in as tokens stream in.
  const pending = document.createElement("div");
  pending.className = "ai-item";
  pending.innerHTML = `
    <div class="ai-q">🙋 ${escapeHtml(prompt)}</div>
    <div class="ai-a ai-pending">…</div>`;
  aiHistory.appendChild(pending);
  aiHistory.scrollTop = aiHistory.scrollHeight;
  const answerEl = pending.querySelector(".ai-a");

  try {
    const res = await fetch(`/api/jobs/${jobId}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
    });
    if (!res.ok || !res.body) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.detail || "Errore nella richiesta all'AI");
    }

    const via = decodeURIComponent(res.headers.get("X-LLM-Provider") || "");
    if (via) {
      const tag = document.createElement("div");
      tag.className = "ai-via";
      tag.textContent = `via ${via}`;
      pending.appendChild(tag);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let answer = "";
    let firstChunk = true;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      answer += decoder.decode(value, { stream: true });
      if (firstChunk) { answerEl.classList.remove("ai-pending"); firstChunk = false; }
      answerEl.textContent = answer;
      aiHistory.scrollTop = aiHistory.scrollHeight;
    }
    answer += decoder.decode(); // flush any trailing bytes
    answerEl.textContent = answer;

    if (!answer.trim()) {
      answerEl.innerHTML = `<span class="ai-error">Il modello non ha restituito testo.</span>`;
    } else {
      // Keep local state in sync so re-renders (poll/reopen) show this exchange.
      const entry = { prompt, answer, created_at: Date.now() / 1000, provider: via || undefined };
      currentJob.ai_messages = [...(currentJob.ai_messages || []), entry];
      aiPrompt.value = "";
    }
  } catch (err) {
    answerEl.classList.remove("ai-pending");
    answerEl.innerHTML = `<span class="ai-error">❌ ${escapeHtml(err.message)}</span>`;
  } finally {
    aiSendBtn.disabled = false;
    aiPrompt.disabled = false;
    aiSendBtn.textContent = original;
  }
}

aiSendBtn.addEventListener("click", () => sendAiPrompt(aiPrompt.value));
aiPrompt.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    sendAiPrompt(aiPrompt.value);
  }
});
document.querySelectorAll(".ai-preset").forEach((btn) =>
  btn.addEventListener("click", () => sendAiPrompt(btn.dataset.prompt))
);
aiOpenSettings.addEventListener("click", openSettings);

// ---- Settings modal (LLM providers with priority fallback) ----
const PROVIDER_TYPES = {
  ollama: { label: "Ollama", url: "http://host.docker.internal:11434", key: false },
  openai: { label: "API OpenAI-compatibile", url: "https://api.openai.com/v1", key: true },
};
// Default chain for the vision role: the bundled llama.cpp service.
const VISION_DEFAULT = {
  name: "llama.cpp locale", type: "openai", base_url: "http://llama:8080",
  model: "Qwen3-VL-2B-Instruct-GGUF:Q4_K_M", enabled: true,
};

const roles = {
  llm: {
    list: $("llmProviders"), add: $("llmAddProvider"), enabled: llmEnabled,
    fallback: llmFallback, temperature: llmTemperature, providers: [],
  },
  vision: {
    list: $("visionProviders"), add: $("visionAddProvider"), enabled: visionEnabled,
    fallback: visionFallback, temperature: visionTemperature, providers: [],
  },
};

function newProviderId() {
  return Math.random().toString(16).slice(2, 10);
}

function renderProviders(role) {
  const state = roles[role];
  state.list.innerHTML = "";
  if (!state.providers.length) {
    const emptyMsg = document.createElement("p");
    emptyMsg.className = "provider-empty";
    emptyMsg.textContent = "Nessun provider configurato.";
    state.list.appendChild(emptyMsg);
  }
  state.providers.forEach((prov, index) => {
    state.list.appendChild(providerCard(role, prov, index));
  });
}

function providerCard(role, prov, index) {
  const state = roles[role];
  const card = document.createElement("div");
  card.className = "provider-card";
  card.classList.toggle("disabled", !prov.enabled);
  card.innerHTML = `
    <div class="pc-head">
      <span class="pc-priority" title="Priorità: viene provato per ${index + 1}°">#${index + 1}</span>
      <input class="pc-name" type="text" placeholder="Nome (opzionale)" maxlength="60" />
      <label class="pc-enabled" title="Usa questo provider"><input type="checkbox" /> attivo</label>
      <button type="button" class="pc-btn pc-up" title="Aumenta priorità">↑</button>
      <button type="button" class="pc-btn pc-down" title="Diminuisci priorità">↓</button>
      <button type="button" class="pc-btn pc-del" title="Rimuovi">✕</button>
    </div>
    <div class="pc-grid">
      <label class="field"><span>Tipo</span>
        <select class="pc-type">
          <option value="ollama">Ollama</option>
          <option value="openai">API OpenAI-compatibile</option>
        </select>
      </label>
      <label class="field"><span>URL</span>
        <input class="pc-url" type="text" autocomplete="off" spellcheck="false" />
      </label>
      <label class="field"><span>Modello</span>
        <input class="pc-model" type="text" autocomplete="off" spellcheck="false"
               placeholder="es. llama3.1:8b, gpt-4o-mini" />
      </label>
      <label class="field pc-key-field"><span>API key</span>
        <input class="pc-key" type="password" autocomplete="new-password" spellcheck="false" />
      </label>
    </div>
    <div class="pc-foot">
      <button type="button" class="btn-ghost pc-test">🔌 Prova connessione</button>
      <span class="pc-status"></span>
      <button type="button" class="link pc-clear-key hidden">rimuovi chiave salvata</button>
    </div>`;

  const q = (sel) => card.querySelector(sel);
  const name = q(".pc-name");
  const enabled = q(".pc-enabled input");
  const type = q(".pc-type");
  const url = q(".pc-url");
  const model = q(".pc-model");
  const key = q(".pc-key");
  const keyField = q(".pc-key-field");
  const status = q(".pc-status");
  const clearKey = q(".pc-clear-key");

  const datalist = document.createElement("datalist");
  datalist.id = `models-${role}-${prov.id}`;
  model.setAttribute("list", datalist.id);
  card.appendChild(datalist);

  name.value = prov.name || "";
  enabled.checked = prov.enabled !== false;
  type.value = prov.type || "ollama";
  url.value = prov.base_url || "";
  model.value = prov.model || "";
  key.value = prov.api_key || "";

  const syncType = () => {
    const meta = PROVIDER_TYPES[type.value];
    url.placeholder = meta.url;
    keyField.classList.toggle("hidden", !meta.key);
    key.placeholder = prov.has_api_key && !prov.clear_api_key
      ? "•••••• salvata (lascia vuoto per mantenerla)"
      : "facoltativa per server locali";
    clearKey.classList.toggle("hidden", !meta.key || !prov.has_api_key || !!prov.clear_api_key);
  };
  syncType();

  name.addEventListener("input", () => { prov.name = name.value; });
  enabled.addEventListener("change", () => {
    prov.enabled = enabled.checked;
    card.classList.toggle("disabled", !prov.enabled);
  });
  type.addEventListener("change", () => {
    const previous = PROVIDER_TYPES[prov.type || "ollama"];
    // Swap the suggested URL when it was left at the other type's default.
    if (!url.value.trim() || url.value.trim() === previous.url) {
      url.value = PROVIDER_TYPES[type.value].url;
      prov.base_url = url.value;
    }
    prov.type = type.value;
    syncType();
  });
  url.addEventListener("input", () => { prov.base_url = url.value.trim(); });
  model.addEventListener("input", () => { prov.model = model.value.trim(); });
  key.addEventListener("input", () => { prov.api_key = key.value.trim(); });
  clearKey.addEventListener("click", () => {
    prov.clear_api_key = true;
    prov.api_key = "";
    key.value = "";
    syncType();
    status.textContent = "La chiave verrà rimossa al salvataggio.";
  });

  q(".pc-up").disabled = index === 0;
  q(".pc-down").disabled = index === state.providers.length - 1;
  q(".pc-up").addEventListener("click", () => moveProvider(role, index, -1));
  q(".pc-down").addEventListener("click", () => moveProvider(role, index, 1));
  q(".pc-del").addEventListener("click", () => {
    state.providers.splice(index, 1);
    renderProviders(role);
  });

  q(".pc-test").addEventListener("click", async (e) => {
    const btn = e.currentTarget;
    btn.disabled = true;
    status.className = "pc-status";
    status.textContent = "Verifica…";
    try {
      const res = await fetch("/api/llm/test", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ role, provider: providerPayload(prov) }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || "Verifica non riuscita");
      status.textContent = (data.ok ? "✅ " : "❌ ") + data.message;
      status.classList.add(data.ok ? "ok" : "err");
      datalist.innerHTML = "";
      for (const m of data.models || []) datalist.appendChild(new Option(m, m));
      if (data.models && data.models.length && !model.value) model.focus();
    } catch (err) {
      status.textContent = "❌ " + err.message;
      status.classList.add("err");
    } finally {
      btn.disabled = false;
    }
  });
  return card;
}

function moveProvider(role, index, delta) {
  const list = roles[role].providers;
  const target = index + delta;
  if (target < 0 || target >= list.length) return;
  [list[index], list[target]] = [list[target], list[index]];
  renderProviders(role);
}

function providerPayload(prov) {
  return {
    id: prov.id,
    name: (prov.name || "").trim(),
    type: prov.type || "ollama",
    base_url: (prov.base_url || "").trim(),
    model: (prov.model || "").trim(),
    api_key: prov.api_key || "",
    clear_api_key: !!prov.clear_api_key,
    enabled: prov.enabled !== false,
  };
}

function applyRoleConfig(role, cfg) {
  const state = roles[role];
  state.enabled.checked = !!cfg.enabled;
  state.fallback.checked = cfg.fallback !== false;
  state.temperature.value = cfg.temperature != null ? cfg.temperature : 0.3;
  state.providers = (cfg.providers || []).map((p) => ({ ...p, api_key: "" }));
  if (!state.providers.length && role === "vision") {
    state.providers.push({ ...VISION_DEFAULT, id: newProviderId() });
  }
  renderProviders(role);
}

for (const role of Object.keys(roles)) {
  roles[role].add.addEventListener("click", () => {
    roles[role].providers.push({
      id: newProviderId(), name: "", type: "ollama",
      base_url: PROVIDER_TYPES.ollama.url, model: "", enabled: true,
    });
    renderProviders(role);
    const cards = roles[role].list.querySelectorAll(".provider-card");
    const last = cards[cards.length - 1];
    if (last) last.querySelector(".pc-model").focus();
  });
}

function refreshConfigHints() {
  if (!aiPanel.classList.contains("hidden")) aiConfigHint.classList.toggle("hidden", llmReady);
  if (currentJob && currentJob.has_video) visionConfigHint.classList.toggle("hidden", visionReady);
}

async function loadConfig() {
  try {
    const [textRes, visionRes] = await Promise.all([
      fetch("/api/config"),
      fetch("/api/vision-config"),
    ]);
    if (textRes.ok) {
      const cfg = await textRes.json();
      applyRoleConfig("llm", cfg);
      llmReady = !!cfg.ready;
    }
    if (visionRes.ok) {
      const cfg = await visionRes.json();
      applyRoleConfig("vision", cfg);
      visionReady = !!cfg.ready;
    }
    refreshConfigHints();
  } catch (_) { /* keep defaults */ }
}

function openSettings() {
  settingsStatus.textContent = "";
  visionSettingsStatus.textContent = "";
  retentionStatus.textContent = "";
  settingsModal.classList.remove("hidden");
  loadConfig();
  loadRetention();
}
function closeSettings() {
  settingsModal.classList.add("hidden");
}

settingsBtn.addEventListener("click", openSettings);
settingsClose.addEventListener("click", closeSettings);
settingsModal.addEventListener("click", (e) => {
  if (e.target === settingsModal) closeSettings();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !settingsModal.classList.contains("hidden")) closeSettings();
});

async function saveRole(role, button, statusEl, endpoint) {
  const state = roles[role];
  button.disabled = true;
  statusEl.textContent = "Salvataggio…";
  try {
    const res = await fetch(endpoint, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        enabled: state.enabled.checked,
        fallback: state.fallback.checked,
        temperature: parseFloat(state.temperature.value) || 0.3,
        providers: state.providers.map(providerPayload),
      }),
    });
    const cfg = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(cfg.detail || "Errore nel salvataggio");
    applyRoleConfig(role, cfg);
    if (role === "llm") llmReady = !!cfg.ready;
    else visionReady = !!cfg.ready;
    refreshConfigHints();
    statusEl.textContent = cfg.active ? `✅ Salvato · primo provider: ${cfg.active}` : "✅ Salvato";
  } catch (err) {
    statusEl.textContent = "❌ " + err.message;
  } finally {
    button.disabled = false;
  }
}

settingsSave.addEventListener("click", () =>
  saveRole("llm", settingsSave, settingsStatus, "/api/config"));
visionSettingsSave.addEventListener("click", () =>
  saveRole("vision", visionSettingsSave, visionSettingsStatus, "/api/vision-config"));

visionOpenSettings.addEventListener("click", openSettings);

// ---- Views: transcription / archive ----
let archiveItems = [];
let archiveCategory = null; // null = category overview
const NO_CATEGORY = "__none__";

function showView(name) {
  const archive = name === "archive";
  transcribeView.classList.toggle("hidden", archive);
  archiveView.classList.toggle("hidden", !archive);
  navTranscribe.classList.toggle("active", !archive);
  navArchive.classList.toggle("active", archive);
  if (archive) loadArchive();
}

navTranscribe.addEventListener("click", () => showView("transcribe"));
navArchive.addEventListener("click", () => {
  archiveCategory = null;
  archiveSearch.value = "";
  showView("archive");
});

async function loadArchive() {
  const res = await fetch("/api/archive").catch(() => null);
  if (!res || !res.ok) return;
  archiveItems = await res.json();
  renderArchive();
}

function categoryKey(item) {
  return item.category ? item.category : NO_CATEGORY;
}
function categoryName(key) {
  return key === NO_CATEGORY ? "Senza categoria" : key;
}
function categoryColor(key) {
  let h = 0;
  for (const ch of key) h = (h * 31 + ch.codePointAt(0)) >>> 0;
  return SPEAKER_COLORS[h % SPEAKER_COLORS.length];
}

function renderArchive() {
  const query = archiveSearch.value.trim().toLowerCase();
  archiveGrid.innerHTML = "";
  archiveEmpty.classList.toggle("hidden", archiveItems.length > 0);

  // Breadcrumbs
  archiveCrumbs.innerHTML = "";
  const root = document.createElement("button");
  root.type = "button";
  root.className = "crumb";
  root.textContent = "🗂️ Archivio";
  root.addEventListener("click", () => { archiveCategory = null; renderArchive(); });
  archiveCrumbs.appendChild(root);
  if (archiveCategory !== null) {
    const sep = document.createElement("span");
    sep.className = "crumb-sep";
    sep.textContent = "›";
    const current = document.createElement("span");
    current.className = "crumb current";
    current.textContent = categoryName(archiveCategory);
    archiveCrumbs.append(sep, current);
  }

  // Search spans every category; otherwise show either categories or one category.
  if (query) {
    const hits = archiveItems.filter((it) => it.filename.toLowerCase().includes(query) &&
      (archiveCategory === null || categoryKey(it) === archiveCategory));
    archiveSummary.textContent = `${hits.length} risultat${hits.length === 1 ? "o" : "i"} per «${archiveSearch.value.trim()}»`;
    archiveGrid.className = "archive-grid records";
    hits.forEach((it) => archiveGrid.appendChild(recordCard(it, true)));
    return;
  }

  if (archiveCategory === null) {
    const groups = new Map();
    for (const it of archiveItems) {
      const key = categoryKey(it);
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(it);
    }
    const keys = [...groups.keys()].sort((a, b) => {
      if (a === NO_CATEGORY) return 1;
      if (b === NO_CATEGORY) return -1;
      return a.localeCompare(b, "it", { sensitivity: "base" });
    });
    const total = archiveItems.length;
    archiveSummary.textContent = total
      ? `${total} registrazion${total === 1 ? "e" : "i"} in ${keys.length} categori${keys.length === 1 ? "a" : "e"}`
      : "";
    archiveGrid.className = "archive-grid categories";
    for (const key of keys) archiveGrid.appendChild(categoryCard(key, groups.get(key)));
    return;
  }

  const items = archiveItems.filter((it) => categoryKey(it) === archiveCategory);
  const duration = items.reduce((sum, it) => sum + (it.duration || 0), 0);
  archiveSummary.textContent = `${items.length} registrazion${items.length === 1 ? "e" : "i"} · ${formatLongDuration(duration)} totali`;
  archiveGrid.className = "archive-grid records";
  items.forEach((it) => archiveGrid.appendChild(recordCard(it, false)));
  if (!items.length) archiveSummary.textContent = "Nessuna registrazione in questa categoria.";
}

function categoryCard(key, items) {
  const color = categoryColor(key);
  const duration = items.reduce((sum, it) => sum + (it.duration || 0), 0);
  const latest = Math.max(...items.map((it) => it.created_at));
  const card = document.createElement("button");
  card.type = "button";
  card.className = "cat-card";
  card.style.setProperty("--cat", color);
  card.innerHTML = `
    <div class="cat-icon">${key === NO_CATEGORY ? "📂" : "🏷️"}</div>
    <div class="cat-name"></div>
    <div class="cat-count">${items.length} registrazion${items.length === 1 ? "e" : "i"}</div>
    <div class="cat-meta">⏱️ ${formatLongDuration(duration)} · ultima ${formatDate(latest)}</div>
    <ul class="cat-preview"></ul>`;
  card.querySelector(".cat-name").textContent = categoryName(key);
  const preview = card.querySelector(".cat-preview");
  for (const it of items.slice(0, 3)) {
    const li = document.createElement("li");
    li.textContent = it.filename;
    preview.appendChild(li);
  }
  card.addEventListener("click", () => {
    archiveCategory = key;
    renderArchive();
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
  return card;
}

function recordCard(item, showCategory) {
  const card = document.createElement("article");
  card.className = "rec-card";
  card.tabIndex = 0;
  card.innerHTML = `
    <div class="rec-top">
      <span class="rec-icon">${item.source_url ? "▶️" : item.has_video ? "🎬" : "🎧"}</span>
      <div class="rec-title-wrap">
        <h3 class="rec-title"></h3>
        <div class="rec-date">${formatDate(item.created_at)}</div>
      </div>
    </div>
    <div class="rec-chips"></div>
    <div class="rec-actions">
      <button type="button" class="dl rec-open">Apri</button>
      <button type="button" class="dl rec-info">ℹ️ Dettagli</button>
      <span class="rec-dls"></span>
    </div>`;
  card.querySelector(".rec-title").textContent = item.filename;
  const chips = card.querySelector(".rec-chips");
  if (showCategory) chips.appendChild(chip(`🏷️ ${categoryName(categoryKey(item))}`));
  if (item.duration) chips.appendChild(chip(`⏱️ ${formatLongDuration(item.duration)}`));
  if (item.detected_language) chips.appendChild(chip(`🌐 ${item.detected_language.toUpperCase()}`));
  if (item.speaker_count) chips.appendChild(chip(`👥 ${item.speaker_count}`));
  if (item.whisper_model) chips.appendChild(chip(`🧠 ${item.whisper_model}`));
  if (!item.media_available) chips.appendChild(chip("🗑️ file originale eliminato", "err"));
  const dls = card.querySelector(".rec-dls");
  for (const [fmt, url] of Object.entries(item.formats || {})) {
    const a = document.createElement("a");
    a.className = "dl dl-small";
    a.href = url;
    a.textContent = fmt.toUpperCase();
    a.addEventListener("click", (e) => e.stopPropagation());
    dls.appendChild(a);
  }
  card.querySelector(".rec-open").addEventListener("click", (e) => {
    e.stopPropagation();
    openFromArchive(item.id);
  });
  card.querySelector(".rec-info").addEventListener("click", (e) => {
    e.stopPropagation();
    openDetails(item.id);
  });
  card.addEventListener("click", () => openDetails(item.id));
  card.addEventListener("keydown", (e) => {
    if (e.key === "Enter") openDetails(item.id);
  });
  return card;
}

function openFromArchive(jobId) {
  closeDetails();
  showView("transcribe");
  openJob(jobId);
  window.scrollTo({ top: 0, behavior: "smooth" });
}

archiveSearch.addEventListener("input", renderArchive);

// ---- Recording details ----
let detailsJobId = null;

async function openDetails(jobId) {
  detailsJobId = jobId;
  detailsTitle.textContent = "Dettagli";
  detailsBody.innerHTML = `<p class="stage-label">Caricamento…</p>`;
  detailsModal.classList.remove("hidden");
  const res = await fetch(`/api/jobs/${jobId}/details`).catch(() => null);
  if (detailsJobId !== jobId) return;
  if (!res || !res.ok) {
    detailsBody.innerHTML = `<p class="stage-label">❌ Dettagli non disponibili.</p>`;
    return;
  }
  renderDetails(await res.json());
}

function closeDetails() {
  detailsModal.classList.add("hidden");
  detailsJobId = null;
}

function renderDetails(d) {
  detailsTitle.textContent = d.filename;
  const lang = (code) => (code ? code.toUpperCase() : null);
  const yesNo = (v) => (v ? "Sì" : "No");
  const statusText = {
    queued: "In coda", processing: "In elaborazione", completed: "Completato",
    failed: "Errore", cancelled: "Annullato",
  }[d.status] || d.status;
  const visual = {
    not_started: "Non eseguita", queued: "In coda", processing: "In corso",
    completed: `Completata (${{ base: "base", medium: "medio", advanced: "avanzato" }[d.visual_detail] || d.visual_detail})`,
    failed: "Non riuscita",
  }[d.visual_status] || d.visual_status;

  const sections = [
    ["📄 Generale", [
      ["Titolo", d.filename],
      ["Stato", statusText + (d.error ? ` — ${d.error}` : "")],
      ["Categoria", d.category || "Senza categoria"],
      ["Creato il", formatDate(d.created_at)],
      ["ID", d.id, "mono"],
    ]],
    ["💾 File", [
      ["Origine", d.source === "youtube" ? "YouTube" : "File caricato"],
      ["URL", d.source_url, "link"],
      ["File salvato", d.stored_file, "mono"],
      ["Dimensione", d.file_size != null ? formatSize(d.file_size) : null],
      ["File originale", d.media_available ? "Disponibile" : "Eliminato / non disponibile"],
      ["Traccia video", yesNo(d.has_video)],
    ]],
    ["📝 Trascrizione", [
      ["Durata", d.duration ? formatLongDuration(d.duration) : null],
      ["Lingua rilevata", lang(d.detected_language)],
      ["Lingua richiesta", lang(d.requested_language) || "Automatica"],
      ["Modello Whisper", d.whisper_model || d.requested_model],
      ["Parlanti richiesti", d.requested_speakers || "Automatico"],
      ["Parlanti rilevati", d.speaker_count],
      ["Frasi", d.segment_count],
      ["Parole", d.word_count],
    ]],
    ["🔎 Analisi", [
      ["Analisi scene", d.has_video ? visual : "—"],
      ["Domande all'AI", d.ai_message_count],
    ]],
  ];

  detailsBody.innerHTML = "";
  for (const [title, rows] of sections) {
    const section = document.createElement("section");
    section.className = "details-section";
    const h = document.createElement("h3");
    h.textContent = title;
    const dl = document.createElement("dl");
    for (const [label, value, kind] of rows) {
      if (value === null || value === undefined || value === "") continue;
      const dt = document.createElement("dt");
      dt.textContent = label;
      const dd = document.createElement("dd");
      if (kind === "link") {
        const a = document.createElement("a");
        a.href = value;
        a.target = "_blank";
        a.rel = "noopener noreferrer";
        a.textContent = value;
        dd.appendChild(a);
      } else {
        dd.textContent = String(value);
        if (kind === "mono") dd.classList.add("mono");
      }
      dl.append(dt, dd);
    }
    section.append(h, dl);
    detailsBody.appendChild(section);
  }

  if (d.speakers && d.speakers.length) {
    const section = document.createElement("section");
    section.className = "details-section";
    section.innerHTML = `<h3>👥 Parlanti</h3>`;
    const table = document.createElement("table");
    table.className = "details-table";
    table.innerHTML = "<thead><tr><th></th><th>Voce</th><th>Nome</th><th>Frasi</th><th>Tempo</th></tr></thead>";
    const tbody = document.createElement("tbody");
    const total = d.speakers.reduce((sum, sp) => sum + sp.seconds, 0) || 1;
    d.speakers.forEach((sp, i) => {
      const tr = document.createElement("tr");
      const pct = Math.round((sp.seconds / total) * 100);
      tr.innerHTML = `
        <td><span class="dot" style="background:${SPEAKER_COLORS[i % SPEAKER_COLORS.length]}"></span></td>
        <td class="mono"></td><td></td><td>${sp.segments}</td>
        <td>${formatLongDuration(sp.seconds)} <span class="muted">(${pct}%)</span></td>`;
      tr.children[1].textContent = sp.label;
      tr.children[2].textContent = sp.name || "—";
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    section.appendChild(table);
    detailsBody.appendChild(section);
  }
}

detailsClose.addEventListener("click", closeDetails);
detailsModal.addEventListener("click", (e) => {
  if (e.target === detailsModal) closeDetails();
});
detailsOpenBtn.addEventListener("click", () => {
  if (detailsJobId) openFromArchive(detailsJobId);
});
jobDetailsBtn.addEventListener("click", () => {
  if (currentJob) openDetails(currentJob.id);
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !detailsModal.classList.contains("hidden")) closeDetails();
});

// ---- Retention settings ----
function showRetention(data) {
  retentionMedia.value = data.media_days;
  retentionJobs.value = data.job_days;
  retentionUsage.textContent = `Spazio occupato dai file caricati: ${formatSize(data.uploads_bytes || 0)}`;
}

async function loadRetention() {
  const res = await fetch("/api/retention").catch(() => null);
  if (res && res.ok) showRetention(await res.json());
}

retentionSave.addEventListener("click", async () => {
  retentionSave.disabled = true;
  retentionStatus.textContent = "Salvataggio…";
  try {
    const res = await fetch("/api/retention", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        media_days: parseInt(retentionMedia.value, 10) || 0,
        job_days: parseInt(retentionJobs.value, 10) || 0,
      }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || "Errore nel salvataggio");
    showRetention(data);
    retentionStatus.textContent = "✅ Salvato";
  } catch (err) {
    retentionStatus.textContent = "❌ " + err.message;
  } finally {
    retentionSave.disabled = false;
  }
});

cleanupNowBtn.addEventListener("click", async () => {
  if (!confirm("Applicare ora la pulizia con le impostazioni salvate?")) return;
  cleanupNowBtn.disabled = true;
  retentionStatus.textContent = "Pulizia in corso…";
  try {
    const res = await fetch("/api/retention/cleanup", { method: "POST" });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || "Errore durante la pulizia");
    showRetention(data);
    retentionStatus.textContent =
      `✅ ${data.deleted_jobs} job eliminati, ${data.deleted_media} file rimossi (${formatSize(data.freed_bytes)})`;
    loadHistory();
  } catch (err) {
    retentionStatus.textContent = "❌ " + err.message;
  } finally {
    cleanupNowBtn.disabled = false;
  }
});

// ---- Init ----
loadHistory();
loadCategories();
loadConfig();
loadModels();

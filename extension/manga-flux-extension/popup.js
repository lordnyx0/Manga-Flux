const $ = (id) => document.getElementById(id);

// DOM refs
const apiBaseInput = $('apiBase');
const apiTokenInput = $('apiToken');
const themeSelect = $('themeSelect');
const mangaIdInput = $('mangaId');
const chapterIdInput = $('chapterId');
const styleReferenceUrlInput = $('styleReferenceUrl');
const styleReferenceFileInput = $('styleReferenceFile');
const styleReferenceHintEl = $('styleReferenceHint');
const pageImagesHintEl = $('pageImagesHint');
const engineInput = $('engine');
const strengthInput = $('strength');
const strengthDisplay = $('strengthDisplay');
const outputRootInput = $('outputRoot');
const metaPathInput = $('metaPath');
const outputDirInput = $('outputDir');
const metadataDirInput = $('metadataDir');
const batchOutputDirInput = $('batchOutputDir');
const expectedPagesInput = $('expectedPages');

const saveBtn = $('saveBtn');
const healthBtn = $('healthBtn');
const captureImagesBtn = $('captureImagesBtn');
const clearImagesBtn = $('clearImagesBtn');
const runChapterBtn = $('runChapterBtn');
const runBtn = $('runBtn');
const batchBtn = $('batchBtn');
const clearHistoryBtn = $('clearHistoryBtn');
const settingsToggle = $('settingsToggle');
const settingsClose = $('settingsClose');
const settingsPanel = $('settingsPanel');

const statusEl = $('status');
const outputEl = $('output');
const historyList = $('historyList');
const thumbGrid = $('thumbGrid');
const imagesCount = $('imagesCount');

const HISTORY_KEY = 'history';
const IMAGES_KEY = 'chapterPageItems';
const MIN_CAPTURE_WIDTH = 320;
const MIN_CAPTURE_HEIGHT = 320;
const THUMB_SIZE = 120; // px for preview thumbnails

let chapterPageItems = [];
let styleReferenceUpload = null;
let captureSourceUrl = '';
let captureCookieHeader = '';

// ---------------------------------------------------------------------------
// Theme
// ---------------------------------------------------------------------------

function applyTheme(themeMode) {
  let mode = themeMode;
  if (themeMode === 'auto') {
    mode = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  document.documentElement.setAttribute('data-theme', mode);
}

// ---------------------------------------------------------------------------
// Status bar
// ---------------------------------------------------------------------------

function setStatus(ok, text) {
  statusEl.className = 'status-bar ' + (ok ? 'ok' : 'err');
  statusEl.textContent = ok ? `✓ ${text}` : `✗ ${text}`;
}

function setStatusLoading(text) {
  statusEl.className = 'status-bar';
  statusEl.textContent = `⏳ ${text}`;
}

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------

document.querySelectorAll('.tab-btn').forEach((btn) => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach((b) => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach((p) => p.classList.remove('active'));
    btn.classList.add('active');
    $('tab-' + btn.dataset.tab).classList.add('active');
  });
});

// ---------------------------------------------------------------------------
// Settings panel
// ---------------------------------------------------------------------------

settingsToggle.addEventListener('click', () => settingsPanel.classList.remove('hidden'));
settingsClose.addEventListener('click', () => settingsPanel.classList.add('hidden'));

// ---------------------------------------------------------------------------
// Strength slider label
// ---------------------------------------------------------------------------

strengthInput.addEventListener('input', () => {
  strengthDisplay.textContent = Number(strengthInput.value).toFixed(2);
});

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------

function getHeaders() {
  const headers = { 'Content-Type': 'application/json' };
  const token = apiTokenInput.value.trim();
  if (token) headers['X-API-Token'] = token;
  return headers;
}

async function requestJSON(url, payload, actionName) {
  const response = await fetch(url, {
    method: 'POST',
    headers: getHeaders(),
    body: JSON.stringify(payload),
  });
  const body = await response.json();
  const ok = response.ok;
  setStatus(ok, ok ? `${actionName} concluído.` : `${actionName} falhou (HTTP ${response.status}).`);
  outputEl.textContent = JSON.stringify(body, null, 2);
  // Switch to log tab on completion
  document.querySelectorAll('.tab-btn').forEach((b) => b.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach((p) => p.classList.remove('active'));
  document.querySelector('[data-tab="log"]').classList.add('active');
  $('tab-log').classList.add('active');
  await pushHistory({
    ts: new Date().toISOString(),
    action: actionName,
    status: ok ? 'ok' : `http_${response.status}`,
  });
}

// ---------------------------------------------------------------------------
// Image fetching — fetch executed INSIDE the tab via executeScript
// ---------------------------------------------------------------------------
//
// fetch() in the extension popup sends "Origin: chrome-extension://..."
// which Cloudflare / WordPress hotlink protection blocks with 403.
// executeScript runs in the page context → same origin → no suspicious
// Origin header → cookies already present → identical to normal browser load.

/**
 * Fetches a thumbnail preview of imgUrl inside the tab and returns a data URL.
 * Used at capture time so the popup can show real previews.
 */
async function fetchThumbInTab(tabId, url) {
  const results = await chrome.scripting.executeScript({
    target: { tabId },
    args: [url, THUMB_SIZE],
    func: async (imgUrl, maxSize) => {
      try {
        const resp = await fetch(imgUrl, {
          credentials: 'include',
          headers: { 'Accept': 'image/avif,image/webp,image/apng,image/*,*/*;q=0.8' },
        });
        if (!resp.ok) return { error: `HTTP ${resp.status}` };
        const blob = await resp.blob();

        // Decode into an ImageBitmap and draw at thumbnail size
        const bitmap = await createImageBitmap(blob);
        const ratio = Math.min(maxSize / bitmap.width, maxSize / bitmap.height, 1);
        const w = Math.round(bitmap.width * ratio);
        const h = Math.round(bitmap.height * ratio);
        const canvas = new OffscreenCanvas(w, h);
        canvas.getContext('2d').drawImage(bitmap, 0, 0, w, h);
        const thumbBlob = await canvas.convertToBlob({ type: 'image/jpeg', quality: 0.65 });

        const ab = await thumbBlob.arrayBuffer();
        const bytes = new Uint8Array(ab);
        let binary = '';
        const CHUNK = 8192;
        for (let i = 0; i < bytes.length; i += CHUNK) {
          binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
        }
        return { dataUrl: 'data:image/jpeg;base64,' + btoa(binary) };
      } catch (e) {
        return { error: String(e) };
      }
    },
  });

  const r = results?.[0]?.result;
  if (r?.dataUrl) return r.dataUrl;
  return null; // preview unavailable, show placeholder
}

/**
 * Fetches a full image inside the tab and returns {content_base64, filename_ext}.
 */
async function fetchImageInTab(tabId, url, referer) {
  const results = await chrome.scripting.executeScript({
    target: { tabId },
    args: [url, referer || ''],
    func: async (imgUrl, imgReferer) => {
      try {
        const headers = { 'Accept': 'image/avif,image/webp,image/apng,image/*,*/*;q=0.8' };
        if (imgReferer) headers['Referer'] = imgReferer;
        const resp = await fetch(imgUrl, { credentials: 'include', headers });
        if (!resp.ok) return { error: `HTTP ${resp.status}` };

        const ct = resp.headers.get('content-type') || '';
        const ext = ct.includes('jpeg') || ct.includes('jpg') ? 'jpg'
          : ct.includes('png') ? 'png'
            : ct.includes('webp') ? 'webp'
              : (imgUrl.split('?')[0].split('.').pop().toLowerCase().replace(/[^a-z]/g, '') || 'jpg');

        const buffer = await resp.arrayBuffer();
        const bytes = new Uint8Array(buffer);
        let binary = '';
        const CHUNK = 8192;
        for (let i = 0; i < bytes.length; i += CHUNK) {
          binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
        }
        return { base64: btoa(binary), ext };
      } catch (e) {
        return { error: String(e) };
      }
    },
  });

  const result = results?.[0]?.result;
  if (!result) throw new Error(`executeScript sem resultado para: ${url}`);
  if (result.error) throw new Error(`${result.error} — ${url}`);
  if (!result.base64) throw new Error(`base64 vazio para: ${url}`);
  return { content_base64: result.base64, filename_ext: result.ext || 'jpg' };
}

/**
 * Downloads all chapter images inside the tab and returns page_uploads[].
 */
async function buildPageUploads(tabId, items, referer, onProgress) {
  const uploads = [];
  for (let i = 0; i < items.length; i++) {
    onProgress(i, items.length);
    const item = items[i];
    const padded = String(i + 1).padStart(3, '0');

    if (item.source === 'upload' && item.content_base64) {
      uploads.push({ filename: item.filename || `page_${padded}.png`, content_base64: item.content_base64 });
    } else if (item.source === 'url' && item.url) {
      const { content_base64, filename_ext } = await fetchImageInTab(tabId, item.url, referer);
      uploads.push({ filename: `page_${padded}.${filename_ext}`, content_base64 });
    } else {
      throw new Error(`Item ${i + 1} inválido: source="${item.source}"`);
    }
  }
  onProgress(items.length, items.length);
  return uploads;
}

// ---------------------------------------------------------------------------
// Cookie helper
// ---------------------------------------------------------------------------

async function buildCookieHeaderForUrl(url) {
  try {
    if (!url || !/^https?:\/\//.test(url)) return '';
    const cookies = await chrome.cookies.getAll({ url });
    if (!Array.isArray(cookies) || !cookies.length) return '';
    return cookies
      .filter((c) => c && c.name && c.value !== undefined)
      .map((c) => `${c.name}=${c.value}`)
      .join('; ');
  } catch (_) { return ''; }
}

// ---------------------------------------------------------------------------
// Thumbnails
// ---------------------------------------------------------------------------

function renderThumbnails() {
  thumbGrid.innerHTML = '';
  imagesCount.textContent = `${chapterPageItems.length} imagens`;

  chapterPageItems.forEach((itemData, idx) => {
    const item = document.createElement('div');
    item.className = 'thumb-item';

    const img = document.createElement('img');
    const src = itemData.thumbDataUrl || itemData.previewUrl || itemData.url || '';

    if (src.startsWith('data:')) {
      // Already a fetched thumbnail — display directly
      img.src = src;
    } else if (src) {
      // External URL — try loading, fallback to placeholder
      img.src = src;
      img.referrerPolicy = 'no-referrer';
      img.addEventListener('error', () => { img.src = makePlaceholder(idx + 1); }, { once: true });
    } else {
      img.src = makePlaceholder(idx + 1);
    }

    img.alt = `page_${idx + 1}`;
    img.loading = 'lazy';

    const num = document.createElement('span');
    num.className = 'thumb-num';
    num.textContent = idx + 1;

    const removeBtn = document.createElement('button');
    removeBtn.className = 'thumb-remove';
    removeBtn.textContent = '✕';
    removeBtn.title = 'Remover';
    removeBtn.addEventListener('click', async (e) => {
      e.stopPropagation();
      chapterPageItems = chapterPageItems.filter((_, i) => i !== idx);
      await chrome.storage.local.set({ [IMAGES_KEY]: chapterPageItems });
      renderThumbnails();
      await saveSettings();
    });

    item.appendChild(img);
    item.appendChild(num);
    item.appendChild(removeBtn);
    thumbGrid.appendChild(item);
  });
}

function makePlaceholder(n) {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="80" height="120">
    <rect width="100%" height="100%" fill="#1c1c21"/>
    <text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle"
      font-size="18" font-family="monospace" fill="#2a2a32">${n}</text>
  </svg>`;
  return 'data:image/svg+xml;charset=UTF-8,' + encodeURIComponent(svg);
}

function normalizeStoredItems(rawItems) {
  if (!Array.isArray(rawItems)) return [];
  return rawItems.map((item) => {
    if (typeof item === 'string') return { source: 'url', url: item, previewUrl: item };
    if (!item || typeof item !== 'object') return null;
    if (item.source === 'url' && typeof item.url === 'string') return item;
    if (item.source === 'upload' && item.content_base64) return item;
    return null;
  }).filter(Boolean);
}

// ---------------------------------------------------------------------------
// Storage
// ---------------------------------------------------------------------------

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => { const o = String(reader.result || ''); resolve(o.includes(',') ? o.split(',', 2)[1] : o); };
    reader.onerror = () => reject(reader.error || new Error('Falha ao ler arquivo'));
    reader.readAsDataURL(file);
  });
}

async function pushHistory(entry) {
  const data = await chrome.storage.local.get([HISTORY_KEY]);
  const history = Array.isArray(data[HISTORY_KEY]) ? data[HISTORY_KEY] : [];
  history.unshift(entry);
  const trimmed = history.slice(0, 30);
  await chrome.storage.local.set({ [HISTORY_KEY]: trimmed });
  renderHistory(trimmed);
}

function renderHistory(history) {
  historyList.innerHTML = '';
  if (!history.length) { historyList.innerHTML = '<li style="color:var(--text-3);font-family:var(--font-mono);font-size:10px;padding:8px">(vazio)</li>'; return; }
  for (const item of history) {
    const li = document.createElement('li');
    const ts = item.ts ? item.ts.replace('T', ' ').slice(0, 19) : '';
    const ok = item.status === 'ok';
    li.innerHTML = `<span style="color:${ok ? 'var(--ok)' : 'var(--err)'}">${ok ? '✓' : '✗'}</span> ${ts} <span style="color:var(--text)">${item.action}</span>`;
    historyList.appendChild(li);
  }
}

async function saveSettings() {
  await chrome.storage.local.set({
    apiBase: apiBaseInput.value.trim(),
    apiToken: apiTokenInput.value.trim(),
    theme: themeSelect.value,
    mangaId: mangaIdInput.value.trim(),
    chapterId: chapterIdInput.value.trim(),
    styleReferenceUrl: styleReferenceUrlInput.value.trim(),
    engine: engineInput.value,
    strength: strengthInput.value,
    outputRoot: outputRootInput.value.trim(),
    metaPath: metaPathInput.value.trim(),
    outputDir: outputDirInput.value.trim(),
    metadataDir: metadataDirInput.value.trim(),
    batchOutputDir: batchOutputDirInput.value.trim(),
    expectedPages: expectedPagesInput.value,
    captureSourceUrl,
    styleReferenceUpload,
    [IMAGES_KEY]: chapterPageItems,
  });
}

async function loadSettings() {
  const keys = [
    'apiBase', 'apiToken', 'theme', 'mangaId', 'chapterId', 'styleReferenceUrl',
    'engine', 'strength', 'outputRoot', 'metaPath', 'outputDir', 'metadataDir',
    'batchOutputDir', 'expectedPages', HISTORY_KEY, IMAGES_KEY, 'chapterPageUrls', 'captureSourceUrl',
    'styleReferenceUpload',
  ];
  const data = await chrome.storage.local.get(keys);

  if (data.apiBase) apiBaseInput.value = data.apiBase;
  if (data.apiToken) apiTokenInput.value = data.apiToken;
  themeSelect.value = data.theme || 'auto';
  applyTheme(themeSelect.value);
  if (data.mangaId) mangaIdInput.value = data.mangaId;
  if (data.chapterId) chapterIdInput.value = data.chapterId;
  if (data.styleReferenceUrl) styleReferenceUrlInput.value = data.styleReferenceUrl;
  if (data.engine) engineInput.value = data.engine;
  if (data.strength !== undefined) { strengthInput.value = data.strength; strengthDisplay.textContent = Number(data.strength).toFixed(2); }
  if (data.outputRoot) outputRootInput.value = data.outputRoot;
  if (data.metaPath) metaPathInput.value = data.metaPath;
  if (data.outputDir) outputDirInput.value = data.outputDir;
  if (data.metadataDir) metadataDirInput.value = data.metadataDir;
  if (data.batchOutputDir) batchOutputDirInput.value = data.batchOutputDir;
  if (data.expectedPages !== undefined) expectedPagesInput.value = data.expectedPages;
  if (data.captureSourceUrl) captureSourceUrl = data.captureSourceUrl;
  if (data.styleReferenceUpload) {
    styleReferenceUpload = data.styleReferenceUpload;
    styleReferenceHintEl.textContent = `✓ ${styleReferenceUpload.filename}`;
  }

  const oldUrls = Array.isArray(data.chapterPageUrls)
    ? data.chapterPageUrls.map((url) => ({ source: 'url', url, previewUrl: url }))
    : [];
  chapterPageItems = normalizeStoredItems(data[IMAGES_KEY] || oldUrls);
  renderThumbnails();
  renderHistory(data[HISTORY_KEY] || []);
}

// ---------------------------------------------------------------------------
// Persist on change
// ---------------------------------------------------------------------------

[
  apiBaseInput, apiTokenInput, mangaIdInput, chapterIdInput, styleReferenceUrlInput,
  engineInput, strengthInput, outputRootInput, metaPathInput, outputDirInput,
  metadataDirInput, batchOutputDirInput, expectedPagesInput,
].forEach((el) => el.addEventListener('change', saveSettings));

// ---------------------------------------------------------------------------
// Style reference
// ---------------------------------------------------------------------------

styleReferenceFileInput.addEventListener('change', async () => {
  const file = styleReferenceFileInput.files?.[0];
  if (!file) { styleReferenceUpload = null; styleReferenceHintEl.textContent = 'Upload tem prioridade sobre URL.'; return; }
  try {
    const base64 = await fileToBase64(file);
    styleReferenceUpload = { filename: file.name, contentBase64: base64 };
    styleReferenceHintEl.textContent = `✓ ${file.name}`;
  } catch (error) {
    styleReferenceUpload = null;
    styleReferenceHintEl.textContent = 'Falha ao ler arquivo.';
    setStatus(false, 'Falha ao preparar upload da referência de estilo.');
  }
});

// ---------------------------------------------------------------------------
// Theme
// ---------------------------------------------------------------------------

themeSelect.addEventListener('change', async () => { applyTheme(themeSelect.value); await saveSettings(); });

// ---------------------------------------------------------------------------
// Save / Health
// ---------------------------------------------------------------------------

saveBtn.addEventListener('click', async () => { await saveSettings(); setStatus(true, 'Configurações salvas.'); settingsPanel.classList.add('hidden'); });

healthBtn.addEventListener('click', async () => {
  const apiBase = apiBaseInput.value.trim();
  try {
    setStatusLoading('Verificando API…');
    const response = await fetch(`${apiBase}/health`);
    const payload = await response.json();
    setStatus(response.ok, response.ok ? `API v${payload.version || '?'} — online` : `Erro HTTP ${response.status}`);
    outputEl.textContent = JSON.stringify(payload, null, 2);
    await pushHistory({ ts: new Date().toISOString(), action: 'GET /health', status: response.ok ? 'ok' : `http_${response.status}` });
  } catch (error) {
    setStatus(false, 'API inacessível — verifique se o servidor está rodando.');
    outputEl.textContent = String(error);
  }
});

// ---------------------------------------------------------------------------
// Capture images — fetch thumbnails inside the tab
// ---------------------------------------------------------------------------

captureImagesBtn.addEventListener('click', async () => {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab?.id) throw new Error('Aba ativa não encontrada');

    captureSourceUrl = typeof tab.url === 'string' ? tab.url : '';
    captureCookieHeader = await buildCookieHeaderForUrl(captureSourceUrl);

    setStatusLoading('Capturando imagens da aba…');

    // Step 1: collect image URLs from the DOM
    const injected = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      args: [MIN_CAPTURE_WIDTH, MIN_CAPTURE_HEIGHT],
      func: (minW, minH) =>
        Array.from(document.images)
          .map((img) => ({ url: img.currentSrc || img.src, w: img.naturalWidth || img.width || 0, h: img.naturalHeight || img.height || 0 }))
          .filter((item) => /^https?:\/\//.test(item.url) && item.w >= minW && item.h >= minH)
          .map((item) => item.url),
    });

    const uniqueUrls = [...new Set(injected?.[0]?.result || [])];
    if (!uniqueUrls.length) {
      setStatus(false, 'Nenhuma imagem encontrada nesta aba.');
      return;
    }

    // Step 2: build items with skeleton placeholders and render immediately
    chapterPageItems = uniqueUrls.map((url) => ({ source: 'url', url, previewUrl: url, thumbDataUrl: null }));
    await saveSettings();
    renderThumbnails();
    setStatusLoading(`Carregando previews (0/${uniqueUrls.length})…`);

    // Step 3: fetch thumbnails inside the tab one by one and update in place
    let loaded = 0;
    for (let i = 0; i < chapterPageItems.length; i++) {
      const thumbDataUrl = await fetchThumbInTab(tab.id, chapterPageItems[i].url);
      if (thumbDataUrl) {
        chapterPageItems[i].thumbDataUrl = thumbDataUrl;
        // Update just this thumbnail in the DOM without full re-render
        const thumbEls = thumbGrid.querySelectorAll('.thumb-item img');
        if (thumbEls[i]) thumbEls[i].src = thumbDataUrl;
      }
      loaded++;
      setStatusLoading(`Carregando previews (${loaded}/${uniqueUrls.length})…`);
    }

    await saveSettings();
    imagesCount.textContent = `${chapterPageItems.length} imagens`;
    pageImagesHintEl.textContent = `${uniqueUrls.length} pág. — ${new URL(captureSourceUrl).hostname}`;
    setStatus(true, `${uniqueUrls.length} imagens capturadas.`);
  } catch (error) {
    setStatus(false, `Captura falhou: ${error.message}`);
    outputEl.textContent = String(error);
  }
});

clearImagesBtn.addEventListener('click', async () => {
  chapterPageItems = [];
  captureSourceUrl = '';
  captureCookieHeader = '';
  await saveSettings();
  renderThumbnails();
  pageImagesHintEl.textContent = '';
  setStatus(true, 'Lista limpa.');
});

// ---------------------------------------------------------------------------
// Run Chapter
// ---------------------------------------------------------------------------

runChapterBtn.addEventListener('click', async () => {
  await saveSettings();

  if (!chapterPageItems.length) {
    setStatus(false, 'Capture as imagens primeiro.');
    return;
  }

  let tab;
  try {
    [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab?.id) throw new Error('Aba do manga precisa estar aberta e ativa.');
  } catch (error) {
    setStatus(false, String(error));
    return;
  }

  const referer = captureSourceUrl || tab.url || '';
  runChapterBtn.disabled = true;

  try {
    // Download full images inside the tab (same origin, session cookies)
    const page_uploads = await buildPageUploads(
      tab.id,
      chapterPageItems,
      referer,
      (done, tot) => setStatusLoading(`Baixando imagens via aba (${done}/${tot})…`),
    );

    setStatusLoading(`Enviando ${page_uploads.length} imagens para o servidor…`);

    // Generate a simple unique ID for this job
    const clientId = crypto.randomUUID();

    const payload = {
      client_id: clientId,
      manga_id: mangaIdInput.value.trim(),
      chapter_id: chapterIdInput.value.trim(),
      engine: engineInput.value,
      strength: Number(strengthInput.value || '1.0'),
      output_root: outputRootInput.value.trim(),
      options: {},
      page_uploads,
    };

    if (styleReferenceUpload?.contentBase64) {
      payload.style_reference_base64 = styleReferenceUpload.contentBase64;
      payload.style_reference_filename = styleReferenceUpload.filename || 'style_reference.png';
    } else if (styleReferenceUrlInput.value.trim()) {
      payload.style_reference_url = styleReferenceUrlInput.value.trim();
    }

    // Start polling the server for progress status
    const baseUrl = apiBaseInput.value.trim();
    const pollInterval = setInterval(async () => {
      try {
        const res = await fetch(`${baseUrl}/v1/pipeline/status?client_id=${clientId}`);
        if (res.ok) {
          const data = await res.json();
          if (data && data.message) {
            setStatusLoading(data.message);
          }
        }
      } catch (e) {
        // Silently ignore ping network errors
      }
    }, 2000);

    try {
      await requestJSON(
        `${baseUrl}/v1/pipeline/run_chapter`,
        payload,
        'POST /v1/pipeline/run_chapter',
      );
    } finally {
      clearInterval(pollInterval);
    }

  } catch (error) {
    setStatus(false, `Falha: ${error.message}`);
    outputEl.textContent = String(error);
    await pushHistory({ ts: new Date().toISOString(), action: 'POST /v1/pipeline/run_chapter', status: `error: ${error.message}` });
  } finally {
    runChapterBtn.disabled = false;
  }
});

// ---------------------------------------------------------------------------
// Pass2 single / batch
// ---------------------------------------------------------------------------

runBtn.addEventListener('click', async () => {
  await saveSettings();
  try {
    await requestJSON(
      `${apiBaseInput.value.trim()}/v1/pass2/run`,
      { meta_path: metaPathInput.value.trim(), output_dir: outputDirInput.value.trim(), engine: engineInput.value, strength: Number(strengthInput.value || '1.0'), options: {} },
      'POST /v1/pass2/run',
    );
  } catch (error) {
    setStatus(false, 'Falha no Pass2 run.');
    outputEl.textContent = String(error);
  }
});

batchBtn.addEventListener('click', async () => {
  await saveSettings();
  try {
    await requestJSON(
      `${apiBaseInput.value.trim()}/v1/pass2/batch`,
      { metadata_dir: metadataDirInput.value.trim(), output_dir: batchOutputDirInput.value.trim(), engine: engineInput.value, strength: Number(strengthInput.value || '1.0'), expected_pages: Number(expectedPagesInput.value || '0'), options: {} },
      'POST /v1/pass2/batch',
    );
  } catch (error) {
    setStatus(false, 'Falha no Pass2 batch.');
    outputEl.textContent = String(error);
  }
});

clearHistoryBtn.addEventListener('click', async () => {
  await chrome.storage.local.set({ [HISTORY_KEY]: [] });
  renderHistory([]);
  setStatus(true, 'Histórico limpo.');
});

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

loadSettings();

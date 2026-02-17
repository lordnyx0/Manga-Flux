const $ = (id) => document.getElementById(id);

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

const statusEl = $('status');
const outputEl = $('output');
const historyList = $('historyList');
const thumbGrid = $('thumbGrid');
const imagesCount = $('imagesCount');

const HISTORY_KEY = 'history';
const IMAGES_KEY = 'chapterPageItems';
const MIN_CAPTURE_WIDTH = 320;
const MIN_CAPTURE_HEIGHT = 320;

let chapterPageItems = [];
let styleReferenceUpload = null;
let captureSourceUrl = '';
let captureCookieHeader = '';

function applyTheme(themeMode) {
  let mode = themeMode;
  if (themeMode === 'auto') {
    mode = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  document.documentElement.setAttribute('data-theme', mode);
}

function setStatus(ok, text) {
  statusEl.className = ok ? 'row ok' : 'row err';
  statusEl.textContent = text;
}

function getHeaders() {
  const headers = { 'Content-Type': 'application/json' };
  const token = apiTokenInput.value.trim();
  if (token) headers['X-API-Token'] = token;
  return headers;
}

function renderThumbnails() {
  thumbGrid.innerHTML = '';
  imagesCount.textContent = `${chapterPageItems.length} imagens selecionadas`;

  chapterPageItems.forEach((itemData, idx) => {
    const item = document.createElement('div');
    item.className = 'thumb-item';

    const img = document.createElement('img');
    img.src = itemData.previewUrl;
    img.alt = `page_${idx + 1}`;
    img.loading = 'lazy';
    img.referrerPolicy = 'no-referrer';
    img.addEventListener('error', () => {
      img.src = 'data:image/svg+xml;charset=UTF-8,' + encodeURIComponent(
        '<svg xmlns="http://www.w3.org/2000/svg" width="220" height="140"><rect width="100%" height="100%" fill="#20262e"/><text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" font-size="12" fill="#d1d5db">Sem preview (bloqueado no site)</text></svg>',
      );
    }, { once: true });

    const actions = document.createElement('div');
    actions.className = 'thumb-actions';

    const removeBtn = document.createElement('button');
    removeBtn.className = 'danger';
    removeBtn.textContent = 'Remover';
    removeBtn.addEventListener('click', async () => {
      chapterPageItems = chapterPageItems.filter((_, i) => i !== idx);
      await chrome.storage.local.set({ [IMAGES_KEY]: chapterPageItems });
      renderThumbnails();
      await saveSettings();
    });

    actions.appendChild(removeBtn);
    item.appendChild(img);
    item.appendChild(actions);
    thumbGrid.appendChild(item);
  });
}


function normalizeStoredItems(rawItems) {
  if (!Array.isArray(rawItems)) return [];
  return rawItems
    .map((item) => {
      if (typeof item === 'string') {
        return {
          source: 'url',
          url: item,
          previewUrl: item,
        };
      }
      if (!item || typeof item !== 'object') return null;
      if (item.source === 'url' && typeof item.url === 'string') {
        return {
          source: 'url',
          url: item.url,
          previewUrl: item.url,
        };
      }
      return null;
    })
    .filter(Boolean);
}



function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const out = String(reader.result || '');
      const base64 = out.includes(',') ? out.split(',', 2)[1] : out;
      resolve(base64);
    };
    reader.onerror = () => reject(reader.error || new Error('Falha ao ler arquivo'));
    reader.readAsDataURL(file);
  });
}
async function pushHistory(entry) {
  const data = await chrome.storage.local.get([HISTORY_KEY]);
  const history = Array.isArray(data[HISTORY_KEY]) ? data[HISTORY_KEY] : [];
  history.unshift(entry);
  const trimmed = history.slice(0, 20);
  await chrome.storage.local.set({ [HISTORY_KEY]: trimmed });
  renderHistory(trimmed);
}

function renderHistory(history) {
  historyList.innerHTML = '';
  if (!history.length) {
    historyList.innerHTML = '<li>(vazio)</li>';
    return;
  }
  for (const item of history) {
    const li = document.createElement('li');
    li.textContent = `${item.ts} - ${item.action} - ${item.status}`;
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
    [IMAGES_KEY]: chapterPageItems,
  });
}

async function loadSettings() {
  const keys = [
    'apiBase', 'apiToken', 'theme', 'mangaId', 'chapterId', 'styleReferenceUrl',
    'engine', 'strength', 'outputRoot', 'metaPath', 'outputDir', 'metadataDir',
    'batchOutputDir', 'expectedPages', HISTORY_KEY, IMAGES_KEY, 'chapterPageUrls',
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
  if (data.strength !== undefined) strengthInput.value = data.strength;
  if (data.outputRoot) outputRootInput.value = data.outputRoot;
  if (data.metaPath) metaPathInput.value = data.metaPath;
  if (data.outputDir) outputDirInput.value = data.outputDir;
  if (data.metadataDir) metadataDirInput.value = data.metadataDir;
  if (data.batchOutputDir) batchOutputDirInput.value = data.batchOutputDir;
  if (data.expectedPages !== undefined) expectedPagesInput.value = data.expectedPages;

  const oldUrls = Array.isArray(data.chapterPageUrls)
    ? data.chapterPageUrls.map((url) => ({ source: 'url', url, previewUrl: url }))
    : [];
  chapterPageItems = normalizeStoredItems(data[IMAGES_KEY] || oldUrls);
  renderThumbnails();
  renderHistory(data[HISTORY_KEY] || []);
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
  await pushHistory({ ts: new Date().toISOString(), action: actionName, status: ok ? 'ok' : `http_${response.status}` });
}

// persist fields quickly so extension keeps state even after popup close/minimize
[
  apiBaseInput, apiTokenInput, mangaIdInput, chapterIdInput, styleReferenceUrlInput,
  engineInput, strengthInput, outputRootInput, metaPathInput, outputDirInput,
  metadataDirInput, batchOutputDirInput, expectedPagesInput,
].forEach((el) => el.addEventListener('change', saveSettings));



styleReferenceFileInput.addEventListener('change', async () => {
  const file = styleReferenceFileInput.files?.[0];
  if (!file) {
    styleReferenceUpload = null;
    styleReferenceHintEl.textContent = 'Use URL ou upload local (upload tem prioridade).';
    return;
  }
  try {
    const base64 = await fileToBase64(file);
    styleReferenceUpload = { filename: file.name, contentBase64: base64 };
    styleReferenceHintEl.textContent = `Upload pronto: ${file.name}`;
  } catch (error) {
    styleReferenceUpload = null;
    styleReferenceHintEl.textContent = 'Falha ao ler arquivo de referência.';
    outputEl.textContent = String(error);
    setStatus(false, 'Falha ao preparar upload da referência de estilo.');
  }
});

themeSelect.addEventListener('change', async () => {
  applyTheme(themeSelect.value);
  await saveSettings();
});

saveBtn.addEventListener('click', async () => {
  await saveSettings();
  setStatus(true, 'Configuração salva com sucesso.');
});

healthBtn.addEventListener('click', async () => {
  const apiBase = apiBaseInput.value.trim();
  try {
    const response = await fetch(`${apiBase}/health`);
    const payload = await response.json();
    setStatus(response.ok, response.ok ? 'API saudável.' : `Erro HTTP ${response.status}`);
    outputEl.textContent = JSON.stringify(payload, null, 2);
    await pushHistory({ ts: new Date().toISOString(), action: 'GET /health', status: response.ok ? 'ok' : `http_${response.status}` });
  } catch (error) {
    setStatus(false, 'Falha ao conectar na API.');
    outputEl.textContent = String(error);
  }
});



async function buildCookieHeaderForUrl(url) {
  try {
    if (!url || !/^https?:\/\//.test(url)) return '';
    const cookies = await chrome.cookies.getAll({ url });
    if (!Array.isArray(cookies) || !cookies.length) return '';
    return cookies
      .filter((c) => c && c.name && c.value !== undefined)
      .map((c) => `${c.name}=${c.value}`)
      .join('; ');
  } catch (_) {
    return '';
  }
}

captureImagesBtn.addEventListener('click', async () => {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab?.id) throw new Error('Aba ativa não encontrada');
    captureSourceUrl = typeof tab.url === 'string' ? tab.url : '';
    captureCookieHeader = await buildCookieHeaderForUrl(captureSourceUrl);

    const injected = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      args: [MIN_CAPTURE_WIDTH, MIN_CAPTURE_HEIGHT],
      func: (minW, minH) => Array.from(document.images)
        .map((img) => ({
          url: img.currentSrc || img.src,
          width: img.naturalWidth || img.width || 0,
          height: img.naturalHeight || img.height || 0,
        }))
        .filter((item) => typeof item.url === 'string' && /^https?:\/\//.test(item.url))
        .filter((item) => item.width >= minW && item.height >= minH)
        .map((item) => item.url),
    });

    const urls = injected?.[0]?.result || [];
    const uniqueUrls = [...new Set(urls)];
    chapterPageItems = uniqueUrls.map((url) => ({ source: 'url', url, previewUrl: url }));
    await saveSettings();
    renderThumbnails();
    pageImagesHintEl.textContent = `Captura ativa: mínimo ${MIN_CAPTURE_WIDTH}x${MIN_CAPTURE_HEIGHT}px. ${uniqueUrls.length} imagem(ns) válida(s).`;
    setStatus(true, `${uniqueUrls.length} imagem(ns) capturada(s) da aba.`);
  } catch (error) {
    setStatus(false, 'Falha ao capturar imagens da aba.');
    outputEl.textContent = String(error);
  }
});

clearImagesBtn.addEventListener('click', async () => {
  chapterPageItems = [];
  await saveSettings();
  renderThumbnails();
  setStatus(true, 'Lista de imagens limpa.');
});

runChapterBtn.addEventListener('click', async () => {
  await saveSettings();
  const apiBase = apiBaseInput.value.trim();
  const payload = {
    manga_id: mangaIdInput.value.trim(),
    chapter_id: chapterIdInput.value.trim(),
    style_reference_url: styleReferenceUrlInput.value.trim(),
    page_urls: chapterPageItems
      .filter((item) => item.source === 'url')
      .map((item) => item.url),
    output_root: outputRootInput.value.trim(),
    engine: engineInput.value,
    strength: Number(strengthInput.value || '1.0'),
    options: {},
  };

  if (styleReferenceUpload?.contentBase64) {
    payload.style_reference_base64 = styleReferenceUpload.contentBase64;
    payload.style_reference_filename = styleReferenceUpload.filename || 'style_reference.png';
  }
  try {
    await requestJSON(`${apiBase}/v1/pipeline/run_chapter`, payload, 'POST /v1/pipeline/run_chapter');
  } catch (error) {
    setStatus(false, 'Falha ao executar POST /v1/pipeline/run_chapter.');
    outputEl.textContent = String(error);
  }
});

runBtn.addEventListener('click', async () => {
  await saveSettings();
  const apiBase = apiBaseInput.value.trim();
  const payload = {
    meta_path: metaPathInput.value.trim(),
    output_dir: outputDirInput.value.trim(),
    engine: engineInput.value,
    strength: Number(strengthInput.value || '1.0'),
    options: {},
  };
  try {
    await requestJSON(`${apiBase}/v1/pass2/run`, payload, 'POST /v1/pass2/run');
  } catch (error) {
    setStatus(false, 'Falha ao executar POST /v1/pass2/run.');
    outputEl.textContent = String(error);
  }
});

batchBtn.addEventListener('click', async () => {
  await saveSettings();
  const apiBase = apiBaseInput.value.trim();
  const payload = {
    metadata_dir: metadataDirInput.value.trim(),
    output_dir: batchOutputDirInput.value.trim(),
    engine: engineInput.value,
    strength: Number(strengthInput.value || '1.0'),
    expected_pages: Number(expectedPagesInput.value || '0'),
    options: {},
  };
  try {
    await requestJSON(`${apiBase}/v1/pass2/batch`, payload, 'POST /v1/pass2/batch');
  } catch (error) {
    setStatus(false, 'Falha ao executar POST /v1/pass2/batch.');
    outputEl.textContent = String(error);
  }
});

clearHistoryBtn.addEventListener('click', async () => {
  await chrome.storage.local.set({ [HISTORY_KEY]: [] });
  renderHistory([]);
  setStatus(true, 'Histórico limpo.');
});

loadSettings();

# API local e extensão (Manga-Flux)

> Nota: a API já estava em desenvolvimento. Este documento consolida o estado atual e os endpoints operacionais.

## Checklist (API + extensão)

### API

- [x] Endpoint `GET /health` implementado
- [x] Endpoint `POST /v1/pass2/run` implementado
- [x] Endpoint `POST /v1/pass2/batch` implementado
- [x] Endpoint `POST /v1/pipeline/run_chapter` implementado (ingestão via URLs de imagens)
- [x] Validação de payload (`engine`, `strength`, `seed_override`, `options`)
- [x] Compatibilidade com metadata cross-platform (paths `\` / `/`)
- [x] Autenticação local opcional (token via `MANGA_FLUX_API_TOKEN` + header `X-API-Token`)

### Extensão

- [x] Popup MV3 criada
- [x] Persistência de configuração no `chrome.storage.local`
- [x] Health-check para `GET /health`
- [x] Form para `POST /v1/pass2/run`
- [x] Form para `POST /v1/pass2/batch`
- [x] Exibição de histórico local de execuções (últimos 20 eventos)
- [x] Captura de imagens da aba atual para envio ao pipeline
- [x] Tema claro/escuro (com modo automático)
- [x] Visualização em miniaturas das imagens capturadas
- [x] Remoção individual de miniaturas
- [x] Persistência de estado para uso mesmo com popup minimizada/fechada

## API local

Subir servidor:

```bash
python api/server.py --host 127.0.0.1 --port 8765
```

Com autenticação por token (opcional):

```bash
MANGA_FLUX_API_TOKEN=seu_token python api/server.py --host 127.0.0.1 --port 8765
```

### Endpoints

- `GET /health`
  - status básico do serviço
  - informa se token está habilitado (`token_required`)

- `POST /v1/pass2/run`
  - executa Pass2 para uma página

- `POST /v1/pass2/batch`
  - executa Pass2 para todos os `page_*.meta.json` em `metadata_dir`

- `POST /v1/pipeline/run_chapter`
  - fluxo fim-a-fim Pass1+Pass2 a partir de URLs de imagens
  - suporta referência de estilo por URL **ou upload (base64)**
  - salva em `output/<manga_id>/chapters/<chapter_id>/...`

Exemplo de body para `run_chapter`:

```json
{
  "manga_id": "my_manga",
  "chapter_id": "chapter_001",
  "style_reference_url": "https://.../style.png",
  "style_reference_base64": "<base64 opcional enviado pela extensão>",
  "style_reference_filename": "style.png",
  "page_urls": [
    "https://.../page1.png",
    "https://.../page2.png"
  ],
  "engine": "dummy",
  "strength": 1.0
}
```

## Extensão Chrome (Companion)

Pasta da extensão:

- `extension/manga-flux-extension`

Carregar no Chrome:

1. Abrir `chrome://extensions`
2. Ativar **Developer mode**
3. Clicar em **Load unpacked**
4. Selecionar `extension/manga-flux-extension`

A popup permite:

- salvar URL/token da API
- executar health-check (`GET /health`)
- executar Pass2 single-page (`POST /v1/pass2/run`)
- executar Pass2 batch (`POST /v1/pass2/batch`)
- executar pipeline de capítulo (`POST /v1/pipeline/run_chapter`)
- visualizar histórico local das últimas execuções
- alternar entre tema claro/escuro/auto
- visualizar miniaturas das imagens capturadas e remover individualmente
- manter configuração/estado após minimizar/fechar popup

## Análise FAISS

- Ver análise e plano de adaptação: `DOCS/FAISS_ADAPTACAO_MANGA_FLUX.md`

## Status consolidado no plano de recuperação

Os itens de API/extensão acima também foram refletidos no checklist principal de recuperação:

- `RECUPERACAO_FUNCIONAL_MINIMA.md`
- `VIABILIDADE.md`

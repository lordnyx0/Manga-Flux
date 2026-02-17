# Extensão — Especificação Inicial (Flux)

## Objetivo

Definir a extensão cliente (browser) para integrar o usuário final ao fluxo Manga-Flux:

- seleção de páginas/capítulos no navegador;
- anexar/upload de imagem de referência para o Pass2;
- envio para processamento Two-Pass via API;
- acompanhamento de progresso e download/preview dos resultados.

> Estado atual: **extensão ainda não implementada neste repositório**. Documento de bootstrap + checklist.

---

## Escopo funcional esperado

1. **Captura/ingestão**
   - selecionar imagens da página atual ou lote;
   - anexar imagem de referência (`style_reference`) obrigatória para o Pass2;
   - preencher parâmetros básicos (estilo, engine, capítulo).

2. **Disparo de processamento**
   - chamar endpoint de criação de job (`POST /v1/jobs/two-pass`);
   - persistir `job_id` localmente para retomada.

3. **Acompanhamento e entrega**
   - monitorar status (`GET /v1/jobs/{job_id}`);
   - listar e baixar artefatos finais;
   - abrir preview da imagem colorizada.

---

## Arquitetura proposta (MV3)

- **`manifest.json`** (MV3)
  - permissões de host para API do Flux;
  - service worker para jobs assíncronos.
- **Popup UI**
  - formulário rápido de execução e progresso.
- **Content script**
  - captura de contexto da página (URLs de imagens, chapter/page hints).
- **Background/service worker**
  - orquestra chamadas HTTP;
  - polling de status;
  - storage local de histórico de jobs.

---

## Fluxo de uso (MVP)

1. usuário abre popup da extensão;
2. seleciona `chapter_id`, `style_reference` e modo de entrada;
3. extensão envia requisição para criar job Two-Pass;
4. extensão acompanha progresso e sinaliza conclusão;
5. usuário abre/download de `page_{NNN}_colorized.png`.

---

## Contratos de integração com API

A extensão depende diretamente da API definida em `DOCS/API.md`:

- `POST /v1/jobs/two-pass` (com `Authorization: Bearer <token>` quando habilitado no backend)
- `GET /v1/jobs/{job_id}`
- `GET /v1/jobs/{job_id}/artifacts`
- `GET /v1/chapters/{chapter_id}/pages`
- `GET /v1/chapters/{chapter_id}/pages/{page_num}/metadata`
- `GET /v1/chapters/{chapter_id}/pages/{page_num}/runmeta/pass1`
- `GET /v1/chapters/{chapter_id}/pages/{page_num}/runmeta/pass2`
- `GET /v1/chapters/{chapter_id}/pages/{page_num}/mask`
- `GET /v1/chapters/{chapter_id}/pages/{page_num}/colorized`

A API bootstrap já expõe `POST /v1/jobs/two-pass`, `GET /v1/jobs/{job_id}`, `GET /v1/jobs/{job_id}/artifacts` e endpoints por capítulo/página; a extensão pode iniciar integração por esses endpoints enquanto o restante da API evolui.

> **Importante:** esta versão não suporta fluxo sem referência; a UI deve bloquear envio quando `style_reference` não for informado.

---

## Checklist de pendências (Extensão)

- [ ] Criar estrutura inicial `browser_extension/` (MV3).
- [ ] Implementar `manifest.json` com permissões mínimas.
- [ ] Implementar popup com formulário de execução Two-Pass.
- [ ] Implementar upload/anexo de `style_reference` no popup.
- [ ] Bloquear submissão sem `style_reference` (regra obrigatória desta versão).
- [ ] Implementar client HTTP para API (com tratamento de erro/retry).
- [ ] Suportar envio de token bearer configurável para rotas de escrita.
- [ ] Implementar polling de status por `job_id`.
- [ ] Implementar visualização/listagem de artefatos gerados.
- [ ] Implementar storage local de histórico de jobs.
- [ ] Adicionar tela/config de endpoint da API e token.
- [ ] Escrever testes E2E mínimos da extensão (playwright/web-ext).
- [ ] Documentar instalação local (modo desenvolvedor) e release.

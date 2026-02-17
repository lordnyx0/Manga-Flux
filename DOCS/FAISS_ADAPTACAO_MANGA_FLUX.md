# Análise e adaptação do fluxo `/manga` (FAISS) para o Manga-Flux atual

## Contexto

No projeto `/manga`, o FAISS costuma ser usado para recuperar embeddings de referência (identidade/personagem/paleta) em fluxos de continuidade entre páginas/capítulos.

No estado atual do Manga-Flux:

- o pipeline Two-Pass já está funcional (Pass1 + Pass2)
- já existe diretório de embeddings (`data/embeddings`)
- a API local agora suporta ingestão de páginas do navegador e execução de capítulo completo

## Adaptação implementada nesta etapa

- Endpoint novo: `POST /v1/pipeline/run_chapter`
  - recebe `manga_id`, `chapter_id`, `style_reference_url`, `page_urls[]`
  - baixa imagens da web
  - executa Pass1 + Pass2 por página
  - grava em: `output/<manga_id>/chapters/<chapter_id>/...`
- Extensão atualizada para:
  - capturar URLs de imagens da aba atual
  - enviar lote via API para processamento de capítulo

## Como isso aproxima do fluxo `/manga`

- **Semântica de capítulo**: output estruturado por manga/capítulo
- **Ingestão web-first**: extensão coleta imagens do browser para pipeline
- **Base para recuperação vetorial**: artefatos por capítulo ficam organizados para futura indexação/retrieval

## Próxima adaptação FAISS (proposta objetiva)

1. Criar módulo `core/identity/faiss_service.py` com fallback numpy quando FAISS não estiver disponível.
2. Gerar embedding por página/personagem no fim do Pass1 e persistir no capítulo.
3. Expor endpoints:
   - `POST /v1/faiss/index_chapter`
   - `POST /v1/faiss/search`
4. Integrar resultado da busca no `options` do Pass2 para reforçar consistência visual entre páginas.

## Risco e mitigação

- **Risco**: FAISS não disponível no ambiente.
- **Mitigação**: fallback em memória com distância coseno (numpy), mantendo contrato da API.

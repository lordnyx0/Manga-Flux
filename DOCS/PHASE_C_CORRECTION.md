# Fase C: Sistema de Correção e Refinamento (Pós-Pass2)

## 1. Visão Geral da Fase C
Com a adoção do `ReferenceLatent` na Fase B (Pass2), conseguimos uma colorização global respeitando as linhas do mangá. No entanto, pipelines de geração full-frame via IA generativa sempre apresentam inconsistências localizadas (ex: sangramento de cor para dentro de balões de fala, personagens específicos com cores erradas, ou pequenos artefatos).

A **Fase C** tem o objetivo de utilizar toda a inteligência e os metadados extraídos na **Fase A (Pass1)** — que identifica painéis, balões, textos e rostos de personagens — para aplicar um sistema automatizado de composição e correção regional em cima do output do FLUX.

## 2. Arquitetura do Pipeline de Correção

O fluxo da Fase C será acoplado logo após o término da geração do FLUX (Pass2) e se baseia em duas frentes principais: **Composição Passiva** e **Inpaint Ativo**.

### 2.1. Composição Passiva (Mask Blending)
Garante que elementos estruturais puros do mangá não sejam arruinados pela colorização.
*   **Recuperação de Texto e Balões:** Usando as máscaras `_text.png` geradas no Pass1, o sistema fará um *Alpha Blending* direto. Os pixels correspondentes a textos e balões na imagem colorida gerada pelo FLUX serão substituídos pelos pixels exatos da página original em alta resolução (ou preenchidos puramente de branco/preto dependendo do balão).
*   **Margens e Calhas (Gutters):** Limpeza das bordas das páginas e dos espaços entre os quadros para garantir que não haja "vazamento" de cores para o fundo da página.

### 2.2. Correção Ativa (Regional Inpaint)
Caso o FLUX erre a colorização de um personagem específico ou falhe num painel complexo, a Fase C poderá invocar o motor de difusão novamente, mas limitando-se a uma área de secção.
*   **Isolamento Analítico:** O sistema lê o `runmeta.json` do Pass1 para encontrar as coordenadas exatas (`[x1, y1, x2, y2]`) do painel ou personagem problemático.
*   **Crop & Masking:** O script recorta essa região específica da imagem colorida (comprimindo a necessidade de VRAM) e recorta a imagem P&B original correspondente.
*   **Micro-Pipeline Inpaint:** Envia para o ComfyUI um novo workflow embutindo a máscara. O Denoise pode ser ajustado para atuar *apenas* dentro das linhas daquele painel/personagem, usando Prompts altamente localizados (ex: "personagem X com cabelo vermelho").
*   **Seamless Stitching:** A região corrigida é colada de volta na página inteira usando algoritmos de *Poisson Blending* ou laplacianos para que as bordas da correção não fiquem visíveis.

---

## 3. Checklist de Implementação - Fase C

### 3.1. Infraestrutura do Compositor (Passiva)
- [ ] Criar o módulo `core/correction/compositor.py`.
- [ ] Implementar a função `apply_text_masks` para sobrepor as regiões textuais da imagem `inputs/` sobre a imagem final `pass2/`.
- [ ] Implementar limpeza de detecção de bordas de painéis (Gutter cleaning/whitening).

### 3.2. Integração do Orchestrator
- [ ] Modificar `core/generation/orchestrator.py` para invocar o Compositor automaticamente após salvar o resultado do FLUX.
- [ ] Salvar a imagem final revisada (ex: `page_001_final.png`) e registrar esse passo no JSON de tracking.

### 3.3. Pipeline de Inpaint (Ativa) - *Avançado*
- [ ] Desenvolver `core/correction/regional_inpaint.py`.
- [ ] Definir o JSON Workflow do ComfyUI focado em *Inpainting* (onde `VAEEncodeForInpaint` ou abordagens similares são usadas com máscaras para FLUX).
- [ ] Integrar recortes de painéis originados dos bounding boxes do YOLO do Pass1.
- [ ] Criar função de mesclagem suavizada (seamless clone) para reinserir as regiões repintadas.

## 4. Requisitos de Hardware e Performance
*   **Composição Passiva:** Rodará puramente em CPU via OpenCV/Pillow, adicionando menos de 1 segundo ao processamento da página.
*   **Correção Ativa:** Reativará a GPU/FLUX para *crops* menores. Por serem pedaços pequenos (ex: 512x512 ao invés da página toda de 1808px), o inpaint consumirá quase metade da VRAM da página completa e rodará consideravelmente mais rápido.

---
*Este documento guiará a próxima evolução do Manga-Flux após estabilizarmos visualmente as cores globais com a arquitetura ReferenceLatent.*

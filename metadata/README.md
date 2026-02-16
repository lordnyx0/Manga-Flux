# Contrato Pass1 -> Pass2 (`.meta.json`)

Cada página processada no Pass1 deve gerar um arquivo:

- `metadata/page_{NNN}.meta.json`

## Chaves obrigatórias

- `page_num` (int)
- `page_image` (str, caminho para imagem da página em P&B)
- `page_seed` (int)
- `page_prompt` (str)
- `style_reference` (str, caminho para imagem de referência de estilo)
- `text_mask` (str, caminho para máscara de texto/balões)

> Exceção controlada: quando `ALLOW_NO_STYLE=1`, o Pass2 aceita metadado sem `style_reference`.

## Exemplo

```json
{
  "page_num": 1,
  "page_image": "tests/data/page_001.png",
  "page_seed": 1918611493,
  "page_prompt": "manga page, cinematic, soft shading",
  "style_reference": "tests/data/style_ref.png",
  "text_mask": "outputs/test_run/masks/page_001_text.png"
}
```

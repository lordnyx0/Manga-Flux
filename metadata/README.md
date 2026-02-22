# Contract Pass1 -> Pass2 (`.meta.json`)

Each page processed in Pass1 must generate a file:

- `metadata/page_{NNN}.meta.json`

## Required keys

- `page_num` (int)
- `page_image` (str, path to the B&W page image)
- `page_seed` (int)
- `page_prompt` (str)
- `style_reference` (str, path to the style reference image)
- `text_mask` (str, path to the text/bubbles mask)

> Controlled exception: when `ALLOW_NO_STYLE=1`, Pass2 accepts metadata without `style_reference`.

## Example

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

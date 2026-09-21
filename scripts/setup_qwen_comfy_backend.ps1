#Requires -Version 5.1
<#
.SYNOPSIS
  Prepara o backend ComfyUI para Qwen-Image-2.1-Uncensored-GGUF (janela com GPU livre).
.DESCRIPTION
  1. Confere ComfyUI + atualiza custom node leejet/ComfyUI-GGUF (ModelQwenImage).
  2. Confere os 3 arquivos (difusão GGUF, text encoder, VAE) e baixa os faltantes.
  3. NÃO roda inferência — o teste de fumaça é um passo separado (docs).
.EXAMPLE
  .\setup_qwen_comfy_backend.ps1 -ComfyDir "C:\ComfyUI" -Quant Q4_K_M
  .\setup_qwen_comfy_backend.ps1 -ComfyDir "C:\ComfyUI" -CheckOnly
#>
param(
  [string]$ComfyDir = "C:\ComfyUI",
  [ValidateSet("Q4_K_M", "Q5_K_M", "Q6_K", "Q4_0")]
  [string]$Quant = "Q4_K_M",
  [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
$Repo = "abenzerps/Qwen-Image-2.1-Uncensored-GGUF"
$GgufFile = "qwen-image-2.1-$($Quant -replace '_','_').gguf"
# Mapeia Quant -> nome real do arquivo no repo (Q4_K_M etc. usam o mesmo padrão).
$GgufMap = @{
  "Q4_K_M" = "qwen-image-2.1-Q4_K_M.gguf"
  "Q5_K_M" = "qwen-image-2.1-Q5_K_M.gguf"
  "Q6_K"   = "qwen-image-2.1-Q6_K.gguf"
  "Q4_0"   = "qwen-image-2.1-Q4_0.gguf"
}
$GgufFile = $GgufMap[$Quant]

$DiffusionDest = Join-Path $ComfyDir "models\diffusion_models\$GgufFile"
$ClipDest      = Join-Path $ComfyDir "models\text_encoders\qwen3vl_8b_int8_convrot.safetensors"
$VaeDest       = Join-Path $ComfyDir "models\vae\qwen_image_2.1_vae_bf16.safetensors"
$GgufNodeDir   = Join-Path $ComfyDir "custom_nodes\ComfyUI-GGUF"

Write-Host "== Qwen backend check =="
Write-Host "ComfyUI: $ComfyDir"
if (-not (Test-Path -LiteralPath (Join-Path $ComfyDir "main.py"))) {
  throw "ComfyUI não encontrado em $ComfyDir (main.py ausente). Ajuste -ComfyDir."
}

# 1. Custom node leejet (ModelQwenImage). city96 antigo causa 'Unknown model architecture!'.
if (Test-Path -LiteralPath $GgufNodeDir) {
  $remote = (git -C $GgufNodeDir remote get-url origin 2>$null)
  Write-Host "ComfyUI-GGUF existente: $remote"
  if ($remote -notmatch "leejet/ComfyUI-GGUF") {
    Write-Warning "Fork antigo (city96?) detectado. Troque por https://github.com/leejet/ComfyUI-GGUF na janela de manutenção."
  } elseif (-not $CheckOnly) {
    git -C $GgufNodeDir pull --ff-only
  }
} else {
  Write-Host "ComfyUI-GGUF ausente em $GgufNodeDir"
  if (-not $CheckOnly) {
    git clone https://github.com/leejet/ComfyUI-GGUF $GgufNodeDir
  }
}

# 2. Arquivos de modelo.
$missing = @()
foreach ($f in @($DiffusionDest, $ClipDest, $VaeDest)) {
  if (Test-Path -LiteralPath $f) { Write-Host "[OK] $f" }
  else { Write-Host "[FALTA] $f"; $missing += $f }
}
if ($CheckOnly) { return }

if ($missing.Count -gt 0) {
  Write-Host ""
  Write-Host "Baixe os faltantes (Q8_0 PROPOSITALMENTE excluído — shape mismatch [136] vs [128]):"
  Write-Host "  huggingface-cli download $Repo $GgufFile --local-dir (models\diffusion_models)"
  Write-Host "  huggingface-cli download $Repo text_encoders/qwen3vl_8b_int8_convrot.safetensors --local-dir (models\text_encoders)"
  Write-Host "  huggingface-cli download $Repo vae/qwen_image_2.1_vae_bf16.safetensors --local-dir (models\vae)"
  Write-Host ""
  Write-Host "Ou rode este script de novo após liberar a GPU/rede para download automático (hf_hub)."
}

Write-Host ""
Write-Host "Próximo (com GPU livre): inicie com 'python main.py --lowvram' e rode o smoke do DOCS/QWEN_MIGRATION.md."

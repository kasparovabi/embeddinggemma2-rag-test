Set-Location $PSScriptRoot
"KOSU_BASLADI $(Get-Date -Format s)" | Out-File durum.txt
New-Item -ItemType Directory -Force emb | Out-Null
$F = @{ BF16 = "embeddinggemma-2-BF16.gguf"; Q4 = "embeddinggemma-2-UD-Q4_K_XL.gguf" }
foreach ($Q in @("BF16", "Q4")) {
  # -c is split across -np slots: 8 x 8192 tokens per slot, so long docs are not rejected.
  $p = Start-Process -FilePath "llama\llama-server.exe" -PassThru -NoNewWindow `
       -RedirectStandardError "srv_$Q.log" -RedirectStandardOutput "srv_${Q}_out.log" `
       -ArgumentList "-m models\$($F[$Q]) --embeddings --pooling mean -c 65536 -np 8 -b 8192 -ub 8192 -ngl 99 --port 8801"
  for ($i = 0; $i -lt 90; $i++) {
    try { if ((Invoke-WebRequest http://127.0.0.1:8801/health -UseBasicParsing -TimeoutSec 3).StatusCode -eq 200) { break } } catch {}
    Start-Sleep 2
  }
  foreach ($D in @("scifact", "nfcorpus", "arguana")) {
    "$(Get-Date -Format T) $Q $D" | Out-File -Append durum.txt
    venv\Scripts\python.exe embed.py $D $Q 8801 *>> embed.log
  }
  $mem = (& nvidia-smi --query-gpu=memory.used --format=csv,noheader)
  "{`"quant`":`"$Q`",`"server_ws_mb`":$([int]((Get-Process -Id $p.Id).WorkingSet64/1MB)),`"gpu_mem`":`"$mem`"}" | Out-File -Append emb\memory.jsonl -Encoding utf8
  Stop-Process -Id $p.Id -Force
  Start-Sleep 3
  "$Q too_long=$((Select-String -Path srv_$Q.log -Pattern 'larger than the max context').Count)" | Out-File -Append durum.txt
}
venv\Scripts\python.exe evaluate.py *> results.txt
"BITTI $(Get-Date -Format s)" | Out-File -Append durum.txt

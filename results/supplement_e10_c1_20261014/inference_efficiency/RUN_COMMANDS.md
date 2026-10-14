# 复现命令

```powershell
Set-Location 'D:\SSPG-Net'
$env:HF_HUB_OFFLINE='1'
& '.\.venv\Scripts\python.exe' 'scripts\benchmark_e10_inference_efficiency_v1.py' --all
```

已有测量输出不会被覆盖；复测请通过 --output 指定新的独立目录。每个模型在独立进程中依次测量。

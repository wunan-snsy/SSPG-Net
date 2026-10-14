# Windows 复现命令

在项目目录运行，使用当前 `.venv` 环境：

```powershell
Set-Location 'D:\SSPG-Net'
$env:HF_HUB_OFFLINE='1'
& '.\.venv\Scripts\python.exe' 'scripts\run_e10_spatial_mixer_v1.py' --preflight
& '.\.venv\Scripts\python.exe' 'scripts\run_e10_spatial_mixer_v1.py' --all
# 安全续跑：完整且 hash 正确的种子跳过；不复写未完成目录。
& '.\.venv\Scripts\python.exe' 'scripts\run_e10_spatial_mixer_v1.py' --resume
# 单独种子（仅适用于尚不存在的种子目录）：
& '.\.venv\Scripts\python.exe' 'scripts\run_e10_spatial_mixer_v1.py' --seed 20261001
& '.\.venv\Scripts\python.exe' 'scripts\summarize_e10_spatial_mixer_v1.py'
```

汇总图、数据和论文补充文本位于 `results/e10_spatial_mixer_v1/`。
完成实验的 `--all` / `--resume` 不重训已完成种子；若工件缺失或 hash 不一致，则停止。未完成目录不得假装成功，也不会自动删除或覆盖。


完成训练和汇总后，交付排版/打包：

```powershell
& '.\.venv\Scripts\python.exe' 'scripts\finalize_e10_delivery_v1.py'
```

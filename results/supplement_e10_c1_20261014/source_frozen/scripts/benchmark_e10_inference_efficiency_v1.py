"""Measure trained models on existing Validation inputs; never train or tune."""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'results/e10_inference_efficiency_v1'
SEED=20261001
MODELS=['FULL','NO_GRAPH','MATCHED_NO_GRAPH','SPATIAL_MIXER_MATCHED']
WARMUP=30
REPEATS=3


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,default=str),encoding='utf-8')
    tmp.replace(path)


def directory(name):
    base=ROOT/'results/clean_revision_v2'
    if name=='SPATIAL_MIXER_MATCHED':
        return ROOT/f'results/e10_spatial_mixer_v1/{SEED}'
    if name=='MATCHED_NO_GRAPH':
        return base/f'budget142_param_matched_no_graph_e2/{SEED}'
    return base/f'budget142_e2/{name.lower()}/{SEED}'


def verify_checkpoint(name):
    folder=directory(name)
    lineage=json.loads((folder/'lineage.json').read_text())
    key=lineage['artifact_hashes']['best.pth'] if name=='SPATIAL_MIXER_MATCHED' else lineage['checkpoint_sha256']
    if sha(folder/'best.pth')!=key:
        raise RuntimeError(f'Checkpoint hash mismatch: {folder}')
    return folder,key


def gpu_snapshot():
    result=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.used,temperature.gpu,power.draw,clocks.current.sm','--format=csv'],capture_output=True,text=True)
    return result.stdout.strip()


def worker(name):
    import numpy as np
    import pandas as pd
    import torch
    import training.train_e53r_joint_v1 as joint
    from preprocessing.active_sspg_dataset import ActiveSSPGDataset
    from scripts.run_e10_spatial_mixer_v1 import install_access_guard
    from models.sspg_joint_model_v2 import SSPGJointModelV2
    from models.sspg_ablation_model_v2 import SSPGAblationModelV2
    from models.sspg_param_matched_no_graph_v2 import SSPGParamMatchedNoGraphV2
    from models.sspg_spatial_mixer_matched_v2 import SSPGSpatialMixerMatchedV2
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    folder,ckpt_sha=verify_checkpoint(name)
    target=OUT/name
    target.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    joint.set_global_seed(SEED)
    # Preserve the E2 runtime defaults explicitly, and record TF32 permissions.
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    install_access_guard(joint,target/'image_access.json')
    manifest=pd.read_csv(joint.MANIFEST_PATH)
    records=joint.build_source_records(manifest,'val')
    assert len(records)==62
    data=ActiveSSPGDataset(records,base_seed=20260923)
    assert len(data)==124
    samples=[]
    input_hash=hashlib.sha256()
    for i in range(len(data)):
        item=data[i]
        x=item['received'].unsqueeze(0).float().contiguous()
        payload=item['payload'].unsqueeze(0).float().contiguous()
        identity=item['sample_key']+'|'+item['attack']
        input_hash.update(identity.encode())
        input_hash.update(x.numpy().tobytes())
        input_hash.update(payload.numpy().tobytes())
        samples.append(dict(received=x.pin_memory(),payload=payload.pin_memory(),
            sample_key=item['sample_key'],attack=item['attack'],truth=item['mask'].numpy().squeeze()>=.5))
    dec=ROOT/'checkpoints/clean_revision_v2/decoder_budget142/best.pth'
    if name=='FULL':
        model=SSPGJointModelV2(str(dec))
    elif name=='NO_GRAPH':
        model=SSPGAblationModelV2(str(dec),ablation='no_graph')
    elif name=='MATCHED_NO_GRAPH':
        model=SSPGParamMatchedNoGraphV2(str(dec))
    else:
        model=SSPGSpatialMixerMatchedV2(str(dec))
    total=sum(p.numel() for p in model.parameters())
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    checkpoint=torch.load(folder/'best.pth',map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    del checkpoint
    gc.collect()
    model=model.float().cuda().eval()
    stored=pd.read_csv(folder/'metrics.csv').iloc[0]
    expected=pd.read_csv(folder/'val_predictions.csv').set_index(['sample_key','attack'])
    threshold=float(stored['detection_threshold'])
    torch.cuda.synchronize()
    resident_allocated=torch.cuda.memory_allocated()/2**20
    snapshot_before=gpu_snapshot()

    def execute(sample,timed=False):
        torch.cuda.synchronize()
        if timed:
            t0=time.perf_counter_ns()
        received=sample['received'].to('cuda',non_blocking=True)
        payload=sample['payload'].to('cuda',non_blocking=True)
        if timed:
            begin=torch.cuda.Event(enable_timing=True)
            end=torch.cuda.Event(enable_timing=True)
            begin.record()
        output=model(received,payload)
        probability=torch.sigmoid(output['detection_logits'])
        detected=probability>=threshold
        mask=torch.sigmoid(model.localization_logits_256(output['localization_grid_logits']))>=.5
        if timed:
            end.record()
        # Wall time includes these output transfers and the final synchronization.
        p_cpu=probability.cpu()
        det_cpu=detected.cpu()
        mask_cpu=mask.cpu()
        torch.cuda.synchronize()
        wall=(time.perf_counter_ns()-t0)/1e6 if timed else None
        gpu=begin.elapsed_time(end) if timed else None
        return float(p_cpu[0]),bool(det_cpu[0]),mask_cpu.numpy().squeeze(),wall,gpu

    timings=[]
    score_difference=[]
    pixel_totals=np.zeros(4,dtype=np.int64)
    with torch.inference_mode():
        for i in range(WARMUP):
            execute(samples[i%len(samples)])
        torch.cuda.reset_peak_memory_stats()
        for repetition in range(REPEATS):
            order=np.random.default_rng(20261012+repetition).permutation(len(samples))
            for index in order:
                sample=samples[int(index)]
                probability,detected,mask,wall,gpu=execute(sample,True)
                timings.append(dict(model=name,checkpoint_seed=SEED,repetition=repetition+1,
                    sample_index=int(index),sample_key=sample['sample_key'],attack=sample['attack'],
                    wall_ms=wall,gpu_interval_ms=gpu))
                if repetition==0:
                    ref=expected.loc[(sample['sample_key'],sample['attack'])]
                    score_difference.append(abs(probability-float(ref['detection_probability'])))
                    truth=sample['truth']
                    pixel_totals+=np.array([(mask&truth).sum(),(mask&~truth).sum(),(~mask&truth).sum(),(~mask&~truth).sum()],dtype=np.int64)
            print(f'{name}: repeat {repetition+1}/{REPEATS}, 124 real inputs measured',flush=True)
    peak_allocated=torch.cuda.max_memory_allocated()/2**20
    peak_reserved=torch.cuda.max_memory_reserved()/2**20
    tp,fp,fn,tn=map(int,pixel_totals)
    pixel_f1=2*tp/(2*tp+fp+fn)
    if max(score_difference)>1e-5 or abs(pixel_f1-float(stored['localization_pixel_f1']))>1e-6:
        raise RuntimeError(f'Inference consistency failure: score max diff {max(score_difference)}, Pixel F1 {pixel_f1}')
    frame=pd.DataFrame(timings)
    frame.to_csv(target/'per_image_timings.csv',index=False)
    summary=dict(model=name,checkpoint_seed=SEED,checkpoint_sha256=ckpt_sha,
        input_tensor_sha256=input_hash.hexdigest(),unique_inputs=124,repetitions=REPEATS,n_measured=len(frame),
        warmup_calls=WARMUP,total_parameters=total,trainable_parameters=trainable,
        wall_mean_ms=float(frame.wall_ms.mean()),wall_sd_ms=float(frame.wall_ms.std(ddof=1)),
        wall_median_ms=float(frame.wall_ms.median()),wall_p95_ms=float(frame.wall_ms.quantile(.95)),
        gpu_mean_ms=float(frame.gpu_interval_ms.mean()),gpu_sd_ms=float(frame.gpu_interval_ms.std(ddof=1)),
        sequential_images_per_second=1000/float(frame.wall_ms.mean()),
        model_resident_allocated_mib=resident_allocated,peak_allocated_mib=peak_allocated,
        peak_reserved_mib=peak_reserved,transient_peak_over_resident_mib=peak_allocated-resident_allocated,
        maximum_detection_score_difference=max(score_difference),verified_pixel_f1=pixel_f1,
        model_eval=True,gradient_enabled=False,batch_size=1,autocast=False,
        weights_and_input_dtype='float32',matmul_tf32=False,cudnn_tf32=True,
        cuda_peak_scope='Per-process PyTorch CUDA allocator; includes model weights, live input and intermediates; does not include graphics apps or CUDA driver/context allocations',
        wall_scope='Pinned preloaded CPU received image and payload -> H2D -> complete verification network -> probabilities and binary detection/localization -> D2H -> CUDA synchronization',
        gpu_scope='CUDA event interval covering complete verification network and GPU postprocessing; excludes H2D and D2H',
        excluded='Image file I/O, resizing, input protection/payload embedding, dataset attack generation, model construction/loading, warmup',
        gpu_snapshot_before=snapshot_before,gpu_snapshot_after=gpu_snapshot())
    write_json(target/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)
    del model
    torch.cuda.empty_cache()


def summarize():
    import pandas as pd
    rows=[json.loads((OUT/name/'summary.json').read_text()) for name in MODELS]
    assert len({r['input_tensor_sha256'] for r in rows})==1
    assert all(r['n_measured']==372 for r in rows)
    for name in MODELS:
        access=json.loads((OUT/name/'image_access.json').read_text())
        assert not access['forbidden_attempts'] and len(access['accessed_source_paths'])==62
    frame=pd.DataFrame(rows)
    frame.to_csv(OUT/'efficiency_summary.csv',index=False)
    all_timing=pd.concat([pd.read_csv(OUT/name/'per_image_timings.csv') for name in MODELS],ignore_index=True)
    all_timing.to_csv(OUT/'all_per_image_timings.csv',index=False)
    report='''# SSPG-Net 推理效率实测报告

## 统一测量条件

- 硬件：NVIDIA GeForce RTX 3060 Laptop GPU，6GB 显存；本机桌面环境。
- 每种架构使用固定种子 20261001 的真实最佳检查点，严格载入并校验 SHA-256；未训练、未选择新的权重。
- 同一 IMD2020 Validation 的 124 个实际输入：62 个保护原图 + 62 个受控篡改图。四种模型输入张量 SHA-256 完全一致，只有 Validation 图片被读取。
- Batch=1，256×256 received image、128-bit payload；模型内部按既有流程使用 ViT 224×224 输入，定位输出上采样为 256×256。
- FP32 权重与输入，无 AMP；matmul TF32 关闭、cuDNN TF32 允许，与既有 E2 运行默认设置一致。eval + inference_mode，cuDNN deterministic=True、benchmark=False，CPU 线程数 4。
- 每模型先预热 30 次，再将全部 124 个输入测量 3 遍（372 次）；每遍按相同固定随机顺序测量。模型依次执行，无并行 GPU 训练任务。
- 完整验证网络包括 payload decoder、冻结 ViT、adapter、证据融合、图/GNN 或相应对照模块、检测及定位 heads，计时后处理包括 sigmoid、阈值化、定位上采样。

## 实测结果

| 模型 | 可训练参数/M | 全部参数/M | 平均墙钟/ms | 中位数/ms | P95/ms | GPU区间均值/ms | 峰值分配/MiB | 峰值保留/MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
'''
    for r in rows:
        report+=f'| {r["model"]} | {r["trainable_parameters"]/1e6:.6f} | {r["total_parameters"]/1e6:.6f} | {r["wall_mean_ms"]:.3f} | {r["wall_median_ms"]:.3f} | {r["wall_p95_ms"]:.3f} | {r["gpu_mean_ms"]:.3f} | {r["peak_allocated_mib"]:.2f} | {r["peak_reserved_mib"]:.2f} |\n'
    full=rows[0]
    report+='\n## 可用于论文的结论\n\n'
    report+=f'FULL 在此硬件与单图输入条件下的平均墙钟时间为 {full["wall_mean_ms"]:.3f} ms/图，GPU 区间均值为 {full["gpu_mean_ms"]:.3f} ms，PyTorch 峰值分配显存为 {full["peak_allocated_mib"]:.2f} MiB。串行等效吞吐率为 {full["sequential_images_per_second"]:.2f} 图/秒，不能视为最大批处理吞吐率。\n\n'
    for r in rows[1:]:
        report+=f'FULL 相对 {r["model"]} 的平均墙钟比值为 {full["wall_mean_ms"]/r["wall_mean_ms"]:.3f}；显存峰值分配差为 {full["peak_allocated_mib"]-r["peak_allocated_mib"]:+.2f} MiB。\n'
    report+='''
## 计时与显存口径、限制

墙钟时间从已预载、已 pin 的 CPU 张量开始，包含 H2D 输入传输、Python 调度、完整网络、GPU 后处理、D2H 结果传输和同步。GPU 区间由 CUDA events 测量，覆盖网络和 GPU 后处理，不含 H2D/D2H。**两者都不包含文件读取、图像 resize、保护嵌入/载荷生成、篡改生成、模型载入和预热**，因此不能写成包含磁盘输入的端到端应用耗时。

显存为单独进程的 PyTorch CUDA allocator 峰值，包含冻结骨干在内的模型权重、输入和中间张量。“保留量”包含缓存分配器持有的显存，不等于当前实际使用量；不能将其和“分配量”混用。CUDA context、驱动和其他桌面程序显存不包含在其中。预热后重置峰值计数，但保留缓存以测量稳定推理。单位 MiB=2^20 bytes。

冻结 ViT 仍然占用参数存储和推理计算；1.4M 是可训练参数量，并非整个模型参数量。此报告直接测量推理成本，不根据可训练参数推断 FLOPs。

这是同一台笔记本、同一 PyTorch/CUDA 环境中的工程测量。桌面图形程序、功耗、温度和 GPU 频率可能影响结果；每模型开始与结束的 GPU 状态见 summary.json。三遍重复和逐图像耗时 SD 描述工程计时变化，不是三个训练种子的统计置信区间。没有启用 TensorRT、编译优化、AMP 或动态批处理，也没有以计时结果调整结构。

全部检查点与输入已校验，第一遍推理的检测分数及 micro Pixel F1 与归档 Validation 结果一致。原训练结果与主论文源文件没有改动。
'''
    (OUT/'inference_efficiency_report.md').write_text(report,encoding='utf-8')
    tex=r'''% Measured inference efficiency supplement; no training or manuscript overwrite.
\paragraph{Inference efficiency.}
We measured the existing best checkpoints from seed 20261001 on an
NVIDIA GeForce RTX 3060 Laptop GPU (6GB), using batch size one, FP32
inputs and weights, no autocast, and evaluation/inference mode.
Matmul TF32 was disabled and cuDNN TF32 was permitted, matching the
existing runtime defaults. Each model was warmed up for 30 calls and
evaluated three times on the same 124 preloaded Validation inputs.
The complete verification pipeline includes the frozen payload decoder
and ViT, evidence fusion, the graph or control module, both heads,
localization upsampling and thresholding. Wall time additionally
includes CPU--GPU input/output transfers and synchronization; file I/O,
resizing, input protection and model loading are excluded.
Peak allocated memory is measured by the PyTorch CUDA allocator and
includes model weights and intermediate tensors, but excludes driver
contexts and other applications. These hardware-specific repeated-input
timings are engineering measurements, not independent-test confidence
intervals or FLOP estimates.

\begin{table*}[!tbp]
\centering
\caption{Inference efficiency on the RTX 3060 Laptop GPU. Mean wall time
is shown with sample SD over 372 timed calls; GPU time is the mean
CUDA-event interval of the network plus GPU postprocessing. Memory
is peak allocated PyTorch CUDA memory (MiB). Checkpoints use the same
training seed; timing repetitions are not training-seed replications.}
\label{tab:e10_inference_efficiency}
\small
\begin{tabular}{@{}lrrrrr@{}}
\toprule
Model & Trainable/M & Total/M & Wall/ms & GPU/ms & Peak MiB \\
\midrule
'''
    for r in rows:
        tex+=r['model'].replace('_',r'\_')+f' & {r["trainable_parameters"]/1e6:.3f} & {r["total_parameters"]/1e6:.3f} & '+f'${r["wall_mean_ms"]:.2f}'+r'\pm'+f'{r["wall_sd_ms"]:.2f}$ & {r["gpu_mean_ms"]:.2f} & {r["peak_allocated_mib"]:.1f}'+r' \\'+'\n'
    tex+=r'''\bottomrule
\end{tabular}
\end{table*}
'''
    (OUT/'paper_inference_efficiency.tex').write_text(tex,encoding='utf-8')
    (OUT/'RUN_COMMANDS.md').write_text("# 复现命令\n\n```powershell\nSet-Location 'D:\\SSPG-Net'\n$env:HF_HUB_OFFLINE='1'\n& '.\\.venv\\Scripts\\python.exe' 'scripts\\benchmark_e10_inference_efficiency_v1.py' --all\n```\n\n已有测量输出不会被覆盖；复测请通过 --output 指定新的独立目录。每个模型在独立进程中依次测量。\n",encoding='utf-8')
    write_json(OUT/'status.json',{'state':'COMPLETE','completed_models':MODELS,'retraining':False,
        'input_consistency':'PASS','archived_inference_consistency':'PASS'})
    write_json(OUT/'sha256_index.json',{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='sha256_index.json'})
    print(frame[['model','wall_mean_ms','gpu_mean_ms','peak_allocated_mib','peak_reserved_mib']].to_string(index=False),flush=True)


def pipeline():
    import torch
    if OUT.exists():
        raise RuntimeError('Efficiency output exists; use --output for a new independent measurement')
    OUT.mkdir(parents=True)
    inputs={name:verify_checkpoint(name)[1] for name in MODELS}
    protocol=dict(experiment='E10-INF-EFF-v1',checkpoint_seed=SEED,models=MODELS,warmup=WARMUP,
        repetitions=REPEATS,n_inputs=124,batch_size=1,python=sys.executable,platform=platform.platform(),
        torch_version=torch.__version__,cuda_version=torch.version.cuda,checkpoint_hashes=inputs,
        benchmark_source_sha256=sha(Path(__file__)),gpu_at_start=gpu_snapshot(),
        timing_scope='Complete verifier + postprocessing; wall includes H2D/D2H; input generation/I/O excluded',
        retraining=False,test_accessed=False,casia_accessed=False,e9_accessed=False)
    write_json(OUT/'protocol.json',protocol)
    completed=[]
    for name in MODELS:
        write_json(OUT/'status.json',{'state':'RUNNING','model':name,'completed_models':completed})
        with (OUT/f'{name}.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--model',name,'--output',str(OUT)],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,HF_HUB_OFFLINE='1',PYTHONIOENCODING='utf-8'))
        if result.returncode:
            write_json(OUT/'status.json',{'state':'FAILED','model':name,'completed_models':completed,'returncode':result.returncode})
            raise RuntimeError(f'Benchmark failed: {name}; inspect its log')
        completed.append(name)
        print('Measured:',name,flush=True)
    summarize()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--all',action='store_true')
    mode.add_argument('--model',choices=MODELS)
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args()
    OUT=args.output.resolve()
    if args.all:
        pipeline()
    else:
        worker(args.model)

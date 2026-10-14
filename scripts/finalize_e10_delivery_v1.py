"""Finalize E10 artifacts after training and rendered visual QA."""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import zipfile
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_e10_spatial_mixer_v1 import OUT,SEEDS,verify_frozen,verify_completed,sha,write_json


def main():
    design=verify_frozen()
    assert all(verify_completed(seed) for seed in SEEDS)
    for path in OUT.glob('*access.json'):
        assert not json.loads(path.read_text())['forbidden_attempts']
    for seed in SEEDS:
        assert not json.loads((OUT/str(seed)/'image_access.json').read_text())['forbidden_attempts']
    for stem in ['figure_e10_comparison','figure_e10_parameters']:
        collision=json.loads((OUT/f'{stem}.collision.json').read_text())
        assert collision['summary']['fail']==0
    # Keep the frozen training/analysis scripts unchanged; this delivery-only
    # adjustment prevents 125 mm artwork being shrunk into an 89 mm column.
    path=OUT/'paper_e10_results.tex'
    tex=path.read_text(encoding='utf-8')
    tex=tex.replace(r'\begin{figure}[!tbp]',r'\begin{figure*}[!tbp]')
    tex=tex.replace(r'\includegraphics[width=\columnwidth]{figure_e10_parameters.pdf}',
                    r'\includegraphics[width=0.70\textwidth]{figure_e10_parameters.pdf}')
    tex=tex.replace(r'\end{figure}',r'\end{figure*}')
    path.write_text(tex,encoding='utf-8')
    logs=[]
    for _ in range(2):
        result=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','-file-line-error','paper_e10_preview.tex'],
            cwd=OUT,capture_output=True,text=True,encoding='utf-8',errors='replace')
        logs.append(result.stdout+result.stderr)
        if result.returncode:
            raise RuntimeError(result.stdout+result.stderr)
    (OUT/'preview_compile.log').write_text('\n'.join(logs),encoding='utf-8')
    import pymupdf
    qa_dir=OUT/'visual_qa'
    qa_dir.mkdir(exist_ok=True)
    doc=pymupdf.open(OUT/'paper_e10_preview.pdf')
    for i,page in enumerate(doc):
        page.get_pixmap(matrix=pymupdf.Matrix(1.5,1.5)).save(qa_dir/f'page{i+1}.png')
    source_dir=OUT/'source'
    source_dir.mkdir(exist_ok=True)
    source_files=['models/sspg_spatial_mixer_matched_v2.py','scripts/run_e10_spatial_mixer_v1.py',
        'scripts/summarize_e10_spatial_mixer_v1.py','scripts/finalize_e10_delivery_v1.py',
        'tests/test_e10_spatial_mixer_v1.py']
    for rel in source_files:
        shutil.copy2(ROOT/rel,source_dir/Path(rel).name)
    raw=pd.read_csv(OUT/'e10_per_seed_metrics.csv')
    summary=pd.read_csv(OUT/'e10_comparison_summary.csv')
    paired=pd.read_csv(OUT/'e10_paired_differences.csv')
    difference=paired['full_minus_spatial_localization_pixel_f1']
    generated_files=sorted(str(p.relative_to(OUT)) for p in OUT.rglob('*') if p.is_file())
    report='''# E10-SCM-v1 实验审计报告

## 完成情况

三种子训练、预检、原始预测复算、检查点/指标哈希核查、图件布局及目视检查均已完成。
空间卷积混合器共 1,416,100 个可训练参数，FULL 为 1,416,490 个；差异 390 个（0.0275%）。扩张宽度 832 仅按参数总数确定，并在训练前冻结。

## 各种子的真实结果

| 种子 | 最佳轮次 | 完成轮数 | AUC | Balanced Accuracy | Detection F1 | Sensitivity | Specificity | micro Pixel F1 | micro IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
'''
    new=raw[raw['model'].eq('SPATIAL_MIXER_MATCHED')]
    for r in new.itertuples():
        values=[r.detection_auc,r.detection_balanced_accuracy,r.detection_f1,r.detection_sensitivity,
                r.detection_specificity,r.localization_pixel_f1,r.localization_pixel_iou]
        report+=f'| {r.seed} | {int(r.best_epoch)} | {int(r.epochs_completed)} | '+' | '.join(f'{v:.6f}' for v in values)+' |\n'
    report+='\n## 与既有对照比较\n\n| 模型 | 可训练参数 | Pixel F1 均值 ± 样本 SD | AUC 均值 ± 样本 SD |\n|---|---:|---:|---:|\n'
    for r in summary.itertuples():
        report+=f'| {r.model} | {r.trainable_parameters:,} | {r.localization_pixel_f1_mean:.6f} ± {r.localization_pixel_f1_sd:.6f} | {r.detection_auc_mean:.6f} ± {r.detection_auc_sd:.6f} |\n'
    report+='\n三个 FULL − SPATIAL_MIXER_MATCHED Pixel F1 配对差值：'+', '.join(f'{v:+.6f}' for v in difference)+f'。均值 {difference.mean():+.6f}，样本 SD {difference.std(ddof=1):.6f}。\n'
    report+='''
## 谨慎结论

本次三个种子均支持 FULL 在固定 IMD2020 Validation 划分上具有优于此两层局部卷积混合器的定位表现。空间混合器的均值高于既有参数匹配逐 token MLP，但低于原始较小 NO_GRAPH；不能据此宣称所有普通空间交互模型都不如图模型，也不能把差异唯一归因于图拓扑。

Validation 参与检查点和检测阈值选择，不是新的盲测。n=3 的 sample SD 不是独立测试集 CI，未计算或编造 p 值。参数数量匹配不等于 FLOPs 或运行时间匹配；局部卷积与图模型的信息传播范围也不同。未对 Test、CASIA2、COCO E9 进行读取图片/推理或性能选择，未向 GitHub 推送。

## 审计证据

- 原 E2 FULL、NO_GRAPH、Matched NO_GRAPH、NO_PROVENANCE 的表14数值重算一致，历史指标和检查点均符合 lineage 哈希。
- 新检查点保存的 AdamW 一阶矩张量直接计数均为 1,416,100 个已优化参数。
- 每个新种子 124 条预测包含 source id、检测分数及四项定位像素计数；检测 ROC/Youden J/混淆矩阵及全体像素计数重算一致。
- 新增 mixer 测试确认相邻及两跳 patch 的传播、超过两层 3×3 范围的无直接依赖、跨样本独立、前向形状、全部可训练模块梯度、冻结 decoder/ViT 无梯度和 eval 状态、可复现输出。
- 原 head 旧测试使用 LayerNorm 不变的通道常数扰动而产生错误失败；项目已有 R1 修正版，通过该修正版核查，保留原失败日志。未改变历史模型或历史测试文件。
- Train/Validation 全量数据管线集成测试通过，无 canonical key/group/path 交叉。运行时图片文件访问白名单仅含该两组源图，所有访问记录无禁止尝试。
- 两张最终矢量图的字体检查、布局检查及碰撞检查通过，并逐面板目视检查。主性能图纵轴为 0～1，配对差值图包含零线和双向范围；没有删掉种子或样本。
- 静态图源检查的 LaTeX 下标警告来自论文片段字符串，图中无对应数学下标；最终图的实际字体分别最低 6.3/6.5 pt。PNG 为 600 dpi 预览，PDF/SVG 为主要矢量交付；实际宽度 180/125 mm。
- 补充 TeX 单独编译为预览。参数图以跨栏 70% 文本宽度显示，避免小字体缩入单栏。既有全文 TeX、图6/7及历史实验工件均未修改。
- 本地无 v1.1.0 tag，远程 ref 查询受 TLS 失败影响；以实际本地 E2 工件为准，Git 分支、commit、已有改动和源文件哈希均在 design.json 中记录。

## 新增代码完整路径

'''
    report+='\n'.join('- '+str(ROOT/rel) for rel in source_files)
    report+='\n\n## 输出目录\n\n'+str(OUT)+'\n\n复现命令见 RUN_COMMANDS.md。三个种子目录分别保留 best.pth、训练日志、协议、逐样本预测、指标与 lineage。所有文件 SHA-256 见 sha256_index.json。\n'
    (OUT/'audit_report.md').write_text(report,encoding='utf-8')
    commands=(OUT/'RUN_COMMANDS.md').read_text(encoding='utf-8')
    if 'finalize_e10_delivery_v1.py' not in commands:
        commands+='\n\n完成训练和汇总后，交付排版/打包：\n\n```powershell\n& \'.\\.venv\\Scripts\\python.exe\' \'scripts\\finalize_e10_delivery_v1.py\'\n```\n'
        (OUT/'RUN_COMMANDS.md').write_text(commands,encoding='utf-8')
    write_json(OUT/'status.json',dict(state='COMPLETE',completed_seeds=SEEDS,
        numerical_audit='PASS',preflight='PASS',visual_qa='PASS',figure_geometry_audit='PASS',
        test_accessed=False,casia_accessed=False,e9_accessed=False))
    exclude={'sha256_index.json','E10_delivery.zip'}
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name not in exclude]
    write_json(OUT/'sha256_index.json',{str(p.relative_to(OUT)):sha(p) for p in files})
    with zipfile.ZipFile(OUT/'E10_delivery.zip','w',zipfile.ZIP_DEFLATED) as package:
        for path in OUT.rglob('*'):
            if path.is_file() and path.name!='E10_delivery.zip' and path.suffix not in ['.pth','.tmp','.aux','.out']:
                package.write(path,path.relative_to(OUT))
    print('COMPLETE:',OUT,flush=True)


if __name__=='__main__':
    main()

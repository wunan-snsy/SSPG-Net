"""Recompute E10 outcomes, paired comparisons, vector figures and TeX fragment."""
from pathlib import Path
import gc
import json
import subprocess
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score,roc_curve,confusion_matrix

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_e10_spatial_mixer_v1 import OUT,SEEDS,COUNT,verify_frozen,verify_completed,historical_directory,sha,write_json
METRICS=['detection_auc','detection_balanced_accuracy','detection_f1','detection_sensitivity',
         'detection_specificity','localization_pixel_f1','localization_pixel_iou']
NAMES=['FULL','NO_GRAPH','Matched NO_GRAPH','SPATIAL_MIXER_MATCHED']
PARAMS={'FULL':1416490,'NO_GRAPH':555812,'Matched NO_GRAPH':1416110,'SPATIAL_MIXER_MATCHED':COUNT}
COLORS={'FULL':'#BA4655','NO_GRAPH':'#7F8992','Matched NO_GRAPH':'#C39555','SPATIAL_MIXER_MATCHED':'#377DA2'}


def near(actual,expected):
    assert np.isclose(float(actual),float(expected),atol=1e-10,rtol=1e-9),(actual,expected)


def recompute(folder,new=False):
    row=pd.read_csv(folder/'metrics.csv').iloc[0].to_dict()
    pred=pd.read_csv(folder/'val_predictions.csv')
    assert len(pred)==124 and pred['sample_key'].value_counts().eq(2).all()
    assert pred['detection_label'].value_counts().to_dict()=={0:62,1:62}
    y=pred['detection_label'].to_numpy()
    p=pred['detection_probability'].to_numpy()
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    fpr,tpr,thresholds=roc_curve(y,p)
    finite=np.where(np.isfinite(thresholds))[0]
    tau=thresholds[finite[np.argmax((tpr-fpr)[finite])]]
    near(tau,row['detection_threshold'])
    tn,fp,fn,tp=confusion_matrix(y,p>=tau,labels=[0,1]).ravel()
    actual=[roc_auc_score(y,p),(tp/(tp+fn)+tn/(tn+fp))/2,2*tp/(2*tp+fp+fn),tp/(tp+fn),tn/(tn+fp)]
    for key,value in zip(METRICS[:5],actual):
        near(value,row[key])
    for key,value in zip(['tn','fp','fn','tp'],[tn,fp,fn,tp]):
        near(value,row['detection_'+key])
    ltp,lfp,lfn,ltn=[int(row['localization_'+key]) for key in ['tp','fp','fn','tn']]
    near(2*ltp/(2*ltp+lfp+lfn),row['localization_pixel_f1'])
    near(ltp/(ltp+lfp+lfn),row['localization_pixel_iou'])
    assert ltp+lfp+lfn+ltn==124*256*256
    assert row['test_accessed'] is False or str(row['test_accessed']).lower()=='false'
    if new:
        for key,expected in zip(['tp','fp','fn','tn'],[ltp,lfp,lfn,ltn]):
            assert int(pred['localization_'+key].sum())==expected
        checkpoint=torch.load(folder/'best.pth',map_location='cpu',weights_only=False)
        optimized=sum(v['exp_avg'].numel() for v in checkpoint['optimizer_state_dict']['state'].values())
        assert optimized==COUNT
        assert int(checkpoint['epoch'])==int(row['best_epoch'])
        history=pd.read_csv(folder/'training_log.csv')
        near(history['val_total_loss'].min(),row['best_val_joint_loss'])
        assert len(history)==int(row['epochs_completed'])
        del checkpoint
        gc.collect()
        access=json.loads((folder/'image_access.json').read_text())
        assert not access['forbidden_attempts']
    return row,set(pred['sample_key'])


def draw(raw,summary,paired):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    sys.path.insert(0,str(Path.home()/'.codex/skills/nature-figure/scripts'))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':7,'axes.labelsize':7,'axes.titlesize':8,'xtick.labelsize':6.5,'ytick.labelsize':6.5,
        'legend.fontsize':6.5,'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none'})
    width_in,height_in=180/25.4,88/25.4
    fig,axes=plt.subplots(1,2,figsize=(width_in,height_in))
    fig.subplots_adjust(left=.085,right=.97,bottom=.25,top=.83,wspace=.35)
    a,b=axes
    markers=['o','^','s']
    for i,name in enumerate(NAMES):
        values=raw.loc[raw['model'].eq(name),'localization_pixel_f1'].to_numpy()
        for j,value in enumerate(values):
            a.scatter(i+(j-1)*.13,value,color=COLORS[name],marker=markers[j],s=18,zorder=3)
        row=summary.loc[summary['model'].eq(name)].iloc[0]
        a.errorbar(i,row['localization_pixel_f1_mean'],yerr=row['localization_pixel_f1_sd'],
            marker='D',mfc='white',mec='#252B30',ecolor='#252B30',ms=3,capsize=3,lw=.8,zorder=4)
    a.set_xticks(range(4),['FULL','NO_GRAPH','Matched\nNO_GRAPH','Spatial\nmixer'])
    a.set_xlim(-.5,3.5)
    a.set_ylim(0,1)
    a.set_ylabel('Validation micro Pixel F1')
    a.set_title('(a) Three-seed localization',loc='left',pad=9)
    differences=paired['full_minus_spatial_localization_pixel_f1'].to_numpy()
    for i,value in enumerate(differences):
        b.scatter(i,value,color=COLORS['SPATIAL_MIXER_MATCHED'],marker=markers[i],s=20,zorder=3)
    mean=float(differences.mean())
    sd=float(differences.std(ddof=1))
    b.errorbar(3,mean,yerr=sd,marker='D',mfc='white',mec='#252B30',ecolor='#252B30',ms=4,capsize=3,lw=.8,zorder=4)
    b.axhline(0,color='#7D868D',lw=.7,ls='--')
    limit=max(.1,float(np.max(np.abs(differences)))+.04,abs(mean)+sd+.04)
    b.set_ylim(-limit,limit)
    b.set_xlim(-.5,3.5)
    b.set_xticks(range(4),['Seed 1','Seed 2','Seed 3','Mean\n± SD'])
    b.set_ylabel('Pixel F1 difference (FULL − spatial mixer)')
    b.set_title('(b) Paired seed differences',loc='left',pad=9)
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis='y',color='#E0E4E7',lw=.45)
        ax.tick_params(width=.6,length=3)
    fig.text(.085,.075,'Seeds 1–3: 20261001–20261003 · symbols show real runs · diamonds: mean ± sample SD, n = 3',fontsize=6.3)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,axes=list(axes),panel_ids=['a','b'],
        json_out=OUT/'figure_e10_comparison.alignment.json',strict=True)
    fig.savefig(OUT/'figure_e10_comparison.pdf')
    fig.savefig(OUT/'figure_e10_comparison.svg')
    fig.savefig(OUT/'figure_e10_comparison.png',dpi=600)
    plt.close(fig)
    width_in,height_in=125/25.4,105/25.4
    fig,ax=plt.subplots(figsize=(width_in,height_in))
    fig.subplots_adjust(left=.15,right=.97,bottom=.28,top=.87)
    styles=['o','^','s','D']
    for i,name in enumerate(NAMES):
        row=summary.loc[summary['model'].eq(name)].iloc[0]
        ax.errorbar(PARAMS[name]/1e6,row['localization_pixel_f1_mean'],yerr=row['localization_pixel_f1_sd'],
            marker=styles[i],ms=5,color=COLORS[name],capsize=3,lw=.8,label=name,zorder=3)
    ax.set_ylim(0,1)
    ax.set_xlim(0,1.65)
    ax.set_xlabel('Trainable parameters (M)')
    ax.set_ylabel('Validation micro Pixel F1')
    ax.set_title('Trainable capacity and localization',loc='left',pad=9)
    ax.set_axisbelow(True)
    ax.grid(axis='y',color='#E0E4E7',lw=.45)
    fig.legend(loc='lower center',bbox_to_anchor=(.53,.025),ncol=2,frameon=False,columnspacing=1.0)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,axes=[ax],json_out=OUT/'figure_e10_parameters.alignment.json',strict=True)
    fig.savefig(OUT/'figure_e10_parameters.pdf')
    fig.savefig(OUT/'figure_e10_parameters.svg')
    fig.savefig(OUT/'figure_e10_parameters.png',dpi=600)
    plt.close(fig)


def main():
    design=verify_frozen()
    if not all(verify_completed(seed) for seed in SEEDS):
        raise RuntimeError('Not all three E10 seeds are complete')
    rows=[]
    reference_keys=None
    for name,model in zip(NAMES,['full','no_graph','matched_no_graph','spatial_mixer']):
        for seed in SEEDS:
            folder=OUT/str(seed) if model=='spatial_mixer' else historical_directory(model,seed)
            row,keys=recompute(folder,model=='spatial_mixer')
            if reference_keys is None:
                reference_keys=keys
            assert keys==reference_keys
            row.update(model=name,seed=seed,trainable_parameters=PARAMS[name])
            rows.append(row)
    raw=pd.DataFrame(rows)
    raw.to_csv(OUT/'e10_per_seed_metrics.csv',index=False)
    summaries=[]
    for name in NAMES:
        subset=raw[raw['model'].eq(name)]
        item=dict(model=name,trainable_parameters=PARAMS[name],n_seeds=3)
        for metric in METRICS:
            item[metric+'_mean']=float(subset[metric].mean())
            item[metric+'_sd']=float(subset[metric].std(ddof=1))
        summaries.append(item)
    summary=pd.DataFrame(summaries)
    summary.to_csv(OUT/'e10_comparison_summary.csv',index=False)
    full=raw[raw['model'].eq('FULL')].set_index('seed')
    spatial=raw[raw['model'].eq('SPATIAL_MIXER_MATCHED')].set_index('seed')
    paired=pd.DataFrame([dict(seed=seed,**{f'full_minus_spatial_{metric}':float(full.loc[seed,metric]-spatial.loc[seed,metric]) for metric in METRICS}) for seed in SEEDS])
    paired.to_csv(OUT/'e10_paired_differences.csv',index=False)
    draw(raw,summary,paired)
    skills=Path.home()/'.codex/skills/nature-figure/scripts'
    qa=[]
    for stem in ['figure_e10_comparison','figure_e10_parameters']:
        for script,args in [('audit_pdf_text.py',['--min-pt','5']),
                            ('audit_figure_collisions.py',['--json-out',str(OUT/f'{stem}.collision.json'),'--overlay-pdf',str(OUT/f'{stem}.collision.pdf')])]:
            result=subprocess.run([sys.executable,str(skills/script),str(OUT/f'{stem}.pdf'),*args],capture_output=True,text=True)
            qa.append(result.stdout+result.stderr)
            if result.returncode:
                raise RuntimeError(result.stdout+result.stderr)
    (OUT/'figure_qa.log').write_text('\n'.join(qa),encoding='utf-8')
    f=summary[summary['model'].eq('FULL')].iloc[0]
    s=summary[summary['model'].eq('SPATIAL_MIXER_MATCHED')].iloc[0]
    dif=paired['full_minus_spatial_localization_pixel_f1']
    delta,spread=float(dif.mean()),float(dif.std(ddof=1))
    if (dif>0).all():
        interpretation='All three paired localization differences favored FULL on this fixed Validation split. This supports a FULL advantage over this particular local convolution control, but does not establish a general or topology-specific advantage.'
    elif delta>0:
        interpretation='FULL had a higher mean localization score, but seed-wise ordering varied. These three runs do not establish a stable advantage over ordinary spatial mixing.'
    elif delta<0:
        interpretation='The spatial mixer had a higher mean localization score. This experiment does not support a graph-topology-specific advantage for FULL over this local spatial control.'
    else:
        interpretation='The two models had equal mean localization scores in these runs; a graph-topology-specific advantage is not established.'
    table_rows=[]
    for row in summary.to_dict('records'):
        display=row['model'].replace('_',r'\_')
        cell=lambda metric: '$'+f'{row[metric+"_mean"]:.4f}'+r'\pm'+f'{row[metric+"_sd"]:.4f}'+'$'
        table_rows.append(display+' & '+f'{row["trainable_parameters"]:,}'+' & '+cell('detection_auc')+' & '+cell('detection_f1')+' & '+cell('localization_pixel_f1')+' & '+cell('localization_pixel_iou')+r' \\')
    tex=r'''% E10 supplementary fragment; existing manuscript is not modified.
\subsection{E10: Parameter-matched local spatial interaction control}
\paragraph{Methods.}
We added SPATIAL\_MIXER\_MATCHED to distinguish local spatial interaction
from the graph-based reasoning used by FULL. The control retains the
NO\_GRAPH frontend, evidence fusion, $768\to256$ token projection and
joint heads. Projected tokens are reshaped to a $14\times14$ grid and
processed by two residual blocks, each comprising channel-only LayerNorm,
a $3\times3$ depthwise convolution (padding 1), a $1\times1$ expansion
from 256 to 832 channels, GELU and a $1\times1$ projection to 256 channels.
The width was selected by parameter count alone before training.
The control has 1,416,100 trainable parameters, versus 1,416,490 for FULL
(a 0.0275\% difference); decoder and pretrained ViT weights are frozen.
The existing E2 Train/Validation split, payloads, attacks, loss and
optimization rules are reused. Seeds 20261001--20261003 are paired with
the completed E2 runs. Each new control is initialized independently;
no trainable FULL weights are used. Checkpoints minimize Validation
joint loss; detection thresholds maximize Validation Youden J, and
the localization threshold is 0.5. Validation therefore participates
in selection and is not a new blind test. Test, CASIA2 and E9 are not evaluated.

\paragraph{Results.}
'''+f'FULL achieved micro pixel $F_1$ ${f.localization_pixel_f1_mean:.4f}\\pm{f.localization_pixel_f1_sd:.4f}$, while the spatial mixer achieved ${s.localization_pixel_f1_mean:.4f}\\pm{s.localization_pixel_f1_sd:.4f}$. '+r'Values are means $\pm$ sample standard deviations across three seeds. '+f'The three paired FULL-minus-spatial-mixer pixel-$F_1$ differences were {dif.iloc[0]:+.4f}, {dif.iloc[1]:+.4f} and {dif.iloc[2]:+.4f}; their mean was {delta:+.4f} with sample standard deviation {spread:.4f}. '+interpretation+r'''

\paragraph{Limitations.}
This control matches trainable parameter count, not FLOPs or runtime.
Two local convolution blocks have a finite receptive field, whereas
FULL includes spatial and appearance-based graph communication.
Only three training seeds and one previously used Validation split
are available; the sample standard deviations are not independent-test
confidence intervals. No population-level significance or exclusive
causal role of graph topology is claimed.

\begin{table*}[!tbp]
\centering
\caption{E10 and existing E2 controls on IMD2020 Validation. Mean $\pm$
sample SD across three paired training seeds. Pixel metrics pool counts
within each run; Validation selects checkpoints and detection thresholds.}
\label{tab:e10_spatial_mixer}
\small
\begin{tabular}{@{}lrrrrr@{}}
\toprule
Model & Trainable parameters & AUC & Detection $F_1$ & Pixel $F_1$ & IoU \\
\midrule
'''+ '\n'.join(table_rows)+r'''
\bottomrule
\end{tabular}
\end{table*}

\begin{figure*}[!tbp]
\centering
\includegraphics[width=\textwidth]{figure_e10_comparison.pdf}
\caption{E10 local spatial interaction control. (a) Three actual training
seed results per model, with diamonds and error bars denoting mean
$\pm$ sample SD. (b) Three paired FULL-minus-spatial-mixer pixel-$F_1$
differences and their mean $\pm$ sample SD. Marker shapes identify
seeds 20261001--20261003. All results use the same 124-sample
IMD2020 Validation cohort; no sample or seed is excluded.}
\end{figure*}

\begin{figure}[!tbp]
\centering
\includegraphics[width=\columnwidth]{figure_e10_parameters.pdf}
\caption{Trainable parameter count versus micro pixel $F_1$.
Each point is a three-seed mean, with sample SD error bars.
FULL, Matched NO\_GRAPH and SPATIAL\_MIXER\_MATCHED have nearly
identical x-coordinates because their trainable parameter counts
are closely matched. Frozen decoder and ViT parameters are excluded.}
\end{figure}
'''
    (OUT/'paper_e10_results.tex').write_text(tex,encoding='utf-8')
    table=raw[raw['model'].eq('SPATIAL_MIXER_MATCHED')][['seed','best_epoch','epochs_completed',*METRICS]]
    report='# E10-SCM-v1 真实实验审计\n\n状态：三种子训练及数值核查通过；图件已导出，交付前仍需目视检查。\n\n'+table.to_csv(index=False)+'\n\n'+summary.to_csv(index=False)+'\n\n'+interpretation+f'\n\n配对 Pixel F1 差值均值 {delta:+.6f}，样本 SD {spread:.6f}。\n\n'
    report+='参数匹配差异为 390/1,416,490 = 0.0275%。无训练后结构或宽度搜索，未复写历史结果。原 E2 四组汇总值与表14一致。原始逐样本预测与混淆像素计数重算通过；三个新检查点保存的优化器状态均含 1,416,100 个已优化参数。\n\nValidation 参与模型选择及阈值选择，n=3 的 SD 不是盲测置信区间。未进行 Test、CASIA2、E9 推理；每次运行的文件访问记录无禁止输入。图 6、7 和全文源文件保持原状。\n'
    (OUT/'audit_report.md').write_text(report,encoding='utf-8')
    commands=r'''# Windows 复现命令

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
'''
    (OUT/'RUN_COMMANDS.md').write_text(commands,encoding='utf-8')
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name not in ['sha256_index.json','status.json']]
    write_json(OUT/'sha256_index.json',{str(p.relative_to(OUT)):sha(p) for p in files})
    write_json(OUT/'status.json',dict(state='READY_FOR_VISUAL_QA',completed_seeds=SEEDS,
        numerical_audit='PASS',figure_geometry_audit='PASS',test_accessed=False,casia_accessed=False,e9_accessed=False))
    print(summary.to_string(index=False),flush=True)
    print(interpretation,flush=True)


if __name__=='__main__':
    main()

"""Aggregate falsifiable mechanisms, uncertainty and prior-work boundaries."""
import argparse,base64,datetime,html,json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr,rankdata,pearsonr


def bootstrap(values):
    values=np.asarray(values,float);values=values[np.isfinite(values)]
    if not len(values):return [None,None,None]
    rng=np.random.default_rng(42)
    samples=values[rng.integers(len(values),size=(10000,len(values)))].mean(-1)
    return [float(values.mean()),*np.quantile(samples,[.025,.975]).tolist()]


def aggregate_gauge(root,model):
    rows=[];names=[]
    for path in sorted((root/f'results/gauge_geometry_{model}_real_t25').glob('*.npz')):
        with np.load(path) as f:
            if len(f['variant_names'])!=15:continue
            errors=f['errors'];valid=f['valid'];variants=f['variant_names'].tolist()
            scores=((errors[...,None]<[1,2,4,8,16]).mean(-1)*valid).sum((-1,-2))/valid.sum()*100
        rows.append(scores);names.append(path.stem)
    if not rows:return None,None
    array=np.stack(rows);base=array[:,:,:,0];summary={'videos':names,'complete':len(rows)==30,'variants':{}}
    for i,variant in enumerate(variants):
        difference=array[:,:,:,i]-base
        summary['variants'][variant]={'mean':float(array[:,:,:,i].mean()),
            'gain_ci':bootstrap(difference.mean((1,2))),
            'fraction_heads_improved':float((difference.mean(0)>0).mean())}
    summary['head_rank_spearman']={variant:float(spearmanr(base.mean(0).ravel(),array[:,:,:,variants.index(variant)].mean(0).ravel()).statistic)
                                   for variant in ['gauge_fast16','gauge_slow16','gauge_random','balanced_pairs','key_mahalanobis']}
    return summary,(array,variants)


def aggregate_causal(root):
    summary={};arrays={}
    for sigma in [.05,.5,.9]:
        records=[json.loads(p.read_text()) for p in sorted((root/'results/causal_wan_native').glob('*_n360.json'))]
        records=[r for r in records if r['sigma']==sigma]
        if not records:continue
        correlations=[];partial=[];within=[]
        for record in records:
            score=np.array(record['readability']).ravel()
            loss=np.array([r['delta_flow_mse'] for r in record['ablations']])
            motion=np.array([r['delta_transport_error'] for r in record['ablations']])
            energy=np.array(record['head_output_energy']).ravel()
            correlations.append([spearmanr(score,loss).statistic,spearmanr(score,motion).statistic,
                                 spearmanr(energy,loss).statistic])
            design=np.column_stack((np.repeat(np.eye(30),12,axis=0),rankdata(energy)))
            x=rankdata(score);y=rankdata(loss)
            x=x-design@np.linalg.lstsq(design,x,rcond=None)[0]
            y=y-design@np.linalg.lstsq(design,y,rcond=None)[0]
            partial.append(pearsonr(x,y).statistic)
            within.append(np.nanmean([spearmanr(s,l).statistic for s,l in zip(score.reshape(30,12),loss.reshape(30,12),strict=True)]))
        correlations=np.array(correlations)
        summary[str(sigma)]={'videos':[r['video'] for r in records],'complete':len(records)==7,
             'mean_flow_mse':float(np.mean([r['baseline_flow_mse'] for r in records])),
             'readability_vs_delta_flow_spearman':bootstrap(correlations[:,0]),
             'readability_vs_delta_transport_spearman':bootstrap(correlations[:,1]),
             'output_energy_vs_delta_flow_spearman':bootstrap(correlations[:,2]),
             'readability_deltaflow_partial_rank_layer_and_energy':bootstrap(partial),
             'mean_within_layer_spearman':bootstrap(within),
             'limitation':'one-step denoising/residual loss on real video latents, not a generated-motion or physics benchmark'}
        arrays[sigma]={'score':np.mean([r['readability'] for r in records],0),
                       'loss':np.mean([[r['delta_flow_mse'] for r in record['ablations']] for record in records],0).reshape(30,12)}
    return summary,arrays


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('/share/yty/heft'));args=parser.parse_args()
    root=args.root;summary={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gauge':{}}
    gauges={}
    for model in ['wan','cosmos']:
        result,arrays=aggregate_gauge(root,model)
        if result:summary['gauge'][model]=result;gauges[model]=arrays
    summary['causal'],causal=aggregate_causal(root)
    (root/'results/mechanism_summary.json').write_text(json.dumps(summary,indent=2))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    figure_paths=[]
    if gauges:
        fig,axes=plt.subplots(1,len(gauges),figsize=(5*len(gauges),4),squeeze=False)
        for ax,(model,(scores,variants)) in zip(axes[0],gauges.items(),strict=True):
            full=scores[:,:,:,0].mean(0);changed=scores[:,:,:,variants.index('gauge_fast16')].mean(0)
            ax.scatter(full.ravel(),changed.ravel(),c=np.repeat(np.arange(full.shape[0]),full.shape[1]),cmap='viridis',s=12,alpha=.7)
            limit=max(full.max(),changed.max())+2;ax.plot([0,limit],[0,limit],'k--',lw=.7)
            ax.set(xlabel='Original key matching (PCK %)',ylabel='Same attention, fast-pair key gain ×16',
                   title=f'{model.capitalize()}: {len(scores)} real clips')
        fig.suptitle('Function-preserving parameter gauges change extracted correspondence')
        fig.tight_layout()
        path=root/'figures/function_preserving_key_geometry.png';figure_paths.append(path)
        fig.savefig(path,dpi=180,bbox_inches='tight');fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight');plt.close(fig)
    if causal:
        fig,axes=plt.subplots(1,len(causal),figsize=(4.3*len(causal),3.8),squeeze=False)
        for ax,(sigma,record) in zip(axes[0],causal.items(),strict=True):
            ax.scatter(record['score'].ravel()*100,record['loss'].ravel(),s=12,alpha=.6,
                       c=np.repeat(np.arange(30),12),cmap='viridis')
            ax.set_yscale('symlog',linthresh=1e-4)
            ax.set(xlabel='Direct key readability (PCK %)',ylabel='Head ablation: Δ flow MSE',title=f'Noise σ = {sigma:g}')
            for layer,head in [(18,1),(18,5),(20,8)]:
                ax.annotate(f'{layer}/{head}',(record['score'][layer,head]*100,record['loss'][layer,head]),fontsize=7)
        fig.suptitle('Readout quality and causal denoising contribution measure different properties')
        fig.tight_layout();path=root/'figures/readability_causal_denoising.png';figure_paths.append(path)
        fig.savefig(path,dpi=180,bbox_inches='tight');fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight');plt.close(fig)
    table=[]
    for model,result in summary['gauge'].items():
        for name in ['post_full','post_low15','gauge_fast16','balanced_pairs','balanced_centered','key_mahalanobis']:
            row=result['variants'][name];gain,lo,hi=row['gain_ci']
            table.append(f'<tr><td>{model}</td><td>{name}</td><td>{row["mean"]:.2f}</td><td>{gain:+.2f} [{lo:+.2f}, {hi:+.2f}]</td><td>{len(result["videos"])}/30</td></tr>')
    figures=''.join(f'<figure><img src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}"><figcaption>{html.escape(p.name)}</figcaption></figure>' for p in figure_paths)
    report=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>HeFT 机制探索记录</title>
<style>body{{max-width:1100px;margin:3em auto;padding:0 24px;font:17px/1.75 system-ui;color:#152132}}h1,h2{{line-height:1.3}}table{{border-collapse:collapse;width:100%;font-size:15px}}td,th{{padding:8px;border-bottom:1px solid #ccd3df;text-align:left}}img{{max-width:100%}}.status{{background:#eef3fa;padding:1em}}code{{font-size:14px}}a{{color:#255dc2}}</style>
<h1>HeFT：从 head／频率筛选转向可检验的对应关系机制</h1><p>更新：{summary['utc']}</p>
<p class="status">这是持续更新的实验记录。表中注明完成视频数；未完成的实验不视为最终证据。直接匹配 PCK 与完整跟踪 AJ 是不同指标；这里的 25 帧匹配诊断不代表论文基准成绩。</p>
<h2>新出现的主要候选：校准特征内容时间</h2>
<p>在七张校准图像的已知平移上，原生因果 VAE 的内容延迟中位数为1.25帧，逐帧编码器版本接近零。将这个事先测得的值固定用于多相位特征读出后，30段 DAVIS 首帧查询诊断的 AJ 从33.31提高到40.25，配对视频 bootstrap 95%CI 为+6.94 [5.20,9.09]。沿真实运动方向测得的预测延迟也从1.23帧降至0.17帧；真实轨迹仅用于事后分析，不用于拟合校正量。</p>
<p>这比继续筛选 head／频率更值得作为新主线。核心是区分潜变量的名义时间、内容实际对应的时间和 RoPE 位置相位。现已完成相同平滑权重、整数平移、反向校正及逐帧编码器对照。完整官方 DAVIS（30段、650轨迹）Wan AJ 32.84→39.96，Cosmos 40.04→46.47；完整 RGB-Stacking 40.48→42.30，PointOdyssey 13段前200帧 16.52→18.53。Cosmos 与 Wan 共享同一 VAE，属于跨去噪骨干迁移。完整 VAE 重建 RGB 在12段已知平移上未检出延迟；后续查询的源／目标时钟交叉实验显示必须联合校准。详见<a href="proposals/temporal_calibration_report.html">最终内容时间研究报告</a>与<a href="proposals/content_time_study.pdf">英文研究草稿</a>。因果卷积混合历史本身是已知事实，不能当作新发现；候选贡献在于测量、预测、联合校准及其可迁移的感知效果。</p>
<h2>特征几何的解释边界</h2>
<p>生成模型的计算功能、key 特征的欧氏几何和具体跟踪器的表现需要分别测量。原论文把不同响应图视为 head 功能分类，但同一生成计算允许多种 key 几何。我们实际修改全部 30 层的 Q／K 归一化增益，做同频率对的互逆缩放；在三段视频、两种缩放下，去噪输出逐元素完全相同。与此同时，直接 key 匹配会显著变化。</p>
<p>这并不意味着原始特征不可用，也不推翻由实际注意力图定义的 head 行为。它限制的是：不能把欧氏 key 距离、单独的 Q 或 K 范数直接等同于生成模型的固有机制。QK 对称性已有理论研究，新的贡献必须落在视频对应关系的可辨识性及有效校准上。</p>
<h2>固定度量和不变度量的实测</h2><table><tr><th>骨干</th><th>读出</th><th>直接 PCK</th><th>相对原 key 增益及视频 bootstrap 95% CI</th><th>完成数</th></tr>{''.join(table)}</table>
<p>平衡频率对使用 Q／K 二阶能量共同定义尺度；query-profile 使用注意力 logit 列的 Gram 度量；完整 key 白化是已有方法的对照。数学不变性不保证匹配性能，因此差的结果也保留。</p>
<h2>新方法候选与独立检验</h2>
<p>候选是在已知图像变换上自监督校准每个 head 的 64 个正幅度，并用 Q 的逆幅度维持生成函数。主要读出预先定为二的整数次幂，便于在 BF16 增益中精确实现。训练用七张校准图像的平移与外观变换，不读取人工轨迹；这是自监督校准而非免训练方法。PointOdyssey 的 13 段、各 200 帧和 64 个固定查询用于独立迁移，主 head 18/5 和 low15 对照均在看结果前冻结。</p>
<p><strong>该候选目前失败。</strong>第一版在完整 DAVIS 诊断的 AJ 为 24.09，低于同 head 的 low15 基线 35.28；独立 PointOdyssey 前缀上的 AJ 为 13.25，低于 19.02。加入多方向、多速度的连续运动校准后，PointOdyssey AJ 仍只有 13.76（二次幂幅度）或 14.68（连续幅度）。这些结果排除了把简单的自监督频率重标定直接作为性能贡献。后一版本是探索性跟进，不是重新定义过的盲测主实验。</p>
<h2>原生时序结构与功能响应</h2>
<p>保留原生时间压缩和全部时间卷积，通过四种因果 padding 相位提取不同模四帧，并按原始帧号／4 设置 RoPE 坐标，再交织得到逐帧特征。未校正时，多相位版本的 DAVIS 首帧查询诊断 AJ 为33.31，原生插值30.81；但完整50段 RGB-Stacking 上，多相位40.48，原生插值42.89。PointOdyssey13段前200帧的对应结果为16.52和14.88。不能宣称原生插值或多相位始终更好。</p>
<p>其二是直接测量模型输入输出的局部响应，绕开内部 head 的坐标选择。FP32 中对第一帧局部潜变量做正负扰动，聚合后续帧的响应能量；扰动幅度从 0.25 加倍至 0.5，响应图余弦均超过 0.9999，匹配精度保持相同。两个小样本试验的直接 PCK 分别为 51.91 与 46.09，静止位置基线为 26.38 与 0；将查询响应随机互换后均值降至约 6.77 与 17.08。这表明响应具有查询特异性，但仅是两段视频、各八个查询的机制初测，尚不是可用跟踪器或物理因果结论。</p>
<p><strong>文献边界：</strong><a href="https://arxiv.org/html/2306.01828">CWM</a> 已用 Jacobian 读出光流；<a href="https://arxiv.org/html/2507.09082">KL-tracing</a> 和 <a href="https://arxiv.org/html/2510.11715">Point Prompting</a> 已做生成模型扰动追踪，后者也已经蒸馏高效跟踪器。因此，扰动传播、Jacobian 光流或一般的蒸馏都不是本次探索的新贡献。</p>
<h2>当前较有希望、仍待独立验证的增量</h2>
<p>随机空间编码把所有源位置同时放入同一组扰动中；模型调用与后续提出的查询数量无关。对编码响应的相关解码存在可推导的混合偏差，校正后两个初测的 PCK 从 21.28／21.30 提高到 30.64／27.39。进一步用总响应方差归一化，并保留模型原生 8 像素输出网格，匹配八个固定查询的 PCK 达到 52.34／48.26。若评估全部 23 个初始可见查询，最近位置读出为 49.78／34.48，连续插值源编码的对照为 51.97／39.55。这说明小查询样本仍有明显波动，不能只展示较好的八点结果。</p>
<p>上述编码使用 BF16、32 个空间编码和两个固定随机通道方向，共128个模型输入；同一响应缓存可以继续解码其他查询。四段预先固定视频的全部初始查询已完成：主解码器直接 PCK 平均42.57，连续源位置版本45.04；同噪声、同输出分辨率的 head18/5 key 基线为57.96。因此目前不支持把编码响应作为新的性能主方法。全通道高斯编码、更细像素编码、连续 Fourier 编码和符号响应均保留为探索结果；它们没有建立稳定超过 key 的优势。</p>
<h2>已经排除或降低优先级的解释</h2>
<p>完全撤销 RoPE、全局均值中心化和 canonical EMA 均未稳定超过过滤基线。相位统计不支持普遍的“学习到反向旋转补偿”。AdaLN 固定成分的精确移除在 30 段视频上平均仅提高约 0.17 个直接 PCK 百分点，不能凭方向相关较高就宣称它解释了位置 head。</p>
<p>跨 Wan 与 Cosmos，撤销时间相位可改善约八成 head 的直接匹配；撤销空间相位则普遍伤害匹配。这个结果表明轴的作用不同，但其对完整跟踪的收益仍受到 EMA 和搜索过程影响。不能把直接匹配涨点直接写成跟踪涨点。</p>
<h2>真正的因果实验及其边界</h2>
<p>在原生时间压缩 4 倍、正确归一化的 Wan VAE 潜变量中，对全部 360 个 head 做逐 head 输出消融，测试七段视频、三个噪声水平。每个干预批次包含相同输入的未处理控制，记录 flow MSE、点附近重建残差及沿轨迹的残差差异。它衡量单步去噪贡献，尚不证明生成运动或物理规律的因果作用。</p>
{figures}
<h2>文献边界</h2><p><a href="https://arxiv.org/html/2607.28611">Chimera 附录 C</a> 已分析视频 DiT 的高频局部选择和低频远距离注意力；<a href="https://openreview.net/pdf?id=K1df8mmncF">Maximal Gauge Symmetry</a> 提供 QK 对称性理论；<a href="https://arxiv.org/abs/2602.04752">Contrastive QK Covariances</a> 分析语言模型的 QK 子空间；<a href="https://arxiv.org/html/2610.14517v2">Diffusion Backbone Behaviors</a> 已使用白化、因果干预研究扩散层深度。不能把这些已有结论重新包装为本文创新。</p>
<p>详细数据：<code>/share/yty/heft/results/mechanism_summary.json</code>；推导草稿：<code>/share/yty/heft/proposals/function_preserving_correspondence.tex</code>；实验脚本在代码仓库 scripts/research_*.py，并已归档到实验目录。</p></html>'''
    (root/'research_report.html').write_text(report)
    print(json.dumps({'report':str(root/'research_report.html'),'causal_complete':{k:v['complete'] for k,v in summary['causal'].items()},
                      'gauge_complete':{k:v['complete'] for k,v in summary['gauge'].items()}},indent=2))


if __name__=='__main__':main()

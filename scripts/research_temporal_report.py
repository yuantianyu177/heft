"""Content-time calibration evidence, complete outcomes and paired uncertainty."""
import base64,datetime,html,json
from pathlib import Path
import numpy as np

def ci(x):
    x=np.asarray(x,float);x=x[np.isfinite(x)]
    if not len(x):return None
    rng=np.random.default_rng(42);sample=x[rng.integers(len(x),size=(10000,len(x)))].mean(1)
    return [float(x.mean()),*np.quantile(sample,[.025,.975]).tolist()]

def load(folder,variant='post_full',head=5):
    return {d['video']:d for p in sorted(folder.glob(f'*_l18h{head:02d}_{variant}.json')) for d in [json.loads(p.read_text())]}

def lag(folder,head=5):
    values={}
    for p in sorted(folder.glob(f'*_l18h{head:02d}_post_full.npz')):
        with np.load(p) as f:
            gt=f['gt']*256;pred=f['tracks']*256
            velocity=(gt[:,2:]-gt[:,:-2])/2;error=pred[:,1:-1]-gt[:,1:-1]
            squared=(velocity**2).sum(-1);valid=~f['gt_occluded']
            if 'gt_valid' in f:valid&=f['gt_valid']
            valid=valid[:,1:-1]&valid[:,:-2]&valid[:,2:]&(squared>.25)
            query_frames=f['query_points'][:,0,None] if 'query_points' in f else np.zeros((len(gt),1))
            valid&=np.arange(1,gt.shape[1]-1)[None]>query_frames
            projected=-(error*velocity).sum(-1)/np.maximum(squared,1e-20)
            if valid.any():values[p.stem.rsplit('_l18h',1)[0]]=float(np.median(projected[valid]))
    return {'video_median_lag_mean_ci95':ci(list(values.values())),'videos':len(values),'per_video_median':values,'speed_threshold_metric_pixels_per_frame':.5,
        'definition':'negative prediction error projected onto central GT velocity, divided by velocity squared; medians over all eligible points, no error truncation'}

def main():
    root=Path('/share/yty/heft');cal=json.loads((root/'results/vae_temporal_alignment.json').read_text());summary={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    calibration={}
    for mode,rows in cal['results'].items():
        a=np.array([r['delay_nominal_minus_effective'] for r in rows]);names=sorted(set(r['video'] for r in rows))
        image_means=[np.mean([r['delay_nominal_minus_effective'][1:] for r in rows if r['video']==n]) for n in names]
        calibration[mode]={'median_delay':float(np.median(a[:,1:])),'mean_delay_by_time':a.mean(0).tolist(),
            'image_bootstrap_mean_delay_ci95':ci(image_means),'images':names,'range':[float(a[:,1:].min()),float(a[:,1:].max())]}
    summary['calibration']=calibration
    for field,filename in [('codec_clock','codec_clock_summary.json'),('head_time_probe','head_time_probe_summary.json'),
        ('query_initialization','clock_initialization_interaction.json'),('query_time_subsets','query_clock_summary.json')]:
        path=root/'results'/filename
        if path.exists():summary[field]=json.loads(path.read_text())
    cog_path=root/'results/cogvideox_vae_temporal_alignment_calibration7.json'
    if cog_path.exists():
        cog=json.loads(cog_path.read_text());summary['second_encoder_calibration']={}
        for mode,rows in cog['results'].items():
            a=np.array([r['delay_nominal_minus_effective'][1:] for r in rows])
            summary['second_encoder_calibration'][mode]={'images':len(cog['images']),'median_delay':float(np.median(a)),'mean_delay':float(a.mean())}
    vp=root/'results/vae_temporal_alignment_validation.json'
    if vp.exists():
        validation=json.loads(vp.read_text());rows=validation['results']['native'];summary['calibration_validation']={}
        for speed in validation['speeds_input_pixels_per_frame']:
            a=[r['delay_nominal_minus_effective'][1:] for r in rows if max(map(abs,r['velocity_pixels_per_frame']))==speed]
            summary['calibration_validation'][str(speed)]={'median':float(np.median(a)),'mean':float(np.mean(a))}
    datasets={
        'Cosmos DAVIS complete official':(root/'results',30,{
            'raw':'benchmark_cosmos_raw','raw_low15':('benchmark_cosmos_raw','post_low15'),
            'native_polyphase':'benchmark_cosmos_polyphase','polyphase_corrected':'benchmark_cosmos_polyphase_corrected',
            'polyphase_smooth025':'benchmark_cosmos_polyphase_smooth025'}),
        'DAVIS complete official':(root/'results',30,{
            'raw':'benchmark_davis_raw','raw_low15':('benchmark_davis_raw','post_low15'),
            'native_polyphase':'benchmark_davis_polyphase','native_polyphase_low15':('benchmark_davis_polyphase','post_low15'),
            'polyphase_corrected':'benchmark_davis_polyphase_corrected','polyphase_corrected_low15':('benchmark_davis_polyphase_corrected','post_low15'),
            'interpolate_corrected':'benchmark_davis_interpolate_corrected'}),
        'DAVIS diagnostic':(root/'results',30,{
            'raw':'tracking_wan_full_grid','native_interpolate':'tracking_wan_native_interpolate_full','native_polyphase':'tracking_wan_native_polyphase_full',
            'interpolate_corrected':'tracking_wan_native_interpolate_latency125_full','polyphase_corrected':'tracking_wan_native_polyphase_latency125_full',
            'interpolate_phase_corrected':'tracking_wan_native_interpolate_latency125_phase_full','interpolate_reverse':'tracking_wan_native_interpolate_latency_m125_full',
            'raw_sham':'tracking_wan_raw_latency125_full','polyphase_smooth025':'tracking_wan_native_polyphase_latency025_full',
            'polyphase_integer100':'tracking_wan_native_polyphase_latency100_full','polyphase_reverse075':'tracking_wan_native_polyphase_latencym075_full'}),
        'RGB-Stacking complete':(root/'rgb_stacking/results',50,{
            'raw':'benchmark_raw','matched':'benchmark_normalized_mode_time025','native_interpolate':'benchmark_native_interpolate','native_polyphase':'benchmark_native_polyphase',
            'polyphase_corrected':'latency_transfer_polyphase','polyphase_smooth025':'latency_transfer_polyphase_smooth025',
            'interpolate_corrected':'latency_transfer_interpolate','raw_sham':'latency_transfer_raw_sham',
            'polyphase_ema0':'benchmark_ema0_native_polyphase','interpolate_ema0':'benchmark_ema0_native_interpolate',
            'raw_low15':('benchmark_raw','post_low15'),'native_polyphase_low15':('benchmark_native_polyphase','post_low15'),
            'polyphase_corrected_low15':('latency_transfer_polyphase','post_low15')}),
        'PointOdyssey 200-frame prefixes':(root/'pointodyssey/results',13,{
            'raw':'transfer_gauge_v1','matched':'native_transfer_matched','native_interpolate':'native_transfer_interpolate','native_polyphase':'native_transfer_polyphase',
            'polyphase_corrected':'latency_transfer_polyphase','polyphase_smooth025':'latency_transfer_polyphase_smooth025',
            'interpolate_corrected':'latency_transfer_interpolate','raw_sham':'latency_transfer_raw_sham',
            'raw_low15':('transfer_gauge_v1','post_low15'),'native_polyphase_low15':('native_transfer_polyphase','post_low15'),
            'polyphase_corrected_low15':('latency_transfer_polyphase','post_low15')})}
    summary['datasets']={};table=[]
    for label,(base,expected,folders) in datasets.items():
        head=6 if label.startswith('Cosmos') else 5
        records={key:load(base/value[0],value[1],head) if isinstance(value,tuple) else load(base/value,head=head) for key,value in folders.items()};out={'expected_videos':expected,'head':[18,head],'configurations':{},'paired_comparisons':{}}
        manifest=json.loads((base.parent/'data/davis_records/manifest.json').read_text())
        out['provided_tracks']=sum(v['points'] for v in manifest)
        primary_folder=base/folders['native_polyphase']
        out['eligible_queries']=0
        for prediction in primary_folder.glob(f'*_l18h{head:02d}_post_full.npz'):
            with np.load(prediction) as saved:out['eligible_queries']+=len(saved['gt'])
        for key,records_k in records.items():
            n=len(records_k)
            if not n:continue
            a=np.array([[d['metrics'][m] for m in ['average_jaccard','average_pts_within_thresh','occlusion_accuracy']] for d in records_k.values()])
            row={'n':n,'complete':n==expected,'AJ':float(a[:,0].mean()),'delta_average':float(a[:,1].mean()),'OA':float(a[:,2].mean())};out['configurations'][key]=row
            if n==expected:table.append(f'<tr><td>{label}</td><td>{key}</td><td>{n}</td><td>{row["AJ"]:.2f}</td><td>{row["delta_average"]:.2f}</td></tr>')
        for before,after in [('native_polyphase','polyphase_corrected'),('native_interpolate','interpolate_corrected'),
            ('polyphase_smooth025','polyphase_corrected'),('raw','raw_sham'),('native_polyphase','polyphase_integer100'),
            ('native_polyphase','polyphase_reverse075'),('native_polyphase','native_interpolate'),('polyphase_ema0','interpolate_ema0'),
            ('native_polyphase_low15','polyphase_corrected_low15'),('raw_low15','polyphase_corrected'),('raw_low15','polyphase_corrected_low15')]:
            if len(records.get(before,{}))!=expected or len(records.get(after,{}))!=expected:continue
            assert records[before].keys()==records[after].keys(),(label,before,after)
            delta=np.array([[records[after][v]['metrics'][m]-records[before][v]['metrics'][m] for m in ['average_jaccard','average_pts_within_thresh']] for v in sorted(records[before])])
            out['paired_comparisons'][after+' minus '+before]={'AJ_mean_ci95':ci(delta[:,0]),'delta_average_mean_ci95':ci(delta[:,1])}
        if len(records.get('polyphase_corrected',{}))==expected:
            out['projected_prediction_lag']={k:lag(base/folders[k],head) for k in ['raw','native_polyphase','polyphase_corrected']}
        summary['datasets'][label]=out
    (root/'results/temporal_calibration_summary.json').write_text(json.dumps(summary,indent=2))
    from research_clock_figures import make_clock_figures
    figure=make_clock_figures(root,summary,cal)[0]
    image=base64.b64encode(figure.read_bytes()).decode()
    factorial=base64.b64encode((root/'figures/query_target_clock_factorial.png').read_bytes()).decode()
    main_rows=[]
    for key,label in [('DAVIS complete official','Wan / DAVIS 官方完整协议'),('Cosmos DAVIS complete official','Cosmos / DAVIS 官方完整协议'),
                      ('RGB-Stacking complete','Wan / RGB-Stacking 完整50段'),('PointOdyssey 200-frame prefixes','Wan / PointOdyssey 13段前200帧')]:
        d=summary['datasets'][key];a=d['configurations']['native_polyphase'];b=d['configurations']['polyphase_corrected']
        change,low,high=d['paired_comparisons']['polyphase_corrected minus native_polyphase']['AJ_mean_ci95']
        main_rows.append(f'<tr><td>{label}</td><td>{a["AJ"]:.2f}</td><td>{b["AJ"]:.2f}</td><td>+{change:.2f} [{low:.2f},{high:.2f}]</td></tr>')
    report=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>内容时间校准实验</title><style>body{{max-width:1100px;margin:3em auto;padding:0 24px;font:17px/1.7 system-ui;color:#172337}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:8px;border-bottom:1px solid #cbd3df;text-align:left}}img{{max-width:100%}}.status{{padding:1em;background:#eef3fa}}</style>
<h1>建议的新主线：视频扩散特征的内容时间校准</h1><p>更新：{summary['utc']}</p>
<p class="status">已经完成机制测量、对照和跨模型／数据验证。新贡献候选是为冻结的视频生成特征建立可测的内容时间，再联合校准查询与目标读出。原有head／频率结果适合作为辅助分析；单纯继续筛选head或通道难以承载更强的新颖性。以下结果是固定配置的配对比较，不宣称最优跟踪器或保证论文录用。</p>
<p><a href="content_time_study.pdf">英文研究草稿（PDF）</a> · <a href="content_time_study.tex">LaTeX 源文件</a></p>
<h2>主要结果</h2><p>每行前后来自完全相同的原生特征缓存，只改变读出时间，保持head、搜索和EMA。1.25帧来自七张图像的已知平移，未使用人工轨迹拟合；迁移时不重新估计。置信区间为视频配对bootstrap 95%。</p>
<table><tr><th>模型／数据协议</th><th>原生未校准AJ</th><th>校准AJ</th><th>增益及95%CI</th></tr>{''.join(main_rows)}</table>
<h2>可检验的机制</h2><p>原生 Wan VAE 的时间压缩为四倍，潜变量的名义时间为0、4、8等。用七张校准图像的四方向恒速平移，对比相同编码时间的静态参考，可以测量潜变量内容对应的实际位移时间。非零时间的延迟中位数为1.25帧；移除时间下采样模块的版本接近零。因果结构和时间滤波已有研究，这里的候选增量是为密集感知测量、校准这个内容时间，而不是宣称发现因果卷积会混合历史。</p>
<p>读出在t+1.25采样特征，保持每段第一帧的参考描述子；模型权重和去噪计算不变。末端不足的部分暂时夹到最后一帧。0.25帧控制使用相同的插值权重，可以检验平滑是否足以解释收益；整数一帧控制没有额外插值平滑；反方向和逐帧编码器校正是机制对照。</p>
<img src="data:image/png;base64,{image}">
<h2>内部特征与完整编解码使用不同的内容时间</h2><p>用未修改的上游VAE实现及同一份ModelScope权重核验，三段视频的编码分布参数与研究恢复版本逐元素完全一致。在三张图像、四方向平移的编解码实验中，潜变量延迟中位数为1.25帧；完整VAE重建RGB的延迟在0.25帧测量分辨率下均为零。这支持“跳过解码器读取内部特征时暴露了时序接口偏差”的解释。它只验证了这些平移重建，不证明任意生成视频的运动都正确。</p>
<p>同样的七图校准协议下，CogVideoX VAE测得0.75帧，逐帧版本中位数为零。1.25不是通用常数；这些比较使用各自输入接口和分辨率，不单独隔离架构因素。当前完整下游迁移是Wan和Cosmos两个去噪器共享同一VAE，尚无CogVideoX下游跟踪验证。</p>
<h2>查询与目标需要联合校准</h2><p>固定八帧时长、所有目标可见、不使用EMA或局部搜索。新出现查询的直接PCK由47.59提高到61.63；只校准查询为46.50，只校准目标为41.94。联合校准的配对交互项为+20.78个百分点，95%CI[13.31,28.89]。在持续可见轨迹上人为把查询设到第8帧，也观察到联合校准收益。它说明校准的是描述子配对接口，不能仅解释为统一平移最终轨迹。</p>
<img src="data:image/png;base64,{factorial}">
<p>完整官方DAVIS中，后出现查询的AJ为25.66；只平移已有轨迹为26.72，增益CI跨零；校准描述子后为34.78。最初“公共延迟会在后续查询中简单抵消”的预测没有充分得到真实数据支持，不能包装成已证明结论。理想匀速、平移等变假设有解释作用，实际特征、可见性与分块初始化更复杂。首帧查询的官方包装与旧诊断在60个对照中预测逐元素相同。</p>
<h2>按论文贡献组织</h2><p>建议用“可测时间模型→预测真实偏差→无人工轨迹校准→联合读出→独立迁移”作为主论证链。去噪器不需要重训；原生时间卷积得以保留，四种因果padding相位提供逐帧特征。head选择用于取得已有对应信息：三个对应head在无EMA直接匹配中分别提高约9–10个PCK点，位置型head基本不变。时钟校准不能为缺少对应信息的head凭空提供跟踪能力。</p>
<details><summary>展开完整配置及对照结果</summary><table><tr><th>数据协议</th><th>读出</th><th>视频数</th><th>AJ</th><th>δ平均</th></tr>{''.join(table)}</table></details>
<h2>协议与创新边界</h2><p>标为“complete official”的DAVIS结果覆盖30段完整视频和全部650条轨迹，使用官方first-visible协议；“diagnostic”仅保留首帧可见查询。RGB-Stacking使用全部50段、250帧以及1500条标注中全部1498个可初始化查询；另2条全程不可见，无法按first-visible协议生成查询。PointOdyssey是全部13段的前200帧、每段64个固定查询，不能称为完整PointOdyssey榜单成绩。Wan固定18/5 head，Cosmos固定18/6，搜索参数16/48/24、EMA0.05。另列EMA0是对照，不按数据集选择最好的配置。</p>
<p>最初DAVIS时间校正试验的主分支是原生插值，多相位是次分支；观察到后者改善更大后，才把多相位校正设为独立迁移试验的主方法。这个选择历史保存在设计文件中。迁移结果之前冻结1.25帧、相位数和head。尚未证明对所有VAE、所有运动或其他感知任务成立。</p>
<p><a href="https://arxiv.org/abs/2411.06449">IV-VAE</a>已研究因果结构带来的帧间信息不平衡；<a href="https://arxiv.org/abs/2506.17220">DiffTrack</a>通过移除时间压缩恢复逐帧对应；<a href="https://arxiv.org/html/2605.12587">TrackCraft3R</a>也涉及时间位置对齐。因此，新贡献需要落在实测内容时间、可迁移的无人工标签校准及其反事实对照，而不是泛称时间对齐。</p>
<p>这些实验尚不证明一个固定配置在所有数据集最优。例如RGB-Stacking的无EMA原生插值对照高于主配置。低频过滤仍具有数据依赖性；固定原low15并加入校准时，Wan在官方DAVIS为42.01、RGB为36.46、PointOdyssey前缀为23.54。主对照使用同一特征缓存，避免把潜变量归一化、后验模式或时间卷积恢复的复合变化都归于时钟校准。</p>
<p>投稿完善项包括完整PointOdyssey／更大视频集合、独立编码器下游迁移、启动和末端边界及独立运行的成本比较。当前实现四相位需要额外编码器调用，没有声称速度优于逐帧版本。理论的标量时间滤波只是充分解释模型，实际延迟随内容和运动变化。更广泛的新颖性核查仍应继续，当前证据支持一条更强的新主线，不等于完成投稿准备。</p>
<p>原始统计：<code>/share/yty/heft/results/temporal_calibration_summary.json</code>。其他机制探索与负结果见<a href="../research_report.html">总实验记录</a>。</p></html>'''
    (root/'proposals/temporal_calibration_report.html').write_text(report)
    print(json.dumps({'calibration':calibration,'datasets':summary['datasets']},indent=2))

if __name__=='__main__':main()

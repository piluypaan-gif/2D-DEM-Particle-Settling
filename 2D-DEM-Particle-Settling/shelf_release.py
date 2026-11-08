"""二维离散元教学算例：支撑隔板上预沉降，瞬时撤板后下落、碰撞及堆积。

复用 dem_particles.py 的法向线性弹簧-阻尼接触模型。
本程序没有模拟隔板实际移动/抽出时的摩擦，也不包含颗粒间切向摩擦及转动。
单位采用 SI；粒子惯性质量按球体密度估计，属于二维截面演示而非真实二维厚度标定。

运行：python shelf_release.py --output shelf_results
仅验证：python shelf_release.py --test-only
跳过 GIF：python shelf_release.py --no-gif
"""
from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path
import numpy as np
from numba import njit
from dem_particles import compute_forces, DENSITY, STIFFNESS, GRAVITY, WIDTH, HEIGHT, DEFAULT_DT, validate

HEIGHTS = (0.10, 0.15, 0.20)  # 隔板顶面离容器底部的高度，m
DURATION = 3.2
PREPARE_DURATION = 3.0
PREPARE_ZETA = 0.70
RELEASE_ZETA = 0.35


@njit
def force_with_shelf(pos, vel, radius, mass, kn, zeta, gravity, width, shelf_y, shelf_enabled):
    """在原有的颗粒/壁面接触之外，增加从下方支撑颗粒的无摩擦水平隔板。"""
    forces, energy, maxratio, contacts = compute_forces(
        pos, vel, radius, mass, kn, zeta, gravity, width, True
    )
    shelf_force = 0.0
    if shelf_enabled:
        for i in range(len(radius)):
            delta = shelf_y + radius[i] - pos[i, 1]
            if delta > 0.0:
                cn = 2.0 * zeta * math.sqrt(kn * mass[i])
                fy = max(0.0, kn * delta - cn * vel[i, 1])
                forces[i, 1] += fy
                shelf_force += fy
                energy += 0.5 * kn * delta * delta
                maxratio = max(maxratio, delta / (2 * radius[i]))
                contacts += 1
    return forces, energy, maxratio, contacts, shelf_force


def make_initial_particles(n=30, seed=17, shelf_height=0.15):
    if n != 30:
        raise ValueError('当前演示固定30个颗粒；扩展数量时需要重新检查初始无重叠。')
    rng = np.random.default_rng(seed)
    radius = np.full(n, .0165) * rng.uniform(.94,1.06,n)
    pos = np.zeros((n,2))
    for i in range(n):
        row,col=divmod(i,6)
        pos[i,0] = .05 + .043 * col + .006 * (row%2) + rng.uniform(-.002,.002)
        pos[i,1] = shelf_height + .030 + .042 * row + rng.uniform(-.002,.002)
    for i in range(n):
        if pos[i,0] - radius[i] <=0 or pos[i,0]+radius[i]>=WIDTH:
            raise ValueError('初始颗粒在容器外。')
        for j in range(i+1,n):
            if np.linalg.norm(pos[i]-pos[j]) <=radius[i]+radius[j]:
                raise ValueError('初始颗粒互相重叠。')
    vel=np.zeros_like(pos)
    mass=DENSITY*(4/3)*math.pi*radius**3
    return pos,vel,radius,mass


def integrate(pos0, vel0, radius, mass, *, dt, duration, zeta, shelf_y, shelf_enabled,
              sample_every=1000, keep_snaps=True):
    steps=int(round(duration/dt))
    if abs(steps*dt-duration)>dt*1e-5:
        raise ValueError('模拟时长需要是时间步长的整数倍。')
    pos=pos0.copy(); vel=vel0.copy()
    rows=[]; snaps=[]; max_overlap=0.0
    for step in range(steps+1):
        forces, elastic, overlap, contacts, shelf_support = force_with_shelf(
            pos, vel, radius, mass, STIFFNESS, zeta, GRAVITY, WIDTH,
            shelf_y, shelf_enabled)
        if overlap>max_overlap: max_overlap=float(overlap)
        if step%sample_every==0 or step==steps:
            kinetic=float(0.5*np.sum(mass[:,None]*vel*vel))
            potential=float(np.sum(mass*GRAVITY*pos[:,1]))
            rows.append((step*dt,kinetic,potential,elastic,kinetic+potential+elastic,
                        float(np.mean(pos[:,1])),int(contacts),float(overlap),shelf_support))
            if keep_snaps: snaps.append(pos.copy())
        if step==steps: break
        vel += dt*forces/mass[:,None]
        pos += dt*vel
        if not np.isfinite(pos).all() or not np.isfinite(vel).all():
            raise FloatingPointError(f'非有限值：step={step}')
    return dict(final_pos=pos,final_vel=vel,history=np.asarray(rows),
                snapshots=np.asarray(snaps) if keep_snaps else None,
                max_overlap_ratio=max_overlap)


def csv_file(path, names, data):
    with open(path,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(names);w.writerows(data)


def check_shelf_unit():
    """单颗粒静置于隔板，验证支撑力、无侧向力、移除后的重力响应。"""
    r=np.array([.0165]); m=DENSITY*4/3*math.pi*r**3
    zeta=.7; h=.15
    p=np.array([[WIDTH/2, h+r[0]-m[0]*GRAVITY/STIFFNESS]])
    v=np.zeros((1,2))
    f, e, ov, nc, support = force_with_shelf(p,v,r,m,STIFFNESS,zeta,GRAVITY,WIDTH,h,True)
    f_after, *_ = force_with_shelf(p,v,r,m,STIFFNESS,zeta,GRAVITY,WIDTH,h,False)
    return [('shelf_support_relative_error',abs(support-m[0]*GRAVITY)/(m[0]*GRAVITY),abs(support-m[0]*GRAVITY)/(m[0]*GRAVITY)<1e-9),
            ('stationary_net_force_N',float(np.linalg.norm(f)),np.linalg.norm(f)<1e-9),
            ('released_vertical_acceleration_error_m_s2',abs(f_after[0,1]/m[0]+GRAVITY),abs(f_after[0,1]/m[0]+GRAVITY)<1e-10),
            ('no_shelf_horizontal_force_N',abs(f[0,0]),abs(f[0,0])<1e-12)]


def prepare_state(dt=DEFAULT_DT, duration=PREPARE_DURATION):
    shelf_h=.15
    p,v,r,m=make_initial_particles(shelf_height=shelf_h)
    out=integrate(p,v,r,m,dt=dt,duration=duration,zeta=PREPARE_ZETA,shelf_y=shelf_h,
                  shelf_enabled=True,sample_every=max(1,round(.04/dt)),keep_snaps=False)
    fp=out['final_pos']; fv=out['final_vel']
    f, se, over, ct, sf=force_with_shelf(fp,fv,r,m,STIFFNESS,PREPARE_ZETA,GRAVITY,WIDTH,shelf_h,True)
    ke=.5*np.sum(m[:,None]*fv*fv)
    mg=np.sum(m)*GRAVITY
    # 不考虑墙体的反力直接判断隔板反力总和接近粒子总重（侧壁无摩擦，地面离隔板有间距）。
    relative_support_error=abs(sf-mg)/mg
    normalized_netforce=np.max(np.linalg.norm(f,axis=1)/(m*GRAVITY))
    checks=[('preparation_kinetic_J',float(ke),ke<1e-5),
            ('preparation_support_relative_error',float(relative_support_error),relative_support_error<.015),
            ('preparation_max_unbalanced_force_over_weight',float(normalized_netforce),normalized_netforce<.06),
            ('preparation_max_overlap_over_diameter',out['max_overlap_ratio'],out['max_overlap_ratio']<.05)]
    return fp,fv,r,m,checks,out


def simulate_release(prepared, radius, mass, h, *, dt=DEFAULT_DT, duration=DURATION, zeta=RELEASE_ZETA,
                     sample_period=.02):
    p=prepared.copy()
    p[:,1]+=h-.15
    v=np.zeros_like(p)
    # 对所有高度完全复用同一个已平衡的颗粒组态。隔板在 t=0 瞬间撤去。
    result=integrate(p,v,radius,mass,dt=dt,duration=duration,zeta=zeta,shelf_y=h,
                     shelf_enabled=False,sample_every=max(1,int(round(sample_period/dt))),keep_snaps=True)
    result['start_pos']=p
    result['shelf_height']=h
    return result


def metrics(result,radius,mass,threshold=1e-4,minimum_tail=.40):
    h=result['history']; k=h[:,1]; t=h[:,0]
    peak_idx=int(np.argmax(k))
    # 动能阈值需持续至结束，且满足足够长的观察窗口。
    settle=float('nan')
    for i in range(len(t)):
        if t[-1]-t[i]>=minimum_tail and np.all(k[i:]<=threshold):
            settle=float(t[i]);break
    # 以从初始到最终耗散能量的95%作为较稳定的动态过程指标；
    # 不同于最终动能持续低于阈值的严格静止时间。
    E=h[:,4]; target=E[0]-.95*(E[0]-E[-1])
    idx=np.nonzero(E<=target)[0]
    energy95=float(t[idx[0]]) if len(idx) else float('nan')
    return dict(peak_ke=float(k[peak_idx]),peak_time=float(t[peak_idx]),
                final_ke=float(k[-1]),settled_time=settle,energy95_time=energy95,
                final_mean_height=float(h[-1,5]),max_overlap=result['max_overlap_ratio'])


def plot_figures(results,radius,outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    outdir=Path(outdir)
    fig,axes=plt.subplots(1,3,figsize=(12,5.5),sharex=True,sharey=True)
    for ax,(h,rst) in zip(axes,results.items()):
        ax.plot([0,0,WIDTH,WIDTH],[HEIGHT,0,0,HEIGHT],color='#34465b',lw=1.5)
        for (x,y),rad in zip(rst['final_pos'],radius):
            ax.add_patch(Circle((x,y),rad,edgecolor='#174f6e',facecolor='#83b8d4',lw=.6))
        ax.axhline(h,linestyle=':',color='#b3814d',lw=1.2,label='removed shelf level')
        ax.set_xlim(-.008,WIDTH+.008);ax.set_ylim(-.02,HEIGHT)
        ax.set_aspect('equal');ax.set_xlabel('x (m)');ax.set_title(f'Shelf at {h:.2f} m')
    axes[0].set_ylabel('y (m)');fig.suptitle('Final positions after instantaneous shelf removal (no friction)')
    fig.tight_layout();fig.savefig(outdir/'final_packing.png',dpi=170);plt.close(fig)

    fig,axes=plt.subplots(2,1,figsize=(9,6.8),sharex=True)
    for h,rst in results.items():
        data=rst['history']; label=f'h = {h:.2f} m'
        axes[0].plot(data[:,0],data[:,1],label=label)
        axes[1].plot(data[:,0],data[:,4],label=label)
    axes[0].set_ylabel('Kinetic energy (J)');axes[1].set_ylabel('Total mechanical energy (J)')
    axes[1].set_xlabel('Time after release (s)')
    for ax in axes: ax.grid(alpha=.22);ax.legend()
    fig.suptitle('Height comparison: contact damping ratio = 0.35')
    fig.tight_layout();fig.savefig(outdir/'energy_comparison.png',dpi=165);plt.close(fig)

    fig,ax=plt.subplots(figsize=(8.5,4.6))
    for h,rst in results.items():
        data=rst['history'];ax.plot(data[:,0],data[:,5],label=f'h = {h:.2f} m')
    ax.set_xlabel('Time after release (s)');ax.set_ylabel('Mean particle height (m)')
    ax.set_title('Mean particle height');ax.grid(alpha=.2);ax.legend();fig.tight_layout()
    fig.savefig(outdir/'mean_height.png',dpi=165);plt.close(fig)


def make_gif(results,radius,path):
    from PIL import Image,ImageDraw
    w=780;H=545; gap=12; colw=(w-4*gap)//3
    scale=(colw-22)/WIDTH
    imgs=[]
    prepared_count=6
    tframes=results[.15]['history'][:,0]
    indices=list(range(0,len(tframes),max(1,len(tframes)//65)))
    if indices[-1]!=len(tframes)-1:indices.append(len(tframes)-1)
    # 释放前暂停8帧显示支撑板；实际计算从移板之后的 t=0 开始。
    for frameid in range(-prepared_count,len(indices)):
        im=Image.new('RGB',(w,H),(252,252,252));d=ImageDraw.Draw(im)
        for j,(sh,res) in enumerate(results.items()):
            ox=gap+j*(colw+gap)
            bottom=H-42
            d.line([(ox+8,23),(ox+8,bottom),(ox+8+WIDTH*scale,bottom),(ox+8+WIDTH*scale,23)],fill=(35,48,62),width=3)
            if frameid<0:
                # 这是预沉降后已平衡状态，不是随机悬浮颗粒。
                particles=res['start_pos'];t=0.0
                y=bottom-sh*scale
                d.line([(ox+8,y),(ox+8+WIDTH*scale,y)],fill=(168,109,47),width=5)
            else:
                index=indices[frameid]
                particles=res['snapshots'][index];t=float(tframes[index])
                # 虚线淡示原隔板高度，不代表仍有物理边界。
                y=bottom-sh*scale
                for xx in range(int(ox+8),int(ox+8+WIDTH*scale),11):
                    d.line([(xx,y),(min(xx+4,ox+8+WIDTH*scale),y)],fill=(210,206,201),width=1)
            for xy,rad in zip(particles,radius):
                x=ox+8+xy[0]*scale;y=bottom-xy[1]*scale;rr=rad*scale
                d.ellipse((x-rr,y-rr,x+rr,y+rr),fill=(104,164,199),outline=(31,74,102),width=1)
            d.text((ox+16,6),f'Shelf {sh:.2f} m',fill=(30,30,30))
        lab='Shelf present: particles at rest' if frameid<0 else f'After removal: t = {t:.2f} s'
        d.text((gap,H-27),lab,fill=(32,32,32))
        imgs.append(im.convert("P", palette=Image.Palette.ADAPTIVE, colors=64))
    imgs[0].save(path,save_all=True,append_images=imgs[1:],duration=110,loop=0,optimize=False)


def main():
    par=argparse.ArgumentParser(description='隔板瞬时移除：二维颗粒释放与沉降模拟')
    par.add_argument('--output',default='shelf_results')
    par.add_argument('--dt',type=float,default=DEFAULT_DT)
    par.add_argument('--duration',type=float,default=DURATION)
    par.add_argument('--test-only',action='store_true')
    par.add_argument('--no-gif',action='store_true')
    args=par.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    base_checks,_=validate()
    unit_checks=check_shelf_unit()
    checks=base_checks+unit_checks
    csv_file(out/'unit_checks.csv',['test','value','passed'],checks)
    for name,value,passed in checks:
        print(('PASS' if passed else 'FAIL'),name, f'{value:.9g}',flush=True)
    if not all(x[-1] for x in checks):
        raise SystemExit('基础力学检查未通过，不运行释放计算。')
    p,v,r,m,prechecks,preout=prepare_state(dt=args.dt)
    csv_file(out/'preparation_checks.csv',['test','value','passed'],prechecks)
    for name,value,passed in prechecks:
        print(('PASS' if passed else 'FAIL'),name,f'{value:.9g}',flush=True)
    if not all(x[-1] for x in prechecks):
        raise SystemExit('隔板上的初始堆积尚未平衡，停止计算。')
    if args.test_only:
        print('单元与隔板平衡检验通过。')
        return
    results={}
    for height in HEIGHTS:
        print(f'Simulating shelf h={height:.2f} m',flush=True)
        result=simulate_release(p,r,m,height,dt=args.dt,duration=args.duration)
        results[height]=result
        csv_file(out/f'height_{int(round(height*100)):02d}cm_history.csv',
                 ['time_s','kinetic_J','gravity_J','spring_J','total_J','mean_height_m','contacts','overlap_at_sample','shelf_reaction_N'],
                 result['history'])
        print(f'  peak KE={np.max(result["history"][:,1]):.5f} J, overlap/diameter max={result["max_overlap_ratio"]:.2%}',flush=True)
    plot_figures(results,r,out)
    if not args.no_gif:
        print('Rendering GIF',flush=True)
        make_gif(results,r,out/'shelf_release.gif')
    sm=[]
    for height,result in results.items():
        met=metrics(result,r,m)
        sm.append((height,met['peak_ke'],met['peak_time'],met['energy95_time'],met['final_ke'],met['settled_time'],
                   met['final_mean_height'],met['max_overlap']))
    csv_file(out/'summary.csv',
             ['release_height_m','peak_kinetic_energy_J','time_of_peak_energy_s','time_to_95pct_total_energy_dissipation_s','final_kinetic_energy_J',
              'settled_time_if_sustained_below_1e-4_J_s','final_mean_particle_height_m','max_overlap_over_diameter'],sm)
    # 全过程时间步长对照。对碰撞多体系统，逐粒轨迹可能因接触顺序出现分离，
    # 因此主要比较峰值动能、整体能量、最终平均高度和耗散进度。
    T=args.duration;step=.02
    print(f'Running full-duration timestep check: {T:.2f} s, h=0.15 m',flush=True)
    a=simulate_release(p,r,m,.15,dt=args.dt,duration=T,sample_period=step)
    b=simulate_release(p,r,m,.15,dt=args.dt/2,duration=T,sample_period=step)
    ea=a['history'][:,4];eb=b['history'][:,4]
    ka=a['history'][:,1];kb=b['history'][:,1]
    # 初始总机械能正值并已包含重力势能。
    e0=float(ea[0]);ma=a['history'][:,5];mb=b['history'][:,5]
    diffs=dict(max_energy_norm=float(np.max(np.abs(ea-eb))/max(abs(e0),1e-12)),
               final_mean_height_rel=float(abs(ma[-1]-mb[-1])/max(abs(ma[-1]),abs(mb[-1]),1e-12)),
               peak_ke_rel=float(abs(np.max(ka)-np.max(kb))/max(np.max(ka),np.max(kb),1e-12)),
               max_overlap=float(max(res['max_overlap_ratio'] for res in results.values())))
    # 数值能量应当整体非增长；只容许单采样增量至多占初能量的千分之一。
    energy_growth=max(float(np.max(np.diff(res['history'][:,4]))/max(res['history'][0,4],1e-12))
                      for res in results.values())
    tests=[('max_energy_difference_over_initial_energy',diffs['max_energy_norm'],.02,diffs['max_energy_norm']<.02),
           ('final_mean_height_relative_difference',diffs['final_mean_height_rel'],.02,diffs['final_mean_height_rel']<.02),
           ('peak_kinetic_energy_relative_difference',diffs['peak_ke_rel'],.02,diffs['peak_ke_rel']<.02),
           ('positive_energy_increment_over_initial_energy',max(0,energy_growth),.001,energy_growth<.001),
           ('max_overlap_over_diameter',diffs['max_overlap'],.05,diffs['max_overlap']<.05)]
    csv_file(out/'numerical_checks.csv',['item','value','limit','passed'],tests)
    # 这些偏差作为参考，尤其不要把某一固定动能阈值下的沉降时间当作精确可复现常数。
    am=metrics(a,r,m);bm=metrics(b,r,m)
    csv_file(out/'time_step_diagnostics.csv',['indicator','dt','dt_half','absolute_difference'],
             [('time_to_95pct_energy_dissipation_s',am['energy95_time'],bm['energy95_time'],abs(am['energy95_time']-bm['energy95_time'])),
              ('settled_time_below_1e-4_J_s',am['settled_time'],bm['settled_time'],abs(am['settled_time']-bm['settled_time']))])
    print('TIMING SENSITIVITY: energy95',am['energy95_time'],bm['energy95_time'],
          '; KE threshold settle',am['settled_time'],bm['settled_time'],flush=True)
    for nm,val,lim,passed in tests:
        print(('PASS' if passed else 'FAIL'), nm, f'{val:.6g} (limit {lim})',flush=True)
    print('Outputs:',out.resolve())
    if not all(x[-1] for x in tests):
        raise SystemExit('数值验证未通过；不能使用此批结果作为正式分析。')

if __name__=='__main__':main()

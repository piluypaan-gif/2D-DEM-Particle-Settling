"""二维圆盘颗粒沉降：教学型显式 DEM（法向弹簧 + 阻尼，无切向摩擦）。

运行：python dem_particles.py --output results
验证：python dem_particles.py --test-only
依赖：numpy matplotlib pillow；numba 可选（有则加速）。
长度 m、时间 s、质量 kg、力 N。圆盘在二维平面运动，质量按球体密度计算。
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(func):
        return func

GRAVITY = 9.81
DENSITY = 2500.0
STIFFNESS = 200000.0
WIDTH = 0.32
HEIGHT = 0.65   # 仅用于绘图；容器顶部开放
DEFAULT_DT = 0.00002
DEFAULT_DURATION = 6.0


@njit
def compute_forces(pos, vel, radius, mass, kn, zeta, gravity, width, use_walls):
    """每步按当前几何关系建立接触，法向力 F=max(0, kn*overlap - cn*v_n)。

    正法向从粒子 i 指向 j；相对法向速度 v_n=(v_j-v_i)·n；接近时 v_n<0。
    不允许接触产生吸引力。阻尼由临界阻尼比 zeta 决定。
    """
    n = pos.shape[0]
    forces = np.zeros((n, 2))
    spring_energy = 0.0
    max_overlap_ratio = 0.0
    contacts = 0
    for i in range(n):
        forces[i, 1] -= mass[i] * gravity
        # 与固定容器壁的接触，墙体的等效质量视为无穷大。
        if use_walls:
            cn = 2.0 * zeta * math.sqrt(kn * mass[i])
            overlap = radius[i] - pos[i, 0]
            if overlap > 0.0:  # 左墙，正向力沿 +x
                force = max(0.0, kn * overlap - cn * vel[i, 0])
                forces[i, 0] += force
                spring_energy += 0.5 * kn * overlap * overlap
                max_overlap_ratio = max(max_overlap_ratio, overlap / (2.0 * radius[i]))
                contacts += 1
            overlap = radius[i] + pos[i, 0] - width
            if overlap > 0.0:  # 右墙，正向力沿 -x
                force = max(0.0, kn * overlap + cn * vel[i, 0])
                forces[i, 0] -= force
                spring_energy += 0.5 * kn * overlap * overlap
                max_overlap_ratio = max(max_overlap_ratio, overlap / (2.0 * radius[i]))
                contacts += 1
            overlap = radius[i] - pos[i, 1]
            if overlap > 0.0:  # 底壁，正向力沿 +y
                force = max(0.0, kn * overlap - cn * vel[i, 1])
                forces[i, 1] += force
                spring_energy += 0.5 * kn * overlap * overlap
                max_overlap_ratio = max(max_overlap_ratio, overlap / (2.0 * radius[i]))
                contacts += 1

    # 粒子-粒子接触，双向作用力大小相等、方向相反。
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dist2 = dx * dx + dy * dy
            gap = radius[i] + radius[j]
            if dist2 >= gap * gap:
                continue
            dist = math.sqrt(dist2)
            if dist < 1e-14:
                nx, ny = 1.0, 0.0
            else:
                nx, ny = dx / dist, dy / dist
            overlap = gap - dist
            vn = (vel[j, 0] - vel[i, 0]) * nx + (vel[j, 1] - vel[i, 1]) * ny
            meff = mass[i] * mass[j] / (mass[i] + mass[j])
            cn = 2.0 * zeta * math.sqrt(kn * meff)
            force = max(0.0, kn * overlap - cn * vn)
            forces[i, 0] -= force * nx
            forces[i, 1] -= force * ny
            forces[j, 0] += force * nx
            forces[j, 1] += force * ny
            spring_energy += 0.5 * kn * overlap * overlap
            max_overlap_ratio = max(max_overlap_ratio, overlap / gap)
            contacts += 1
    return forces, spring_energy, max_overlap_ratio, contacts


def initial_particles(n=30, seed=7):
    """生成不重叠的30颗粒初始配置。每一工况复用完全一样的初始状态。"""
    if n != 30:
        raise ValueError('课程演示暂固定30颗粒，以便保持可核查的初始位置。')
    rng = np.random.default_rng(seed)
    radius = np.full(n, 0.0165)
    radius *= rng.uniform(0.94, 1.06, n)
    pos = np.empty((n, 2))
    for k in range(n):
        row, col = divmod(k, 6)
        pos[k] = (0.041 + col * 0.047, 0.19 + row * 0.071)
    pos += rng.uniform(-0.003, 0.003, pos.shape)
    for i in range(n):
        for j in range(i+1,n):
            if np.linalg.norm(pos[i]-pos[j]) < radius[i]+radius[j]:
                raise ValueError('初始颗粒重叠，请调整位置。')
    vel = np.zeros_like(pos)
    mass = DENSITY * (4.0 / 3.0) * math.pi * radius**3
    return pos, vel, radius, mass


def simulate(pos0, vel0, radius, mass, *, dt, duration, zeta, gravity=GRAVITY,
             width=WIDTH, use_walls=True, sample_every=250):
    """半隐式欧拉（先更新速度再更新位置），记录抽样轨迹和能量。"""
    steps_float = duration / dt
    steps = int(round(steps_float))
    if abs(steps - steps_float) > 1e-8:
        raise ValueError('duration / dt 应为整数。')
    pos, vel = pos0.copy(), vel0.copy()
    rows, snaps = [], []
    global_max_overlap = 0.0
    for step in range(steps + 1):
        forces, spring, overlap, contacts = compute_forces(
            pos, vel, radius, mass, STIFFNESS, zeta, gravity, width, use_walls)
        global_max_overlap = max(overlap, global_max_overlap)
        if step % sample_every == 0 or step == steps:
            kinetic = 0.5 * np.sum(mass[:, None] * vel * vel)
            potential = np.sum(mass * gravity * pos[:, 1])
            rows.append((step * dt, kinetic, potential, spring,
                         kinetic + potential + spring, float(np.mean(pos[:, 1])),
                         int(contacts), float(overlap)))
            snaps.append(pos.copy())
        if step == steps:
            break
        vel += (forces / mass[:, None]) * dt
        pos += vel * dt
        if not (np.isfinite(pos).all() and np.isfinite(vel).all()):
            raise FloatingPointError(f'数值发散，step={step}')
    return {'final_pos':pos, 'final_vel':vel, 'history':np.array(rows),
            'snapshots':np.stack(snaps), 'max_overlap_ratio':global_max_overlap}


def validate():
    """基础检验不借助预期图形：自由落体、两颗粒动量、反弹及时间步长。"""
    checks = []
    r = np.array([0.0165]); m = DENSITY * 4.0/3.0 * math.pi * r**3
    p = np.array([[WIDTH/2, 0.45]]); v = np.zeros((1,2))
    dt = 1e-4; T = 0.12
    fall = simulate(p,v,r,m,dt=dt,duration=T,zeta=0.3,sample_every=300)
    expected_y = p[0,1] - 0.5 * GRAVITY * T*T
    y_error = abs(fall['final_pos'][0,1] - expected_y)
    v_error = abs(fall['final_vel'][0,1] + GRAVITY*T)
    checks.append(('free_fall_position_error_m',y_error, y_error < 1e-4))
    checks.append(('free_fall_velocity_error_m_s',v_error, v_error < 1e-9))

    # 单球、无重力、无墙间碰撞，底部以1m/s入射，测量回弹速度。
    bounced = []
    for zeta in (0.0, 0.10, 0.35, 0.70):
        br = np.array([0.0165]); bm = DENSITY * 4.0/3.0 * math.pi * br**3
        bp=np.array([[WIDTH/2, 0.052]])
        bv=np.array([[0.0, -1.0]])
        out=simulate(bp,bv,br,bm,dt=5e-6,duration=0.075,zeta=zeta,
                     gravity=0.0,sample_every=1)
        # 不只看最终高度，通过重建逐步位置与速度检测离开底壁时的向上速度。
        h=out['history']
        # 需要在第二次实现中用最终速度；0.075秒时仍在向上运动且已离墙。
        e_measured=out['final_vel'][0,1]
        bounced.append((zeta, e_measured))
    checks.append(('elastic_rebound_speed',bounced[0][1],abs(bounced[0][1]-1.0)<0.03))
    checks.append(('damping_rebound_monotonic',
                   bounced[-1][1], all(bounced[i][1]>bounced[i+1][1] for i in range(3))))
    checks.append(('no_negative_rebound',bounced[-1][1],bounced[-1][1]>=-1e-5))

    # 无外力、无墙，孤立两球碰撞总动量守恒。
    rr=np.array([0.0165,0.0165]); mm=DENSITY*4.0/3.0*math.pi*rr**3
    pp=np.array([[0.12,0.25],[0.21,0.25]])
    vv=np.array([[0.5,0.0],[-0.5,0.0]])
    momentum_before=np.sum(mm[:,None]*vv,axis=0)
    twob=simulate(pp,vv,rr,mm,dt=5e-6,duration=0.14,zeta=0.35,gravity=0.0,
                  use_walls=False,sample_every=20)
    momentum_after=np.sum(mm[:,None]*twob['final_vel'],axis=0)
    momentum_err=np.linalg.norm(momentum_after-momentum_before)
    checks.append(('two_ball_momentum_error_kg_m_s',momentum_err,momentum_err<1e-10))
    initial_energy=0.5*np.sum(mm[:,None]*vv**2)
    final_energy=0.5*np.sum(mm[:,None]*twob['final_vel']**2)
    checks.append(('two_ball_damping_energy_loss_J',initial_energy-final_energy,
                   (initial_energy-final_energy)>0))

    return checks,bounced


def write_csv(path, headings, rows):
    with open(path,'w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(headings);writer.writerows(rows)


def visualize(results, r, outdir, duration):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    outdir=Path(outdir)
    cases=list(results)
    fig,axs=plt.subplots(1,3,figsize=(12,5.5),sharex=True,sharey=True)
    for ax,(label, res) in zip(axs,results.items()):
        for xy,rad in zip(res['final_pos'],r):
            ax.add_patch(Circle(xy,rad,edgecolor='black',linewidth=.45,facecolor='#6a9dbb'))
        ax.set_xlim(-0.006,WIDTH+0.006)
        ax.set_ylim(-0.01,HEIGHT)
        ax.plot([0,0,WIDTH,WIDTH], [HEIGHT,0,0,HEIGHT], color='black',linewidth=1.4)
        ax.set_aspect('equal')
        ax.set_title(label)
        ax.set_xlabel('x (m)')
    axs[0].set_ylabel('y (m)')
    fig.suptitle(f'Final packing after {duration:g} s (no tangential friction)')
    fig.tight_layout();fig.savefig(outdir/'final_packing.png',dpi=165)
    plt.close(fig)

    fig,axs=plt.subplots(2,1,figsize=(9,7),sharex=True)
    for name,res in results.items():
        hist=res['history']
        axs[0].plot(hist[:,0],hist[:,1],label=name)
        axs[1].plot(hist[:,0],hist[:,4],label=name)
    axs[0].set_ylabel('Kinetic energy (J)')
    axs[1].set_ylabel('Total mechanical energy (J)')
    axs[1].set_xlabel('Time (s)')
    axs[0].grid(alpha=.25);axs[1].grid(alpha=.25)
    axs[0].legend();axs[1].legend()
    fig.suptitle('DEM: damping ratio comparison')
    fig.tight_layout();fig.savefig(outdir/'energy_curves_full.png',dpi=160)
    axs[1].set_xlim(0,min(1.4,duration))
    fig.savefig(outdir/'energy_curves.png',dpi=160)
    plt.close(fig)

    fig, ax=plt.subplots(figsize=(9,4))
    for name,res in results.items():
        h=res['history'];ax.plot(h[:,0],h[:,5],label=name)
    ax.set_xlabel('Time (s)');ax.set_ylabel('Mean particle center height (m)')
    ax.set_title('Particle settling');ax.legend();ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(outdir/'mean_height.png',dpi=160);plt.close(fig)


def make_animation(mid_result, radii, path, dt_frame, width=WIDTH, height=HEIGHT):
    """无需 ParaView：直接用 matplotlib 已安装的 Pillow 依赖生成 GIF。"""
    from PIL import Image, ImageDraw, ImageFont
    w,h=480,710
    scale=(w-30)/width
    bottom=h-45
    images=[]
    snaps=mid_result['snapshots']
    # 动画只展示落下和碰撞最明显的前1.6秒，不重复播放其后的静止阶段。
    last=min(len(snaps)-1, int(round(1.6/dt_frame)))
    skip=max(1, int(np.ceil((last+1)/60)))
    for idx in range(0,last+1,skip):
        im=Image.new('RGB',(w,h),'white')
        d=ImageDraw.Draw(im)
        d.line([(15,20),(15,bottom),(w-15,bottom),(w-15,20)],fill=(20,20,20),width=3)
        for xy,r in zip(snaps[idx],radii):
            x=15+xy[0]*scale
            y=bottom-xy[1]*scale
            rad=r*scale
            d.ellipse((x-rad,y-rad,x+rad,y+rad),fill=(98,154,191),outline=(28,68,101),width=2)
        d.text((16,h-30),f'Time = {idx*dt_frame:.2f} s | medium damping',fill=(10,10,10))
        images.append(im)
    images[0].save(path,save_all=True,append_images=images[1:],duration=max(60,int(1000*dt_frame*skip)),loop=0,optimize=True)


def main():
    parser=argparse.ArgumentParser(description='二维 DEM 颗粒重力沉降和碰撞，带自动验证')
    parser.add_argument('--output',default='results',help='输出文件夹')
    parser.add_argument('--dt',type=float,default=DEFAULT_DT,help='积分时间步长 s')
    parser.add_argument('--duration',type=float,default=DEFAULT_DURATION,help='物理总时间 s')
    parser.add_argument('--test-only',action='store_true',help='仅运行基础物理验证')
    parser.add_argument('--no-gif',action='store_true',help='跳过 GIF 绘制')
    args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    checks,bounce=validate()
    write_csv(out/'verification.csv',['test','value','pass'],checks)
    write_csv(out/'bounce.csv',['damping_ratio','rebound_speed_m_s'],bounce)
    for name,value,ok in checks:
        print(f"{'PASS' if ok else 'FAIL'} | {name}: {value:.8g}")
    if not all(ok for _,_,ok in checks):
        raise SystemExit('基础物理验证未通过，停止正式工况。')
    if args.test_only:
        return
    p,v,r,m=initial_particles()
    results={}
    for name,zeta in [('low zeta=0.10',.10),('medium zeta=0.35',.35),('high zeta=0.70',.70)]:
        print('Running',name,flush=True)
        sample=max(1,round(0.05/args.dt))
        res=simulate(p,v,r,m,dt=args.dt,duration=args.duration,zeta=zeta,sample_every=sample)
        results[name]=res
        h=res['history']
        write_csv(out/(name.split()[0]+'_history.csv'),
                  ['time_s','kinetic_J','gravity_J','spring_J','total_J','mean_height_m','contact_count','max_overlap_ratio_at_sample'],h)
        print(f'  final KE={h[-1,1]:.5g} J, final mean_y={h[-1,5]:.5g} m; max overlap/diameter={res["max_overlap_ratio"]:.4f}',flush=True)
    visualize(results,r,out,args.duration)
    if not args.no_gif:
        make_animation(results['medium zeta=0.35'],r,out/'settling_animation.gif',
                       dt_frame=max(1,round(0.05/args.dt))*args.dt)
    # 检查时间步长：物理总时间不变、两次计算的初始位置完全一致。
    shorter=min(2.0,args.duration)
    a=simulate(p,v,r,m,dt=args.dt,duration=shorter,zeta=.35,sample_every=max(1,round(.05/args.dt)))
    b=simulate(p,v,r,m,dt=args.dt/2,duration=shorter,zeta=.35,sample_every=max(1,round(.05/(args.dt/2))))
    Ea,Eb=a['history'][-1,4],b['history'][-1,4]
    ya,yb=a['history'][-1,5],b['history'][-1,5]
    # 总能量分母用初始机械能，避免最终接近0导致相对误差放大。
    E0=a['history'][0,4]
    rel_energy=abs(Ea-Eb)/max(abs(E0),1e-12)
    rel_height=abs(ya-yb)/max(abs(ya),abs(yb),1e-12)
    write_csv(out/'dt_check.csv',
              ['dt_s','time_s','total_energy_J','mean_height_m','max_overlap_ratio'],
              [(args.dt,shorter,Ea,ya,a['max_overlap_ratio']),
               (args.dt/2,shorter,Eb,yb,b['max_overlap_ratio'])])
    max_energy_diff=float(np.max(np.abs(a['history'][:,4]-b['history'][:,4]))/max(abs(E0),1e-12))
    verification_ok = (max_energy_diff < 0.01 and rel_height < 0.01
                       and all(x['max_overlap_ratio'] < 0.05 for x in results.values()))
    write_csv(out/'numeric_check.csv', ['item','value','criterion','pass'],
              [('maximum_sampled_total_energy_difference_over_initial_energy',max_energy_diff,0.01,max_energy_diff<0.01),
               ('final_mean_height_relative_difference',rel_height,0.01,rel_height<0.01),
               ('maximum_overlap_over_particle_diameter',max(x['max_overlap_ratio'] for x in results.values()),0.05,
                all(x['max_overlap_ratio']<0.05 for x in results.values()))])
    summary=[]
    for name,res in results.items():
        h=res['history']
        # 静止判据：此时及其之后每个采样点的总动能持续小于 1e-4 J。
        settled=float('nan')
        for k in range(len(h)):
            if np.all(h[k:,1]<1e-4):
                settled=float(h[k,0]);break
        summary.append((name,float(h[-1,1]),float(h[-1,5]),
                        res['max_overlap_ratio'],settled))
    write_csv(out/'summary.csv',
              ['case','final_kinetic_J','final_mean_height_m','max_overlap_over_diameter',
               'settled_time_at_kinetic_below_1e-4_J_s'],summary)
    print(f'DT CHECK at t={shorter:g}s: final rel E={rel_energy:.4%}; final rel height={rel_height:.4%}; max sampled rel E={max_energy_diff:.4%}')
    print('NUMERICAL CHECK:', 'PASS' if verification_ok else 'FAIL')
    print('Output:',out.resolve())
    if not verification_ok:
        raise SystemExit('数值检查未通过：请检查 numeric_check.csv，勿将这些结果当作有效工况。')

if __name__=='__main__':
    main()

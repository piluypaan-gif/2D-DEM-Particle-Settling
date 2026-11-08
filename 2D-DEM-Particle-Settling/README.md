# 二维离散元颗粒隔板释放与重力沉降模拟

基于 **Python** 实现的二维离散元（DEM）教学项目，模拟圆形颗粒在水平隔板支撑下预先堆积、隔板瞬时撤除后自由下落、碰撞及再次沉降的过程。项目以 **释放高度** 为变量，比较颗粒群运动、动能变化与能量耗散规律，可用于《岩土数值分析》课程的编程计算与结果分析。

> **模型范围：** 本项目为基础法向弹簧–阻尼接触模型。隔板在指定时刻直接移除，**不模拟实际水平抽板**；不考虑切向摩擦、颗粒自转或黏结破坏。结果用于教学分析，不代表经过材料参数标定的实际砂石料仓预测。

## 模拟展示

![三组释放高度的颗粒沉降动画](results/shelf_release.gif)

图中颗粒首先在隔板上达到近似静止状态，然后在 `t = 0` 时失去隔板支撑。比较的隔板高度为 **0.10 m、0.15 m、0.20 m**，各工况均使用同一组预沉降完成的颗粒构型，仅改变该构型距地面的高度。撤板后的水平虚线只标记**原隔板位置**，不再对颗粒施加作用力。

![不同释放高度下的最终堆积状态](results/final_packing.png)

## 运行环境与使用方法

推荐 Windows + VS Code，使用 Python 3 运行。项目不依赖 MATLAB、YADE、DDA 或 ParaView。

将仓库下载或克隆到本地，在包含 `shelf_release.py` 的目录打开终端，先安装依赖：

```powershell
python -m pip install -r requirements.txt
```

先执行基础力学及隔板初始平衡检查：

```powershell
python shelf_release.py --test-only
```

随后运行三组释放高度的完整计算：

```powershell
python shelf_release.py --output shelf_results
```

计算结果将保存到新生成的 `shelf_results/` 目录，**不会覆盖仓库中已有的 `results/` 参考结果**。首次运行可能需要额外时间完成 Numba 编译和 GIF 绘制。动画可使用 Edge 或 Chrome 打开。

调试时如果只需要数值结果、不需要生成动画，可以运行：

```powershell
python shelf_release.py --output test_results --no-gif
```

Windows 用户也可尝试双击 `run_windows.bat`。

## 数值模型与程序结构

采用30个半径略有差异的颗粒，位置和速度在二维平面内表示。颗粒质量按三维球体的体积和密度计算。颗粒受重力、颗粒间法向接触力及容器壁/隔板法向支撑力作用，运动方程通过**半隐式欧拉法**积分。

当颗粒间发生重叠时，法向接触力采用线性弹簧–阻尼形式：

$$
F_n=\max(0,\,k_n\delta-c_nv_n),\qquad
c_n=2\zeta\sqrt{k_nm_{\mathrm{eff}}}
$$

其中，\(\delta\) 是接触重叠量，\(v_n\) 是沿法向的分离相对速度（分离为正），\(k_n\) 为法向刚度，\(\zeta\) 为阻尼比；\(m_{\mathrm{eff}}\) 对颗粒间接触取约化质量。

`dem_particles.py` 实现基本接触力、运动积分和单元测试；`shelf_release.py` 负责隔板支撑、预沉降、瞬时撤板、多工况计算与绘图；`requirements.txt` 记录运行依赖；`results/` 保存本项目已完成的计算数据和展示图。

主要工况参数位于 `shelf_release.py` 顶部：

```python
HEIGHTS = (0.10, 0.15, 0.20)  # 隔板高度，m
DURATION = 3.2                # 撤板后模拟时长，s
PREPARE_DURATION = 3.0        # 预沉降时长，s
PREPARE_ZETA = 0.70           # 预沉降阻尼比
RELEASE_ZETA = 0.35           # 正式计算阻尼比
```

重力、密度、法向接触刚度和容器尺寸等参数在 `dem_particles.py` 中定义。默认时间步长为 `2e-5 s`；改变颗粒尺寸或刚度后，应重新验证时间步长和接触重叠量。

## 主要计算结果

| 隔板高度 | 峰值动能 | 95%机械能耗散时间 | 最大重叠量/颗粒直径 |
|:--:|--:|--:|--:|
| 0.10 m | 1.315 J | 0.32 s | 1.60% |
| 0.15 m | 1.718 J | 0.38 s | 1.95% |
| 0.20 m | 2.684 J | 0.44 s | 2.26% |

在本组参数下，隔板释放高度越大，颗粒群的峰值动能越高，达到所定义的95%机械能耗散阈值所需的时间也越长。具体数值及逐时刻记录见 [`results/summary.csv`](results/summary.csv) 和各高度的 `history.csv`。

![三组工况的动能与机械能曲线](results/energy_comparison.png)

## 数值验证与适用范围

程序对单颗粒自由落体、无阻尼反弹、双颗粒动量守恒、阻尼耗散、隔板支撑与撤除响应，以及30颗粒预沉降平衡进行了基本检查。预沉降结束时，系统总动能约为 `3.48 × 10^-9 J`。

针对 **0.15 m** 工况，将时间步长从 `2e-5 s` 减半至 `1e-5 s`，归一化总机械能曲线最大差异约 **0.424%**，最终平均高度相对差异约 **0.225%**。但按照严格动能阈值定义的“静止时间”分别为 **0.62 s** 和 **0.72 s**，仍具有时间步长敏感性，**不应当作已经收敛的精确结论**。

模型不含切向摩擦、滚动、颗粒形状差异及三维颗粒间相互作用，无法直接预测真实料仓卸料流量、堆积角或实际砂石材料力学性质。模拟结果属于**数值计算**，不是实测数据。

## 结果文件与复现

`results/` 内包含已完成的三组工况计算结果，可直接浏览并与重新运行的 `shelf_results/` 对照。其中，`shelf_release.gif` 为动画，`final_packing.png` 为最终堆积图，`energy_comparison.png` 为能量曲线，`mean_height.png` 为平均高度变化，`summary.csv` 为结果汇总，`height_10cm_history.csv` 等为逐时刻采样记录。

`unit_checks.csv`、`preparation_checks.csv`、`numerical_checks.csv` 和 `time_step_diagnostics.csv` 分别保存基础力学、预沉降、数值指标与时间步长对照的检查结果。

**课程报告题目：**《基于 Python 的二维离散元颗粒隔板释放与重力沉降数值模拟》

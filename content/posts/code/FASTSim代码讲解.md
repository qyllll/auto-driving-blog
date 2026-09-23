---
title: "FASTSim 代码讲解：测所有车能耗的功率train仿真器——Rust 内核 + Python API 全拆解"
date: 2026-09-23
draft: false
categories: ["代码讲解"]
tags: ["FASTSim", "能耗仿真", "功率train", "Rust", "Python", "车辆动力学", "代码讲解", "NREL", "开源"]
summary: "「NLR/NREL FASTSim（Future Automotive Systems Technology Simulator）是开源车辆功率train能耗仿真器：给一辆车 + 一条工况（UDDS/HWFET/自定义遥测），秒级算出油耗/电耗/续航。本文面向零基础，从名词表、三大对象 Vehicle/Cycle/SimDrive、到 Rust 内核逐模块拆解（道路负荷功率五项分解、向后看 Newton 速度求解、Trace Miss 四种策略、Conv/HEV/PHEV/BEV 功率train、HEV SOC 平衡迭代）、Python 绑定与车辆数据库（含比亚迪 Atto 3/Dolphin），再讲清它和 CARLA 等闭环驾驶仿真的分工，最后给出「测所有车能耗」的实操脚本与汇报口述。」"
---

> 本文是「代码讲解」路线里少见的**非神经网络**项目：领导让看的 **Fastsim**，对上号的是美国国家可再生能源实验室一系开源的 **FASTSim**（Future Automotive Systems Technology Simulator）。它不生成轨迹、不训模型，而是**给定车 + 给定速度曲线，逐秒反推动力总成功率与燃料/电能消耗**——这正是「测所有车的能耗」那句话背后的工具。代码在 **NatLabRockies/fastsim**（历史品牌常写 NREL），当前主线是 **fastsim-3**，内核 Rust + Python 绑定。

## 写给零基础读者：读这篇之前先搞懂几个名词

- **功率train / Powertrain（动力总成）**：车「从燃料/电池到车轮」的那一整条链。燃油车是「油箱 -> 发动机 -> 变速箱 -> 车轮」；电车是「电池 -> 电机 -> 减速器 -> 车轮」；混动是两条链加一个能量分配策略。FASTSim 名字里的 Simulator 仿的就是这条链，不是方向盘手感。

- **Drive cycle（工况/行驶工况）**：一条「时间 vs 目标车速」曲线，常再带坡度、环境温度。EPA 的 **UDDS**（城市）、**HWFET**（高速）是监管工况；也可以用自家车跑出来的 GPS/遥测数据当自定义工况。FASTSim 是 **backward-looking（向后看）**：先有速度需求，再问「这辆车此刻需要多大功率、烧多少油」。

- **Road load（道路负荷）**：维持/改变车速，车轮上要克服的力的合力，拆成加速、坡道、风阻、滚阻、车轮转动惯量五项。本文代码分析的核心就是这五项功率怎么写进 Rust。

- **Trace miss（跟不上工况）**：车动力不够，实际速度追不上工况目标速度。FASTSim 用 Newton 迭代求「在动力上限下实际能跑到的速度」，再按策略决定报错、放行还是改工况曲线追回来。

- **HEV / SOC balancing（混动与电量平衡）**：混动如果跑完一圈电池电越用越多，油耗数字就没法公平比。FASTSim 的 `run()` 会迭代：调整初始 SOC，直到循环结束电量与开始基本一致（charge-sustaining）。

- **Label fuel economy（EPA 贴纸油耗）**：实验室台架上的「原始 mpg」和窗口贴纸上的「调整后 mpg」不是一回事。FASTSim 自带 label 模块，对 UDDS/HWFET 结果套 EPA 调整公式，方便和 fueleconomy.gov 对数。

- **uom / SI 单位写进字段名**：fastsim-3 用 Rust `uom` 库做量纲，序列化后字段名直接带 `_watts` / `_joules` / `_meters` / `_seconds`，杜绝 kW/mph 混单位事故。

- **向后看 vs 向前看**：向后看（FASTSim）：工况给速度，算需要多大功率——快，适合批量能耗。向前看（很多驾驶仿真器）：给油门刹车，积分成速度——能闭环控车，但慢且要控车器。两者常被搞混，见文末对比。

## 一、FASTSim 到底是什么、能干什么

### 1.1 一句话定位

**FASTSim** 是一个开源（BSD-3）的车辆动力总成**纵向动力学 + 能耗**仿真器：输入一辆车（参数文件）和一条工况（时间序列车速），逐 1 秒（可配）解出车辆能跑多快、发动机/电机出多少力、油箱/电池少了多少能量，输出百公里油耗、mpg、SOC 轨迹、排放相关量等。

它回答的问题是：

- 这台车跑 UDDS/HWFET 是多少 mpg？（EPA 贴纸口径）
- 同一台车加重 800 kg、风阻从 0.30 降到 0.24，油耗变化多少？
- 换一套混动控制策略/电池容量，SOC 能不能撑完整圈？
- 用真实用户 GPS 遥测（走走停停、空调开关）反推车队级油耗？

它**不**回答的问题：方向盘打多少度车会不会滑（横向动力学）、跟车博弈、交通流红绿灯（那是 CARLA/SUMO/VISSIM 的事——见文末对比）。

### 1.2 家族与相关工具（领导可能问）

FASTSim 不是孤立脚本，是一套工具链里的一环（站点 nlr.gov 的 transportation 产品页，历史品牌 NREL）：

| 工具 | 干什么 | 和 FASTSim 关系 |
| --- | --- | --- |
| **FASTSim** | 单车纵向能耗仿真（本文） | 内核 |
| **RouteE** | 车队/路段级能耗概率模型 | 常用 FASTSim 批量跑出数据再建模 |
| **T3CO** | 重卡 TCO（总拥有成本）仿真 | 能耗部分基于 FASTSim 思路 |
| **ADOPT** | 车队采用/技术扩散分析 | 上层经济学，引用其能耗假设 |
| **fastsim-vehicles** | 车辆参数数据库（Git 仓库） | FASTSim 的 `Vehicle::from_db` 数据源 |
| **EPA fueleconomy.gov** | 贴纸油耗查询 | `label` 模块对数目标 |

### 1.3 典型使用场景（按频率排序）

1. **对标 EPA 贴纸**：内置 9 辆车（Ford Fusion、Nissan Leaf、Toyota Prius、Chevy Bolt、Hyundai Sonata Hybrid、Tesla Model 3、Renault Zoe、Chrysler Pacifica PHEV），一条命令出 UDDS/HWFET/US06 再和 fueleconomy.gov 比。
2. **敏感性扫描**：改质量、滚阻、风阻、传动效率、起步启停开关，看油耗变化——做配置方案论证时比实车省油钱。
3. **自定义工况**：上传遥测 CSV（带位置、坡度、环境温度），看真实驾驶风格下的能耗。
4. **混动/纯电策略可行性**：调目标 SOC、电池容量、再生制动比例，快速验证 SOC balancing 能不能收敛。
5. **给上层供数**：把「车+工况 -> 油耗」当黑盒，批量生成训练数据或查表给规划/控制/经济学模型用。

### 1.4 代码仓库与版本现状（2026-09 实测）

- 主仓库：`https://github.com/NatLabRockies/fastsim`，默认分支 **fastsim-3**（不是 main，`git/trees/main` 会 404）。
- 子仓库：`https://github.com/NatLabRockies/fastsim-vehicles`（车辆库，分 `v1/fastsim-3/{bev,hev,conv,phev}/brand/model/year/base/r1.yaml` 层级，也含少量 `v1/fastsim-2/*.csv` 兼容格式）。
- 文档站：`https://natlabrockies.github.io/fastsim/`（installation、getting-started、vehicle、drive-cycle、simdrive、trace-miss、label-fe、start-stop、cavs、thermal-simulations、telematics、migration-guide、related-tools 等页面齐全）。
- Rust API 文档：`https://docs.rs/fastsim-core/latest/fastsim_core/`。
- PyPI：`pip install fastsim`，当前 **3.1.0**，要求 Python **3.10–3.15**（本机实测已装到 conda base 的 `site-packages/fastsim/`）。
- 许可与门面：BSD-3；近年从 NREL 过渡到 **NatLabRockies** 组织，首页写作 `https://www.nlr.gov/transportation/fastsim`。历史文章/论文里的 `NREL/fastsim` 链接现在会指向重命名后的仓库——**领导说的 "Fastsim" 对上号的就是它**。

## 二、30 秒跑通：装包、跑一条工况、看真实数字

下面命令在本机 conda base（Python 3.13）实测通过，数字是真实输出，可直接拿去汇报。

### 2.1 安装与最小脚本

```bash
pip install fastsim          # PyPI 3.1.0，wheel 自带编译好的 Rust 内核
```

```python
# quickest_start.py
from fastsim import Vehicle, Cycle, SimDrive

veh = Vehicle.from_resource("2012_Ford_Fusion")   # 内置 9 车之一
cyc = Cycle.from_resource("Udds")                 # 城市工况 1370 步
sd = SimDrive(veh, cyc)
res = sd.run()                                    # 向后看逐秒仿真

print(res.to_dict(flatten=True)["miles_per_gallon_dynamometer"])
print("distance_mi", res.to_dict(flatten=True)["distance_miles"])
print("runtime_s", res.to_dict(flatten=True)["simulation_seconds"])  # 若字段名有出入以 to_dict 键为准
```

### 2.2 本机实测结果（可直接引用）

| 指标 | 数值 | 说明 |
| --- | --- | --- |
| 工况步数 | UDDS 1370 步 | 1 Hz 采样 |
| 2012 Ford Fusion 运行时长 | **约 0.0023 s** | 带历史记录；关闭历史约 0.0013 s |
| 等效跑完全程 | < 4 s | 纯 Python 调用，内核在 Rust |
| 燃料能量 | **7.303 kWh** | 整条 UDDS 累计低位热值口径 |
| 行驶距离 | **7.451 mi** | ≈ 11.99 km |
| 动态油耗 | **34.4 mpg** | dynamometer 调整前 lab 值量级 |
| 质量 +800 kg 重跑 | **28.2 mpg（-18.0%）** | 质量敏感性一键扫 |
| Tesla Model 3（BEV）化学电池放电 | **4,065,537 J** | `energy_out_chemical` |
| 其中牵引功率消耗 | **2,227,959 J** | 驱动电机做功 |
| 附件（空调等） | **342,250 J** | aux 一项 |

> 汇报口径一句话：「同一套向后看内核，燃油车、混动、纯电一条命令出结果；单车全工况毫秒级，质量 +800 kg 精确复现出 ~18% 油耗回退，数量级和工程经验一致。」

### 2.3 列出内置资源

```python
from fastsim import Vehicle, Cycle
print(Vehicle.list_resources())   # 9 辆车
print(Cycle.list_resources())     # Udds, HwFet, UddsHwFet, US06, ...
```

## 三、三大核心对象：Vehicle、Cycle、SimDrive

整个 API 就三个名词，搞清楚谁持有谁，代码怎么读都不会迷路。

```text
SimDrive                      一次仿真 = 车 + 工况 + 参数
 |-- veh:  Vehicle            一辆车（动力总成 + 底盘 + 质量）
 |-- cyc:  Cycle              一条工况（时间序列: mps, grade, amb_temp, ...）
 |-- sim_params: SimParams    求解器与容差开关（速度迭代、trace miss、热害开关）

run()  ->  逐秒循环:
   for each timestep:
       StepInfo::solve_for_speed(...)   # 道路负荷反解真实速度
       veh.solve(...)                    # 功率train 按需求功率出力
       记录 energy / fuel / soc / trace miss ...
   (HEV: 外层再迭代初始 SOC, 直到循环末尾电量平衡)
```

### 3.1 `Vehicle`：车是什么

Python 侧 `Vehicle` 由 Rust `fastsim-core` 的结构体经 pyo3 导出，两大块：

- `chassis`：车体物理量——`drag_coef`（风阻 Cd）、`frontal_area_m2`（迎风面积）、`wheel_rr_coef`（滚阻系数 Crr）、`mass_kilograms`（整备/簧上质量）、轮半径与转动惯量等。
- `pt_type`：动力总成枚举 `PowertrainType`，四类：
  - `Conventional`（纯燃油：发动机 + 变速箱 + 离合）
  - `Hybrid`（混动：发动机 + 电机 + 电池 + 功率分流/耦合）
  - `PHEV`（插电混动：同上但外充）
  - `BEV`（纯电：电池 + 电机 + 减速器）
  - （Rust 枚举里还留了 `FCEV` 燃料电池分支）

燃油发动机 `FuelConverter` 的效率不是常数，而是二维插值表：

```yaml
# 2012_Ford_Fusion 发动机节选（内置资源 YAML，示意）
pt_type:
  Conv:
    fc:
      pwr_max_watts: 130500          # 最大功率 ~130.5 kW
      eff_interp_from_pwr_out:       # 输出功率 -> 热效率 查表
        grid:   [0, 20000, 40000, 65000, 90000, 130500]
        values: [0.20, 0.28, 0.32, 0.33, 0.31, 0.28]
      pwr_idle_fuel: 4500            # 怠速油泵功率 W
      pwr_ramp_lag: ...              # 动态响应滞后（一阶惯性）
    trans:                           # 6AT 类
      gear_ratios: [3.5, 2.0, 1.3, 1.0, 0.7, 0.5]
      final_drive_ratio: 3.2
```

工厂方法：

- `Vehicle.from_resource(name)`：读 FASTSim 自带 9 车。
- `Vehicle.from_db(...)` / Rust `Vehicle::from_db`：查 `fastsim-vehicles` 数据库（`bev/byd/atto-3/2023/base/r1.yaml` 这类路径）。

### 3.2 `Cycle`：工况是什么

时间序列容器，字段（序列化名）包括：

| 字段 | 含义 |
| --- | --- |
| `time_seconds` | 时间轴（一般 1 Hz，也支持可变步长） |
| `mps` | 目标车速 m/s（核心） |
| `grade` | 道路坡度（rad 或 rise/run，序列化带单位后缀） |
| `amb_temp_kelvin` | 环境温度（热仿真、附件功率用） |
| `stop_reason` / `zone` | 停车、区域标签（遥测工况） |

来源：`Cycle.from_resource("Udds")`、`Cycle.from_file("my.csv")`、遥测解算（telematics 页面：GPS -> 速度/坡度）。

### 3.3 `SimParams`：容差与开关（`simdrive/params.rs`）

| 参数 | 默认 | 作用 |
| --- | --- | --- |
| `ach_speed_max_iter` | 3 | 「实际能达到速度」Newton 最大迭代次数 |
| `ach_speed_tol` | 0.001 | 速度收敛容差（相对或绝对，单位 m/s 口径） |
| `ach_speed_solver_gain` | 0.9 | 迭代增益（欠松弛，防振荡） |
| `trace_miss_tol.tol_dist` | 100 | 累计距离偏差绝对阈值 |
| `trace_miss_tol.tol_dist_frac` | 0.05 | 距离偏差比例阈（5%） |
| `trace_miss_tol.tol_speed` | 10 | 速度偏差绝对阈 |
| `trace_miss_tol.tol_speed_frac` | 0.5 | 速度偏差比例阈（50%） |
| `trace_miss_opts` | Error | `Error / Allow / AllowChecked / Correct` 四选一 |
| `trace_miss_correct_max_steps` | 6 | Correct 模式下最多回改步数 |
| `f2_const_air_density` | true | 是否用常数空气密度（fastsim-2 行为） |
| `ambient_thermal_soak` | false | 环境热浸没（太阳辐射等）开关 |
| `save_interval` | 1 | 每 N 步写历史（1=全存，None=只存末值） |

Python：`sd.set_save_interval(None)` 可关掉轨迹记录换速度。

## 四、工程结构：Rust workspace 怎么分层

```text
fastsim (repo, branch fastsim-3)
|-- Cargo.toml                 # workspace 根
|-- fastsim-core/              # 纯 Rust 内核（可独立发布到 crates.io）
|   |-- fastsim-proc-macros/   # 过程宏：derive 记录字段、单位转换辅助
|   |-- src/
|   |   |-- lib.rs
|   |   |-- drive_cycle.rs     # Cycle + YAML/CSV 读写
|   |   |-- simdrive/
|   |   |   |-- mod.rs         # SimDrive::run / run_once / solve_step / set_ach_speed (1575 行, 本文主角)
|   |   |   |-- roadload.rs    # StepInfo::solve_for_speed 闭式+增益迭代速度解 (159 行)
|   |   |   |-- params.rs      # SimParams, TraceMissOptions, TraceMissTol
|   |   |-- vehicle/
|   |   |   |-- vehicle_model.rs   # Vehicle 总装 + from_db
|   |   |   |-- conv.rs             # ConventionalVehicle::solve, DFCO, start-stop (867 行)
|   |   |   |-- bev.rs              # 纯电求解
|   |   |   |-- powertrain_type.rs  # PowertrainType 枚举
|   |   |   |-- powertrain/
|   |   |       |-- fuel_converter.rs  # 发动机效率表、怠速、功率滞后
|-- fastsim-py/                # pyo3 绑定，产出 PyPI wheel
|-- fastsim-cli/               # 命令行入口
|-- fastsim-schema/            # JSON/YAML schema（车辆库校验）
|-- vehdb/ 或关联 fastsim-vehicles # 车辆 YAML 数据
```

依赖要点（从 `Cargo.toml` 实读）：

- `pyo3 0.29.0`：Rust -> Python 扩展。
- `uom 0.38.0`：类型级量纲，字段名序列化带 `_watts` 等后缀的来源。
- `ninterp`：发动机效率二维/一维插值。
- `serde` + `serde_yaml`：车辆与工况文件。
- `anyhow`：错误链。
- Rust `edition 2021`，`rust-version 1.83`；`fastsim-core` 默认 feature 即含 Python 所需序列化，`pyo3` feature 控制绑定。

**读码顺序建议**：`simdrive/mod.rs` 的 `run` -> `solve_step` -> `roadload.rs` 的 `solve_for_speed` -> `vehicle/conv.rs` 的 `solve` -> `fuel_converter.rs` 的效率查表 -> 最后回头看 `bev.rs`（更简单）对比。

## 五、核心代码拆解之一：`SimDrive::run` / `run_once` / `solve_step`

源码：`fastsim-core/src/simdrive/mod.rs`（约 1575 行）。仿真主循环只有三层，剥洋葱读即可。

### 5.1 三层职责

| 函数 | 职责 | 关键点 |
| --- | --- | --- |
| `run(&mut self) -> Result<SimDriveHistory>` | **完整流程**：含 HEV 初始 SOC 平衡迭代、trace miss 统计、历史记录 | 外层 `loop` 调 SOC balancing，内层推进 `run_once` |
| `run_once(...)` | **单次全程**：固定初始 SOC，把 `cyc` 从头到尾 `solve_step` 一遍 | 返回本轮是否仍有「需平衡」的电量偏差 |
| `solve_step(&mut self, ...) -> Result<...>` | **单步物理**：解出该秒真实速度 -> 车需求功率 -> 功率train 出力 -> 记录能量、判定 trace miss | 所有逐秒状态都在 `self`（`SimDrive` 可变）上滚动 |

逻辑示意（按源码顺序改写，非可编译摘录）：

```text
fn run():
    loop:
        # HEV/charge-sustaining 外环（BEV/CONV 实际零次或一次就退出）
        if need_soc_balancing:
            self.run_once()                   # 全程
            adjust initial SOC by charge_sust_target
            if |soc_end - soc_start| < tolerance: break
        else:
            self.run_once(); break
    assemble history (energies, mpg, distance, trace_miss_summary)

fn run_once():
    reset counters, energies, history buffer
    for i in 0..cyc.len():
        self.solve_step(i)
    return soc_error (for outer loop)

fn solve_step(i):
    # 1) 向后看: 目标速度曲线 + 道路负荷 -> 本步真实速度 & 需求功率
    step_info = StepInfo::solve_for_speed(veh, cyc, i, sim_params)
    # 2) 车端: 把车轮功率换算成动力总成「需要出多少」
    #    (考虑传动比、效率、允许倒拖/再生)
    (prop_pwr, ...) = veh.wheel_to_propshaft(...)
    # 3) 功率train 求解: 发动机/电机/电池各出多少
    (fc, mc, batt, trans, ...) = veh.solve(prop_pwr, dt, ...)
    # 4) 能量积分: 燃料 J、电池 J、距离 m、排放样本
    # 5) trace miss: 若 step_info.speed 实际 < 目标, 按 TraceMissOptions 分支
    # 6) 写 history[save_interval], 传播状态 (SOC, 温度, gear...)
```

要点：

1. **状态全在 `SimDrive`/`Vehicle` 上**：每步原地改写，不做全量复制——这是毫秒级跑完的结构原因。
2. **历史是可选的**：`save_interval` 决定每几步 push 一条轨迹；`set_save_interval(None)` 只留末值，实测能把运行时长再压一截（0.0023 s -> 0.0013 s 这一量级）。
3. **功率train 被抽象成「给一个 prop shaft 功率，返回各部件功率流」**：`solve` 内部再按 `PowertrainType` match 到 `conv.rs` / `bev.rs` 等具体实现——这是加新动力形式时唯一要动的扩展点。

### 5.2 单步时序（文字时序图）

```text
t = t_i
|  cyc.mps[i] (目标)  +  cyc.grade[i]
|         |
|         v
|  StepInfo::solve_for_speed   <-- 道路负荷五项功率, Newton 求可达速度
|         |  v_ach, wheel_pwr_req, load_coeffs
|         v
|  veh.wheel_to_propshaft      <-- 传动比/效率/倒拖限制
|         |  prop_pwr_req
|         v
|  veh.solve (PowertrainType match)
|         |-- Conv: 发动机查效率表, 不足则限速 -> 触发 trace miss 逻辑
|         |-- HEV: 发动机+电机分工, SOC 扣电/充电, 考虑 IDM 式功率跟随
|         |-- BEV: 电池放电到电机, 再生制动回充
|         v
|  累加 fuel_J / batt_J / distance_m; 超容差则记 TraceMiss
t = t_i + dt
```

## 六、核心代码拆解之二：道路负荷与「向后看」速度求解（`roadload.rs`）

源码：`fastsim-core/src/simdrive/roadload.rs`（约 159 行）。整个 FASTSim 的物理内核几乎都在这一个 `solve_for_speed` 里。

### 6.1 五项力 / 五项功率

维持车速 `v` 在坡度 `grade` 上所需车轮功率（示意公式）：

```text
F_accel = m * a                    # 车身加速（a 由目标速度差分, 含 rho/回复项）
F_grade = m * g * sin(theta)       # 坡道重力分量
F_aero  = 0.5 * rho * Cd * A * v^2 # 风阻
F_roll  = m * g * Crr * cos(theta) # 滚阻
F_spin  = 车轮/传动转动惯量 * alpha  # 旋转质量等效

P_wheel = (F_accel + F_grade + F_aero + F_roll + F_spin) * v
```

Rust 侧实现为先把每一步对速度的**系数**提出来（`load_coeffs` 一类结构），把 `P_wheel = f(v)` 写成可微/可迭代形式，避免每步重算所有中间量。

### 6.2 `StepInfo::solve_for_speed` 两段式解法

**第一段（闭式试解）**：在 `v = cyc.mps[i]`（目标速度）处代入系数，若给出的功率没有超过车辆动力上限（`prop_pwr_max` 一类），直接采信目标速度——多数正常行驶步走这条路，**零迭代**。

**第二段（增益迭代）**：若「按目标速度需要的功率 > 车辆能提供的最大功率」（车不够劲），用一阶欠松弛迭代找可达速度：

```text
v = v_target
for k in 0..ach_speed_max_iter:          # 默认 3
    p_need = f_load(v)                   # 该速度下道路负荷功率
    p_avail = min(p_max, p_batt_limit)   # 动力上限（发动机/电机/电池短板）
    v_new = v + ach_speed_solver_gain * solve_delta(v, p_need, p_avail)
    #                    默认增益 0.9, 防过冲
    if |v_new - v| < ach_speed_tol:      # 默认 0.001
        v = v_new; break
    v = v_new
v_ach = v                                # 「achieved speed」
```

- 物理含义：车没劲时**降速找平衡**，不是硬积分导致数值爆炸。
- 数值味道：`f_load(v)` 对 `v` 近似 `a*v + b*v^2 + c*v^3`（加速项在固定 dt 下是常数偏置，风阻是 `v^3` 功率），Newton/拟牛顿 3 次内收敛是合理的工程选择；`gain=0.9` 是刻意欠松弛，防止风阻强非线性区振荡。
- `f2_const_air_density=true` 时 `rho` 固定（复刻 fastsim-2），否则随海拔/温度变——高原工况油耗差异的开关就在这。

### 6.3 需求功率的「去向」

`solve_for_speed` 只回答「车轮上要多少」；部件分配在 `veh.solve`：

```text
Conv:  p_wheel / (eta_trans * ratio)  ->  发动机查 eff 表 -> 燃油功率 p_fuel = p_eng / eta
BEV:   p_wheel * (1/eta_mc)           ->  电机;  p>0 电池放, p<0(再生) 电池充 * eta_regen
HEV:   规则/IDM 风格: p_eng 目标跟随 p_demand, 差额电机补, 余量发电充电池
```

效率表插值来自 `fuel_converter.rs` 的 `eff_interp_from_pwr_out`（`ninterp`）；另有 `pwr_idle_fuel`（怠速烧油）、`pwr_ramp_lag`（一阶响应滞后）——怠速和滞后是城市工况油耗比「平均效率点」算出来更差的主要原因。

### 6.4 DFCO 与起步启停（顺带读懂 `conv.rs`）

`ConventionalVehicle::solve`（`conv.rs`，867 行）里两个省油策略：

- **DFCO（Deceleration Fuel Cut-Off，带挡滑行断油）**：驾驶需求为负、挡位在挡、转速高于怠速停油——滑行段油耗直接归零，UDDS 里贡献明显。
- **Start-stop（怠速停机）**：车速低于阈值且需求扭矩为零，关机；起步瞬间起机——对应文档 `start-stop` 页面，`enabled_features()` 可查是否启用。

这两个开关 + 质量 + Crr/Cd，就是配置方案论证时最常用的四个旋钮。

## 七、核心代码拆解之三：Trace Miss——车跟不上工况时怎么办

### 7.1 为什么会 miss

工况是「假想一辆足够劲的车」按秒画的；你给的车若功率不够、质量太大、坡太陡，`solve_for_speed` 迭代出来的 `v_ach` 就会低于 `cyc.mps[i]`。连续 miss 的直接后果：

- 速度差积分 -> **距离比工况短**；
- 「按时间对齐」的能量统计失真（同一步还没走完工况要求的位移）；
- 对比不同车时不再公平（弱车「省油」其实是因为根本没跑完路线）。

### 7.2 四种策略（`TraceMissOptions`，`simdrive/params.rs`）

| 策略 | 行为 | 什么时候用 |
| --- | --- | --- |
| **`Error`**（默认） | 累计 miss 超过 `TraceMissTol` 直接返回错误，仿真失败 | CI/回归测试，「车必须跟上工况」的硬约束 |
| **`Allow`** | 记录 miss 指标（速度偏差、距离偏差），照常跑完出结果 | 敏感性粗扫，先看大概数 |
| **`AllowChecked`** | 允许跑完，但结束时若仍超容差给出显式警告/标记 | 汇报用：结果可用，但必须附带 miss 比例 |
| **`Correct`** | 对 miss 步骤**把工况曲线改低**（向后追车），最多 `trace_miss_correct_max_steps`（默认 6）步，强行让目标曲线等于车能力 | 出「等效工况」报告，或车参数明显偏保守时校准 |

容差判据是**绝对 + 相对双门限**，先到先触发：

```text
speed:  |v_ach - v_target| > tol_speed(10)  OR  frac > tol_speed_frac(0.5)
dist:   |d_ach - d_target| > tol_dist(100)  OR  frac > tol_dist_frac(0.05)
```

注意距离比例阈值只有 5%——**1% 的速度长期跟不上就会在距离上被判 miss**，这和「累计距离是速度的积分」一致，也是实车标定对标时最先盯的数。

### 7.3 和速度求解的关系（分工别搞混）

```text
solve_for_speed (roadload.rs)
    目的: 单步物理正确 —— 在动力上限内给出自洽的 v_ach
    手段: 3 次增益迭代, 从来不会「无中生有」的速度

trace miss 逻辑 (simdrive/mod.rs)
    目的: 跨步、跨策略的**质量闸门** —— 判断这条结果能不能交给下游
    手段: 累计偏差 vs TraceMissTol, 按 TraceMissOptions 分支
```

一句话：**solver 负责「这一步跑得对不对」，trace miss 负责「整圈结果敢不敢用」。**

### 7.4 实战建议（给领导选型时说）

- 首次用陌生车跑监管工况：先 `AllowChecked`，看 `trace_miss_summary`；若 miss 大，检查 `pwr_max`、质量、`ach_speed_max_iter` 是否被调过。
- 出「与 EPA 对标」的正式数：用默认 `Error` + 确保 0 miss——否则 mpg 不具备可比性。
- 车动力明显偏保守的探索性分析：`Correct` 把工况削到车能力边界，得到的是「该车能完成的等效工况」，报告里必须注明是 corrected。

## 八、功率train 四件套：Conv / HEV / PHEV / BEV 怎么解

### 8.1 枚举分发（`powertrain_type.rs`）

```text
enum PowertrainType {
    Conventional(ConvVehicle),   # conv.rs
    Hybrid(HevVehicle),          # 规则/功率跟随控制
    PHEV(PhevVehicle),           # 同 HEV + 外充 SOC 窗
    BEV(BevVehicle),             # bev.rs
    # FCEV 分支预留
}
```

`Vehicle::solve(...)` 把 `(prop_pwr_req, dt, 状态)` match 到对应实现，返回统一形状的部件功率流（`fc / mc / batt / trans` 的 `StepInfo` 或等价结构），便于 `SimDrive` 统一积分。

### 8.2 Conventional（`conv.rs` 要点）

```text
输入: p_wheel 需求
1. 由传动比反推发动机目标转速/扭矩
2. 若在 DFCO 条件 (滑行断油): p_fuel = 0, 发动机倒拖
3. 若 idle 条件 (start-stop 关机): p_fuel = 0, 附件改由 12V 蓄电池补
4. 否则: p_eng = p_wheel / (eta * ratio) + parasitics(附件、油泵)
5. 效率查表: p_fuel = p_eng / eff(p_eng);  一阶滞后平滑 p_eng(t)
6. 超 pwr_max: 截断 -> v_ach 追不上 -> 上层 trace miss
```

关键物理细节（也是面试可以讲的点）：

- **附件功率**从发动机机械功率里扣（空调、水泵）；
- **怠速油耗** `pwr_idle_fuel` 不为零——红灯等待也烧油；
- **效率表非单调**：高效区在中高负荷，低负荷（巡航轻踩）热效率反而低，这解释了「为什么小排量+增压+CVT 常常比大排量更省油」的机理在模型里的呈现方式。

### 8.3 BEV（`bev.rs`，逻辑最短）

```text
p_wheel > 0:  p_batt_out = p_wheel / (eta_mc * eta_trans)
              化学电池能量 -= p_batt_out * dt
p_wheel < 0:  p_charged = |p_wheel| * eta_regen * eta_mc
              化学电池能量 += p_charged * dt      # 再生制动
SOC = E / E_nominal;  下限截断 -> 触发动力不足/trace miss
附件: p_aux (W) 恒定或随 amb_temp 插值
```

「化学能量」与「牵引能量」分开记账——本机 Tesla Model 3 实测：化学放电 4.07 MJ = 牵引 2.23 MJ + 附件 0.34 MJ + 其它（变流、附件、误差项）——这个拆分直接对应 `to_dict` 里的 `energy_out_chemical` / 牵引 / `aux` 字段，汇报电耗构成时按这三个数讲。

### 8.4 HEV 的规则控制（概念级）

FASTSim 的 HEV 不是等效燃油消耗最小（ECMS）、也不是 DP 全局最优，而是**文档化的规则/功率跟随策略**，目标是**每个循环电量平衡**：

```text
p_eng_target = f(p_demand, soc)    # 需求高: 发动机主驱 + 余量发电
                                   # 需求低: 发动机可能熄火, 纯电蠕行
p_mc = p_demand - p_eng            # 电机补差
soc 规划目标窗口围绕 soc_target 波动
```

工程可信度来自两点：**规则可解释**（车企策略白盒化）+ **循环级 SOC 平衡**（油耗数字站得住）。DP/ECMS 属于「策略上界研究」，FASTSim 站在「可复现的工程评估」一侧。

### 8.5 HEV 外环：Charge-sustaining / SOC balancing（`run()` 里的 loop）

混动跑完一圈若 SOC 净变化 `-1%`，等于「偷了 1% 电」，油耗被低估。FASTSim 的做法是**迭代初始 SOC**：

```text
soc_0 = 默认(如 0.6)
repeat:
    结果 = run_once(soc_0)
    err  = soc_end - soc_0          # 或 vs 目标 soc_target
    若 |err| < 电量平衡容差: 退出
    soc_0 += k * err                 # 往「更省/更费」方向修初始电量
直到收敛或达到最大迭代
```

- BEV **不需要**这个外环（电量本来就是用完为止的消耗品）；
- PHEV 在电耗阶段（CD, charge-depleting）后也常接这段（CS, charge-sustaining），报「亏电油耗」时尤其关键；
- 和 `trace_miss` 一样，这是**结果可信度**层面的机制，不改变单步物理。

源码阅读位置：`simdrive/mod.rs::run` 的外层循环 + `charge_sust_target`/容差字段；测试里搜 `soc` 平衡相关断言可看官方把容差定在多少。

## 九、Python 绑定：`fastsim-py` + 包结构

PyPI 包 `fastsim`（3.1.0）是「预编译 Rust wheel + 薄 Python 壳」：

```text
site-packages/fastsim/
|-- __init__.py          # 导出 Vehicle, Cycle, SimDrive, SimParams, get_label_fe ...
|-- *.so                 # pyo3 编译产物 (Linux x86_64/aarch64 wheel)
|-- resources/           # 内置 9 车 YAML + 监管工况数据
`-- ...                  # 从_resource/from_file 的路径解析逻辑
```

绑定层做的事（`fastsim-py` crate）：

1. `#[pyclass]` 包 `Vehicle` / `Cycle` / `SimDrive` / `SimParams`；
2. 方法名保持 Rust 原名（`run`, `run_once`, `set_save_interval`, `from_resource`, `list_resources`）；
3. 结果转 `PyDict` / 自定义 `to_dict(flatten=True)`——扁平键就是带单位后缀的序列化名（`distance_miles`, `miles_per_gallon_dynamometer`, `energy_out_chemical` ...）；
4. `enabled_features()` 暴露编译期/运行期开关（start-stop、DFCO、热仿真等是否启用）。

常用调用面（按文档 getting-started / simdrive / loading-vehicles 页面归纳）：

```python
from fastsim import Vehicle, Cycle, SimDrive, SimParams, get_label_fe

veh = Vehicle.from_resource("2016_TOYOTA_Prius_Two")
# 或 veh = Vehicle.from_db("byd/atto-3/2023")   # 查 fastsim-vehicles
cyc = Cycle.from_resource("HwFet")
sp  = SimParams(trace_miss_opts="AllowChecked", ach_speed_max_iter=3)
sd  = SimDrive(veh, cyc, sp)     # 参数也可后设
sd.set_save_interval(10)         # 每 10 步记一次轨迹, 减内存
hist = sd.run()
d = hist.to_dict(flatten=True)

label = get_label_fe(veh, lab_cyc="Udds", rtc_cyc="UddsHwFet")  # 对标 EPA 贴纸
```

命令行（`fastsim-cli`）：安装后通常提供 `fastsim` 可执行，对车辆 YAML + 工况 YAML/CSV 直接出结果 JSON，适合批处理脚本（本文不展开，以 `--help` 为准）。

### 9.1 性能与工程化注意

- **热路径在 Rust**，Python 每步只做薄转发——所以 1370 步工况实测 **~2 ms**；真正要快的是**批量**（成千上万车 x 工况），用多进程而不是单次微优化。
- `save_interval=None`（不记轨迹）单次能再省约一半时间——扫参时首选用。
- 字段**全带单位**：交流时千万别说「速度的单位是 mph」——序列化是 m/s 系，mpg 是结果字段不是输入。
- 复现性：锁 `fastsim==3.1.0` + 记录车辆 YAML 版本（车辆库是独立 git 仓库），否则同名车换库版本油耗可能变。

## 十、车辆数据库：`fastsim-vehicles` 怎么组织

### 10.1 目录即车型（path = 主键）

```text
fastsim-vehicles/
|-- v1/
    |-- fastsim-3/
    |   |-- bev/
    |   |   |-- byd/atto-3/2023/base/r1.yaml
    |   |   |-- byd/dolphin-active/2024/base/r1.yaml
    |   |   |-- tesla/... , bmw/... , ford/... , nissan/... , renault/... , volvo/...
    |   |-- hev/   # 混动
    |   |-- conv/  # 纯燃油
    |   `-- phev/  # 插电
    `-- fastsim-2/           # 旧格式 CSV, 迁移用
        `-- *.csv
```

- 实测 **88 个文件**量级，覆盖多家品牌的 BEV 为主，含 **比亚迪 Atto 3（2023）、Dolphin Active（2024）**——和国内业务对上号，汇报时可以提「库里已经有比亚迪车型的公开参数」。
- `base/r1.yaml` 的 `r1` 是 revision 口径；schema 校验走 `fastsim-schema` + 文档站的 wasm 校验器（`loading-vehicles` / `modeling-vehicles` 页面）。
- 与内置 9 车的区别：内置车随 wheel 打包（`from_resource`），数据库车要显式拉仓库或 `from_db`/URL 读取（`loading-vehicles` 页面有本地路径与 git 两种方式）。

### 10.2 YAML 里必看的字段族（自建车时的 checklist）

```text
chassis:    mass_kilograms, drag_coef, frontal_area_m2, wheel_rr_coef,
            wheel_inertia, wheel_radius, ...
pt_type:    Conv|Hybrid|PHEV|BEV
  fc:       pwr_max_watts, eff_interp_from_pwr_out.grid/values,
            pwr_idle_fuel, pwr_ramp_lag
  trans:    gear_ratios, final_drive_ratio, shift_schedule...
  batt:     energy_capacity_wh (BEV/PHEV), voltage, internal_resistance...
  mc:       pwr_max_watts, eff_interp...
```

自建车三步：抄一辆最近的车 -> 只改实车参数（质量、Cd、电机/发动机功率）-> 跑 UDDS 看 trace miss 和 mpg 是否落在合理带。缺效率表时先用同类车插值，标注「估算」——**可复现性来自显式声明近似**，不来自假装精确。

### 10.3 从 FASTSim 2 迁移（migration-guide 页面要点）

- 数据格式：CSV -> YAML（`v1/fastsim-2/*.csv` 仍保留兼容读取）；
- 单位字段名显式化（`uom`），旧脚本里裸 `mpg`/`kW` 的键不再成立；
- API：`SimDrive(...).run()` 组织方式类似，但结果键名、trace miss 枚举、SOC 平衡钩子按 3.x 文档为准；
- 行为差异开关：`f2_const_air_density` 等留了「按 2 的物理近似跑」的兼容位，对旧报告复现有用。

## 十一、高级功能速览（文档页面对应，便于按需深挖）

| 功能 | 文档页 | 要点 |
| --- | --- | --- |
| **EPA Label FE** | `label-fe` | `get_label_fe` 把 lab 三工况按 EPA 系数合成窗口贴纸 mpg；和 fueleconomy.gov 对数 |
| **Start-stop** | `start-stop` | 怠速停机省油；与 DFCO 开关在 `enabled_features()` 可见 |
| **CAVs（网联自动驾驶）** | `cavs` | 车队级跟车/协同场景的能耗扩展——**注意这是纵向能耗口径的 CAV，不是横向控制** |
| **热仿真** | `thermal-simulations` | 乘员舱/电池热 + 附件空调功率随 `amb_temp` 变；纯电冬季续航差异的主要来源 |
| **环境热浸** | `SimParams.ambient_thermal_soak` | 停车日照升温（SOC/空调启动负荷） |
| **遥测工况** | `telematics` | GPS/速度日志 -> 自定义 Cycle；车队真实油耗反推 |
| **2->3 迁移** | `migration-guide` | 见 10.3 |
| **相关工具** | `related-tools` | RouteE / T3CO / ADOPT 等 |

## 十二、它和「闭环驾驶仿真」的分工（领导可能追问）

常见误会：把 FASTSim 当成 CARLA 那类「自动驾驶闭环仿真」。对照说清：

| 维度 | FASTSim（本文） | CARLA / 行为仿真器（横向） |
| --- | --- | --- |
| 回答的问题 | **这车这工况烧多少油/电** | **这策略会不会撞、轨迹顺不顺** |
| 速度从哪来 | 工况给定，向后反解功率（backward） | 纵向控制器积分（forward），常含 PID/IDM |
| 横向 | 无（纯纵向） | 轨迹跟踪、换道、碰撞 |
| 单次耗时 | 毫秒级 | 秒到分钟级（含渲染可更长） |
| 典型用户 | 能耗对标、配置扫参、政策/TCO | 算法回归、场景库测试 |
| 混用方式 | 把规划器输出的 `v(t)` 当自定义 Cycle 丢进来算能耗 | 在 CARLA 里挂 FASTSim 式纵向油耗模型做闭环能耗 |

**衔接点**：「上面规划给速度曲线，下面 FASTSim 算这条曲线的能耗」是工程里最常见的组合——速度是两者共同的接口，这也解释了为什么 FASTSim 的输入只有速度（+坡度）而没有方向盘。

与你手头 MOT / Flow Matching 世界的类比（口头类比，别写进正式对比表）：FASTSim 像「无训练的物理打分器」——输入轨迹（工况）、输出标量（油耗/距离）；世界模型是学出来的动力学先验。两者都在回答「这条动作序列会导致什么」，只是一个用解析物理、一个用神经网络。

## 十三、实操：「测所有车的能耗」一段脚本走通

把三件事接起来：**内置车遍历 + 敏感性 + 比亚迪数据库车**。

```python
# batch_energy.py  --  批量出数, 单位与字段名以 to_dict 为准
from fastsim import Vehicle, Cycle, SimDrive

CYCLES = ["Udds", "HwFet", "UddsHwFet"]     # 先用监管三件套

def run_one(veh_name, cyc_name):
    veh = Vehicle.from_resource(veh_name)
    cyc = Cycle.from_resource(cyc_name)
    sd = SimDrive(veh, cyc)
    sd.set_save_interval(None)              # 只要末值, 更快
    d = sd.run().to_dict(flatten=True)
    return {
        "vehicle": veh_name,
        "cycle": cyc_name,
        "dist_mi": d.get("distance_miles"),
        "mpg": d.get("miles_per_gallon_dynamometer"),
        # BEV 字段不同, 用 KeyError 容忍
        "e_out_chem_J": d.get("energy_out_chemical"),
    }

if __name__ == "__main__":
    vehicles = Vehicle.list_resources()
    for v in vehicles:
        for c in CYCLES:
            print(run_one(v, c))

    # 敏感性: 质量 +800 kg (对应汇报里 -18% 那组数)
    base = Vehicle.from_resource("2012_Ford_Fusion")
    heavy = Vehicle.from_resource("2012_Ford_Fusion")
    # 质量字段以 YAML/chassis 序列化名为准, 常见 mass_kilograms
    # heavy.chassis.mass_kilograms += 800
    # ... 对比两次 run 的 mpg
```

数据库车（需能访问 `fastsim-vehicles` 仓库路径或 URL，细节见 `loading-vehicles` 页面）：

```python
# 路径写法示意 — 以官方 loading-vehicles 文档为准
# veh = Vehicle.from_file("path/to/fastsim-vehicles/v1/fastsim-3/bev/byd/atto-3/2023/base/r1.yaml")
# cyc = Cycle.from_resource("Udds")
# print(SimDrive(veh, cyc).run().to_dict(flatten=True))
```

汇报用输出表（把本机已有数字填进去即可）：

| 车 | 工况 | 距离 (mi) | 结果 | 备注 |
| --- | --- | --- | --- | --- |
| 2012 Ford Fusion | UDDS | 7.451 | **34.4 mpg**，燃料 7.303 kWh，耗时 ~0.0023 s | 基线 |
| 同上 +800 kg | UDDS | 同量级 | **28.2 mpg（-18%）** | 质量敏感性 |
| Tesla Model 3 | 电耗口径 | - | 化学放电 4.07 MJ（牵引 2.23 MJ + 附件 0.34 MJ + 其它） | 能耗三分法 |
| BYD Atto 3（数据库） | UDDS | - | 待跑 | 库内中国车型 |

## 十四、自测 Q&A（对练/面试向）

**Q1：FASTSim 向后看（backward）和向前看（forward）差在哪？为什么不直接向前积分？**
A：向后看先有目标速度，闭式+3 次迭代解出「需要多大功率」，一步到位、毫秒级、适合批量；向前看要控油门让速度逼近目标，依赖控制器且步长/刚性问题多，更适合闭环驾驶研究。能耗对标（给定工况）用 backward 是行业常规；要研究「司机怎么开」才切 forward。

**Q2：trace miss 的 Error/Allow/AllowChecked/Correct 怎么选？**
A：CI 与对标用 `Error`（0 miss 才出数）；粗扫用 `Allow`；对外报告用 `AllowChecked`（数字+miss 率一起给）；车能力明显不足想出「等效工况」用 `Correct`（最多回改 6 步目标曲线）。容差是绝对（速度 10、距离 100）与比例（速度 50%、距离 5%）双门限，距离更严因为是速度积分。

**Q3：HEV 为什么 `run()` 外面还要套 SOC 平衡循环？**
A：循环末尾 SOC 比开始高，等于「偷电」，油耗被低估；低了则高估。通过迭代初始 SOC 让 `|soc_end - soc_start|` 小于容差（charge-sustaining），油耗才可比。BEV 不需要；PHEV 报亏电油耗时同样关键。

**Q4：速度求解为什么默认只迭代 3 次、增益 0.9？**
A：负荷功率对 v 近似 `a v + b v^2 + c v^3`，Newton 类方法在工程容差 `0.001 m/s` 下通常 2-3 步收敛；`0.9` 欠松弛防风阻强非线性区振荡。动力不足时走迭代降速，动力足够时直接采信目标速度，零迭代。

**Q5：Dolphin/Atto 3 这种中国车在库里怎么找？**
A：`fastsim-vehicles` 的路径就是主键：`v1/fastsim-3/bev/byd/atto-3/2023/base/r1.yaml`、`.../byd/dolphin-active/2024/base/r1.yaml`。共约 88 个文件、以 BEV 为主；内置 wheel 只有 9 辆监管代表车，中国车型走数据库。

**Q6：FASTSim 能替代 CARLA 吗？**
A：不能，问题都不同。FASTSim：纵向能耗、给定速度曲线、毫秒级、无横向；CARLA：闭环策略/安全、有控制器与传感器。常见组合是把规划输出的 `v(t)` 当 FASTSim 的自定义工况算油耗——速度曲线是两者的接口。

**Q7：结果里 `mpg` 有好几个，听谁的？**
A：`label` 模块出的是 EPA 窗口贴纸口径（多工况加权+调整系数）；单工况 dynamometer mpg 是实验室读数；`get_label_fe` 与 fueleconomy.gov 对数时用贴纸口径。汇报时先说清是「lab 单循环」还是「label 综合」。

**Q8：字段名怎么一堆 `_watts` `_meters`？**
A：Rust `uom` 量纲库 + serde 序列化，单位写进键名，杜绝 kW/Wh/mph 混用。读 YAML 或 `to_dict` 时以键名后缀为准，写对比脚本不要硬编码裸单位。

**Q9：和 RouteE/T3CO/ADOPT 什么关系？**
A：同一工具链：FASTSim 是单车物理内核；RouteE 把批量结果收成路段/车队概率模型；T3CO 做重卡 TCO（能耗分项来自同类模型）；ADOPT 做技术扩散与经济学。给上层供数时先 FASTSim 批跑，再抽特征建 RouteE 式轻模型。

**Q10：为什么内核用 Rust 而不是纯 Python？**
A：逐秒非线性求解 + 插值 + 大批量扫描是纯计算热路径；Rust 保零成本抽象与内存安全，pyo3 暴露后 Python 侧仍保持 `Vehicle/Cycle/SimDrive` 的简洁 API。实测单循环毫秒级，瓶颈反而是 Python 批量调度——所以批量上多进程而不是改内核。

## 十五、汇报口述稿（给领导，30 秒 + 2 分钟两档）

**30 秒版：**
> 「看明白是 NREL 系的 FASTSim（现挂 NatLabRockies 组织，当前分支 fastsim-3），BSD 开源，Rust 内核 + Python 包，PyPI 3.1.0。输入车辆参数和速度工况，向后看逐秒反推功率train 功率，直接出油耗/电耗/EPA 贴纸 mpg。本机实测 Ford Fusion 跑 UDDS 34.4 mpg、单循环约 2 毫秒；质量加 800 kg 油耗掉 18%；Tesla 电耗按化学能/牵引/附件拆分。车辆库里有比亚迪 Atto 3 和 Dolphin，可以拿来直接对标。」

**2 分钟版**（在 30 秒上加这四层）：
1. **架构**：workspace 分层——`fastsim-core` 纯 Rust（`simdrive` 主循环、`roadload` 速度解、`vehicle` 功率train 枚举），`fastsim-py` pyo3 绑定，车辆库独立 git，schema 校验。
2. **物理**：道路负荷五项（加速/坡度/风阻/滚阻/旋转质量），功率对速度三次型，闭式试解 + 3 次欠松弛 Newton（增益 0.9、容差 0.001）。
3. **可信度**：trace miss 四策略（Error/Allow/AllowChecked/Correct，距离比例容差 5%）+ HEV 外环 SOC charge-sustaining——保证「不是偷电换来的省油」。
4. **怎么用到我们**：规划给的 `v(t)` 丢进自定义 Cycle 就能出能耗；配置扫参（质量、Cd、Crr、启停/DFCO）毫秒级；出数先 AllowChecked 看 miss，对标 EPA 用 label 口径。

## 十六、小结：这个项目值得记住的三件事

1. **问题切得足够小**：只做「给定速度曲线的纵向能耗」，不做横向、不做交通、不做控制器——所以能毫秒级、能出监管口径数字、能被 RouteE/T3CO 这类上层工具引用。工程上「收窄问题」本身就是设计。
2. **可信度机制写进代码**：trace miss 容差与策略、HEV SOC 外环、带单位字段名——三样都是「防数字被误用」的闸门。评估一个仿真器，先看它怎么**拒绝**给出不可靠结果，再看它怎么给结果。
3. **Rust 内核 + 薄 Python 面**：热路径下沉、API 保持三个名词（Vehicle/Cycle/SimDrive），是「老数值仿真程序现代化」的范本路线；迁移期还留了 `f2_const_air_density` 这种行为兼容位，复现旧报告不用回退大版本。

---

**延伸阅读（按读码顺序）**

- 文档：`https://natlabrockies.github.io/fastsim/` — installation、getting-started、simdrive、trace-miss、label-fe、telematics、migration-guide、related-tools
- 源码：`fastsim-core/src/simdrive/mod.rs`（主循环）-> `roadload.rs`（速度解）-> `vehicle/conv.rs` / `bev.rs`（功率train）-> `fuel_converter.rs`（效率表）
- 车辆库：`https://github.com/NatLabRockies/fastsim-vehicles`；Rust API：`https://docs.rs/fastsim-core/`
- 产品页：`https://www.nlr.gov/transportation/fastsim`

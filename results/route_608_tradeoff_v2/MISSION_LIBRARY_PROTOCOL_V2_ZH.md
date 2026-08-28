# URSA–BO04 多 A/B 任务库协议 V2：稳定所选路线门

日期：2026-08-28（PDT）

## 1. 版本定位与证据边界

V2 是在 V1 参考评分完成后的 post-audit expanded panel，不追溯改写或替换 V1。V1 已显示，把“任一备选路线不可飞”当成整项任务无效会系统性删除固定翼规划器本应正常处理的任务。V2 只修正这一输入侧任务有效性定义；BO04、URSA 四个固定方法、飞行器、几何、AGL、bank、stall、CL、推力、deadline、fine/coarse 间距和参考评分定义均不改变。

V2 不得称为在 V1 结果之外预注册的原始确认试验。论文必须同时报告 V1 的小样本结果及其门设计问题，并把 V2 写成由输入侧门审计触发的扩大验证。V2 的任务选择仍不得读取 FuXi 输出或参考能量。

## 2. 防止直接复用 V1 已评分地形

从冻结且尚未访问输出的 `formal_reference_manifest_v3.json` 读取 V1 的 15 个 exact-DEM group 身份，并在 V2 中全部排除。排除规则只使用 V1 参考清单中的地形身份，不读取 V1 `reference_score.json`。V2 仅使用 confirmation/reserve 中其余 exact-DEM groups；输出引用身份在 V2 input manifest 冻结后才允许数值访问。

## 3. 输入侧稳定性门

对 raw、hard warning、continuous attenuation、S-only 和 matched-uniform 每个规划世界分别要求：

1. fine（1.5 m）与 coarse（3.0 m）所选路线均存在且相同；
2. 三条候选路线各自的可行/不可行分类在 fine/coarse 间一致；
3. 最终所选路线在 fine/coarse 两个间距下均可飞；
4. 不要求所有未选备选路线均可飞。稳定拒绝一个不可飞备选属于规划器的正常可行性处理，不是删除整个任务的理由。

人口网格按上述 raw 门入库。challenge 还必须满足：raw 稳定选择 `ridge_downstream`；`direct_mid` 在 raw fine/coarse 下可飞；downstream 相对所有可飞替代路线具有正输入侧能量余量；相对 direct 具有正额外距离和正额外时间。不可飞替代路线不进入能量最小值，但其失败原因完整保留。

## 4. 不变的任务、规划与参考定义

- population：`f={0.25,0.50,0.75}`，`L={600,1200,1800} m`；
- challenge：`f=0.1,...,0.9`，`L={600,900,1200,1500,1800} m`；
- 同一 exact DEM 的多个 A/B 是相关任务，不是独立地形；
- signed BO04 仅作为载体审计，规划器信用为 `0.875 max(w_BO04,0)`；
- hard warning、continuous attenuation、S-only、matched-uniform 的公式与 `eta=0.05` 不变；
- 三段 C2 连续高度平台、共同初末位置/空气状态、9.7 m/s 空速、15 度 bank、50 N 推力、30--214.29 m AGL、600 s deadline 不变；
- FuXi-W 为主世界，FuXi-full 为敏感性世界；每条冻结几何在 full/suppress-w 和 fine/coarse 中独立重新配平，不能称为 fixed-control replay；
- 数值等价阈值仍为 `tau=max(1 J,0.001 min(J_a,J_b))`。

## 5. 对称风险—机会报告

每个方法必须同时报告：

- reference-false downstream corridor 数量与修正率；
- reference-valid raw route 数量与被放弃率；
- 其中 reference-valid lift opportunity（raw 接近 oracle 且同几何 `J_suppress-w-J_full-w>0`）的保留/放弃；
- finite gross benefit、gross harm、net benefit、benefit/harm ratio、最坏 finite harm；
- raw 可飞但方法路线参考不可飞的 hard harm，以及相反方向的 hard benefit；
- corrected false corridor 的距离/时间变化，不预设一定为正；
- exact-DEM group macro 和 5000 次 group-cluster bootstrap；confirmation、reserve 和 combined 分开报告。

不得从四个固定方法中事后挑选一个重新称为“预注册 primary”。若 V2 显示原 hard warning 取舍失败，正文必须将其写成算法限制或否证结果；任何后续新方法需要新的版本与独立任务冻结。

## 6. 审计与失败处理

V2 input manifest 必须绑定本协议、V1 input-only 代码、V2 wrapper、共同几何代码、2251-panel 和 V1 output-blind reference manifest。input 阶段 `output_npz_numeric_open_count=0`。所有几何、raw、方法和参考不可达状态保留；不得删掉不利任务、降低门或反复重跑到成功。


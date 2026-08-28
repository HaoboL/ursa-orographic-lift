# URSA–BO04 多 A/B 任务库决策日志 V1

日期：2026-08-28（PDT）

## D001：随机全任务均值不能作为主要验证问题

原 325-task 设计及其非训练分层中，raw 选路改变只发生在少数任务，导致总体百分比主要被大量未触发任务稀释。该结果能描述既有任务库，却没有直接、高效地检验论文真正攻击的机制：BO04 把受上游遮挡的下游山脊估成强正升力，规划器因此为错误的 downstream route 付出额外距离和时间。

决定：把主要 estimand 改为输入条件化的 route-regret reduction；随机/固定任务域全分母结果降为触发率和总体背景。原因不是追求更大节能数字，而是让实验分母与预先声明的科学问题一致。

## D002：采用 scenario-based stress testing / rare-event enrichment

文献依据包括航空 adaptive stress testing、Scenic 的约束情景分布、自动驾驶 unsafe-cut-in accelerated evaluation 和 rare-event importance sampling。共同原则是：先定义感兴趣的罕见失效或输入条件，再集中生成能暴露它的情景；若要回推基准总体概率，必须另有 base distribution 与统计校正。

决定：V1 建立两个冻结集合。

1. population grid：每个严格双山脊 pair 的 3×3 A/B 网格，用于声明任务域中的 raw-downstream 触发比例和全分母后果；
2. challenge grid：每个严格 pair 的 9×5 密网格，仅按 raw BO04 稳定选择有额外距离/时间代价的 downstream route 入库，用于条件化主性能检验。

不得使用 FuXi、reference energy、URSA 改路或 `Delta J` 决定入库。结果驱动的 AST/CE 搜索若进行，只能作为独立 discovery/falsification 证据。

## D003：不伪造总体概率，不把同地形多任务当独立环境

项目没有真实固定翼 UAV A/B 任务分布，因此 challenge set 不使用虚构 importance weight，也不宣称代表现实任务平均节能或自然发生率。population grid 只能估计其声明合成任务域中的触发比例。

同一 exact-DEM group 的不同 A/B 是不同 mission，但共享地形和参考风场。headline 统计先在组内汇总，再对 exact-DEM group 等权；置信区间按 group cluster bootstrap。这样，多做目标任务增加机制覆盖，而不会凭任务复制虚增独立样本量。

## D004：学术叙事边界

正文可写“mechanism-conditioned scenario-based stress test”或“input-conditioned rare-event challenge set”。不得写“随机验证证明普遍节能”，也不得把筛入 challenge 本身称为 reference-confirmed failure。只有打开冻结参考后，才能把 downstream detour 判为真实误绕行并统计 URSA 的纠正率和能量后果。

## D005：扩大面板不能替代既有风险闭环

用户要求新 challenge 实验同时处理前序审计暴露的攻击点。决定把载体错名、FuXi 轴语义、结果泄漏、同地形伪独立、瞬时折角、非 bank-aware 能量、AGL/期限、reference replay 误称、S/E 公式身份、通用收缩替代解释、最坏伤害遗漏和四种能量量混写全部转成第 8.1 节的机器门与报告字段。任一门失败时保留失败状态，不通过删任务、改阈值或仅展示有利子集恢复 headline。

## D006：干运行否决“共同绝对高度”，改为共同边界的连续地形随行剖面

正式 manifest 冻结前的第一次干运行显示：首个确认地形的 52 个候选全部因共同绝对高度无法同时维持 30--210 m AGL 而排除。随后对前 12 个确认地形的诊断干运行中，11 个地形的 621 个候选均被同一门排除；余下 1 个地形只有 3 个几何通过，且 raw fine/coarse 赢家不稳定。故这不是个别极端地形，而是协议把复杂地形任务系统性删去的设计错误。

决定：在 formal freeze 前撤销共同绝对高度，改为三路共享相同 A/B 位置及 100 m AGL 初末状态、内部各自采用冻结的 100 m AGL 地形随行三次样条。地形起伏导致的 `gamma`、爬升、下降、`CL`、推力和推进能耗进入现有连续固定翼逆动力学；30 m 最小 AGL、214.29 m 风廓线支持上限、15 度 bank、stall、推力、600 s deadline 和 1.5/3.0 m 一致性仍是硬门。三路初末空气航向和地速垂直分量必须机器验证相同。

依据：连续空间曲线、显式初末姿态、曲率/扭率及地形约束是既有 3D UAV 路径验证的标准结构（Chen et al., 2019, DOI: 10.3390/app9092621）；固定翼转弯路径也应把 bank、stall margin 与气动能耗共同处理（Reyner and Liem, 2026, DOI: 10.3390/drones10060426）。这次修改由无输出访问的输入侧干运行失败触发，不是观察参考节能结果后放宽门。旧干运行目录完整保留。

## D007：逐地形 100 m AGL 样条不可飞，改为预瞄式三段 C2 间隔平台

D006 的第一版实现按 DEM 短波起伏逐点维持约 100 m AGL。单组干运行的 52 个候选全部不可飞：主要原因是 `thrust_limit`，并出现最高 31.49 度的 bank 需求；扩展到同一 12 组输入侧诊断后，已完成的 11 组仍无入库任务，raw 路线失败账本累计以 `thrust_limit`（1330 条路线）和 `bank_limit`（381 条路线）为主，另 1 组在端点三维固定空速风三角处失败。该失败说明逐像元追地形会把 DEM 的短波起伏错误地变成固定翼的高频爬升/俯冲命令。

决定：保留 D006 的共同 A/B、显式三维动力学和全部飞行门，但把内部高度剖面改为预瞄式三段 C2 结构。每条路线在进入走廊前以 quintic smoothstep 平滑爬升到一个路线特定的绝对高度平台，穿越走廊后平滑返回共同 100 m AGL 终点；平台高度的可行区间由 1.5 m 稠密地形剖面对 30--214.29 m AGL 约束逐点求交，区间内按“典型内部 AGL 接近 100 m”的冻结规则选取。这样不追随短波 DEM 噪声，仍提前为山脊获得高度，并把实际爬升/下降、bank-aware 配平和推进能耗全部计入。旧 D006 实现及失败干运行完整保留，本次修改仍发生在参考输出打开和 formal manifest 冻结之前。

## D008：用冻结飞行包线扩大 `L`，不是降低推力门

三段平台的首个单组干运行仍无入库任务。输入侧诊断显示，该组 upstream 路线需要约 172 m 的预爬升；冻结飞行器在 9.7 m/s、零倾侧时 10 度爬升可配平（45.40 N），12 度已超过 50 N 推力上限（52.44 N）。quintic 过渡的峰值垂直坡度为 `1.875 Delta z/L`，原 `L<=720 m` 对 172 m 爬升要求约 24 度峰值坡度，因此推力失败由任务长度与既定飞行包线不相容导致。

决定：在 formal freeze 和参考访问前，把 population `L` 改为 `{600,1200,1800} m`，challenge `L` 改为 `{600,900,1200,1500,1800} m`。上限 1800 m 使上述代表性爬升的 quintic 峰值坡度降至约 10.2 度；短 `L` 仍保留以完整记录不可飞任务。未改变 50 N 推力、15 度 bank、CL、stall、30 m AGL 或 600 s deadline。平台构造使用 31--213.29 m AGL 的 1 m 双侧数值保护带，正式验收仍使用 30--214.29 m，以避免插值/离散误差把理论等号误判为通过。旧 `L` 干运行和失败账本完整保留。

## D009：signed carrier 与规划信用层分离

扩大 `L` 后的 12 组输入侧诊断仍无任务入库：406 个候选在地形平台门排除，206 个进入 raw 动力学门；后者的路线失败以 `thrust_limit` 为主（189 条），另有 2 条 `bank_limit` 和 1 条固定空速判别式失败。逐路线审计显示，主要推力失败来自 signed BO04 的强负值被直接当成下沉气流，而非正升力信用引起的绕行。BO04 的局地负坡度输出没有在本实验中被验证为真实下沉风；把它强制注入飞行动力学，会把目标问题从“虚假正信用诱导绕行”改成“未经验证的负 BO04 是否可飞”。

决定：保留完整 signed BO04 和 URSA“负值不变”的公式/数组审计；路线规划器只消费 `w_credit=0.875 max(w_signed,0)` 的机会型正升力信用层，非正值表示“不提供额外升力信用”，而不是“预测零真实垂直风”。hard warning、continuous、S-only 和 matched-uniform 均通过同一接口。参考阶段仍在完整 FuXi 风场及 suppress-w 世界中独立重新配平每条冻结几何。数组和文稿必须分别写成 `signed BO04 carrier` 与 `BO04 positive-lift credit planner layer`，不得简称成含义不明的“raw BO04 wind”。该更改发生在 formal freeze 与任何本轮输出参考访问之前，并直接收窄到用户指定的正信用失效机制。

## D010：收益与“可用升力误杀”必须对称报告

用户指出，审稿人必然会问 URSA 的保守性是否把本来有效的升力路线也删掉。决定在任何本轮参考输出访问前，把 population library 预注册为对称 harm audit。reference-valid lift opportunity 要求 raw 所选路线既是三路线 reference oracle（允许冻结 numerical tie），又在同一几何上满足 `J_suppress-w-J_FuXi-w>0`。若 URSA 离开该路线且增加参考能量，定义为 valid-lift abandonment。

每个方法必须并列报告 false-corridor correction 与 valid-lift abandonment，以及有效升力保留率、gross benefit、gross harm、net benefit、benefit/harm ratio、最坏伤害和 exact-DEM group-macro 结果。hard warning 与 continuous attenuation 是两个不同保守程度的预注册 operating points，不得事后只展示较有利者。这样，论文的结论是可审计的风险—机会取舍，而不是单向“删得越多越好”。

## D011：冻结稳定数值环境并排除 CPU 1

系统 Miniconda 环境（NumPy 1.25.2、SciPy 1.11.4）在完整单组工作流中出现两次 NumPy `_multiarray_umath` 原生段错误；切换项目 `.venv`（NumPy 1.26.4、SciPy 1.17.1）后，串行和 2 workers 对照通过，但未限制亲和性的 4 workers 再次段错误。三次内核日志均把故障定位到逻辑 CPU 1。保持同一四 case、同一 `.venv`、同一 4 workers，只把亲和性限制到 CPU 4--23 后通过；随后 12 case/8 workers（33.37 s）和 32 case/16 workers（34.00 s）扩展基准均零 worker error。

决定：formal prepare 使用项目 `.venv`、16 workers、每 worker BLAS/OpenMP 1 线程，并通过 `taskset -c 4-23` 排除 CPU 1。manifest 保存解释器、NumPy/SciPy 版本和 worker 配置；启动命令与亲和性写入运行记录。该问题分类为本地数值栈/CPU 亲和性故障，不是 URSA、BO04 或飞行物理失败。若 formal 仍出现任何 worker error，manifest 保持 failed，不自动删 case 或重跑到成功。

## D012：参考评分首次干跑的 JSON 类型错误

参考清单 V1 冻结后，以首个 exact-DEM group（3 个任务）执行串行干跑。全部参考数值计算在 2.41 s 内完成且无 worker error，但写 score 自哈希前失败：机械能平衡通过标志由 NumPy 比较产生 `numpy.bool_`，标准 JSON 编码器拒绝序列化。失败目录未生成 `reference_score.json`，参考清单 V1 与干跑目录完整保留。

决定：只把该标志显式转换为 Python `bool`，不改变几何、风场、配平、阈值、分类或任何数值公式。由于脚本文件身份已经改变，不复用 V1 冻结清单；重新冻结 V2 清单后再做干跑和正式评分。该问题分类为 agent-created serialization bug，不是物理或数值失败。

## D013：自定义不可达异常破坏并行池，物理状态改为结构化记录

V2 首组串行干跑通过后，15-worker 正式评分在 0.82 s 内以 15 个 `BrokenProcessPool` 结束，零组进入结果；4-worker/4-group 的单因素复现同样整体失败。随后保持相同前 4 组仅把 worker 数改为 1：前三组完成，第 4 组明确抛出 `GroundDirectionUnreachable: cross_track_wind_exceeds_fixed_airspeed`。该异常来自参考场下共同端点的固定 9.7 m/s 空速风三角；自定义异常不能按默认协议在子进程侧序列化/父进程侧重建，故一个物理不可达状态破坏了整个池。

决定：不改变空速、端点或任务，捕获世界绑定阶段的固定空速不可达，将三条路线的 fine/coarse 状态统一写成结构化 reference-unavailable/infeasible 行；因没有机械能平衡数值，该 world 的 numeric gate 为 false，退出性能分母但保留任务和失败原因。脚本改变后不复用 V2 清单，冻结 V3；先以同一前 4 组和 4 workers 做最小判别测试，只有零 worker error 才进入正式评分。该问题中的横风不可达是物理/模型包线限制；并行池崩溃是 agent-created exception transport bug，两者分开记录。

## D014：V1 参考结果否证“hard warning 越强越好”

V3 reference manifest 对应的正式评分完成 15/15 exact-DEM groups、39/39 tasks、零 worker error；FuXi-W 主世界 39 个任务全部通过数值门，最大机械能平衡相对残差为 `1.87e-11`。16 个 reference-false downstream corridors 分布在 8 个地形。

hard warning 修正 6/16（37.5%），但在 21 个 reference-valid raw routes 中产生 10 次实质性放弃，其中 7 次新路线在参考场不可飞；7 个狭义 reference-valid lift opportunities 中放弃 1 个。有限比较的 gross benefit/gross harm 为 85.64/137.30 kJ，net 为 -51.67 kJ，最坏 finite harm 为 41.08 kJ。故 V1 不支持把 hard warning 写成安全或净节能改进。

固定敏感性中，continuous attenuation 修正 5/16，gross benefit/harm 为 58.49/11.46 kJ，但仍有 3 次 hard harm；S-only 修正 4/16，本面板观察到 gross harm=0、hard harm=0；matched-uniform 修正 3/16 且 net=-2.72 kJ。S-only 的结果是预先冻结的敏感性/消融，不得在看到结果后改称原始 primary；它说明空间 sheltering 信息可能有用，也说明 operational exposure hard mask 的保守性代价必须正面处理。

## D015：V1 的 all-routes-feasible 门过严，建立 post-audit V2

V1 held-out 输入账本中，14512 个候选先在几何门退出，另有 5428 个任务因 `raw_world_gate_failed` 退出；路线级失败主要是未选备选路线的 `thrust_limit`。V1 的 `stable_world` 同时要求三条路线全部可飞，即使所选路线 fine/coarse 稳定可飞且某条输家稳定不可飞，也删除整项任务。按真实规划语义，稳定拒绝不可飞备选是正常可行性处理，不应作为任务无效条件。

不访问任何新增参考输出的账本重算表明，改为“所选路线 fine/coarse 稳定可飞 + 每条路线可行性分类稳定”后，可恢复 2041 个 held-out raw-stable candidates；其中 population 候选 391 个，满足 downstream/direct 均可飞及正能量余量、额外距离和时间的 challenge triggers 365 个。

决定：冻结 V2 协议，只修改任务有效性门，不改 BO04、四种方法、飞行器、几何或任何物理/数值阈值。为避免直接复用 V1 已评分地形，V2 从 output-blind `formal_reference_manifest_v3.json` 读取并排除其 15 个 exact-DEM identities，不读取 V1 score；只处理剩余 559 个 confirmation/reserve groups。V2 明确标为 post-audit expanded panel，V1 负结果和门缺陷继续保留，不能以 V2 静默替换。

## D016：V2 对称取舍结果否决 hard warning，并支持连续衰减作为风险受控工作点

V2 在不改地图、方法、飞行器或物理阈值的条件下完成 161/161 个 exact-DEM groups、608/608 个任务的评分，零 worker error；FuXi-W 主世界全部 608 个任务通过数值门，最大机械能平衡相对残差为 `2.49e-10`。expanded panel 中有 52 个 reference-false corridors、406 个 reference-valid raw routes，其中 129 个同时满足狭义 reference-valid lift opportunity 定义。

hard warning 修正 52/52 个 false corridors，却放弃 145/406 个 reference-valid routes（35.7%）和 39/129 个 valid-lift opportunities（30.2%）；有限比较的 gross benefit/gross harm 为 933.18/1270.70 kJ，net 为 -337.52 kJ，另有 43 个“raw reference 可行、方法路线 reference 不可行”的 hard harms，最坏有限伤害为 76.48 kJ。该结果与 V1 方向一致，禁止把 hard warning 描述为“更安全”或“净节能”。

continuous attenuation 修正 29/52（55.8%），放弃 21/406 个 reference-valid routes（5.17%）和 3/129 个 valid-lift opportunities（2.33%）；gross benefit/gross harm 为 617.71/62.91 kJ，net 为 +554.80 kJ，仍有 11 个 hard harms，最坏有限伤害 15.12 kJ。matched-uniform 修正 26/52、放弃 55/406，gross benefit/gross harm 为 565.78/240.72 kJ，net 为 +325.06 kJ，且有 27 个 hard harms。S-only 仅修正 7/52、放弃 6/406，net 为 +48.71 kJ，仍观察到 1 个 hard harm。

决定：论文的主结论改为“二元 hard mask 暴露严重的机会成本；连续 attenuation 在冻结工作点上提供更好的经验性纠错—误杀折中”，而不是宣称任何方法普遍安全或节能。continuous attenuation 仍是预先冻结的 operating point，并非结果后调参，但其 11 个 hard harms 使它只能作为经过审计的候选工作点，不能写成部署保证。正文与补充材料必须并列给出纠错率、有效路线/有效升力放弃率、gross benefit、gross harm、hard harm、最坏有限伤害，并同时展示 confirmation/reserve 和 cluster-bootstrap 不确定性；不得只报 net benefit 或只报 challenge 子集。

## D017：组级绘图分子必须显式受条件分母约束

公开发布前的哈希复核发现，`false_corridor_correction` 行级标志还记录了少量发生在 reference-false 子集之外的 route departures。初版绘图脚本直接按组求和该标志，却以 `reference_false_corridor` 为分母，因而 13 个小组的显示率超过 1；总体正式分析和公开重算器已经使用二者交集，论文表中的 29/52、52/52、26/52 和 7/52 不受影响。有效路线与有效升力放弃标志没有观察到分母外事件，但仍应使用同一条件计数结构。

决定：组级画像的三个分子统一定义为“条件分母标志 AND 对应事件标志”，并加入 `numerator <= denominator` 失败门。重画后三个组级比率均落在 [0,1]，视觉审计无异常；旧图和旧画像不得进入公开包或投稿 PDF。该问题分类为 agent-created visualization aggregation bug，不是实验结果或物理模型变化。

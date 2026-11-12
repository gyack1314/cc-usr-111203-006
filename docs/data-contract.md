# 冻结参数及温度口径

temperature和counterfactual使用摄氏度，center/scale作用于两条序列和固定温度对比；q(l)中的l/L只依lag_max，不随缺失有效日数改变。beta顺序固定为b一次×q常数、b一次×q线性、b二次×q常数、b二次×q线性。V在相同坐标系中，不能只更改温度单位而不相应更改参数。固定温度对比每lag用同一温度与参考值，累计等于对各lag设计差求和。

count为当前日期观测人数，非负整数；每日设计依赖该日和前L日。counterfactual必须覆盖同样lag日，任一缺失该对比不可估。完整输入中最初L天可以作为lag历史而不输出正式负担，它们的count需列为未覆盖，不隐去。参数抽样只给回归参数不确定性，公开z和draws/seed由调用者固定。

domain只数资料和核对形状，不构造张量设计、不算RR或归属数。

## 设计与可估性（climate_data.design）

- 温度先标准化x=(temperature-center)/scale，scale必须为正；暴露基b(x)=(x,x平方)，滞后基q(l)=(1,l/L)，l=0..L。
- 单个lag的设计向量为(x, x·l/L, x平方, x平方·l/L)，日期t的设计向量是各lag张量积之和；坐标索引0..3的含义与beta一一对应，输出coordinate_map供回链。
- 按真实连续公历日轴（datetime.date逐日timedelta）构造，不跨年重置滞后；资料中的日轴缺口由calendar_gap_days列出，缺口日不做最近记录替代。
- 日期t可估当且仅当t-l（l=0..L）全部存在，且这些日子的temperature与counterfactual均非null；任一不满足则该日期不可估，原因按lag与source_day逐项记录（lag_day_absent、missing_temperature、missing_counterfactual），缺测沿之后L个日期的lag窗传播。
- 实际序列对比的d_t为实际设计与完整反事实设计之差；固定温度对比的d_l每lag使用同一对温度。design_trace逐lag保留原始摄氏度、标准化x、两个设计向量与差，独立核对单位缩放与坐标顺序。

## 风险、归属数与不确定性（climate_data.risk, climate_data.matrices）

- 冻结beta与4×4对称半正定V，不重新拟合。logRR=d'beta，方差=d'Vd；RR在log尺度构造区间（±z·SE）后指数化。
- 固定温度对比逐lag给出logRR/RR及区间，累计值由累计设计差直接构造，且各lag logRR之和必须等于累计logRR（输出identity_check）。
- 协方差先校验形状、有限性与对称，再做Jacobi特征分解；负特征值（相对矩阵量级的psd_tolerance之外）直接拒绝，奇异但半正定接受，绝不加对角抖动或掩盖负特征值。
- 可归属数AN_t=n_t·(1-exp(-logRR_t))，允许为负；总量只对可估日期求和，未覆盖人数与天数单列，不改分母。
- 总量方差使用总梯度g=Σ[n_t·exp(-logRR_t)·d_t]计算g'Vg，保留共享beta的跨日期相关；naive_daily_variance_sum仅作对照，不是合法总量方差。
- 固定种子（标准库random.Random(seed)）多元正态抽样beta_draw=beta+Q·diag(√λ)·z；λ=0的方向零方差严格保持，不注入噪声。抽样分位数区间只刻画回归参数不确定性，不含计数波动、拟合外混杂或因果结论。

## 命令

`inspect`只输出原始摘要；`design`输出日轴、可估性与逐lag设计溯源；`analyze`输出固定温度对比、每日与总量负担、解析区间、抽样敏感性、特征值与适用范围声明。所有输出附来源SHA-256，JSON禁止NaN/Infinity。

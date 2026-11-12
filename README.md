# 热暴露分布滞后对比与负担不确定性

这是独立的气候资料Python项目。`data/`保存合成资料，`climate_data.io`负责严格读取、公历日期和来源字节摘要，`climate_data.domain`提供现有描述统计，`climate_data.cli`将摘要、设计与风险分析输出为JSON。所有观测值、站点、人群、情景与金额都是虚构。字段和研究约定见[资料约定](docs/data-contract.md)。

当前Python 3.11及以上可直接使用，没有第三方Python依赖。项目根目录运行：

```sh
python -m unittest discover -s tests -v
python -m compileall -q climate_data tests
python -m climate_data.cli inspect data/normal.json
python -m climate_data.cli design data/normal.json
python -m climate_data.cli analyze data/normal.json
```

模块分工：`climate_data.design`按真实连续日轴构造b(x)=(x,x²)与q(l)=(1,l/L)的张量积设计，处理标准化、缺测沿lag窗传播与逐lag坐标溯源；`climate_data.matrices`用标准库Jacobi分解做对称半正定校验、二次型与固定种子多元正态抽样（奇异V的零方差方向保持零方差）；`climate_data.risk`在不重新拟合的前提下计算逐lag与累计logRR/RR、可归属数、梯度法总量方差g'Vg与抽样敏感性。

关键约定：V非法（非对称或负特征值）直接拒绝，不加对角噪声；各lag贡献之和等于累计logRR；总量只对可估日期求和，最初L天lag历史与缺测污染日的count单列为未覆盖；区间只传播冻结回归系数的参数不确定性，不含死亡计数波动、拟合外混杂或因果证明。`indefinite-covariance.json`在domain形状层合法、在analyze半正定层拒绝；`invalid-`前缀文件在domain层即不合法。

`inspect`显示当前原始摘要，不覆盖资料；`design`与`analyze`同样只读。来源字节SHA-256随输出给出，结果同时标明输入身份、量纲和适用范围。

容器使用固定基础镜像，可运行`docker build -t climate-material .`，随后`docker run --rm climate-material`检查资料。`docker run --rm climate-material python -m unittest discover -s tests -v`执行已有检查。运行不依赖远端账号或实时服务。

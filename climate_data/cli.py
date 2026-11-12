"""资料摘要、滞后设计与冻结参数风险分析的命令行入口。

来源文件保持只读；inspect输出原始摘要，design输出张量积设计与逐lag溯源，
analyze输出logRR/RR、可归属数、跨日期总量不确定性与固定种子抽样敏感性。
所有JSON禁止NaN/Infinity，并附来源字节SHA-256。
"""
import argparse
import json

from .design import build_series_design
from .domain import summarize
from .io import load
from .risk import analyze


def _print(payload):
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(
        description='热暴露分布滞后对比与负担不确定性（合成资料，只读分析）'
    )
    parser.add_argument('command', choices=['inspect', 'design', 'analyze'])
    parser.add_argument('file')
    a = parser.parse_args()
    v, digest = load(a.file)
    if a.command == 'inspect':
        _print({'source_sha256': digest, 'summary': summarize(v)})
    elif a.command == 'design':
        _print({'source_sha256': digest, 'design': build_series_design(v)})
    else:
        _print({'source_sha256': digest, 'analysis': analyze(v)})


if __name__ == '__main__':
    main()

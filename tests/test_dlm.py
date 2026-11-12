"""核对张量积设计、反事实对比、可归属数与不确定性的全部业务约定。

数值用手工推导的小例子独立核对，不依赖第三方库；合成资料位于data/。
"""
import json
import math
import subprocess
import sys
import unittest
from pathlib import Path

from climate_data.design import (
    COORDINATE_TERMS,
    build_constant_contrast,
    build_series_design,
    lag_design,
    standardize,
)
from climate_data.io import load
from climate_data.matrices import (
    mvnorm_sample,
    quadratic,
    symmetric_eigen,
    validate_symmetric_psd,
)
from climate_data.risk import (
    analyze,
    attributable_number,
    sequence_results,
    validate_frozen,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'


def load_name(name):
    return load(DATA / name)[0]


class BasisAndDesignTests(unittest.TestCase):
    def test_standardization_and_scaling(self):
        self.assertEqual(standardize(30, 20, 10), 1.0)
        self.assertEqual(standardize(30, 20, 5), 2.0)
        self.assertEqual(standardize(18, 20, 10), -0.2)
        with self.assertRaises(ValueError):
            standardize(30, 20, 0)

    def test_lag_design_order_and_basis(self):
        # b(x)=(x,x^2)，q(l)=(1,l/L)，顺序xq0,xq1,x2q0,x2q1
        self.assertEqual(lag_design(1.0, 0, 3), [1.0, 0.0, 1.0, 0.0])
        self.assertEqual(lag_design(1.0, 3, 3), [1.0, 1.0, 1.0, 1.0])
        self.assertEqual(lag_design(2.0, 1, 2), [2.0, 1.0, 4.0, 2.0])
        self.assertEqual(len(COORDINATE_TERMS), 4)
        with self.assertRaises(ValueError):
            lag_design(1.0, 4, 3)

    def test_constant_contrast_hand_calculation(self):
        v = load_name('normal.json')
        cc = build_constant_contrast(30, 20, v)
        # x_T=1, x_ref=0；参考侧设计全为0，累计为四个lag向量之和[4,2,4,2]
        self.assertEqual(cc['cumulative_design_diff'], [4.0, 2.0, 4.0, 2.0])
        self.assertEqual(cc['lags'][0]['design_diff'], [1.0, 0.0, 1.0, 0.0])
        self.assertEqual(cc['lags'][3]['design_diff'], [1.0, 1.0, 1.0, 1.0])
        # 零温度差必须给出零设计
        zero = build_constant_contrast(20, 20, v)
        self.assertEqual(zero['cumulative_design_diff'], [0.0] * 4)

    def test_series_hand_calculation(self):
        v = load_name('normal.json')
        design = build_series_design(v)
        # 2021-12-30的lag窗为Dec30(29)/Dec29(23)/Dec28(20)/Dec27(18)，
        # 反事实全为20(x=0)，手工累计差=[1.0,-0.1,0.94,0.07]
        actual = design['design_by_day']['2021-12-30']
        for got, want in zip(actual, [1.0, -0.1, 0.94, 0.07]):
            self.assertAlmostEqual(got, want, places=14)


class CalendarAndMissingnessTests(unittest.TestCase):
    def test_first_l_days_are_history_and_crosses_year(self):
        design = build_series_design(load_name('normal.json'))
        self.assertEqual(design['estimable_days'][0], '2021-12-30')
        # 最初L=3天作为lag历史，不输出正式负担但count列为未覆盖
        self.assertEqual(design['uncovered']['days'], 3)
        self.assertEqual(design['uncovered']['people'], 10 + 11 + 12)
        self.assertEqual(design['covered_people'], 76)
        # 跨年不重置：2022-01-01的lag 1落在2021-12-31
        jan1 = next(p for p in design['provenance'] if p['day'] == '2022-01-01')
        sources = {row['lag']: row['source_day'] for row in jan1['lag_history']}
        self.assertEqual(sources, {
            0: '2022-01-01', 1: '2021-12-31',
            2: '2021-12-30', 3: '2021-12-29',
        })

    def test_missing_temperature_propagates_through_lag_window(self):
        design = build_series_design(load_name('lag-gap.json'))
        by_day = {p['day']: p for p in design['provenance']}
        # 缺温日本身lag0即不可估
        self.assertFalse(by_day['2021-12-31']['estimable'])
        self.assertEqual(by_day['2021-12-31']['reasons'][0]['lag'], 0)
        # 不做最近记录替代：之后L天全部因缺日传播而不可估
        for day in ('2022-01-01', '2022-01-02', '2022-01-03'):
            self.assertFalse(by_day[day]['estimable'], day)
        jan2 = by_day['2022-01-02']
        self.assertIn(2, [r['lag'] for r in jan2['reasons']])
        # L=3，缺温日Dec31污染其自身及之后3天；Dec30与Jan4、Jan5可估
        self.assertEqual(
            design['estimable_days'],
            ['2021-12-30', '2022-01-04', '2022-01-05'],
        )
        # 总人数109；覆盖10+12+10=32，未覆盖77，绝不改分母
        self.assertEqual(design['covered_people'], 32)
        self.assertEqual(design['uncovered']['people'], 77)
        self.assertEqual(design['uncovered']['days'], 7)

    def test_calendar_gap_is_detected_not_filled(self):
        v = load_name('normal.json')
        v['daily'] = [r for r in v['daily'] if r['day'] != '2021-12-30']
        design = build_series_design(v)
        self.assertEqual(design['calendar_gap_days'], ['2021-12-30'])
        # 缺口日及其后L天均不可估
        for day in ('2021-12-31', '2022-01-01', '2022-01-02'):
            p = next(x for x in design['provenance'] if x['day'] == day)
            self.assertFalse(p['estimable'])
            self.assertTrue(
                any(r['reason'] == 'lag_day_absent' for r in p['reasons'])
            )

    def test_missing_counterfactual_makes_contrast_inestimable(self):
        v = load_name('normal.json')
        v['daily'][0]['counterfactual'] = None
        design = build_series_design(v)
        # Dec27缺反事实；Dec30的lag3恰好引用Dec27，故Dec30也不可估
        by_day = {p['day']: p for p in design['provenance']}
        self.assertFalse(by_day['2021-12-30']['estimable'])


class CovarianceValidationTests(unittest.TestCase):
    def test_singular_psd_accepted_and_reconstructed(self):
        v = load_name('normal.json')
        eigval, eigvec = validate_symmetric_psd(v['covariance'], 4, 1e-12)
        self.assertIn(0.0, eigval)  # 秩亏但半正定，接受
        n = 4
        for i in range(n):
            for j in range(n):
                rebuilt = sum(
                    eigvec[i][k] * eigval[k] * eigvec[j][k] for k in range(n)
                )
                self.assertAlmostEqual(rebuilt, v['covariance'][i][j], places=14)

    def test_indefinite_and_asymmetric_rejected(self):
        bad = load_name('indefinite-covariance.json')
        with self.assertRaises(ValueError):
            validate_symmetric_psd(bad['covariance'], 4, 1e-12)
        with self.assertRaises(ValueError):
            validate_frozen(bad)
        asymmetric = [[1.0, 0.5], [0.2, 1.0]]
        with self.assertRaises(ValueError):
            validate_symmetric_psd(asymmetric, 2, 1e-12)

    def test_jacobi_eigenvalues_known_case(self):
        eigval, _ = symmetric_eigen([[2.0, 1.0], [1.0, 2.0]])
        self.assertAlmostEqual(eigval[0], 1.0, places=14)
        self.assertAlmostEqual(eigval[1], 3.0, places=14)


class RiskAndBurdenTests(unittest.TestCase):
    def test_constant_contrast_logrr_identity_and_intervals(self):
        v = load_name('normal.json')
        out = analyze(v)
        hot = next(c for c in out['constant_contrasts']
                   if c['temperature'] == 30.0 and c['reference'] == 20.0)
        # 手工累计logRR=4*.08+2*(-.03)+4*.02+2*.01=0.36
        self.assertAlmostEqual(hot['cumulative']['log_rr'], 0.36, places=14)
        self.assertTrue(hot['identity_check']['matches'])
        self.assertAlmostEqual(
            sum(b['log_rr'] for b in hot['lags']), 0.36, places=14
        )
        # 手工方差=16*.0004+4*.0009+16*.0001=0.0116
        self.assertAlmostEqual(hot['cumulative']['variance'], 0.0116, places=14)
        z = v['confidence_z']
        rr, lo, hi = (
            hot['cumulative']['rr'],
            hot['cumulative']['rr_ci']['low'],
            hot['cumulative']['rr_ci']['high'],
        )
        self.assertAlmostEqual(rr, math.exp(0.36), places=14)
        self.assertAlmostEqual(lo, math.exp(0.36 - z * math.sqrt(0.0116)), places=12)
        self.assertAlmostEqual(hi, math.exp(0.36 + z * math.sqrt(0.0116)), places=12)
        self.assertLess(lo, rr)
        self.assertLess(rr, hi)

    def test_zero_contrast_collapses_to_null(self):
        v = load_name('zero-contrast.json')
        out = analyze(v)
        for c in out['constant_contrasts']:
            if c['temperature'] == c['reference']:
                self.assertEqual(c['cumulative']['log_rr'], 0.0)
                self.assertEqual(c['cumulative']['variance'], 0.0)
                self.assertEqual(c['cumulative']['rr'], 1.0)
        # 实际序列与反事实逐日相同：所有d_t=0，总量及其不确定性全为0
        total = out['sequence']['total']
        self.assertEqual(total['attributable_number'], 0.0)
        self.assertEqual(total['variance'], 0.0)
        self.assertEqual(total['ci']['low'], 0.0)
        self.assertEqual(total['ci']['high'], 0.0)
        mc = out['sensitivity']
        self.assertEqual(mc['total_attributable_number']['ci_low_quantile'], 0.0)
        self.assertEqual(mc['total_attributable_number']['ci_high_quantile'], 0.0)

    def test_attributable_number_allows_negative(self):
        self.assertAlmostEqual(attributable_number(10, 0.0), 0.0)
        self.assertLess(attributable_number(10, -0.5), 0.0)
        self.assertAlmostEqual(
            attributable_number(10, 0.5), 10 * (1 - math.exp(-0.5)), places=14
        )

    def test_gradient_total_variance_keeps_shared_coefficient_correlation(self):
        # 构造含非对角协方差的最小资料：L=1，3个温度日，2个可估目标日
        v = {
            'lag_max': 1, 'center': 0, 'scale': 1,
            'beta': [0.1, 0.2, 0.0, 0.0],
            'covariance': [
                [1.0, 0.3, 0.0, 0.0],
                [0.3, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            'daily': [
                {'day': '2030-01-01', 'temperature': 1, 'counterfactual': 0, 'count': 0},
                {'day': '2030-01-02', 'temperature': 2, 'counterfactual': 0, 'count': 2},
                {'day': '2030-01-03', 'temperature': 3, 'counterfactual': 0, 'count': 3},
            ],
            'constant_contrasts': [],
            'confidence_z': 1.96, 'draws': 200, 'seed': 7, 'psd_tolerance': 1e-12,
        }
        frozen = validate_frozen(v)
        design = build_series_design(v)
        self.assertEqual(
            design['design_by_day']['2030-01-02'], [3.0, 1.0, 5.0, 1.0]
        )
        self.assertEqual(
            design['design_by_day']['2030-01-03'], [5.0, 2.0, 13.0, 4.0]
        )
        seq = sequence_results(v, frozen, design)
        d1, d2 = (
            design['design_by_day']['2030-01-02'],
            design['design_by_day']['2030-01-03'],
        )
        g1 = [2 * math.exp(-0.5) * x for x in d1]
        g2 = [3 * math.exp(-0.9) * x for x in d2]
        g = [a + b for a, b in zip(g1, g2)]
        V = frozen['covariance']
        expected = quadratic(V, g)
        total = seq['total']
        self.assertAlmostEqual(total['variance'], expected, places=14)
        self.assertAlmostEqual(total['gradient'], g, places=14)
        # 逐日方差直接相加漏掉2*g1'Vg2，必须与总量不同
        naive = quadratic(V, g1) + quadratic(V, g2)
        self.assertAlmostEqual(total['naive_daily_variance_sum'], naive, places=14)
        self.assertNotAlmostEqual(total['variance'], naive, places=10)
        self.assertAlmostEqual(
            total['variance'] - naive,
            2 * sum(g1[i] * sum(V[i][j] * g2[j] for j in range(4)) for i in range(4)),
            places=12,
        )
        # 点估计总量
        self.assertAlmostEqual(
            total['attributable_number'],
            2 * (1 - math.exp(-0.5)) + 3 * (1 - math.exp(-0.9)),
            places=14,
        )


class SamplingTests(unittest.TestCase):
    def test_seed_determinism(self):
        eigval = [0.0004, 0.0009, 0.0001, 0.0]
        eigvec = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
        mean = [0.08, -0.03, 0.02, 0.01]
        a = mvnorm_sample(mean, eigval, eigvec, 500, 71)
        b = mvnorm_sample(mean, eigval, eigvec, 500, 71)
        self.assertEqual(a, b)
        c = mvnorm_sample(mean, eigval, eigvec, 500, 72)
        self.assertNotEqual(a, c)
        self.assertEqual(len(a), 500)

    def test_singular_direction_has_zero_sampling_variance(self):
        # 第4个特征值为0且基为标准基：该坐标抽样必须恒等于beta[3]
        eigval = [0.0004, 0.0009, 0.0001, 0.0]
        eigvec = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
        samples = mvnorm_sample([0.08, -0.03, 0.02, 0.01], eigval, eigvec, 400, 71)
        self.assertTrue(all(abs(row[3] - 0.01) == 0.0 for row in samples))
        # 有方差的方向确实在变动
        self.assertTrue(any(row[0] != 0.08 for row in samples))
        # 完整分析链路同样保持
        out = analyze(load_name('normal.json'))
        self.assertTrue(out['sensitivity']['null_variance_directions_preserved'])

    def test_sample_empirical_spread_matches_variance(self):
        eigval = [0.04, 0.0]
        eigvec = [[1.0, 0.0], [0.0, 1.0]]
        samples = mvnorm_sample([0.0, 0.0], eigval, eigvec, 4000, 11)
        vals = [row[0] for row in samples]
        emp_var = sum(x * x for x in vals) / len(vals)
        self.assertAlmostEqual(emp_var, 0.04, delta=0.006)
        self.assertTrue(all(row[1] == 0.0 for row in samples))


class EndToEndTests(unittest.TestCase):
    def test_analyze_all_valid_materials_and_json_safe(self):
        for p in sorted(DATA.glob('*.json')):
            # indefinite-covariance按契约应在协方差校验层拒绝，非形状问题
            if p.name.startswith('invalid-') or p.name == 'indefinite-covariance.json':
                continue
            with self.subTest(file=p.name):
                out = analyze(load(p)[0])
                # 禁止NaN/Infinity进入输出
                json.dumps(out, allow_nan=False)
                self.assertEqual(len(out['coordinate_map']), 4)

    def test_invalid_materials_rejected(self):
        for p in sorted(DATA.glob('invalid-*.json')):
            with self.subTest(file=p.name):
                with self.assertRaises((ValueError, KeyError, TypeError)):
                    analyze(load(p)[0])

    def test_cli_commands_isolated_mode(self):
        # -I忽略PYTHONPATH且脚本直接运行不支持包相对导入；用-c显式注入项目
        # 路径后以runpy模块方式执行，仍保留-I对环境路径与用户site的隔离
        runner = (
            "import runpy, sys; "
            "sys.path.insert(0, %r); "
            "sys.argv = ['climate_data.cli'] + sys.argv[1:]; "
            "runpy.run_module('climate_data.cli', run_name='__main__')"
        ) % str(ROOT)
        for command in ('inspect', 'design', 'analyze'):
            proc = subprocess.run(
                [sys.executable, '-I', '-c', runner, command,
                 str(DATA / 'normal.json')],
                cwd=ROOT, capture_output=True, text=True, check=True,
            )
            payload = json.loads(proc.stdout)
            self.assertEqual(len(payload['source_sha256']), 64)
        # 非法协方差在analyze下必须以非零状态退出
        proc = subprocess.run(
            [sys.executable, '-I', '-c', runner, 'analyze',
             str(DATA / 'indefinite-covariance.json')],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertNotEqual(proc.returncode, 0)


if __name__ == '__main__':
    unittest.main()

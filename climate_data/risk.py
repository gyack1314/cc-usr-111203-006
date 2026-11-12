"""冻结回归系数下的对比风险、可归属数与跨日期不确定性。

所有计算只使用调用方冻结的beta与4乘4协方差V，不重新拟合：
- logRR=d'beta，方差=d'Vd，RR区间在log尺度构造后指数化；
- 可归属数AN_t=n_t*(1-exp(-logRR_t))，允许为负；
- 总量方差用总梯度g=sum[n_t*exp(-logRR_t)*d_t]计算g'Vg，
  共享beta导致的跨日期相关因此被保留，绝不逐日方差直接相加；
- 固定种子多元正态抽样仅刻画回归参数不确定性，奇异V的零方差方向
  由特征分解严格保持。区间不包含死亡计数波动或任何拟合外混杂/因果主张。
"""
import math

from .io import integer, number
from .matrices import (
    dot,
    mvnorm_sample,
    quantile,
    validate_symmetric_psd,
)
from .design import build_constant_contrast, build_series_design

COEF_COUNT = 4
SCOPE_STATEMENT = (
    '区间仅传播冻结回归系数的参数不确定性；不含死亡/人数计数波动，'
    '不代表拟合误差以外的混杂调整，也不构成温度与健康结局的因果证明。'
    '资料全部为合成数值。'
)


def validate_frozen(v):
    """校验beta、V与区间/抽样配置，返回可复用的冻结参数与V的特征分解。"""
    beta = v['beta']
    if not isinstance(beta, list) or len(beta) != COEF_COUNT:
        raise ValueError('beta必须为长度4的向量')
    beta = [number(x) for x in beta]
    tol = number(v.get('psd_tolerance', 1e-12))
    if tol <= 0:
        raise ValueError('psd_tolerance必须为正数')
    eigval, eigvec = validate_symmetric_psd(v['covariance'], COEF_COUNT, tol)
    z = number(v['confidence_z'])
    if z < 0:
        raise ValueError('confidence_z必须非负')
    draws = integer(v['draws'], 1)
    if isinstance(v['seed'], bool) or not isinstance(v['seed'], int):
        raise ValueError('seed必须为整数')
    return {
        'beta': beta,
        'covariance': v['covariance'],
        'eigenvalues': eigval,
        'eigenvectors': eigvec,
        'confidence_z': z,
        'draws': draws,
        'seed': v['seed'],
        'psd_tolerance': tol,
    }


def quadratic_variance(V, d):
    """d'Vd的对称求和形式，并对浮点量级内的微小负值给出显式处理。"""
    n = len(d)
    total = sum(V[i][i] * d[i] * d[i] for i in range(n))
    cross = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            cross += V[i][j] * d[i] * d[j]
    total += 2.0 * cross
    abs_scale = sum(abs(V[i][j] * d[i] * d[j]) for i in range(n) for j in range(n))
    if total < 0.0:
        if total < -1e-12 * (1.0 + abs_scale):
            raise ValueError('二次型出现显著负值，协方差不满足半正定')
        return 0.0
    return total


def log_risk(diff, beta, V):
    """返回(logRR, 方差)；零对比时两者均为0。"""
    return dot(diff, beta), quadratic_variance(V, diff)


def rr_band(logrr, variance, z):
    """log尺度区间后指数化，返回(RR, RR下限, RR上限)。"""
    se = math.sqrt(max(0.0, variance))
    return math.exp(logrr), math.exp(logrr - z * se), math.exp(logrr + z * se)


def _contrast_block(diff, beta, V, z):
    logrr, variance = log_risk(diff, beta, V)
    rr, rr_lo, rr_hi = rr_band(logrr, variance, z)
    return {
        'design_diff': list(diff),
        'log_rr': logrr,
        'variance': variance,
        'se': math.sqrt(variance),
        'rr': rr,
        'rr_ci': {'low': rr_lo, 'high': rr_hi},
    }


def constant_contrast_results(v, frozen):
    """固定温度对比：逐lag风险与累计风险，累计由累计设计差直接构造。"""
    beta, V, z = frozen['beta'], frozen['covariance'], frozen['confidence_z']
    results = []
    for c in v.get('constant_contrasts', []):
        if not isinstance(c, dict):
            raise ValueError('constant_contrasts每项必须为对象')
        cc = build_constant_contrast(c['temperature'], c['reference'], v)
        lag_blocks = []
        lag_log_sum = 0.0
        for row in cc['lags']:
            block = _contrast_block(row['design_diff'], beta, V, z)
            block['lag'] = row['lag']
            block['source'] = {
                'lag': row['lag'],
                'temperature': row['temperature'],
                'reference': row['reference'],
                'x_temperature': row['x_temperature'],
                'x_reference': row['x_reference'],
                'design_temperature': row['design_temperature'],
                'design_reference': row['design_reference'],
                'coordinates': row['coordinates'],
            }
            lag_blocks.append(block)
            lag_log_sum += block['log_rr']
        cumulative = _contrast_block(cc['cumulative_design_diff'], beta, V, z)
        identity_gap = abs(lag_log_sum - cumulative['log_rr'])
        results.append({
            'temperature': cc['temperature'],
            'reference': cc['reference'],
            'x_temperature': cc['x_temperature'],
            'x_reference': cc['x_reference'],
            'lags': lag_blocks,
            'cumulative': cumulative,
            'identity_check': {
                'sum_lag_log_rr': lag_log_sum,
                'cumulative_log_rr': cumulative['log_rr'],
                'abs_gap': identity_gap,
                'matches': identity_gap <= 1e-12 * (1.0 + abs(cumulative['log_rr'])),
            },
        })
    return results


def attributable_number(count, logrr):
    """n*(1-exp(-logRR))，logRR为负时归属数相应为负。"""
    return count * (1.0 - math.exp(-logrr))


def sequence_results(v, frozen, design):
    """每日对比与归属数，以及梯度法总量（含未覆盖人数单列）。"""
    beta, V, z = frozen['beta'], frozen['covariance'], frozen['confidence_z']
    days = []
    total_an = 0.0
    gradient = [0.0] * COEF_COUNT
    naive_variance_sum = 0.0
    for p in design['provenance']:
        if not p['estimable']:
            continue
        diff = p['design_diff']
        block = _contrast_block(diff, beta, V, z)
        count = p['count']
        an = attributable_number(count, block['log_rr'])
        weight = count * math.exp(-block['log_rr'])
        an_gradient = [weight * x for x in diff]
        an_variance = quadratic_variance(V, an_gradient)
        an_se = math.sqrt(an_variance)
        days.append({
            'day': p['day'],
            'count': count,
            'design_diff': list(diff),
            'log_rr': block['log_rr'],
            'variance': block['variance'],
            'se': block['se'],
            'rr': block['rr'],
            'rr_ci': block['rr_ci'],
            'attributable_number': an,
            'an_gradient': an_gradient,
            'an_variance': an_variance,
            'an_ci': {
                'low': an - z * an_se,
                'high': an + z * an_se,
            },
        })
        total_an += an
        for i in range(COEF_COUNT):
            gradient[i] += an_gradient[i]
        naive_variance_sum += an_variance

    total_variance = quadratic_variance(V, gradient)
    total_se = math.sqrt(total_variance)
    nonestimable = [
        {
            'day': p['day'],
            'count': p['count'],
            'reasons': p['reasons'],
        }
        for p in design['provenance'] if not p['estimable']
    ]
    return {
        'days': days,
        'nonestimable_days': nonestimable,
        'total': {
            'attributable_number': total_an,
            'gradient': gradient,
            'variance': total_variance,
            'se': total_se,
            'ci': {
                'low': total_an - z * total_se,
                'high': total_an + z * total_se,
            },
            'covered_people': design['covered_people'],
            'uncovered_people': design['uncovered']['people'],
            'uncovered_days': design['uncovered']['days'],
            'naive_daily_variance_sum': naive_variance_sum,
            'shared_parameter_variance_retained': (
                '总量方差为g转置Vg；naive_daily_variance_sum仅作对照，'
                '不是合法的总量不确定性。'
            ),
        },
    }


def _tail_probability(z):
    """由双侧z反推尾概率：p_low=1-Phi(z)，p_high=Phi(z)。"""
    phi = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return 1.0 - phi, phi


def _sample_summary(samples, p_lo, p_hi):
    return {
        'mean': sum(samples) / len(samples),
        'median': quantile(samples, 0.5),
        'ci_low_quantile': quantile(samples, p_lo),
        'ci_high_quantile': quantile(samples, p_hi),
    }


def _rr_summary(samples, p_lo, p_hi):
    """log尺度分位数汇总后逐键指数化，与解析区间的构造口径一致。"""
    log_summary = _sample_summary(samples, p_lo, p_hi)
    return log_summary, {
        key: math.exp(value) for key, value in log_summary.items()
    }


def sensitivity(v, frozen, design, sequence):
    """固定种子系数抽样敏感性；只汇总分位数，不把抽样区间表述为解析区间。"""
    beta = frozen['beta']
    samples = mvnorm_sample(
        beta,
        frozen['eigenvalues'],
        frozen['eigenvectors'],
        frozen['draws'],
        frozen['seed'],
    )
    p_lo, p_hi = _tail_probability(frozen['confidence_z'])

    total_samples = []
    per_day_samples = {d['day']: [] for d in sequence['days']}
    for b in samples:
        total = 0.0
        for d in sequence['days']:
            logrr = dot(d['design_diff'], b)
            an = attributable_number(d['count'], logrr)
            per_day_samples[d['day']].append(an)
            total += an
        total_samples.append(total)

    contrast_samples = []
    for c in v.get('constant_contrasts', []):
        cc = build_constant_contrast(c['temperature'], c['reference'], v)
        lag_log = {row['lag']: [] for row in cc['lags']}
        cum_log = []
        for b in samples:
            running = 0.0
            for row in cc['lags']:
                value = dot(row['design_diff'], b)
                lag_log[row['lag']].append(value)
                running += value
            cum_log.append(running)
        lag_blocks = []
        for lag in sorted(lag_log):
            log_summary, rr_summary = _rr_summary(lag_log[lag], p_lo, p_hi)
            lag_blocks.append({
                'lag': lag,
                'log_rr': log_summary,
                'rr': rr_summary,
            })
        cum_log_summary, cum_rr_summary = _rr_summary(cum_log, p_lo, p_hi)
        contrast_samples.append({
            'temperature': cc['temperature'],
            'reference': cc['reference'],
            'lags': lag_blocks,
            'cumulative_log_rr': cum_log_summary,
            'cumulative_rr': cum_rr_summary,
        })

    return {
        'seed': frozen['seed'],
        'draws': frozen['draws'],
        'tail_probability_low': p_lo,
        'tail_probability_high': p_hi,
        'null_variance_directions_preserved': True,
        'total_attributable_number': _sample_summary(total_samples, p_lo, p_hi),
        'per_day_attributable_number': {
            day: _sample_summary(vals, p_lo, p_hi)
            for day, vals in per_day_samples.items()
        },
        'constant_contrasts': contrast_samples,
        'scope': SCOPE_STATEMENT,
    }


def analyze(v):
    """端到端分析：冻结参数校验、设计、解析风险区间与抽样敏感性。"""
    frozen = validate_frozen(v)
    design = build_series_design(v)
    constants = constant_contrast_results(v, frozen)
    sequence = sequence_results(v, frozen, design)
    mc = sensitivity(v, frozen, design, sequence)
    return {
        'scope': SCOPE_STATEMENT,
        'frozen': {
            'lag_max': design['lag_max'],
            'center': design['center'],
            'scale': design['scale'],
            'beta': frozen['beta'],
            'covariance': frozen['covariance'],
            'eigenvalues': frozen['eigenvalues'],
            'confidence_z': frozen['confidence_z'],
            'draws': frozen['draws'],
            'seed': frozen['seed'],
        },
        'coordinate_map': design['coordinate_map'],
        'calendar_gap_days': design['calendar_gap_days'],
        'constant_contrasts': constants,
        'sequence': sequence,
        'design_trace': design['provenance'],
        'sensitivity': mc,
    }

"""暴露-滞后张量积设计、真实连续日轴与可估性传播。

口径（与docs/data-contract.md一致）：
- 温度先标准化 x=(T-center)/scale，scale必须为正；暴露基b(x)=(x,x平方)。
- 滞后基q(l)=(1,l/L)，l=0..L，L为正整数，分母固定为lag_max。
- 每个lag的设计坐标顺序为(x*q0, x*q1, x平方*q0, x平方*q1)，与beta顺序一致；
  日期t的设计向量为各lag向量之和。
- 日期t可估当且仅当t-l（l=0..L）全部存在于连续公历日轴，且这些日子的
  实际温度与反事实温度均非缺测；不跨年重置滞后、不用最近记录替代缺日。
"""
import datetime

from .io import date, integer, number, positive

COORDINATE_TERMS = (
    'x*q0',
    'x*(l/L)',
    'x^2*q0',
    'x^2*(l/L)',
)


def standardize(temperature, center, scale):
    """摄氏度到标准化温度，返回x=(temperature-center)/scale。"""
    return (number(temperature) - number(center)) / positive(scale)


def lag_design(x, lag, lag_max):
    """单个lag的四维设计向量(x, x*l/L, x平方, x平方*l/L)。"""
    integer(lag, 0)
    L = integer(lag_max, 1)
    if lag > L:
        raise ValueError('lag不能超过lag_max')
    linear = lag / L
    return [x, x * linear, x * x, x * x * linear]


def add_vec(u, w):
    return [a + b for a, b in zip(u, w)]


def sub_vec(u, w):
    return [a - b for a, b in zip(u, w)]


def zero_vec():
    return [0.0, 0.0, 0.0, 0.0]


def validate_daily(v):
    """校验daily结构并返回按日期排序的记录列表与连续日轴缺口。"""
    if not isinstance(v['daily'], list) or not v['daily']:
        raise ValueError('daily必须为非空列表')
    rows = {}
    for r in v['daily']:
        if not isinstance(r, dict):
            raise ValueError('daily每行必须为对象')
        d = date(r['day'])
        if d in rows:
            raise ValueError('日期重复:%s' % r['day'])
        integer(r['count'], 0)
        parsed = {'day': d, 'count': r['count']}
        for name in ('temperature', 'counterfactual'):
            cell = r[name]
            if cell is not None:
                number(cell)
            parsed[name] = None if cell is None else float(cell)
        rows[d] = parsed
    ordered = [rows[d] for d in sorted(rows)]
    start, end = ordered[0]['day'], ordered[-1]['day']
    gap_days = []
    cursor = start
    while cursor <= end:
        if cursor not in rows:
            gap_days.append(cursor.isoformat())
        cursor += datetime.timedelta(days=1)
    return ordered, gap_days


def _lag_history(target, lag_max, rows):
    """返回target之前l=0..L各lag日的(lag, 记录或None, 缺测类型)。"""
    history = []
    for lag in range(lag_max + 1):
        source = target - datetime.timedelta(days=lag)
        record = rows.get(source)
        problem = None
        if record is None:
            problem = 'lag_day_absent'
        elif record['temperature'] is None:
            problem = 'missing_temperature'
        elif record['counterfactual'] is None:
            problem = 'missing_counterfactual'
        history.append((lag, source, record, problem))
    return history


def build_series_design(v):
    """构造实际序列相对完整反事实序列的每日设计差与全量溯源。

    返回字典含：排序后的日记录、连续日轴缺口、每个日期的可估性与逐lag
    设计溯源、可估日期的设计差d_t、未覆盖人数及其原因。不可估日期不输出
    任何风险量，其count仅计入uncovered，绝不改变分母。
    """
    L = integer(v['lag_max'], 1)
    center = number(v['center'])
    scale = positive(v['scale'])
    ordered, gap_days = validate_daily(v)
    rows = {r['day']: r for r in ordered}

    provenance = []
    design_by_day = {}
    uncovered = {'people': 0, 'days': 0}
    for r in ordered:
        target = r['day']
        history = _lag_history(target, L, rows)
        problems = [
            {'lag': lag, 'source_day': source.isoformat(), 'reason': problem}
            for lag, source, record, problem in history if problem is not None
        ]
        if problems:
            provenance.append({
                'day': target.isoformat(),
                'estimable': False,
                'count': r['count'],
                'reasons': problems,
            })
            uncovered['people'] += r['count']
            uncovered['days'] += 1
            continue

        total_diff = zero_vec()
        lag_rows = []
        for lag, source, record, _ in history:
            x_actual = standardize(record['temperature'], center, scale)
            x_cf = standardize(record['counterfactual'], center, scale)
            design_actual = lag_design(x_actual, lag, L)
            design_cf = lag_design(x_cf, lag, L)
            diff = sub_vec(design_actual, design_cf)
            total_diff = add_vec(total_diff, diff)
            lag_rows.append({
                'lag': lag,
                'source_day': source.isoformat(),
                'temperature': record['temperature'],
                'counterfactual': record['counterfactual'],
                'x_actual': x_actual,
                'x_counterfactual': x_cf,
                'design_actual': design_actual,
                'design_counterfactual': design_cf,
                'design_diff': diff,
                'coordinates': list(COORDINATE_TERMS),
            })
        provenance.append({
            'day': target.isoformat(),
            'estimable': True,
            'count': r['count'],
            'lag_history': lag_rows,
            'design_diff': total_diff,
        })
        design_by_day[target.isoformat()] = total_diff

    estimable_days = [p['day'] for p in provenance if p['estimable']]
    covered_people = sum(
        p['count'] for p in provenance if p['estimable']
    )
    return {
        'lag_max': L,
        'center': center,
        'scale': scale,
        'ordered_days': [r['day'].isoformat() for r in ordered],
        'calendar_gap_days': gap_days,
        'provenance': provenance,
        'design_by_day': design_by_day,
        'estimable_days': estimable_days,
        'covered_people': covered_people,
        'uncovered': uncovered,
        'coordinate_map': [
            {
                'index': i,
                'term': term,
                'exposure_basis': 'x' if i < 2 else 'x_squared',
                'lag_basis': 'q_constant' if i % 2 == 0 else 'q_linear',
            }
            for i, term in enumerate(COORDINATE_TERMS)
        ],
    }


def build_constant_contrast(temperature, reference, v):
    """固定温度相对固定参考温度的逐lag与累计设计差。

    每个lag使用同一对温度，逐lag差向量之和即累计设计差；溯源逐lag列出
    标准化温度与两个设计向量，供独立核对单位缩放与坐标顺序。
    """
    L = integer(v['lag_max'], 1)
    center = number(v['center'])
    scale = positive(v['scale'])
    x_t = standardize(temperature, center, scale)
    x_r = standardize(reference, center, scale)
    cumulative = zero_vec()
    lag_rows = []
    for lag in range(L + 1):
        design_t = lag_design(x_t, lag, L)
        design_r = lag_design(x_r, lag, L)
        diff = sub_vec(design_t, design_r)
        cumulative = add_vec(cumulative, diff)
        lag_rows.append({
            'lag': lag,
            'temperature': float(number(temperature)),
            'reference': float(number(reference)),
            'x_temperature': x_t,
            'x_reference': x_r,
            'design_temperature': design_t,
            'design_reference': design_r,
            'design_diff': diff,
            'coordinates': list(COORDINATE_TERMS),
        })
    return {
        'temperature': float(number(temperature)),
        'reference': float(number(reference)),
        'x_temperature': x_t,
        'x_reference': x_r,
        'lags': lag_rows,
        'cumulative_design_diff': cumulative,
    }

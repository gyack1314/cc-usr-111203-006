"""清点温度、人数及冻结参数的形状，不构造滞后设计。"""
from .io import date,number,positive,integer

def summarize(v):
    integer(v['lag_max'],1);positive(v['scale']);number(v['center'])
    if len(v['beta'])!=4 or len(v['covariance'])!=4 or any(len(r)!=4 for r in v['covariance']):raise ValueError('系数形状不匹配')
    for x in v['beta']:number(x)
    for row in v['covariance']:
        for x in row:number(x)
    seen=set();missing=0;people=0
    for r in v['daily']:
        d=date(r['day'])
        if d in seen:raise ValueError('日期重复')
        seen.add(d);people+=integer(r['count'])
        for name in ('temperature','counterfactual'):
            if r[name] is None:missing+=1
            else:number(r[name])
    return {'rows':len(seen),'missing_temperature_cells':missing,'observed_counts':people,'coefficient_count':len(v['beta'])}

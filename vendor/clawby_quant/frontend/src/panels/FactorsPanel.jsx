import React, { useMemo } from 'react'
import { Card, Col, Row, Statistic, Tooltip } from 'antd'
import { useI18n } from '../i18n'

// Pull a few headline metrics out of the factor snapshot for at-a-glance reading.
export default function FactorsPanel({ data, bare = false }) {
  const { t } = useI18n()
  const cards = useMemo(() => {
    if (!data) return []
    const idx = {}
    for (const f of data.factors) idx[`${f.factor}|${f.symbol}`] = f.value

    const v = (k) => idx[k]
    const btcFunding = v('funding_agg|BTCUSDT')?.latest
    const btcLsr = v('lsr_global|BTCUSDT')?.latest
    const btcTopLsr = v('lsr_top_pos|BTCUSDT')?.latest
    const btcOi = v('oi_binance|BTCUSDT')?.chg_1h_pct
    const btcTaker = v('taker_flow|BTCUSDT')?.latest
    const liq = v('liq_agg|BTCUSDT')
    const cbPrem = v('coinbase_prem|')?.latest
    const fg = v('fear_greed|')?.latest
    const etf = v('etf_flow_btc|')?.last_day_flow_usd
    const basis = (v('mark_all|') || {})['BTCUSDT']?.basis_pct

    const items = [
      { t: t('fp.funding'), val: btcFunding, fmt: (x) => `${(x * 100).toFixed(4)}%`,
        color: btcFunding > 0.0003 ? '#e5484d' : btcFunding < -0.0001 ? '#00b96b' : undefined,
        tip: t('fp.fundingTip') },
      { t: t('fp.lsr'), val: btcLsr, fmt: (x) => x.toFixed(2), tip: t('fp.lsrTip') },
      { t: t('fp.topLsr'), val: btcTopLsr, fmt: (x) => x.toFixed(2), tip: t('fp.topLsrTip') },
      { t: t('fp.oi'), val: btcOi, fmt: (x) => `${x >= 0 ? '+' : ''}${x.toFixed(2)}%`,
        color: btcOi > 0 ? '#00b96b' : '#e5484d', tip: t('fp.oiTip') },
      { t: t('fp.taker'), val: btcTaker, fmt: (x) => x.toFixed(2),
        color: btcTaker > 1.1 ? '#00b96b' : btcTaker < 0.9 ? '#e5484d' : undefined,
        tip: t('fp.takerTip') },
      { t: t('fp.basis'), val: basis, fmt: (x) => `${x.toFixed(3)}%`, tip: t('fp.basisTip') },
      { t: t('fp.liq'), val: liq,
        fmt: (x) => `${(x.long_1h / 1e6).toFixed(1)}M / ${(x.short_1h / 1e6).toFixed(1)}M`,
        tip: t('fp.liqTip') },
      { t: t('fp.cbprem'), val: cbPrem, fmt: (x) => `${Number(x).toFixed(4)}`,
        color: cbPrem > 0 ? '#00b96b' : '#e5484d', tip: t('fp.cbpremTip') },
      { t: t('fp.fg'), val: fg, fmt: (x) => x.toFixed(0),
        color: fg >= 85 || fg <= 15 ? '#e5484d' : undefined, tip: t('fp.fgTip') },
      { t: t('fp.etf'), val: etf, fmt: (x) => `${x >= 0 ? '+' : ''}${(x / 1e6).toFixed(0)}M`,
        color: etf >= 0 ? '#00b96b' : '#e5484d', tip: t('fp.etfTip') },
    ]
    return items.filter((i) => i.val !== undefined && i.val !== null)
  }, [data, t])

  const grid = (
    <Row gutter={[10, 10]}>
      {cards.map((c) => (
        <Col xs={12} md={8} lg={bare ? 4 : 12} key={c.t}>
          <Tooltip title={c.tip}>
            <Card size="small" style={{ background: '#10151d' }}>
              <Statistic title={<span style={{ fontSize: 12 }}>{c.t}</span>}
                         value={c.fmt(c.val)}
                         valueStyle={{ fontSize: 16, color: c.color }} />
            </Card>
          </Tooltip>
        </Col>
      ))}
    </Row>
  )
  if (bare) return grid
  return <Card size="small" title={t('fp.title')}>{grid}</Card>
}

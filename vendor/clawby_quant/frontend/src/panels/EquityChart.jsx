import React, { useEffect, useMemo, useState } from 'react'
import { Card, Segmented } from 'antd'
import Chart from '../Chart'
import { apiPath } from '../api'
import { useI18n } from '../i18n'

// Paper and live equity are different universes ($10k virtual vs real wallet):
// the curve is stored per mode and NEVER mixed; default view = active mode.
export default function EquityChart({ activeMode }) {
  const { t, lang } = useI18n()
  const [view, setView] = useState(null)          // null until activeMode known
  const [series, setSeries] = useState([])
  const mode = view || activeMode || 'paper'

  useEffect(() => {
    let alive = true
    const load = () => fetch(apiPath(`/api/equity?mode=${mode}`)).then((r) => r.json())
      .then((d) => alive && setSeries(d.series || [])).catch(() => {})
    load()
    const timer = setInterval(load, 30000)
    return () => { alive = false; clearInterval(timer) }
  }, [mode])

  const option = useMemo(() => {
    const ts = series.map((r) => new Date(r.ts * 1000).toLocaleString(
      lang === 'en' ? 'en-US' : 'zh-CN',
      { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' }))
    const eq = series.map((r) => Number(r.equity.toFixed(2)))
    return {
      grid: { left: 60, right: 16, top: 24, bottom: 24 },
      tooltip: { trigger: 'axis' },
      xAxis: { type: 'category', data: ts, axisLine: { lineStyle: { color: '#3a4553' } } },
      yAxis: { type: 'value', scale: true, splitLine: { lineStyle: { color: '#1d2633' } } },
      series: [{
        name: t('dash.equityName'), type: 'line', data: eq, showSymbol: false, smooth: true,
        lineStyle: { color: '#00b96b', width: 2 },
        areaStyle: { color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
          colorStops: [{ offset: 0, color: 'rgba(0,185,107,0.25)' },
                       { offset: 1, color: 'rgba(0,185,107,0)' }] } },
      }],
    }
  }, [series, lang, t])
  return (
    <Card size="small" title={t('dash.equityCurve')}
          extra={(
            <Segmented size="small" value={mode} onChange={setView}
                       options={[{ label: t('hdr.paper'), value: 'paper' },
                                 { label: t('hdr.live'), value: 'live' }]} />
          )}>
      <Chart option={option} height={280} />
    </Card>
  )
}

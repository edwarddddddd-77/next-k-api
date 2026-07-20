import React, { useEffect, useMemo, useState } from 'react'
import { Card, Select, Space } from 'antd'
import Chart from '../Chart'
import { api } from '../api'
import { useI18n } from '../i18n'

const INTERVALS = ['5m', '15m', '1h', '4h']

export default function PriceChart({ universe }) {
  const { t, lang } = useI18n()
  const [symbol, setSymbol] = useState('BTCUSDT')
  const [interval, setItv] = useState('15m')
  const [data, setData] = useState(null)

  useEffect(() => {
    let alive = true
    const load = () => api.klines(symbol, interval).then((d) => alive && setData(d)).catch(() => {})
    load()
    const timer = setInterval(load, 60000)
    return () => { alive = false; clearInterval(timer) }
  }, [symbol, interval])

  const option = useMemo(() => {
    if (!data) return null
    const ks = data.klines || []
    const cat = ks.map((k) => new Date(k.ts * 1000).toLocaleString(
      lang === 'en' ? 'en-US' : 'zh-CN',
      { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' }))
    const ohlc = ks.map((k) => [k.open, k.close, k.low, k.high])
    const vols = ks.map((k) => k.volume)
    const minTs = ks.length ? ks[0].ts : 0
    const marks = (data.marks || []).filter((m) => m.ts >= minTs).map((m) => {
      const idx = ks.findIndex((k, i) => k.ts <= m.ts && (i === ks.length - 1 || ks[i + 1].ts > m.ts))
      const isOpen = m.kind === 'open'
      const isBuy = m.side === 'long' || m.side === 'buy'
      return {
        coord: [idx >= 0 ? idx : cat.length - 1, m.price],
        value: isOpen ? t('dash.openMark') : t('dash.closeMark'),
        itemStyle: { color: isBuy ? '#00b96b' : '#e5484d' },
        symbol: isOpen ? 'triangle' : 'circle',
        symbolRotate: isOpen && !isBuy ? 180 : 0,
        symbolSize: 11,
      }
    })
    return {
      grid: [{ left: 70, right: 16, top: 24, height: '62%' },
             { left: 70, right: 16, top: '78%', height: '14%' }],
      tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
      xAxis: [{ type: 'category', data: cat, gridIndex: 0 },
              { type: 'category', data: cat, gridIndex: 1, axisLabel: { show: false } }],
      yAxis: [{ scale: true, gridIndex: 0, splitLine: { lineStyle: { color: '#1d2633' } } },
              { gridIndex: 1, axisLabel: { show: false }, splitLine: { show: false } }],
      series: [
        { name: symbol, type: 'candlestick', data: ohlc, xAxisIndex: 0, yAxisIndex: 0,
          itemStyle: { color: '#00b96b', color0: '#e5484d',
                       borderColor: '#00b96b', borderColor0: '#e5484d' },
          markPoint: { data: marks, label: { fontSize: 9, color: '#fff' } } },
        { name: t('dash.volume'), type: 'bar', data: vols, xAxisIndex: 1, yAxisIndex: 1,
          itemStyle: { color: '#3a4553' } },
      ],
      dataZoom: [{ type: 'inside', xAxisIndex: [0, 1], start: 55, end: 100 }],
    }
  }, [data, symbol, lang, t])

  return (
    <Card size="small" title={t('dash.priceChart')} extra={
      <Space>
        <Select size="small" value={symbol} onChange={setSymbol} style={{ width: 110 }}
                options={universe.map((s) => ({ value: s, label: s }))} />
        <Select size="small" value={interval} onChange={setItv} style={{ width: 70 }}
                options={INTERVALS.map((i) => ({ value: i, label: i }))} />
      </Space>
    }>
      {option && <Chart option={option} height={280} />}
    </Card>
  )
}

import React from 'react'
import {
  Card, Table, Select, Button, Space, Tag, Typography, Descriptions, message,
} from 'antd'
import { DownloadOutlined, ReloadOutlined } from '@ant-design/icons'
import { api, apiPath } from '../api'
import { useI18n } from '../i18n'

const { Text } = Typography

const fmtPct = (v) => (v == null ? '-' : `${v >= 0 ? '+' : ''}${Number(v).toFixed(2)}%`)

function FactorSnap({ title, snap }) {
  if (!snap || !Object.keys(snap).length) return null
  return (
    <Descriptions size="small" column={4} title={title}
                  labelStyle={{ fontSize: 11 }} contentStyle={{ fontSize: 11 }}
                  style={{ marginBottom: 8 }}>
      {Object.entries(snap).map(([k, v]) => (
        <Descriptions.Item key={k} label={k}>
          {typeof v === 'number' ? Number(v).toPrecision(6) : String(v ?? '-')}
        </Descriptions.Item>
      ))}
    </Descriptions>
  )
}

export default function JournalPage() {
  const { t } = useI18n()
  const [rows, setRows] = React.useState([])
  const [days, setDays] = React.useState([])
  const [loading, setLoading] = React.useState(false)
  const [flt, setFlt] = React.useState({ strategy: '', symbol: '', day: '', mode: '' })

  const load = React.useCallback(async () => {
    setLoading(true)
    try {
      const q = { limit: 500 }
      if (flt.strategy) q.strategy = flt.strategy
      if (flt.symbol) q.symbol = flt.symbol
      if (flt.day) q.day = flt.day
      if (flt.mode) q.mode = flt.mode
      const d = await api.journal(q)
      setRows((d.rows || []).map((r, i) => ({ ...r, _key: i })))
      setDays(d.days || [])
    } catch (e) { message.error(`${t('common.loadFailed')}: ${e.message}`) }
    setLoading(false)
  }, [flt, t])

  React.useEffect(() => { load() }, [load])

  const strategies = Array.from(new Set(rows.map((r) => r.strategy))).sort()
  const symbols = Array.from(new Set(rows.map((r) => r.symbol))).sort()

  const exportUrl = () => {
    const q = new URLSearchParams()
    if (flt.strategy) q.set('strategy', flt.strategy)
    if (flt.symbol) q.set('symbol', flt.symbol)
    if (flt.day) q.set('day', flt.day)
    if (flt.mode) q.set('mode', flt.mode)
    return apiPath(`/api/journal/export?${q}`)
  }

  const columns = [
    { title: t('journal.col.closed'), dataIndex: 'closed_at', width: 148,
      render: (v) => <span style={{ fontSize: 12 }}>{(v || '').replace('T', ' ').replace('Z', '')}</span> },
    { title: t('journal.col.strategy'), dataIndex: 'strategy', width: 128,
      render: (v) => <Tag style={{ fontSize: 11 }}>{v}</Tag> },
    { title: t('journal.col.symbol'), dataIndex: 'symbol', width: 84,
      render: (v) => (v || '').replace('USDT', '') },
    { title: t('journal.col.side'), dataIndex: 'side', width: 56,
      render: (v) => (
        <Tag color={v === 'long' ? 'green' : 'red'} style={{ fontSize: 11 }}>
          {v === 'long' ? t('journal.long') : t('journal.short')}
        </Tag>
      ) },
    { title: t('journal.col.entry'), dataIndex: 'entry_price', width: 96, align: 'right',
      render: (v) => <span style={{ fontSize: 12 }}>{Number(v).toPrecision(6)}</span> },
    { title: t('journal.col.exit'), dataIndex: 'close_price', width: 96, align: 'right',
      render: (v) => <span style={{ fontSize: 12 }}>{Number(v).toPrecision(6)}</span> },
    { title: t('journal.col.pnl'), dataIndex: 'pnl', width: 88, align: 'right',
      sorter: (a, b) => (a.pnl || 0) - (b.pnl || 0),
      render: (v) => <b style={{ color: v >= 0 ? '#00b96b' : '#e5484d', fontSize: 12 }}>{v >= 0 ? '+' : ''}{Number(v).toFixed(2)}U</b> },
    { title: t('journal.col.pnlPct'), dataIndex: 'pnl_pct', width: 76, align: 'right',
      render: (v) => <span style={{ color: v >= 0 ? '#00b96b' : '#e5484d', fontSize: 12 }}>{fmtPct(v)}</span> },
    { title: t('journal.col.hold'), dataIndex: 'hold_minutes', width: 72, align: 'right',
      render: (v) => <span style={{ fontSize: 12 }}>{v >= 60 ? `${(v / 60).toFixed(1)}h` : `${Math.round(v)}m`}</span> },
    { title: 'MFE/MAE', key: 'mfemae', width: 108, align: 'right',
      render: (_, r) => <span style={{ fontSize: 11 }}><span style={{ color: '#00b96b' }}>{fmtPct(r.mfe_pct)}</span> / <span style={{ color: '#e5484d' }}>{fmtPct(r.mae_pct)}</span></span> },
    { title: t('journal.col.reason'), dataIndex: 'close_reason', width: 128,
      render: (v) => <span style={{ fontSize: 12 }}>{v}</span> },
    { title: t('journal.col.mode'), dataIndex: 'mode', width: 64,
      render: (v) => <Tag color={v === 'live' ? 'red' : 'blue'} style={{ fontSize: 11 }}>{v}</Tag> },
  ]

  const expandedRowRender = (r) => (
    <div style={{ padding: '4px 8px' }}>
      <Text type="secondary" style={{ fontSize: 12 }}>{t('journal.signalReason')}: {r.signal_reason || '-'}</Text>
      <div style={{ marginTop: 8 }}>
        <FactorSnap title={t('journal.entrySnap')} snap={r.entry_factors} />
        <FactorSnap title={t('journal.exitSnap')} snap={r.exit_factors} />
      </div>
    </div>
  )

  const pnlSum = rows.reduce((s, r) => s + (r.pnl || 0), 0)
  const wins = rows.filter((r) => (r.pnl || 0) > 0).length

  return (
    <Card size="small"
          title={(
            <Space size="middle">
              {t('journal.title')}
              {rows.length > 0 && (
                <Text type="secondary" style={{ fontSize: 12, fontWeight: 400 }}>
                  {rows.length}{t('journal.stats.trades')} · {t('journal.stats.wr')} {((wins / rows.length) * 100).toFixed(0)}% · {t('journal.stats.total')}
                  <span style={{ color: pnlSum >= 0 ? '#00b96b' : '#e5484d' }}>
                    {' '}{pnlSum >= 0 ? '+' : ''}{pnlSum.toFixed(2)}U
                  </span>
                </Text>
              )}
            </Space>
          )}
          extra={(
            <Space>
              <Select size="small" allowClear placeholder={t('journal.filter.strategy')} style={{ width: 160 }}
                      value={flt.strategy || undefined}
                      onChange={(v) => setFlt({ ...flt, strategy: v || '' })}
                      options={strategies.map((s) => ({ value: s, label: s }))} />
              <Select size="small" allowClear placeholder={t('journal.filter.symbol')} style={{ width: 110 }} showSearch
                      value={flt.symbol || undefined}
                      onChange={(v) => setFlt({ ...flt, symbol: v || '' })}
                      options={symbols.map((s) => ({ value: s, label: s.replace('USDT', '') }))} />
              <Select size="small" allowClear placeholder={t('journal.filter.day')} style={{ width: 130 }}
                      value={flt.day || undefined}
                      onChange={(v) => setFlt({ ...flt, day: v || '' })}
                      options={days.map((d) => ({ value: d, label: d }))} />
              <Select size="small" allowClear placeholder={t('journal.filter.mode')} style={{ width: 100 }}
                      value={flt.mode || undefined}
                      onChange={(v) => setFlt({ ...flt, mode: v || '' })}
                      options={[{ value: 'paper', label: t('hdr.paper') },
                                { value: 'live', label: t('hdr.live') }]} />
              <Button size="small" icon={<ReloadOutlined />} onClick={load} loading={loading} />
              <Button size="small" type="primary" icon={<DownloadOutlined />}
                      onClick={() => window.open(exportUrl(), '_blank')}>{t('journal.export')}</Button>
            </Space>
          )}>
      <Table rowKey="_key" dataSource={rows} columns={columns} size="small" loading={loading}
             pagination={{ pageSize: 50, showSizeChanger: false }}
             expandable={{ expandedRowRender }} scroll={{ x: 1150 }} />
    </Card>
  )
}

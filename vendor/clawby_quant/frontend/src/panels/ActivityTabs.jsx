import React from 'react'
import { Card, Table, Tabs, Tag, Typography } from 'antd'
import { fmtPnl } from '../api'
import { useI18n } from '../i18n'

export default function ActivityTabs({ signals, trades, logs }) {
  const { t, lang } = useI18n()
  const ts = (v) => new Date(v * 1000).toLocaleString(lang === 'en' ? 'en-US' : 'zh-CN', {
    month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' })

  const signalCols = [
    { title: t('act.time'), dataIndex: 'ts', width: 96, render: ts },
    { title: t('dash.col.strategy'), dataIndex: 'strategy', width: 60, render: (v) => v.split('_')[0] },
    { title: t('dash.col.symbol'), dataIndex: 'symbol', width: 80 },
    { title: t('dash.col.side'), dataIndex: 'side', width: 50,
      render: (v) => <Tag color={v === 'long' ? 'green' : 'red'}>{v === 'long' ? t('dash.long') : t('dash.short')}</Tag> },
    { title: t('act.result'), key: 'acted', width: 100,
      render: (_, r) => r.acted
        ? <Tag color="green">{t('act.executed')}</Tag>
        : <Tag color="default">{r.skip_reason || t('act.skipped')}</Tag> },
    { title: t('act.reason'), dataIndex: 'reason', ellipsis: true },
  ]
  const tradeCols = [
    { title: t('act.time'), dataIndex: 'ts', width: 96, render: ts },
    { title: t('dash.col.strategy'), dataIndex: 'strategy', width: 60, render: (v) => v.split('_')[0] },
    { title: t('dash.col.symbol'), dataIndex: 'symbol', width: 80 },
    { title: t('act.kind'), dataIndex: 'kind', width: 55,
      render: (v) => <Tag color={v === 'open' ? 'blue' : 'purple'}>{v === 'open' ? t('dash.openMark') : t('dash.closeMark')}</Tag> },
    { title: t('act.price'), dataIndex: 'price', width: 90, render: (v) => v?.toPrecision(6) },
    { title: t('dash.col.qty'), dataIndex: 'qty', width: 80, render: (v) => v?.toPrecision(4) },
    { title: t('journal.col.pnl'), dataIndex: 'pnl', width: 80,
      render: (v) => v == null ? '-' :
        <span style={{ color: v >= 0 ? '#00b96b' : '#e5484d' }}>{fmtPnl(v)}</span> },
    { title: t('journal.col.mode'), dataIndex: 'mode', width: 60,
      render: (v) => <Tag color={v === 'live' ? 'red' : 'blue'} style={{ fontSize: 10 }}>{v}</Tag> },
  ]
  const logCols = [
    { title: t('act.time'), dataIndex: 'ts', width: 96, render: ts },
    { title: t('act.level'), dataIndex: 'level', width: 55,
      render: (v) => <Tag color={{ error: 'red', warn: 'orange' }[v] || 'default'}>{v}</Tag> },
    { title: t('act.msg'), dataIndex: 'msg',
      render: (v) => <Typography.Text style={{ fontSize: 12 }}>{v}</Typography.Text> },
  ]
  const items = [
    { key: 'signals', label: `${t('act.signals')} (${signals.length})`,
      children: <Table rowKey="id" dataSource={signals} columns={signalCols} size="small"
                       pagination={{ pageSize: 8 }} /> },
    { key: 'trades', label: `${t('act.trades')} (${trades.length})`,
      children: <Table rowKey="id" dataSource={trades} columns={tradeCols} size="small"
                       pagination={{ pageSize: 8 }} /> },
    { key: 'logs', label: t('act.logs'),
      children: <Table rowKey="id" dataSource={logs} columns={logCols} size="small"
                       pagination={{ pageSize: 8 }} /> },
  ]
  return (
    <Card size="small">
      <Tabs items={items} size="small" />
    </Card>
  )
}

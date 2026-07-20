import React from 'react'
import { Card, Table, Tag, Progress, Typography, Button, Popconfirm, message } from 'antd'
import { apiPath, fmtDur, fmtPnl } from '../api'
import { useI18n } from '../i18n'

export default function PositionsTable({ positions, onChanged }) {
  const { t } = useI18n()
  const closeOne = async (pid) => {
    try {
      const r = await fetch(apiPath(`/api/positions/${pid}/close`), { method: 'POST' })
      const d = await r.json()
      if (!r.ok || !d.ok) throw new Error(d.detail || 'failed')
      message.success(t('dash.closed'))
      onChanged?.()
    } catch (e) {
      message.error(`${t('dash.closeFailed')}: ${e.message}`)
    }
  }

  const closeAll = async () => {
    try {
      const r = await fetch(apiPath('/api/positions/close-all'), { method: 'POST' })
      const d = await r.json()
      message[d.ok ? 'success' : 'warning'](
        `${t('dash.closedN')} ${d.closed}${t('dash.n')}${d.failed ? ` · ${t('dash.failedN')} ${d.failed}` : ''}`)
      onChanged?.()
    } catch (e) {
      message.error(`${t('common.opFailed')}: ${e.message}`)
    }
  }

  const columns = [
    { title: t('dash.col.symbol'), dataIndex: 'symbol', render: (v) => <b>{v.replace('USDT', '')}</b> },
    { title: t('dash.col.side'), dataIndex: 'side',
      render: (v) => <Tag color={v === 'long' ? 'green' : 'red'}>{v === 'long' ? t('dash.long') : t('dash.short')}</Tag> },
    { title: t('dash.col.strategy'), dataIndex: 'strategy', render: (v) => v.split('_')[0] },
    { title: t('dash.col.entry'), dataIndex: 'entry_price', render: (v) => v?.toPrecision(6) },
    { title: t('dash.col.mark'), dataIndex: 'mark_price', render: (v) => v?.toPrecision(6) },
    { title: t('dash.col.qty'), dataIndex: 'qty', render: (v) => v?.toPrecision(4) },
    { title: t('dash.col.upnl'), dataIndex: 'unrealized_pnl',
      render: (v) => <span style={{ color: v >= 0 ? '#00b96b' : '#e5484d' }}>{fmtPnl(v)}</span> },
    { title: t('dash.col.excursion'), key: 'excursion', width: 120,
      render: (_, r) => (
        <span style={{ fontSize: 11.5 }}>
          <span style={{ color: '#00b96b' }}>+{(r.mfe_pct || 0).toFixed(2)}%</span>
          {' / '}
          <span style={{ color: '#e5484d' }}>{(r.mae_pct || 0).toFixed(2)}%</span>
        </span>
      ) },
    { title: t('dash.col.age'), key: 'age', width: 160,
      render: (_, r) => {
        const pct = Math.min(100, (r.age_sec / r.max_hold_sec) * 100)
        return (
          <div>
            <Progress percent={pct} size="small" showInfo={false}
                      strokeColor={pct > 80 ? '#e5484d' : '#00b96b'} />
            <Typography.Text type="secondary" style={{ fontSize: 11 }}>
              {fmtDur(r.age_sec)} / {fmtDur(r.max_hold_sec)}
            </Typography.Text>
          </div>
        )
      } },
    { title: t('dash.col.action'), key: 'action', width: 110, align: 'right',
      render: (_, r) => (
        <Popconfirm title={`${t('dash.closeOneConfirm')} ${r.symbol} ${r.side === 'long' ? t('dash.long') : t('dash.short')}?`}
                    okText={t('dash.confirmClose')} cancelText={t('common.cancel')}
                    onConfirm={() => closeOne(r.id)}>
          <Button size="small" danger>{t('dash.closeOne')}</Button>
        </Popconfirm>
      ) },
  ]

  return (
    <Card size="small" title={`${t('dash.positions')} (${positions.length})`}
          extra={positions.length > 0 && (
            <Popconfirm title={`${t('dash.closeAllConfirm')} ${positions.length} ${t('dash.posUnit')}`}
                        okText={t('dash.confirmCloseAll')} cancelText={t('common.cancel')}
                        onConfirm={closeAll}>
              <Button size="small" danger type="primary">{t('dash.closeAll')}</Button>
            </Popconfirm>
          )}>
      <Table rowKey="id" dataSource={positions} columns={columns} size="small"
             pagination={false} locale={{ emptyText: t('dash.noPositions') }} />
    </Card>
  )
}

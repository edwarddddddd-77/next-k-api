import React, { useState } from 'react'
import {
  Card, Row, Col, Table, InputNumber, Button, message, Statistic,
  Tag, Alert,
} from 'antd'
import { api, fmtUsd } from '../api'
import { usePoll } from '../hooks'
import { useI18n } from '../i18n'

function GlobalRisk({ data, onSaved }) {
  const { t } = useI18n()
  const g = data?.global || {}
  const [v, setV] = useState({})
  React.useEffect(() => { setV(g) }, [data]) // eslint-disable-line
  const fields = [
    ['risk_per_trade_pct', t('risk.f.rpt'), t('risk.f.rptTip')],
    ['max_gross_leverage', t('risk.f.lev'), t('risk.f.levTip')],
    ['max_position_pct_per_coin', t('risk.f.coinCap'), t('risk.f.coinCapTip')],
    ['max_concurrent_coins', t('risk.f.conc'), t('risk.f.concTip')],
    ['daily_loss_halt_pct', t('risk.f.halt'), t('risk.f.haltTip')],
    ['event_quiet_minutes', t('risk.f.quiet'), t('risk.f.quietTip')],
  ]
  const save = async () => {
    try { await api.setGlobalRisk(v); message.success(t('risk.globalSaved')); onSaved?.() }
    catch (e) { message.error(`${t('config.saveFailed')}: ${e.message}`) }
  }
  return (
    <Card size="small" title={t('risk.global')}>
      <Row gutter={[16, 12]}>
        {fields.map(([k, label, tip]) => (
          <Col xs={12} md={8} key={k}>
            <div style={{ fontSize: 12, color: '#7d8590', marginBottom: 4 }}>{label}</div>
            <InputNumber size="small" style={{ width: '100%' }} value={v[k]}
                         onChange={(x) => setV((s) => ({ ...s, [k]: x }))} />
            <div style={{ fontSize: 11, color: '#4d5661', marginTop: 2 }}>{tip}</div>
          </Col>
        ))}
      </Row>
      <Button type="primary" size="small" style={{ marginTop: 14 }} onClick={save}>
        {t('risk.saveGlobal')}
      </Button>
    </Card>
  )
}

function StrategyRiskTable({ data, onSaved }) {
  const { t } = useI18n()
  const [edits, setEdits] = useState({})
  const rows = data?.strategies || []
  const setField = (id, key, val) =>
    setEdits((s) => ({ ...s, [id]: { ...(s[id] || {}), [key]: val } }))
  const rowRisk = (r) => ({ ...(r.risk || {}), ...(edits[r.id] || {}) })

  const save = async (r) => {
    try {
      await api.strategyConfig(r.id, { risk: rowRisk(r) })
      message.success(`${r.id} ${t('risk.riskSaved')}`)
      setEdits((s) => { const n = { ...s }; delete n[r.id]; return n })
      onSaved?.()
    } catch (e) { message.error(`${t('config.saveFailed')}: ${e.message}`) }
  }

  const num = (r, key, props = {}) => (
    <InputNumber size="small" style={{ width: 96 }} value={rowRisk(r)[key]}
                 onChange={(x) => setField(r.id, key, x)} {...props} />
  )

  const columns = [
    { title: t('dash.col.strategy'), key: 'name', width: 190, fixed: 'left',
      render: (_, r) => (
        <span><b>{r.id.split('_')[0]}{r.id.includes('__') ? `#${r.id.split('__')[1]}` : ''}</b>{' '}
          {r.enabled ? <Tag color="green" style={{ fontSize: 10 }}>{t('common.enabled')}</Tag>
                     : <Tag style={{ fontSize: 10 }}>{t('common.disabled')}</Tag>}</span>
      ) },
    { title: t('risk.col.capital'), key: 'capital', width: 120,
      render: (_, r) => num(r, 'capital_usd', { min: 0, step: 100, placeholder: t('risk.ph.allAccount') }) },
    { title: t('risk.col.lev'), key: 'lev', width: 90,
      render: (_, r) => num(r, 'leverage', { min: 1, max: 125 }) },
    { title: t('risk.col.maxpos'), key: 'maxpos', width: 130,
      render: (_, r) => num(r, 'max_position_usd', { min: 0, step: 100, placeholder: t('risk.ph.capLev') }) },
    { title: t('risk.col.rpt'), key: 'rpt', width: 110,
      render: (_, r) => num(r, 'risk_per_trade_pct', { min: 0, step: 0.1, placeholder: t('risk.ph.global') }) },
    { title: t('risk.col.used'), dataIndex: 'used_notional', width: 100,
      render: (v) => fmtUsd(v, 0) },
    { title: '', key: 'save', width: 70, fixed: 'right',
      render: (_, r) => (
        <Button size="small" type={edits[r.id] ? 'primary' : 'default'}
                disabled={!edits[r.id]} onClick={() => save(r)}>{t('common.save')}</Button>
      ) },
  ]

  return (
    <Card size="small" title={t('risk.perStrategy')} style={{ marginTop: 12 }}>
      <Alert type="info" showIcon style={{ marginBottom: 12, fontSize: 12 }}
             message={t('risk.hint')} />
      <Table rowKey="id" dataSource={rows} columns={columns} size="small"
             pagination={false} scroll={{ x: 900 }} />
    </Card>
  )
}

export default function RiskPage() {
  const { t } = useI18n()
  const [risk, refresh] = usePoll(api.risk, 10000)
  return (
    <>
      <Row gutter={[12, 12]}>
        <Col xs={24} md={8}>
          <Card size="small">
            <Statistic title={t('dash.equity')} value={risk?.equity ?? '-'}
                       formatter={(v) => (v === '-' ? '-' : fmtUsd(Number(v)))}
                       valueStyle={{ color: '#00b96b' }} />
          </Card>
        </Col>
        <Col xs={24} md={16}>
          <GlobalRisk data={risk} onSaved={refresh} />
        </Col>
      </Row>
      <StrategyRiskTable data={risk} onSaved={refresh} />
    </>
  )
}

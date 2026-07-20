import React, { useEffect, useState } from 'react'
import {
  Card, Row, Col, Input, Button, message, Descriptions, Tag, Space,
  Select, Alert, Radio,
} from 'antd'
import { api, apiPath, fmtUsd } from '../api'
import { useI18n } from '../i18n'

const CRED_FIELDS = [
  { key: 'clawby_api_key', label: 'Clawby API Key', span: 24 },
  { key: 'binance_api_key', label: 'Binance API Key', span: 12 },
  { key: 'binance_secret_key', label: 'Binance Secret Key', span: 12 },
  { key: 'bitget_api_key', label: 'Bitget API Key', span: 8 },
  { key: 'bitget_secret_key', label: 'Bitget Secret Key', span: 8 },
  { key: 'bitget_passphrase', label: 'Bitget Passphrase', span: 8 },
  { key: 'okx_api_key', label: 'OKX API Key', span: 8 },
  { key: 'okx_secret_key', label: 'OKX Secret Key', span: 8 },
  { key: 'okx_passphrase', label: 'OKX Passphrase', span: 8 },
]

function Credentials() {
  const { t } = useI18n()
  const [cur, setCur] = useState(null)
  const [form, setForm] = useState({})
  const [testing, setTesting] = useState(false)
  const [test, setTest] = useState(null)

  const load = () => api.credentials().then(setCur).catch(() => {})
  useEffect(() => { load() }, [])

  const save = async () => {
    const payload = {}
    for (const f of CRED_FIELDS) if (form[f.key]) payload[f.key] = form[f.key].trim()
    if (!Object.keys(payload).length) return message.info(t('config.nothingToSave'))
    try {
      await api.setCredentials(payload)
      message.success(t('config.credsSaved'))
      setForm({}); load()
    } catch (e) { message.error(`${t('config.saveFailed')}: ${e.message}`) }
  }

  const runTest = async () => {
    setTesting(true); setTest(null)
    try { setTest(await api.testCredentials('both')) }
    catch (e) { message.error(`${t('config.testFailed')}: ${e.message}`) }
    setTesting(false)
  }

  const badge = (ok, masked) => (ok
    ? <Tag color="green">{masked}</Tag> : <Tag>{t('config.notSet')}</Tag>)

  const venueResult = (name, r) => r && (
    <div style={{ fontSize: 12, marginBottom: 4 }}>
      {name}: {r.ok
        ? (
          <Tag color="green">
            {t('config.connOk')}
            {r.plan != null && ` · plan=${r.plan} · ${t('config.balance')}${r.balance}`}
            {r.can_trade != null && ` · ${t('config.canTrade')}=${String(r.can_trade)}`}
            {r.wallet_usdt != null && ` · ${t('config.wallet')}${fmtUsd(Number(r.wallet_usdt))}`}
            {r.available != null && ` · ${t('config.avail')}${fmtUsd(Number(r.available))}`}
          </Tag>
        )
        : <Tag color="red">{t('config.failed')}: {r.error}</Tag>}
    </div>
  )

  return (
    <Card size="small" title={t('config.creds')}>
      <Alert type="warning" showIcon style={{ marginBottom: 12, fontSize: 12 }}
             message={t('config.credsWarn')} />
      {cur && (
        <Descriptions size="small" column={2} bordered style={{ marginBottom: 14 }}
                      labelStyle={{ width: 140 }}>
          <Descriptions.Item label="Clawby">{badge(cur.has_clawby, cur.clawby_api_key)}</Descriptions.Item>
          <Descriptions.Item label="Binance">{badge(cur.has_binance, cur.binance_api_key)}</Descriptions.Item>
          <Descriptions.Item label="Bitget">{badge(cur.has_bitget, cur.bitget_api_key)}</Descriptions.Item>
          <Descriptions.Item label="OKX">{badge(cur.has_okx, cur.okx_api_key)}</Descriptions.Item>
        </Descriptions>
      )}
      <Row gutter={[12, 12]}>
        {CRED_FIELDS.map((f) => (
          <Col xs={24} md={f.span} key={f.key}>
            <div style={{ fontSize: 12, color: '#7d8590', marginBottom: 4 }}>{f.label}</div>
            <Input.Password placeholder={t('config.keepEmpty')} autoComplete="off"
                            value={form[f.key] || ''}
                            onChange={(e) => setForm((s) => ({ ...s, [f.key]: e.target.value }))} />
          </Col>
        ))}
      </Row>
      <Space style={{ marginTop: 14 }}>
        <Button type="primary" size="small" onClick={save}>{t('config.saveCreds')}</Button>
        <Button size="small" loading={testing} onClick={runTest}>{t('config.testConn')}</Button>
      </Space>
      {test && (
        <div style={{ marginTop: 12 }}>
          {venueResult('Clawby', test.clawby)}
          {venueResult('Binance', test.binance)}
          {venueResult('Bitget', test.bitget)}
          {venueResult('OKX', test.okx)}
        </div>
      )}
    </Card>
  )
}

function ExecutorExchange() {
  const { t } = useI18n()
  const [info, setInfo] = useState(null)

  const load = () => fetch(apiPath('/api/executor-exchange')).then((r) => r.json())
    .then(setInfo).catch(() => {})
  useEffect(() => { load() }, [])

  const switchTo = async (venue) => {
    try {
      const r = await fetch(apiPath('/api/executor-exchange'), {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ exchange: venue }),
      })
      const d = await r.json().catch(() => ({}))
      if (!r.ok) throw new Error(d.detail || r.status)
      message.success(`${t('config.executorOk')}: ${venue}`)
      load()
    } catch (e) { message.error(`${t('common.opFailed')}: ${e.message}`) }
  }

  if (!info) return null
  return (
    <Card size="small" title={t('config.executor')} style={{ marginTop: 12 }}>
      <Alert type="info" showIcon style={{ marginBottom: 12, fontSize: 12 }}
             message={t('config.executorHint')} />
      <Radio.Group value={info.exchange} buttonStyle="solid" size="small"
                   onChange={(e) => switchTo(e.target.value)}>
        {(info.supported || []).map((v) => (
          <Radio.Button key={v.id} value={v.id} disabled={!v.has_credentials}>
            {v.id.toUpperCase()}
            {!v.has_credentials && ` (${t('config.noCreds')})`}
          </Radio.Button>
        ))}
      </Radio.Group>
      <div style={{ marginTop: 8, fontSize: 12, color: '#7d8590' }}>
        {t('config.executorCur')}: <Tag color="green">{info.exchange.toUpperCase()}</Tag>
      </div>
    </Card>
  )
}

function UniverseConfig() {
  const { t } = useI18n()
  const [uni, setUni] = useState([])
  const [known, setKnown] = useState([])
  const [sel, setSel] = useState([])

  useEffect(() => {
    api.universe().then((d) => { setUni(d.universe); setKnown(d.known); setSel(d.universe) }).catch(() => {})
  }, [])

  const save = async () => {
    if (!sel.length) return message.error(t('config.atLeastOne'))
    try {
      await api.setUniverse(sel)
      message.success(t('config.universeSaved'))
      setUni(sel)
    } catch (e) { message.error(`${t('config.saveFailed')}: ${e.message}`) }
  }

  const options = Array.from(new Set([...known, ...uni])).map((s) => ({ value: s, label: s }))

  return (
    <Card size="small" title={t('config.universe')} style={{ marginTop: 12 }}>
      <Alert type="info" showIcon style={{ marginBottom: 12, fontSize: 12 }}
             message={t('config.universeHint')} />
      <Select mode="tags" style={{ width: '100%' }} value={sel} onChange={setSel}
              options={options} placeholder={t('config.universePh')}
              tokenSeparators={[',', ' ']} />
      <Button type="primary" size="small" style={{ marginTop: 12 }} onClick={save}>
        {t('config.saveUniverse')}
      </Button>
      <div style={{ marginTop: 8, fontSize: 12, color: '#7d8590' }}>
        {t('config.current')}: {uni.join(' · ')}
      </div>
    </Card>
  )
}

export default function ConfigPage() {
  return (
    <Row gutter={[12, 12]}>
      <Col span={24}><Credentials /></Col>
      <Col span={24}><ExecutorExchange /></Col>
      <Col span={24}><UniverseConfig /></Col>
    </Row>
  )
}

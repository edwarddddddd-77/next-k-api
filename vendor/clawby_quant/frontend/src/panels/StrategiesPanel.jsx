import React from 'react'
import {
  Card, Table, Switch, Tag, Typography, InputNumber, Input, Descriptions, Select,
  Button, Modal, Popconfirm, Space, message,
} from 'antd'
import { PlusOutlined, DeleteOutlined, ExclamationCircleOutlined } from '@ant-design/icons'
import { useI18n } from '../i18n'

const { Text } = Typography

const STR_INTERVALS = { '1s': 1, '5s': 5, '15s': 15, '30s': 30, '1m': 60, '5m': 300,
                        '15m': 900, '30m': 1800, '1h': 3600, '4h': 14400 }
const toSec = (v) => (typeof v === 'number' ? v : STR_INTERVALS[String(v)] ?? 900)

const TYPE_COLORS = { 均值回归: 'cyan', 动量突破: 'magenta', 趋势跟踪: 'volcano',
                      仓位情绪: 'geekblue', 微观结构反转: 'purple', 链上事件驱动: 'gold',
                      时段性资金流: 'lime', 事件驱动动量: 'orange', 高频均值回归: 'red',
                      高频动量: 'red' }

function IntervalEditor({ row, onSave }) {
  const [val, setVal] = React.useState(toSec(row.scan_interval))
  React.useEffect(() => { setVal(toSec(row.scan_interval)) }, [row.scan_interval])
  const commit = () => {
    const v = Math.max(1, Math.round(val || 1))
    if (v !== toSec(row.scan_interval)) onSave(row.id, v)
  }
  return (
    <InputNumber size="small" min={1} max={86400} value={val} style={{ width: 88 }}
                 addonAfter="s" onChange={setVal} onBlur={commit} onPressEnter={commit} />
  )
}

function SymbolsInline({ row, universe, known, onSave }) {
  const { t } = useI18n()
  const [val, setVal] = React.useState(row.symbols || [])
  React.useEffect(() => { setVal(row.symbols || []) }, [row.symbols])
  const uni = universe || []
  const uniSet = new Set(uni)
  const options = Array.from(new Set([...uni, ...(known || [])])).map((s) => ({
    value: s, label: s.replace('USDT', ''), disabled: !uniSet.has(s),
  }))
  const commitIfChanged = (open) => {
    if (open) return
    const a = JSON.stringify([...(row.symbols || [])].sort())
    const b = JSON.stringify([...val].sort())
    if (a !== b) onSave(row.id, val)
  }
  const tagRender = ({ label, value, closable, onClose }) => {
    const isOverflow = value == null            // the "+N ..." collapsed counter
    const ok = isOverflow || uniSet.has(value)
    return (
      <Tag closable={closable} onClose={onClose}
           color={ok ? undefined : 'red'}
           title={ok ? (value || '') : `${value} ${t('strat.notInUniverse')}`}
           style={{ marginInlineEnd: 2, fontSize: 11 }}>
        {label}
      </Tag>
    )
  }
  return (
    <Select mode="multiple" size="small" style={{ minWidth: 170, width: '100%' }}
            value={val} onChange={setVal} options={options} tagRender={tagRender}
            onDropdownVisibleChange={commitIfChanged}
            maxTagCount={3} placeholder={t('strat.globalPool')}
            popupMatchSelectWidth={280} showSearch optionFilterProp="value"
            dropdownRender={(menu) => (
              <>
                <Space style={{ padding: '4px 8px' }} size={8}>
                  <Button size="small" type="link" style={{ padding: 0 }}
                          onClick={() => setVal([...uni])}>
                    {t('strat.selectAll')}({uni.length})
                  </Button>
                  <Button size="small" type="link" style={{ padding: 0 }}
                          onClick={() => setVal([])}>
                    {t('strat.clearFollow')}
                  </Button>
                </Space>
                {menu}
              </>
            )} />
  )
}

function ParamsEditor({ row, onSave }) {
  const { t } = useI18n()
  const [draft, setDraft] = React.useState({ ...(row.params || {}) })
  React.useEffect(() => { setDraft({ ...(row.params || {}) }) }, [row.params])
  const entries = Object.entries(row.params || {})
  if (!entries.length) return <Text type="secondary">{t('strat.noParams')}</Text>
  const changed = JSON.stringify(draft) !== JSON.stringify(row.params || {})
  return (
    <Space wrap size={[16, 8]}>
      {entries.map(([k, v]) => (
        <span key={k} style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
          <Text type="secondary" style={{ fontSize: 12 }}>{k}</Text>
          {typeof v === 'boolean' ? (
            <Switch size="small" checked={!!draft[k]}
                    onChange={(nv) => setDraft({ ...draft, [k]: nv })} />
          ) : typeof v === 'number' ? (
            <InputNumber size="small" value={draft[k]} style={{ width: 110 }}
                         step={Math.abs(v) >= 100 ? 1 : Math.abs(v) >= 1 ? 0.1 : 0.0001}
                         onChange={(nv) => setDraft({ ...draft, [k]: nv })} />
          ) : (
            <Input size="small" value={draft[k]} style={{ width: 120 }}
                   onChange={(e) => setDraft({ ...draft, [k]: e.target.value })} />
          )}
        </span>
      ))}
      <Button size="small" type="primary" disabled={!changed}
              onClick={() => onSave(row.id, draft)}>{t('strat.saveParams')}</Button>
    </Space>
  )
}

function FactorDeps({ factors, compact }) {
  const { t } = useI18n()
  if (!factors || !factors.length) {
    return <Text type="secondary" style={{ fontSize: 11 }}>-</Text>
  }
  return (
    <Space size={[4, 4]} wrap>
      {factors.map((f) => (
        <Tag key={f.name} color={f.enabled ? 'default' : 'red'}
             title={`${f.label} · ${f.enabled ? t('strat.factorOn') : t('strat.factorOff')}`}
             style={{ fontSize: 10.5, marginInlineEnd: 0 }}>
          {compact ? f.name : `${f.name} ${f.enabled ? '✓' : '✕'}`}
        </Tag>
      ))}
    </Space>
  )
}

function CreateModal({ open, templates, onOk, onCancel }) {
  const { t, pick } = useI18n()
  const [base, setBase] = React.useState()
  const [name, setName] = React.useState('')
  React.useEffect(() => { if (open) { setBase(undefined); setName('') } }, [open])
  return (
    <Modal title={t('strat.createTitle')} open={open} onCancel={onCancel} destroyOnClose
           okText={t('common.create')} cancelText={t('common.cancel')}
           okButtonProps={{ disabled: !base }} onOk={() => onOk(base, name)}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginTop: 8 }}>
        <div>
          <Text type="secondary" style={{ fontSize: 12 }}>{t('strat.createTpl')}</Text>
          <Select style={{ width: '100%', marginTop: 4 }} value={base} onChange={setBase}
                  placeholder={t('strat.createTplPh')}
                  options={(templates || []).map((tp) => ({
                    value: tp.id, label: `${tp.id.split('_')[0]} ${pick(tp.meta, 'name') || tp.id}` }))} />
        </div>
        <div>
          <Text type="secondary" style={{ fontSize: 12 }}>{t('strat.createName')}</Text>
          <Input style={{ marginTop: 4 }} value={name} maxLength={60}
                 onChange={(e) => setName(e.target.value)} placeholder={t('strat.createNamePh')} />
        </div>
      </div>
    </Modal>
  )
}

export default function StrategiesPanel({ data, universe, known, onToggle, onIntervalSave,
                                          onSymbolsSave, onParamsSave, onCreate, onDelete }) {
  const { t, pick } = useI18n()
  const rows = data?.strategies || []
  const templates = data?.templates || []
  const [createOpen, setCreateOpen] = React.useState(false)
  const [modal, modalCtx] = Modal.useModal()

  // enabling a strategy whose factor deps are OFF -> confirm dialog (需求4)
  const guardedToggle = (row, enabled) => {
    const off = (row.factors || []).filter((f) => !f.enabled)
    if (enabled && off.length) {
      modal.confirm({
        title: t('strat.enableWarnTitle'),
        icon: <ExclamationCircleOutlined />,
        content: (
          <div>
            <div style={{ marginBottom: 8 }}>{t('strat.enableWarnBody')}</div>
            <Space size={[4, 4]} wrap>
              {off.map((f) => <Tag key={f.name} color="red">{f.name} · {f.label}</Tag>)}
            </Space>
          </div>
        ),
        okText: t('strat.enableWarnGo'), cancelText: t('common.cancel'),
        onOk: () => onToggle(row.id, true),
      })
      return
    }
    onToggle(row.id, enabled)
  }

  const columns = [
    { title: t('strat.col.name'), key: 'name', width: 190,
      render: (_, r) => (
        <span>
          <b>{r.id.split('_')[0]}{r.id.includes('__') ? `#${r.id.split('__')[1]}` : ''}</b>{' '}
          {r.display_name || pick(r.meta, 'name') || r.id}
          {r.meta?.type && (
            <Tag color={TYPE_COLORS[r.meta.type.split('·')[0]] || 'default'}
                 style={{ marginLeft: 6, fontSize: 11 }}>{r.meta.type}</Tag>
          )}
        </span>
      ) },
    { title: t('strat.col.desc'), key: 'desc',
      render: (_, r) => (
        <Text type="secondary" style={{ fontSize: 12, whiteSpace: 'normal',
                                        wordBreak: 'break-word', display: 'block' }}>
          {pick(r.meta, 'logic') || '-'}
        </Text>
      ) },
    { title: t('strat.col.factors'), key: 'deps', width: 150,
      render: (_, r) => <FactorDeps factors={r.factors} compact /> },
    { title: t('strat.col.symbols'), key: 'symbols', width: 190,
      render: (_, r) => (
        <SymbolsInline row={r} universe={universe} known={known} onSave={onSymbolsSave} />
      ) },
    { title: t('strat.col.interval'), key: 'interval', width: 106,
      render: (_, r) => <IntervalEditor row={r} onSave={onIntervalSave} /> },
    { title: t('strat.col.stats'), key: 'stats', width: 145,
      render: (_, r) => {
        const s = r.stats
        if (!s || !s.closed) return <Text type="secondary" style={{ fontSize: 12 }}>{t('strat.noTrades')}</Text>
        const wr = ((s.wins / s.closed) * 100).toFixed(0)
        return (
          <span style={{ fontSize: 12 }}>
            {s.closed}{t('strat.trades')}·{t('strat.winRate')}{wr}%·
            <span style={{ color: s.pnl >= 0 ? '#00b96b' : '#e5484d' }}>
              {s.pnl >= 0 ? '+' : ''}{s.pnl.toFixed(1)}U</span>
          </span>
        )
      } },
    { title: t('strat.col.switch'), key: 'enabled', width: 56, align: 'center',
      render: (_, r) => (
        <Switch size="small" checked={!!r.enabled} onChange={(v) => guardedToggle(r, v)} />
      ) },
    { title: '', key: 'del', width: 44, align: 'center',
      render: (_, r) => (
        <Popconfirm title={`${t('strat.deleteTitle')} ${r.id}?`}
                    description={t('strat.deleteDesc')}
                    okText={t('common.delete')} cancelText={t('common.cancel')}
                    okButtonProps={{ danger: true }}
                    onConfirm={() => onDelete(r.id)}>
          <Button size="small" type="text" danger icon={<DeleteOutlined />} />
        </Popconfirm>
      ) },
  ]

  const expandedRowRender = (r) => {
    const m = r.meta || {}
    return (
      <Descriptions size="small" column={1} bordered
                    labelStyle={{ width: 100, fontSize: 12 }}
                    contentStyle={{ fontSize: 12.5 }} style={{ margin: '4px 0' }}>
        <Descriptions.Item label={t('strat.entry')}>{pick(m, 'entry') || '-'}</Descriptions.Item>
        <Descriptions.Item label={t('strat.exit')}>{pick(m, 'exit') || '-'}</Descriptions.Item>
        <Descriptions.Item label={t('strat.deps')}>
          <FactorDeps factors={r.factors} />
        </Descriptions.Item>
        <Descriptions.Item label={t('strat.risk')}>
          <Text type="warning" style={{ fontSize: 12.5 }}>{pick(m, 'risk') || '-'}</Text>
        </Descriptions.Item>
        <Descriptions.Item label={t('strat.params')}>
          <ParamsEditor row={r} onSave={onParamsSave} />
        </Descriptions.Item>
      </Descriptions>
    )
  }

  return (
    <Card size="small" title={t('strat.title')}
          extra={(
            <Button size="small" type="primary" icon={<PlusOutlined />}
                    onClick={() => setCreateOpen(true)}>{t('strat.new')}</Button>
          )}>
      {modalCtx}
      <Table rowKey="id" dataSource={rows} columns={columns} size="small"
             pagination={false} expandable={{ expandedRowRender }} scroll={{ x: 1220 }} />
      <CreateModal open={createOpen} templates={templates}
                   onCancel={() => setCreateOpen(false)}
                   onOk={async (base, name) => {
                     try { await onCreate(base, name); setCreateOpen(false) }
                     catch (e) { message.error(e.message) }
                   }} />
    </Card>
  )
}

export { toSec }

import React from 'react'
import { message } from 'antd'
import { api } from '../api'
import { usePoll } from '../hooks'
import { useI18n } from '../i18n'
import StrategiesPanel from '../panels/StrategiesPanel'

export default function StrategiesPage() {
  const [strategies, refresh] = usePoll(api.strategies, 15000)
  const [universe] = usePoll(api.universe, 60000)
  const { t } = useI18n()

  const wrap = (fn, okKey) => async (...args) => {
    try { await fn(...args); if (okKey) message.success(t(okKey)); refresh() }
    catch (e) { message.error(`${t('common.opFailed')}: ${e.message}`); throw e }
  }

  const onToggle = async (id, enabled) => {
    try {
      await api.toggleStrategy(id, enabled)
      message.success(`${id} ${enabled ? t('strat.enabledMsg') : t('strat.disabledMsg')}`)
      refresh()
    } catch (e) { message.error(`${t('common.opFailed')}: ${e.message}`) }
  }

  return (
    <StrategiesPanel data={strategies} universe={universe?.universe || []}
                     known={universe?.known || []}
                     onToggle={onToggle}
                     onIntervalSave={wrap((id, sec) => api.strategyConfig(id, { scan_interval_sec: sec }), 'strat.intervalSaved')}
                     onSymbolsSave={wrap((id, symbols) => api.strategyConfig(id, { symbols }), 'strat.symbolsSaved')}
                     onParamsSave={wrap((id, params) => api.strategyConfig(id, { params }), 'strat.paramsSaved')}
                     onCreate={wrap((base, name) => api.createStrategy(base, name), 'strat.created')}
                     onDelete={wrap((id) => api.deleteStrategy(id), 'strat.deleted')} />
  )
}

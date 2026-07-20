import React, { useState } from 'react'
import { Layout, Menu, Tag, Space, Typography, Segmented } from 'antd'
import {
  DashboardOutlined, ThunderboltOutlined, FunctionOutlined,
  SafetyOutlined, SettingOutlined, BookOutlined,
} from '@ant-design/icons'
import { api } from './api'
import { usePoll } from './hooks'
import { useI18n } from './i18n'
import { ModeSwitch } from './panels/StatusBar'
import Dashboard from './pages/Dashboard'
import StrategiesPage from './pages/StrategiesPage'
import FactorsPage from './pages/FactorsPage'
import RiskPage from './pages/RiskPage'
import ConfigPage from './pages/ConfigPage'
import JournalPage from './pages/JournalPage'

const { Header, Sider, Content } = Layout

const PAGES = {
  dashboard: { key: 'menu.dashboard', icon: <DashboardOutlined />, comp: Dashboard },
  strategies: { key: 'menu.strategies', icon: <ThunderboltOutlined />, comp: StrategiesPage },
  factors: { key: 'menu.factors', icon: <FunctionOutlined />, comp: FactorsPage },
  journal: { key: 'menu.journal', icon: <BookOutlined />, comp: JournalPage },
  risk: { key: 'menu.risk', icon: <SafetyOutlined />, comp: RiskPage },
  config: { key: 'menu.config', icon: <SettingOutlined />, comp: ConfigPage },
}

export default function App() {
  const [page, setPage] = useState(() => {
    const q = new URLSearchParams(window.location.search).get('page')
    return PAGES[q] ? q : 'dashboard'
  })
  const [collapsed, setCollapsed] = useState(false)
  const [status] = usePoll(api.status, 5000)
  const { t, lang, setLang } = useI18n()

  const manageAgeMs = status?.last_manage_ms ? status.now_ms - status.last_manage_ms : null
  const engineLabel = manageAgeMs == null ? '-'
    : manageAgeMs < 2000 ? `${(manageAgeMs / 1000).toFixed(1)}s`
    : `${Math.round(manageAgeMs / 1000)}s`

  const PageComp = PAGES[page].comp

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider collapsible collapsed={collapsed} onCollapse={setCollapsed}
             theme="dark" style={{ background: '#0d1117' }}>
        <div style={{ height: 48, margin: 12, display: 'flex', alignItems: 'center',
                      gap: 8, color: '#00b96b', fontWeight: 700, fontSize: 16, paddingLeft: 8 }}>
          <ThunderboltOutlined style={{ fontSize: 20 }} />
          {!collapsed && 'Quant Bot'}
        </div>
        <Menu theme="dark" mode="inline" selectedKeys={[page]}
              style={{ background: 'transparent' }}
              onClick={(e) => setPage(e.key)}
              items={Object.entries(PAGES).map(([k, v]) => ({ key: k, icon: v.icon, label: t(v.key) }))} />
      </Sider>

      <Layout>
        <Header style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                         background: '#10151d', borderBottom: '1px solid #1d2633', paddingInline: 20 }}>
          <Space size="middle">
            <Typography.Title level={4} style={{ margin: 0 }}>{t(PAGES[page].key)}</Typography.Title>
            {status && (
              <Space size="small">
                <Tag color={status.mode === 'live' ? 'red' : 'blue'}>
                  {status.mode === 'live' ? t('hdr.live') : t('hdr.paper')}
                </Tag>
                {status.halted && <Tag color="volcano">{t('hdr.halted')}</Tag>}
                {status.event_quiet && <Tag color="orange">{t('hdr.quiet')}</Tag>}
                <Tag color={manageAgeMs != null && manageAgeMs < 5000 ? 'green' : 'volcano'}>
                  {t('hdr.engine')} 0.5s×2 · {engineLabel} {t('hdr.ago')}
                </Tag>
                <Tag color={status.ws_symbols > 0 ? 'green' : 'default'}>
                  {t('hdr.px')} {status.ws_symbols} {t('hdr.coins')}
                </Tag>
              </Space>
            )}
          </Space>
          <Space size="middle">
            <Segmented size="small" value={lang}
                        options={[{ label: '中文', value: 'zh' }, { label: 'EN', value: 'en' }]}
                        onChange={setLang} />
            <ModeSwitch status={status} />
          </Space>
        </Header>

        <Content style={{ padding: 16 }}>
          <PageComp status={status} />
        </Content>
      </Layout>
    </Layout>
  )
}

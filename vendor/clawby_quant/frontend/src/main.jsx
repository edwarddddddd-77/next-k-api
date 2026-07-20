import React from 'react'
import ReactDOM from 'react-dom/client'
import { ConfigProvider, theme } from 'antd'
import zhCN from 'antd/locale/zh_CN'
import enUS from 'antd/locale/en_US'
import App from './App'
import { I18nProvider, useI18n } from './i18n'

function LocaleShell() {
  const { lang } = useI18n()
  return (
    <ConfigProvider
      locale={lang === 'en' ? enUS : zhCN}
      theme={{
        algorithm: theme.darkAlgorithm,
        token: { colorPrimary: '#00b96b', colorBgBase: '#0a0e14', fontSize: 13 },
      }}
    >
      <App />
    </ConfigProvider>
  )
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <I18nProvider>
    <LocaleShell />
  </I18nProvider>
)

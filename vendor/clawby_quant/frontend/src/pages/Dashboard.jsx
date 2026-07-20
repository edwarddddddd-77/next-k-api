import React from 'react'
import { Row, Col } from 'antd'
import { api } from '../api'
import { usePoll } from '../hooks'
import { StatusCards } from '../panels/StatusBar'
import EquityChart from '../panels/EquityChart'
import PriceChart from '../panels/PriceChart'
import PositionsTable from '../panels/PositionsTable'
import ActivityTabs from '../panels/ActivityTabs'

export default function Dashboard({ status }) {
  const [positions, refreshPositions] = usePoll(api.positions, 5000)
  const [signals] = usePoll(api.signals, 15000)
  const [trades] = usePoll(api.trades, 15000)   // backend filters to active mode
  const [logs] = usePoll(api.logs, 15000)

  return (
    <>
      <StatusCards status={status} positions={positions?.positions || []} />

      <Row gutter={[12, 12]} style={{ marginTop: 12 }}>
        <Col xs={24} lg={10}><EquityChart activeMode={status?.mode} /></Col>
        <Col xs={24} lg={14}><PriceChart universe={status?.universe || []} /></Col>
      </Row>

      <Row gutter={[12, 12]} style={{ marginTop: 12 }}>
        <Col span={24}>
          <PositionsTable positions={positions?.positions || []} onChanged={refreshPositions} />
        </Col>
      </Row>

      <Row gutter={[12, 12]} style={{ marginTop: 12 }}>
        <Col span={24}>
          <ActivityTabs signals={signals?.signals || []} trades={trades?.trades || []}
                        logs={logs?.logs || []} />
        </Col>
      </Row>
    </>
  )
}

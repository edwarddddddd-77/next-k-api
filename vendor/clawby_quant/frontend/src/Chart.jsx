import React, { useEffect, useRef } from 'react'
import * as echarts from 'echarts'

// Thin ECharts wrapper: re-renders on option change, resizes with container.
export default function Chart({ option, height = 260 }) {
  const ref = useRef(null)
  const chartRef = useRef(null)

  useEffect(() => {
    if (!ref.current) return
    chartRef.current = echarts.init(ref.current, 'dark', { renderer: 'canvas' })
    const ro = new ResizeObserver(() => chartRef.current?.resize())
    ro.observe(ref.current)
    return () => { ro.disconnect(); chartRef.current?.dispose() }
  }, [])

  useEffect(() => {
    if (chartRef.current && option) {
      chartRef.current.setOption({ backgroundColor: 'transparent', ...option }, true)
    }
  }, [option])

  return <div ref={ref} style={{ width: '100%', height }} />
}

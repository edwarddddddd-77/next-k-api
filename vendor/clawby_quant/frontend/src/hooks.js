import { useCallback, useEffect, useState } from 'react'

// Poll an async fn on an interval; returns [data, refresh].
export function usePoll(fn, intervalMs, deps = []) {
  const [data, setData] = useState(null)
  const tick = useCallback(() => { fn().then(setData).catch(() => {}) }, deps) // eslint-disable-line
  useEffect(() => {
    tick()
    const t = setInterval(tick, intervalMs)
    return () => clearInterval(t)
  }, [tick, intervalMs])
  return [data, tick]
}

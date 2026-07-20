import React from 'react'
import { api } from '../api'
import { usePoll } from '../hooks'
import FactorLibrary from '../panels/FactorLibrary'

export default function FactorsPage() {
  const [factors] = usePoll(api.factors, 30000)
  return <FactorLibrary factorsData={factors} />
}

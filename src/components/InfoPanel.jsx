import { useMemo } from 'react'

function polylinePoints(values, width, height, pad = 10) {
  if (!values.length) return ''
  const minV = Math.min(...values)
  const maxV = Math.max(...values)
  const span = maxV - minV || 1

  const xStep = (width - pad * 2) / Math.max(1, values.length - 1)
  return values
    .map((v, idx) => {
      const x = pad + idx * xStep
      const y = pad + (1 - (v - minV) / span) * (height - pad * 2)
      return `${x.toFixed(2)},${y.toFixed(2)}`
    })
    .join(' ')
}

export default function InfoPanel(props) {
  const {
    outputValue,
    errorValue,
    lossValue,
    iterationCount,
    lossHistory,
    hiddenSizesLabel,
  } = props

  const graph = useMemo(() => {
    const width = 320
    const height = 120
    const points = polylinePoints(lossHistory, width, height, 14)
    return { width, height, points }
  }, [lossHistory])

  return (
    <aside className="panel">
      <div className="panelHeader">
        <div>
          <div className="panelTitle">Training Stats</div>
          <div className="panelSub">{hiddenSizesLabel}</div>
        </div>
        <div className="panelSub">Iteration {iterationCount}</div>
      </div>
      <div className="panelBody">
        <div className="statList">
          <div className="stat">
            <div className="statLabel">Output</div>
            <div className="statValue">{outputValue == null ? '—' : outputValue.toFixed(4)}</div>
          </div>
          <div className="stat">
            <div className="statLabel">Error (target - output)</div>
            <div className="statValue">{errorValue.toFixed(4)}</div>
          </div>
          <div className="stat">
            <div className="statLabel">Loss (MSE simplified)</div>
            <div className="statValue">{lossValue.toFixed(6)}</div>
          </div>

          <div className="lossGraph" title="Loss history (recent)">
            {lossHistory.length ? (
              <svg width="100%" height="100%" viewBox={`0 0 ${graph.width} ${graph.height}`}>
                <defs>
                  <linearGradient id="lossStroke" x1="0" y1="0" x2="1" y2="0">
                    <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.9" />
                    <stop offset="60%" stopColor="#8b5cf6" stopOpacity="0.9" />
                    <stop offset="100%" stopColor="#22c55e" stopOpacity="0.9" />
                  </linearGradient>
                </defs>
                <polyline
                  points={graph.points}
                  fill="none"
                  stroke="url(#lossStroke)"
                  strokeWidth="2.3"
                  strokeLinejoin="round"
                  strokeLinecap="round"
                />
              </svg>
            ) : (
              <div style={{ color: 'rgba(255,255,255,0.65)', fontSize: 12 }}>
                Run or Train to see loss history
              </div>
            )}
          </div>

          <div className="smallNote">
            Tip: hover a connection on the canvas to see the current weight value. You’ll see weight
            thickness and color change during backprop.
          </div>
        </div>
      </div>
    </aside>
  )
}


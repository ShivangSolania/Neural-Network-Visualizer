import { useEffect, useMemo, useRef, useState } from 'react'

const clamp = (v, min, max) => Math.min(max, Math.max(min, v))

function smoothstep(t) {
  const x = clamp(t, 0, 1)
  return x * x * (3 - 2 * x)
}

function lerp(a, b, t) {
  return a + (b - a) * t
}

function distPointToSegment(px, py, ax, ay, bx, by) {
  const abx = bx - ax
  const aby = by - ay
  const apx = px - ax
  const apy = py - ay
  const ab2 = abx * abx + aby * aby
  const t = ab2 === 0 ? 0 : clamp((apx * abx + apy * aby) / ab2, 0, 1)
  const cx = ax + abx * t
  const cy = ay + aby * t
  const dx = px - cx
  const dy = py - cy
  return Math.sqrt(dx * dx + dy * dy)
}

function activationGlowValue(type, a) {
  // Inputs can be negative; keep glow pleasant for both activations.
  if (type === 'relu') return clamp(a, 0, 1.2) / 1.2
  return clamp(a, 0, 1)
}

export default function CanvasVisualizer(props) {
  const { network, activationType, inputValues, forwardResult, animation, timing } = props

  const canvasRef = useRef(null)
  const wrapRef = useRef(null)

  const rafRef = useRef(0)
  const activeAnimRef = useRef(null)

  const [tooltip, setTooltip] = useState(null) // {x,y,text}
  const hoveredKeyRef = useRef(null)

  const dimsRef = useRef({ w: 800, h: 600, dpr: 1 })
  const layoutRef = useRef(null)

  const baseColors = useMemo(() => {
    return {
      pos: '#3b82f6',
      neg: '#ef4444',
      pos2: 'rgba(59,130,246,0.85)',
      neg2: 'rgba(239,68,68,0.85)',
      forwardDot: '#60a5fa',
      backwardPos: '#22c55e',
      backwardNeg: '#f472b6',
    }
  }, [])

  const computeLayout = (w, h, net) => {
    const padL = 70
    const padR = 40
    const padT = 48
    const padB = 44

    const x0 = padL
    const x1 = padL + (w - padL - padR) * 0.33
    const x2 = padL + (w - padL - padR) * 0.66
    const x3 = w - padR

    const layerYs = (count) => {
      if (count <= 1) return [(padT + h - padB) / 2]
      const step = (h - padT - padB) / (count - 1)
      return Array.from({ length: count }, (_, i) => padT + i * step)
    }

    const y0 = layerYs(net.inputCount)
    const y1 = layerYs(net.hidden1Count)
    const y2 = layerYs(net.hidden2Count)
    const y3 = layerYs(net.outputCount)

    const neuronR = clamp(Math.min(w, h) * 0.018, 11, 18)

    const neuronPositions = {
      input: y0.map((y, i) => ({ x: x0, y, idx: i })),
      h1: y1.map((y, j) => ({ x: x1, y, idx: j })),
      h2: y2.map((y, k) => ({ x: x2, y, idx: k })),
      out: y3.map((y, m) => ({ x: x3, y, idx: m })),
    }

    return { x0, x1, x2, x3, neuronR, neuronPositions }
  }

  const getEffectiveWeights = (netFrom, weightDeltas, alphas) => {
    const { dw1, dw2, dw3 } = weightDeltas || { dw1: null, dw2: null, dw3: null }

    const a1 = alphas.alphaW1 ?? 0
    const a2 = alphas.alphaW2 ?? 0
    const a3 = alphas.alphaW3 ?? 0

    // When no deltas are provided, return netFrom weights.
    if (!dw1 || !dw2 || !dw3) {
      return netFrom.weights
    }

    const w1 = netFrom.weights.w1.map((row, j) =>
      row.map((w, i) => w + (dw1[j][i] || 0) * a1),
    )
    const w2 = netFrom.weights.w2.map((row, k) =>
      row.map((w, j) => w + (dw2[k][j] || 0) * a2),
    )
    const w3 = netFrom.weights.w3.map((row, m) =>
      row.map((w, k) => w + (dw3[m][k] || 0) * a3),
    )

    return { w1, w2, w3 }
  }

  const drawNetwork = (ctx, opts) => {
    const { w, h, dpr } = dimsRef.current

    ctx.save()
    ctx.clearRect(0, 0, w, h)

    // Background glow / grid
    ctx.globalAlpha = 1
    ctx.fillStyle = 'rgba(0,0,0,0.08)'
    ctx.fillRect(0, 0, w, h)

    // Subtle grid
    ctx.globalAlpha = 0.18
    ctx.strokeStyle = 'rgba(255,255,255,0.08)'
    ctx.lineWidth = 1
    const grid = 40
    for (let gx = 0; gx <= w; gx += grid) {
      ctx.beginPath()
      ctx.moveTo(gx, 0)
      ctx.lineTo(gx, h)
      ctx.stroke()
    }
    for (let gy = 0; gy <= h; gy += grid) {
      ctx.beginPath()
      ctx.moveTo(0, gy)
      ctx.lineTo(w, gy)
      ctx.stroke()
    }
    ctx.restore()

    const { layout, weights, forward, backward, signal } = opts
    const { neuronPositions, neuronR } = layout

    const sourceInputs = forward?.input || inputValues
    const a1 = forward?.a1 || []
    const a2 = forward?.a2 || []

    const alphaConnBase = 0.22
    const pulse = signal?.pulse ?? 0
    const forwardPhase = signal?.forwardPhase ?? null
    const backwardPhase = signal?.backwardPhase ?? null

    const drawConnection = (ax, ay, bx, by, weight, signColor, dot, thicknessScale) => {
      const mag = Math.abs(weight)
      const thickness = 1 + mag * 3.1 * thicknessScale
      const alpha = alphaConnBase + clamp(mag * 0.12, 0, 0.42) + pulse * 0.25

      // Connection line
      ctx.save()
      ctx.globalAlpha = alpha
      ctx.strokeStyle = signColor
      ctx.lineWidth = thickness
      ctx.beginPath()
      ctx.moveTo(ax, ay)
      ctx.lineTo(bx, by)
      ctx.stroke()

      // Outer glow for emphasis
      ctx.globalAlpha = clamp(alpha * 0.55, 0, 0.55)
      ctx.lineWidth = thickness + 2.2
      ctx.strokeStyle = signColor
      ctx.stroke()

      if (dot) {
        const { t, size, dotColor, dir = 1 } = dot
        const x = dir === 1 ? lerp(ax, bx, t) : lerp(bx, ax, t)
        const y = dir === 1 ? lerp(ay, by, t) : lerp(by, ay, t)
        ctx.globalAlpha = clamp(0.95 * dot.size, 0, 1)
        ctx.fillStyle = dotColor
        ctx.shadowColor = dotColor
        ctx.shadowBlur = 18 + 18 * dot.size
        ctx.beginPath()
        ctx.arc(x, y, 3 + 6 * dot.size, 0, Math.PI * 2)
        ctx.fill()
        ctx.shadowBlur = 0
      }

      ctx.restore()
    }

    // --- Connections (draw behind neurons) ---
    // w1: input -> h1
    for (let j = 0; j < network.hidden1Count; j++) {
      for (let i = 0; i < network.inputCount; i++) {
        const wgt = weights.w1[j][i]
        const signColor = wgt >= 0 ? baseColors.pos2 : baseColors.neg2
        const a = neuronPositions.input[i]
        const b = neuronPositions.h1[j]

        const isForward =
          forwardPhase === 'w1' && signal?.t != null
        const isBackward = backwardPhase === 'w1' && signal?.t != null

        const shouldDot = isForward || isBackward
        const dot = shouldDot
          ? (() => {
              const t = (signal.t + (i + j) * 0.07) % 1
              if (isForward) {
                const intensity = clamp(Math.abs(sourceInputs[i]) * Math.abs(wgt), 0, 1.5) / 1.5
                return {
                  t,
                  size: intensity,
                  dotColor: wgt >= 0 ? baseColors.forwardDot : '#fb7185',
                  dir: 1,
                }
              }
              const err = backward?.hidden1Delta?.[j] ?? 0
              const intensity = clamp(Math.abs(err) * 0.9 + Math.abs(wgt) * 0.2, 0, 1.2) / 1.2
              return {
                t,
                size: intensity,
                dotColor: err >= 0 ? baseColors.backwardPos : baseColors.backwardNeg,
                dir: -1,
              }
            })()
          : null

        drawConnection(a.x, a.y, b.x, b.y, wgt, signColor, dot, 1.0)
      }
    }

    // w2: h1 -> h2
    for (let k = 0; k < network.hidden2Count; k++) {
      for (let j = 0; j < network.hidden1Count; j++) {
        const wgt = weights.w2[k][j]
        const signColor = wgt >= 0 ? baseColors.pos2 : baseColors.neg2
        const a = neuronPositions.h1[j]
        const b = neuronPositions.h2[k]

        const isForward = forwardPhase === 'w2' && signal?.t != null
        const isBackward = backwardPhase === 'w2' && signal?.t != null

        const dot = isForward || isBackward
          ? (() => {
              const t = (signal.t + (k + j) * 0.06) % 1
              if (isForward) {
                const intensity = clamp(Math.abs(a1[j] ?? 0) * Math.abs(wgt), 0, 1.6) / 1.6
                return {
                  t,
                  size: intensity,
                  dotColor: wgt >= 0 ? baseColors.forwardDot : '#fb7185',
                  dir: 1,
                }
              }
              const err = backward?.hidden2Delta?.[k] ?? 0
              const intensity = clamp(Math.abs(err) * 0.9 + Math.abs(wgt) * 0.18, 0, 1.2) / 1.2
              return {
                t,
                size: intensity,
                dotColor: err >= 0 ? baseColors.backwardPos : baseColors.backwardNeg,
                dir: -1,
              }
            })()
          : null

        drawConnection(a.x, a.y, b.x, b.y, wgt, signColor, dot, 1.0)
      }
    }

    // w3: h2 -> out (only one output)
    for (let m = 0; m < network.outputCount; m++) {
      for (let k = 0; k < network.hidden2Count; k++) {
        const wgt = weights.w3[m][k]
        const signColor = wgt >= 0 ? baseColors.pos2 : baseColors.neg2
        const a = neuronPositions.h2[k]
        const b = neuronPositions.out[m]

        const isForward = forwardPhase === 'w3' && signal?.t != null
        const isBackward = backwardPhase === 'w3' && signal?.t != null

        const dot = isForward || isBackward
          ? (() => {
              const t = (signal.t + (k + m) * 0.08) % 1
              if (isForward) {
                const intensity = clamp(Math.abs(a2[k] ?? 0) * Math.abs(wgt), 0, 1.7) / 1.7
                return {
                  t,
                  size: intensity,
                  dotColor: wgt >= 0 ? baseColors.forwardDot : '#fb7185',
                  dir: 1,
                }
              }
              const err = backward?.outputDelta?.[m] ?? 0
              const intensity = clamp(Math.abs(err) * 0.95 + Math.abs(wgt) * 0.16, 0, 1.25) / 1.25
              return {
                t,
                size: intensity,
                dotColor: err >= 0 ? baseColors.backwardPos : baseColors.backwardNeg,
                dir: -1,
              }
            })()
          : null

        drawConnection(a.x, a.y, b.x, b.y, wgt, signColor, dot, 1.0)
      }
    }

    // --- Neurons on top ---
    const drawNeuron = (cx, cy, value, isInput) => {
      const glow = activationGlowValue(activationType, value)
      const radius = neuronR * (0.9 + glow * 0.6)
      const core = isInput ? 'rgba(59,130,246,0.95)' : 'rgba(139,92,246,0.95)'

      // If value is "inactive" in ReLU, reduce alpha.
      const alpha = isInput ? 0.75 + glow * 0.2 : 0.55 + glow * 0.45
      ctx.save()
      ctx.globalAlpha = alpha
      ctx.shadowColor = core
      ctx.shadowBlur = 20 + glow * 40
      ctx.fillStyle = core
      ctx.beginPath()
      ctx.arc(cx, cy, radius, 0, Math.PI * 2)
      ctx.fill()

      // Inner ring
      ctx.shadowBlur = 0
      ctx.globalAlpha = 0.95
      ctx.strokeStyle = 'rgba(255,255,255,0.25)'
      ctx.lineWidth = 1.5
      ctx.beginPath()
      ctx.arc(cx, cy, radius * 0.58, 0, Math.PI * 2)
      ctx.stroke()
      ctx.restore()
    }

    for (let i = 0; i < network.inputCount; i++) {
      const a = neuronPositions.input[i]
      drawNeuron(a.x, a.y, sourceInputs[i] ?? 0, true)
    }
    for (let j = 0; j < network.hidden1Count; j++) {
      const a = neuronPositions.h1[j]
      drawNeuron(a.x, a.y, a1[j] ?? 0, false)
    }
    for (let k = 0; k < network.hidden2Count; k++) {
      const a = neuronPositions.h2[k]
      drawNeuron(a.x, a.y, a2[k] ?? 0, false)
    }
    for (let m = 0; m < network.outputCount; m++) {
      const a = neuronPositions.out[m]
      drawNeuron(a.x, a.y, forward?.a3?.[m] ?? 0, false)
    }

    // Labels (just layer markers to keep it clean)
    const drawLabel = (text, x, y) => {
      ctx.save()
      ctx.globalAlpha = 0.9
      ctx.fillStyle = 'rgba(255,255,255,0.75)'
      ctx.font = '12px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono"'
      ctx.textAlign = 'center'
      ctx.fillText(text, x, y - neuronR - 10)
      ctx.restore()
    }

    drawLabel('Input', layout.x0, h - padBottom(layout))
    drawLabel('Hidden 1', layout.x1, h - padBottom(layout))
    drawLabel('Hidden 2', layout.x2, h - padBottom(layout))
    drawLabel('Output', layout.x3, h - padBottom(layout))
  }

  const padBottom = () => 54

  // Animations (named as requested)
  const animateForward = (elapsed) => {
    const seg = timing.forwardMs / 3
    const segIdx = clamp(Math.floor(elapsed / seg), 0, 2)
    const local = (elapsed - segIdx * seg) / seg
    return {
      phase: segIdx === 0 ? 'w1' : segIdx === 1 ? 'w2' : 'w3',
      t: clamp(local, 0, 1),
      done: elapsed >= timing.forwardMs,
    }
  }

  const animateBackward = (elapsedBack) => {
    const seg = timing.backwardMs / 3
    const segIdx = clamp(Math.floor(elapsedBack / seg), 0, 2)
    const local = (elapsedBack - segIdx * seg) / seg
    return {
      phase: segIdx === 0 ? 'w3' : segIdx === 1 ? 'w2' : 'w1',
      t: clamp(local, 0, 1),
      done: elapsedBack >= timing.backwardMs,
      segIdx,
    }
  }

  const computeBackAlphas = (elapsedBack) => {
    const seg = timing.backwardMs / 3
    const aW3 = elapsedBack <= 0 ? 0 : elapsedBack < seg ? smoothstep(elapsedBack / seg) : 1
    const aW2 =
      elapsedBack <= seg
        ? 0
        : elapsedBack < seg * 2
          ? smoothstep((elapsedBack - seg) / seg)
          : 1
    const aW1 =
      elapsedBack <= seg * 2
        ? 0
        : elapsedBack < seg * 3
          ? smoothstep((elapsedBack - seg * 2) / seg)
          : 1
    return { alphaW1: aW1, alphaW2: aW2, alphaW3: aW3 }
  }

  const drawIdle = () => {
    const ctx = canvasRef.current?.getContext('2d')
    if (!ctx) return
    const dims = dimsRef.current
    const layout = layoutRef.current || computeLayout(dims.w, dims.h, network)
    layoutRef.current = layout

    drawNetwork(ctx, {
      layout,
      weights: network.weights,
      forward: forwardResult || lastForwardFromAnimationIfAny(animation, network, inputValues),
      backward: null,
      signal: { forwardPhase: null, backwardPhase: null, t: 0, pulse: 0 },
    })
  }

  const lastForwardFromAnimationIfAny = (anim, net, inputs) => {
    // Prefer forwardAfter for train so the idle view matches the latest weights.
    if (anim?.mode === 'train' && anim?.forwardAfter && anim?.forwardAfter?.a3) return anim.forwardAfter
    if (anim?.mode && anim?.forward && anim?.forward?.a3) return anim.forward
    // Fallback: show raw inputs on input layer; keep hidden/out dark.
    return {
      input: inputs,
      a1: Array(net.hidden1Count).fill(0),
      a2: Array(net.hidden2Count).fill(0),
      a3: [0],
    }
  }

  const startAnimation = () => {
    if (!canvasRef.current) return
    const ctx = canvasRef.current.getContext('2d')
    if (!ctx) return

    if (rafRef.current) cancelAnimationFrame(rafRef.current)
    const payload = animation
    activeAnimRef.current = {
      active: true,
      startTs: performance.now(),
      payload,
    }

    const loop = (now) => {
      const st = activeAnimRef.current
      if (!st || !st.active) return

      const elapsed = now - st.startTs
      const dims = dimsRef.current

      const layout = layoutRef.current || computeLayout(dims.w, dims.h, payload.networkFrom || network)
      layoutRef.current = layout

      const isTrain = payload.mode === 'train'
      const isRun = payload.mode === 'run'

      // Determine signal/phase and weight interpolation.
      let forwardSignal = null
      let backwardSignal = null
      let signal = { forwardPhase: null, backwardPhase: null, t: 0, pulse: 0 }
      let alphas = { alphaW1: 0, alphaW2: 0, alphaW3: 0 }

      const netFrom = payload.networkFrom || network
      const netTo = payload.networkTo || network
      const weightDeltas = payload.weightDeltas || null

      if (elapsed <= timing.forwardMs || isRun) {
        if (elapsed <= timing.forwardMs) {
          const f = animateForward(elapsed)
          forwardSignal = f.phase
          signal.forwardPhase = forwardSignal
          signal.t = f.t
          signal.pulse = 0.7 * Math.sin(Math.min(1, f.t) * Math.PI)
        } else {
          // Forward done; just keep last forward glow briefly.
          const f = animateForward(timing.forwardMs - 0.001)
          signal.forwardPhase = f.phase
          signal.t = 1
          signal.pulse = 0.3
        }
      }

      if (isTrain) {
        const elapsedBack = elapsed - timing.forwardMs
        if (elapsedBack >= 0) {
          const b = animateBackward(elapsedBack)
          backwardSignal = b.phase
          signal.backwardPhase = backwardSignal
          signal.t = b.t
          signal.pulse = 0.8 * Math.sin(Math.min(1, b.t) * Math.PI)
          alphas = computeBackAlphas(elapsedBack)
        }
      }

      const effectiveWeights = getEffectiveWeights(netFrom, weightDeltas, alphas)

      drawNetwork(ctx, {
        layout,
        weights: effectiveWeights,
        forward:
          payload.forward ||
          forwardResult ||
          lastForwardFromAnimationIfAny(animation, network, inputValues),
        backward: payload.backward || null,
        signal,
      })

      const done = elapsed >= (isTrain ? timing.totalMs : timing.forwardMs + 70)
      if (!done) {
        rafRef.current = requestAnimationFrame(loop)
      } else {
        activeAnimRef.current.active = false
        // Ensure we end on the final network (post-training weights).
        const finalNet = isTrain ? (payload.networkTo || netTo) : netTo
        layoutRef.current = computeLayout(dims.w, dims.h, finalNet)
        const forwardFinal =
          isTrain && payload.forwardAfter ? payload.forwardAfter : payload.forward
        drawNetwork(ctx, {
          layout: layoutRef.current,
          weights: finalNet.weights,
          forward: forwardFinal,
          backward: null,
          signal: { forwardPhase: null, backwardPhase: null, t: 0, pulse: 0 },
        })
      }
    }

    rafRef.current = requestAnimationFrame(loop)
  }

  useEffect(() => {
    // Resize observer to keep the canvas responsive.
    const el = wrapRef.current
    if (!el) return
    const ro = new ResizeObserver(() => {
      const canvas = canvasRef.current
      if (!canvas) return
      const rect = el.getBoundingClientRect()
      const dpr = window.devicePixelRatio || 1
      dimsRef.current = { w: Math.floor(rect.width * dpr), h: Math.floor(rect.height * dpr), dpr }
      canvas.width = dimsRef.current.w
      canvas.height = dimsRef.current.h
      layoutRef.current = computeLayout(dimsRef.current.w, dimsRef.current.h, network)
      drawIdle()
    })

    ro.observe(el)
    return () => ro.disconnect()
  }, [network, activationType]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    // Re-draw idle whenever inputs/network change (if not animating).
    if (activeAnimRef.current?.active) return
    drawIdle()
  }, [network, inputValues, activationType, animation.id]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (animation?.mode === 'idle') return
    startAnimation()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [animation.id])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const handleMove = (e) => {
      const anim = animation
      const payloadNet = anim?.networkFrom || network
      const layout = layoutRef.current
      if (!layout) return
      const rect = canvas.getBoundingClientRect()
      const x = ((e.clientX - rect.left) / rect.width) * dimsRef.current.w
      const y = ((e.clientY - rect.top) / rect.height) * dimsRef.current.h

      // If not animating, use effectiveWeights from current network.
      const weights = payloadNet.weights
      const threshold = clamp(Math.min(dimsRef.current.w, dimsRef.current.h) * 0.014, 8, 18)

      let best = null
      let bestD = Infinity

      const check = (key, ax, ay, bx, by, value) => {
        const d = distPointToSegment(x, y, ax, ay, bx, by)
        if (d < threshold && d < bestD) {
          bestD = d
          best = { key, value }
        }
      }

      // Only compute hover weights for the "from" network to keep it stable.
      for (let j = 0; j < network.hidden1Count; j++) {
        for (let i = 0; i < network.inputCount; i++) {
          const a = layout.neuronPositions.input[i]
          const b = layout.neuronPositions.h1[j]
          check(
            `w1:${j}:${i}`,
            a.x,
            a.y,
            b.x,
            b.y,
            weights.w1[j][i],
          )
        }
      }
      for (let k = 0; k < network.hidden2Count; k++) {
        for (let j = 0; j < network.hidden1Count; j++) {
          const a = layout.neuronPositions.h1[j]
          const b = layout.neuronPositions.h2[k]
          check(
            `w2:${k}:${j}`,
            a.x,
            a.y,
            b.x,
            b.y,
            weights.w2[k][j],
          )
        }
      }
      for (let m = 0; m < network.outputCount; m++) {
        for (let k = 0; k < network.hidden2Count; k++) {
          const a = layout.neuronPositions.h2[k]
          const b = layout.neuronPositions.out[m]
          check(
            `w3:${m}:${k}`,
            a.x,
            a.y,
            b.x,
            b.y,
            weights.w3[m][k],
          )
        }
      }

      const key = best?.key || null
      if (key !== hoveredKeyRef.current) {
        hoveredKeyRef.current = key
        if (!best) {
          setTooltip(null)
          return
        }
        const rectW = canvas.getBoundingClientRect()
        setTooltip({
          x: e.clientX - rectW.left,
          y: e.clientY - rectW.top,
          text: `weight ${best.key.replaceAll(':', ', ') } = ${best.value.toFixed(4)}`,
        })
      }
    }

    const onLeave = () => {
      hoveredKeyRef.current = null
      setTooltip(null)
    }

    canvas.addEventListener('mousemove', handleMove)
    canvas.addEventListener('mouseleave', onLeave)
    return () => {
      canvas.removeEventListener('mousemove', handleMove)
      canvas.removeEventListener('mouseleave', onLeave)
    }
  }, [animation, network]) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div ref={wrapRef} style={{ width: '100%', height: '100%', position: 'relative' }}>
      <div className="legend">
        <div className="legendItem">
          <span className="dot" />
          Positive weights
        </div>
        <div className="legendItem">
          <span className="dot red" />
          Negative weights
        </div>
        <div className="legendItem">
          <span
            className="dot"
            style={{ background: baseColors.backwardPos, boxShadow: '0 0 22px rgba(34,197,94,0.55)' }}
          />
          Backprop error signals
        </div>
      </div>

      {tooltip ? <div className="tooltip" style={{ left: tooltip.x, top: tooltip.y }}>{tooltip.text}</div> : null}

      <canvas ref={canvasRef} />
    </div>
  )
}


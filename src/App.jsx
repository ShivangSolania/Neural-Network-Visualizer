import { useEffect, useMemo, useRef, useState } from 'react'
import ControlsPanel from './components/ControlsPanel.jsx'
import CanvasVisualizer from './components/CanvasVisualizer.jsx'
import InfoPanel from './components/InfoPanel.jsx'

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

function relu(x) {
  return Math.max(0, x)
}

const clamp = (v, min, max) => Math.min(max, Math.max(min, v))

function sigmoid(x) {
  // Keep it stable enough for visualization.
  const z = clamp(x, -60, 60)
  return 1 / (1 + Math.exp(-z))
}

function activationFn(type, x) {
  return type === 'relu' ? relu(x) : sigmoid(x)
}

function randWeight(scale = 0.8) {
  return (Math.random() * 2 - 1) * scale
}

// --- Core required functions (named as requested) ---

function initNetwork(inputCount) {
  const hidden1Count = Math.max(2, Math.min(8, inputCount + 2))
  const hidden2Count = Math.max(2, Math.min(7, Math.floor(hidden1Count * 0.75) + 1))
  const outputCount = 1

  // Fully connected weights:
  // w1: hidden1 x input, w2: hidden2 x hidden1, w3: output x hidden2
  const w1 = Array.from({ length: hidden1Count }, () =>
    Array.from({ length: inputCount }, () => randWeight(0.9)),
  )
  const w2 = Array.from({ length: hidden2Count }, () =>
    Array.from({ length: hidden1Count }, () => randWeight(0.8)),
  )
  const w3 = Array.from({ length: outputCount }, () =>
    Array.from({ length: hidden2Count }, () => randWeight(0.7)),
  )

  const b1 = Array.from({ length: hidden1Count }, () => randWeight(0.4))
  const b2 = Array.from({ length: hidden2Count }, () => randWeight(0.35))
  const b3 = Array.from({ length: outputCount }, () => randWeight(0.25))

  return {
    inputCount,
    hidden1Count,
    hidden2Count,
    outputCount,
    weights: { w1, w2, w3 },
    biases: { b1, b2, b3 },
  }
}

function forwardPass(network, inputValues, activationType) {
  const { hidden1Count, hidden2Count, outputCount, weights, biases } = network

  const a0 = [...inputValues]
  const z1 = Array.from({ length: hidden1Count }, (_, j) => {
    let sum = biases.b1[j]
    for (let i = 0; i < network.inputCount; i++) sum += weights.w1[j][i] * a0[i]
    return sum
  })
  const a1 = z1.map((z) => activationFn(activationType, z))

  const z2 = Array.from({ length: hidden2Count }, (_, k) => {
    let sum = biases.b2[k]
    for (let j = 0; j < hidden1Count; j++) sum += weights.w2[k][j] * a1[j]
    return sum
  })
  const a2 = z2.map((z) => activationFn(activationType, z))

  const z3 = Array.from({ length: outputCount }, (__, m) => {
    let sum = biases.b3[m]
    for (let k = 0; k < hidden2Count; k++) sum += weights.w3[m][k] * a2[k]
    return sum
  })
  const a3 = z3.map((z) => activationFn(activationType, z))

  return {
    input: a0,
    z1,
    a1,
    z2,
    a2,
    z3,
    a3,
    output: a3[0],
  }
}

function calculateError(target, output) {
  // error = target - output (per spec)
  return target - output
}

function backpropagate(network, forwardResult, target) {
  // Spec-aligned error: error = target - output
  const { hidden1Count, hidden2Count, outputCount, weights } = network
  const { a3 } = forwardResult

  const output = a3[0]
  const error = calculateError(target, output)

  // For visualization: propagate a pseudo-error backward through the weight signs/magnitudes.
  const outputDelta = Array.from({ length: outputCount }, () => error)
  const hidden2Delta = Array.from({ length: hidden2Count }, (_, k) => error * weights.w3[0][k])
  const hidden1Delta = Array.from({ length: hidden1Count }, (_, j) => {
    let sum = 0
    for (let k = 0; k < hidden2Count; k++) {
      sum += hidden2Delta[k] * weights.w2[k][j]
    }
    return sum
  })

  return { error, outputDelta, hidden2Delta, hidden1Delta }
}

function updateWeights(network, forwardResult, deltas, learningRate) {
  // Spec-aligned simplified rule:
  // weight += learning_rate * error * input_value
  // where input_value is the *source neuron* activation for that connection.
  const { inputCount, hidden1Count, hidden2Count, outputCount } = network
  const { input, a1, a2 } = forwardResult
  const { error } = deltas

  const dw1 = Array.from({ length: hidden1Count }, () =>
    Array.from({ length: inputCount }, () => 0),
  )
  const dw2 = Array.from({ length: hidden2Count }, () =>
    Array.from({ length: hidden1Count }, () => 0),
  )
  const dw3 = Array.from({ length: outputCount }, () =>
    Array.from({ length: hidden2Count }, () => 0),
  )

  // Create new weights (do not mutate the input network).
  const w1 = network.weights.w1.map((row, j) =>
    row.map((w, i) => {
      const delta = learningRate * error * input[i]
      dw1[j][i] = delta
      return w + delta
    }),
  )

  const w2 = network.weights.w2.map((row, k) =>
    row.map((w, j) => {
      const delta = learningRate * error * a1[j]
      dw2[k][j] = delta
      return w + delta
    }),
  )

  const w3 = network.weights.w3.map((row, m) =>
    row.map((w, k) => {
      const delta = learningRate * error * a2[k]
      dw3[m][k] = delta
      return w + delta
    }),
  )

  const b1 = network.biases.b1.map((b) => b + learningRate * error)
  const b2 = network.biases.b2.map((b) => b + learningRate * error)
  const b3 = network.biases.b3.map((b) => b + learningRate * error)

  return {
    networkTo: {
      ...network,
      weights: { w1, w2, w3 },
      biases: { b1, b2, b3 },
    },
    weightDeltas: { dw1, dw2, dw3 },
  }
}

export default function App() {
  const [inputCount, setInputCount] = useState(3)
  const [inputValues, setInputValues] = useState([0.2, -0.1, 0.4])

  const [activationType, setActivationType] = useState('sigmoid') // 'relu'|'sigmoid'
  const [learningRate, setLearningRate] = useState(0.35)
  const [targetOutput, setTargetOutput] = useState(1)

  const [maxIterations, setMaxIterations] = useState(25)
  const [speed, setSpeed] = useState(1) // animation speed multiplier

  const [network, setNetwork] = useState(() => initNetwork(3))
  const [lastForward, setLastForward] = useState(null)
  const [lastError, setLastError] = useState(0)
  const [lastLoss, setLastLoss] = useState(0)
  const [iterationCount, setIterationCount] = useState(0)
  const [lossHistory, setLossHistory] = useState([])

  const [isTraining, setIsTraining] = useState(false)
  const [isPaused, setIsPaused] = useState(false)

  // Canvas animation request payload.
  const [animation, setAnimation] = useState({ id: 0, mode: 'idle' })
  const animationIdRef = useRef(0)
  const trainingAbortRef = useRef(false)
  const isPausedRef = useRef(false)
  const stepOnceRef = useRef(false)

  const timing = useMemo(() => {
    // Keep these relatively short to make training feel responsive.
    const forwardMs = 900 / speed
    const backwardMs = 1050 / speed
    const totalMs = forwardMs + backwardMs + 60
    return { forwardMs, backwardMs, totalMs }
  }, [speed])

  const hiddenSizesLabel = useMemo(() => {
    return `H1:${network.hidden1Count} · H2:${network.hidden2Count}`
  }, [network.hidden1Count, network.hidden2Count])

  const resetAll = () => {
    trainingAbortRef.current = true
    setIsTraining(false)
    setIsPaused(false)
    isPausedRef.current = false
    stepOnceRef.current = false
    setIterationCount(0)
    setLossHistory([])
    setLastForward(null)
    setLastError(0)
    setLastLoss(0)

    const nextNet = initNetwork(inputCount)
    setNetwork(nextNet)

    const nextInputs = Array.from({ length: inputCount }, (_, i) => {
      const existing = inputValues[i]
      if (Number.isFinite(existing)) return existing
      return randWeight(0.2)
    })
    setInputValues(nextInputs)
  }

  const requestAnim = (payload) => {
    animationIdRef.current += 1
    setAnimation({ id: animationIdRef.current, ...payload })
  }

  const handleInputCountChange = (nextCount) => {
    trainingAbortRef.current = true
    setIsTraining(false)
    setIsPaused(false)
    isPausedRef.current = false
    setIterationCount(0)
    setLossHistory([])
    setLastForward(null)

    setInputCount(nextCount)
    const nextInputs = Array.from({ length: nextCount }, (_, i) => {
      const v = inputValues[i]
      return Number.isFinite(v) ? v : randWeight(0.25)
    })
    setInputValues(nextInputs)
    setNetwork(initNetwork(nextCount))
  }

  const handleRun = () => {
    trainingAbortRef.current = false
    const forward = forwardPass(network, inputValues, activationType)
    const error = calculateError(targetOutput, forward.output)
    const loss = (forward.output - targetOutput) * (forward.output - targetOutput)

    setLastForward(forward)
    setLastError(error)
    setLastLoss(loss)

    // Forward-only animation, no weights change.
    requestAnim({
      mode: 'run',
      networkFrom: network,
      networkTo: network,
      forward,
      backward: null,
      weightDeltas: null,
    })
  }

  const runTrainStep = (iterationIndex) => {
    const networkFrom = network
    const forwardBefore = forwardPass(networkFrom, inputValues, activationType)
    const deltas = backpropagate(networkFrom, forwardBefore, targetOutput)
    const { networkTo, weightDeltas } = updateWeights(
      networkFrom,
      forwardBefore,
      deltas,
      learningRate,
    )

    const forwardAfter = forwardPass(networkTo, inputValues, activationType)
    const lossAfter =
      (forwardAfter.output - targetOutput) * (forwardAfter.output - targetOutput)

    setLastForward(forwardAfter)
    setLastError(calculateError(targetOutput, forwardAfter.output))
    setLastLoss(lossAfter)
    setIterationCount(iterationIndex + 1)
    setLossHistory((prev) => {
      const next = [...prev, lossAfter]
      // Keep it manageable for the graph.
      return next.slice(-160)
    })

    // Trigger the animation from "before weights" to "after weights".
    requestAnim({
      mode: 'train',
      networkFrom,
      networkTo,
      forward: forwardBefore,
      forwardAfter,
      backward: deltas,
      weightDeltas,
    })

    setNetwork(networkTo)
  }

  const handleTrain = async () => {
    if (isTraining) return
    trainingAbortRef.current = false
    setIsTraining(true)
    setIsPaused(false)
    isPausedRef.current = false
    stepOnceRef.current = false

    // Snapshot starting iteration to keep UI consistent.
    const startIter = iterationCount
    for (let i = 0; i < maxIterations; i++) {
      if (trainingAbortRef.current) break
      while (isPausedRef.current) {
        // Check frequently so the pause feels responsive.
        // eslint-disable-next-line no-await-in-loop
        await sleep(80)
        if (trainingAbortRef.current) break
        if (stepOnceRef.current) break
      }
      if (trainingAbortRef.current) break

      const isStep = stepOnceRef.current
      stepOnceRef.current = false

      const currentIndex = startIter + i
      runTrainStep(currentIndex)
      // eslint-disable-next-line no-await-in-loop
      await sleep(timing.totalMs)
    }

    setIsTraining(false)
    setIsPaused(false)
    isPausedRef.current = false
  }

  const handlePause = () => {
    if (!isTraining) return
    setIsPaused(true)
    isPausedRef.current = true
    // Keep step mode explicit; don't carry over an old step request.
    stepOnceRef.current = false
  }

  const handleResume = () => {
    if (!isTraining) return
    setIsPaused(false)
    isPausedRef.current = false
    stepOnceRef.current = false
  }

  const canPause = isTraining && !isPaused
  const canResume = isTraining && isPaused
  const canStep = isTraining && isPaused

  const handleStep = () => {
    if (!isTraining) return
    if (!isPausedRef.current) return
    stepOnceRef.current = true
  }

  // Keep the right panel (output/error/loss) in sync with the current inputs + weights.
  // During training we intentionally let the training loop drive these values to match each iteration.
  useEffect(() => {
    if (isTraining) return
    const forward = forwardPass(network, inputValues, activationType)
    const error = calculateError(targetOutput, forward.output)
    const loss = (forward.output - targetOutput) * (forward.output - targetOutput)
    setLastForward(forward)
    setLastError(error)
    setLastLoss(loss)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isTraining, network, inputValues, activationType, targetOutput])

  return (
    <div className="appShell">
      <ControlsPanel
        inputCount={inputCount}
        onInputCountChange={handleInputCountChange}
        inputValues={inputValues}
        onInputValueChange={(idx, v) => {
          setInputValues((prev) => {
            const next = [...prev]
            next[idx] = v
            return next
          })
        }}
        targetOutput={targetOutput}
        onTargetOutputChange={setTargetOutput}
        learningRate={learningRate}
        onLearningRateChange={setLearningRate}
        activationType={activationType}
        onActivationChange={setActivationType}
        maxIterations={maxIterations}
        onMaxIterationsChange={setMaxIterations}
        speed={speed}
        onSpeedChange={setSpeed}
        onRun={handleRun}
        onTrain={handleTrain}
        onPause={handlePause}
        onResume={handleResume}
        onStep={handleStep}
        isTraining={isTraining}
        isPaused={isPaused}
        hiddenSizesLabel={hiddenSizesLabel}
        onReset={resetAll}
        canPause={canPause}
        canResume={canResume}
        canStep={canStep}
      />

      <div className="panel">
        <div className="panelHeader">
          <div>
            <div className="panelTitle">Forward + Backprop Visualizer</div>
            <div className="panelSub">Forward: left → right · Backprop: right → left</div>
          </div>
          <div className="panelSub">
            {activationType === 'relu' ? 'ReLU' : 'Sigmoid'} activations
          </div>
        </div>
        <div className="panelBody canvasWrap">
          <div className="canvasCard">
            <CanvasVisualizer
              network={network}
              activationType={activationType}
              inputValues={inputValues}
              targetOutput={targetOutput}
              forwardResult={lastForward}
              animation={animation}
              timing={timing}
            />
          </div>
        </div>
      </div>

      <InfoPanel
        outputValue={lastForward?.output ?? null}
        errorValue={lastError}
        lossValue={lastLoss}
        iterationCount={iterationCount}
        lossHistory={lossHistory}
        hiddenSizesLabel={hiddenSizesLabel}
      />
    </div>
  )
}

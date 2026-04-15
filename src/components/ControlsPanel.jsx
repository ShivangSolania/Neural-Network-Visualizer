function NumberInput({ value, onChange, step = 0.01 }) {
  return (
    <input
      type="number"
      value={Number.isFinite(value) ? value : ''}
      step={step}
      onChange={(e) => onChange(e.target.value === '' ? 0 : Number(e.target.value))}
    />
  )
}

export default function ControlsPanel(props) {
  const {
    inputCount,
    onInputCountChange,
    inputValues,
    onInputValueChange,
    targetOutput,
    onTargetOutputChange,
    learningRate,
    onLearningRateChange,
    activationType,
    onActivationChange,
    maxIterations,
    onMaxIterationsChange,
    speed,
    onSpeedChange,
    onRun,
    onTrain,
    onPause,
    onResume,
    onStep,
    isTraining,
    isPaused,
    hiddenSizesLabel,
    onReset,
    canPause,
    canResume,
    canStep,
  } = props

  return (
    <aside className="panel">
      <div className="panelHeader">
        <div>
          <div className="panelTitle">Controls</div>
          <div className="panelSub">{hiddenSizesLabel}</div>
        </div>
        <div className="panelSub">{isTraining ? (isPaused ? 'Paused' : 'Training') : 'Ready'}</div>
      </div>
      <div className="panelBody">
        <div className="group">
          <div className="labelRow">
            <div className="label">Input neurons</div>
            <div className="valuePill">{inputCount}</div>
          </div>
          <input
            type="range"
            min={2}
            max={8}
            step={1}
            value={inputCount}
            onChange={(e) => onInputCountChange(Number(e.target.value))}
          />
          <div className="smallNote">
            Change input size anytime. The network re-initializes so you can explore how
            structure affects learning.
          </div>
        </div>

        <div className="group">
          <div className="labelRow">
            <div className="label">Input values</div>
            <div className="valuePill">{inputValues.length} values</div>
          </div>
          <div className="inputsList">
            {inputValues.map((v, idx) => (
              <div className="row" key={idx}>
                <div className="valuePill" style={{ width: 54, textAlign: 'center' }}>
                  x{idx}
                </div>
                <NumberInput
                  value={v}
                  step={0.05}
                  onChange={(nv) => onInputValueChange(idx, nv)}
                />
              </div>
            ))}
          </div>
        </div>

        <div className="group">
          <div className="labelRow">
            <div className="label">Target output</div>
            <div className="valuePill">{targetOutput.toFixed(3)}</div>
          </div>
          <NumberInput value={targetOutput} step={0.05} onChange={onTargetOutputChange} />
          <div className="smallNote">
            The app uses a simplified error: <span className="kbd">error = target - output</span>.
            Training nudges weights so the output moves toward your target.
          </div>
        </div>

        <div className="grid2">
          <div className="group">
            <div className="labelRow">
              <div className="label">Learning rate</div>
              <div className="valuePill">{learningRate.toFixed(3)}</div>
            </div>
            <input
              type="range"
              min={0.01}
              max={1.0}
              step={0.01}
              value={learningRate}
              onChange={(e) => onLearningRateChange(Number(e.target.value))}
            />
          </div>

          <div className="group">
            <div className="labelRow">
              <div className="label">Activation</div>
              <div className="valuePill">{activationType === 'relu' ? 'ReLU' : 'Sigmoid'}</div>
            </div>
            <select value={activationType} onChange={(e) => onActivationChange(e.target.value)}>
              <option value="sigmoid">Sigmoid</option>
              <option value="relu">ReLU</option>
            </select>
          </div>
        </div>

        <div className="grid2">
          <div className="group">
            <div className="labelRow">
              <div className="label">Iterations</div>
              <div className="valuePill">{maxIterations}</div>
            </div>
            <input
              type="range"
              min={1}
              max={80}
              step={1}
              value={maxIterations}
              onChange={(e) => onMaxIterationsChange(Number(e.target.value))}
            />
          </div>

          <div className="group">
            <div className="labelRow">
              <div className="label">Animation speed</div>
              <div className="valuePill">{speed.toFixed(2)}x</div>
            </div>
            <input
              type="range"
              min={0.6}
              max={1.8}
              step={0.05}
              value={speed}
              onChange={(e) => onSpeedChange(Number(e.target.value))}
            />
          </div>
        </div>

        <div className="btnRow">
          <button className="btnPrimary" onClick={onRun} disabled={isTraining}>
            Run (forward)
          </button>
          <button className="btnGreen" onClick={onTrain} disabled={isTraining}>
            Train
          </button>
          <button className="btnDanger" onClick={onPause} disabled={!canPause}>
            Pause
          </button>
          <button onClick={onResume} disabled={!canResume}>
            Resume
          </button>
          <button onClick={onStep} disabled={!canStep}>
            Step (1 iter)
          </button>
          <button onClick={onReset} disabled={isTraining}>
            Reset network
          </button>
        </div>
      </div>
    </aside>
  )
}


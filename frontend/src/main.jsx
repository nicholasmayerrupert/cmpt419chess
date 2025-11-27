import React, { useEffect, useMemo, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { Chessboard } from 'react-chessboard'

const API = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

/* ---------------- Backend bridge ---------------- */

function useChessState() {
  const [state, setState] = useState(null)
  const [error, setError] = useState(null)

  const fetchState = async () => {
    try {
      const res = await fetch(`${API}/state`)
      const data = await res.json()
      setState(data); setError(null)
    } catch (e) {
      setError(String(e))
    }
  }

  useEffect(() => { fetchState() }, [])

  const newGame = async () => {
    try {
      const res = await fetch(`${API}/new_game`, { method: 'POST' })
      const data = await res.json()
      setState(data); setError(null)
    } catch (e) { setError(String(e)) }
  }

  const undo = async () => {
    try {
      const res = await fetch(`${API}/undo`, { method: 'POST' })
      const data = await res.json()
      setState(data)
    } catch (e) { setError(String(e)) }
  }

  const redo = async () => {
    try {
      const res = await fetch(`${API}/redo`, { method: 'POST' })
      const data = await res.json()
      setState(data)
    } catch (e) { setError(String(e)) }
  }

  const makeMove = async (uci) => {
    try {
      const ply = Array.isArray(state?.history_san) ? state.history_san.length : 0
      const req_id = (crypto?.randomUUID?.() ?? `${Date.now()}:${Math.random()}`)

      const res = await fetch(`${API}/move`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ uci: uci, ply, req_id })
      })
      const txt = await res.text()
      if (!res.ok) {
        let detail = 'Move failed'
        try { detail = JSON.parse(txt)?.detail ?? detail } catch {}
        throw new Error(detail)
      }
      const data = JSON.parse(txt)
      setState(data)
      return true
    } catch (e) {
      setError(String(e))
      return false
    }
  }

  return { state, error, newGame, undo, redo, makeMove, setState, fetchState }
}

/* ---------------- Eval helpers/bar (unchanged from your good version) ---------------- */

function clampCp(cp) { const v = Number(cp ?? 0); return Math.max(-800, Math.min(800, v)) }
function cpToPct(cpBottom) { const cpC = clampCp(cpBottom); return 1 / (1 + Math.pow(10, -cpC / 400)) }
function cloneScore(s){ if(!s) return null; if('mate'in s && s.mate!=null) return {mate:+s.mate}; if('cp'in s&&s.cp!=null) return {cp:+s.cp}; return {cp:0} }
function invertScore(s){ if(!s) return null; if('mate'in s && s.mate!=null) return {mate:-(+s.mate)}; if('cp'in s&&s.cp!=null) return {cp:-(+s.cp)}; return {cp:0} }
function whiteToBottom(scoreWhite, orientation){ return !scoreWhite?null : (orientation==='white'? cloneScore(scoreWhite): invertScore(scoreWhite)) }
function stmToBottom(scoreSTM, evalTurn, orientation){ if(!scoreSTM) return null; const whitePOV = evalTurn==='w'? cloneScore(scoreSTM): invertScore(scoreSTM); return whiteToBottom(whitePOV, orientation) }
function scoreToCpNumber(score){ if(!score) return 0; if('mate'in score && score.mate!=null){ const sign = score.mate>=0?1:-1; return sign*2000 } return +(score.cp||0) }
function formatLabel(score){ if(!score) return '—'; if('mate'in score && score.mate!=null) return score.mate>0?`#${score.mate}`:`#${-score.mate}`; if('cp'in score && score.cp!=null) return (score.cp/100).toFixed(2); return '—' }

function EvalBar({ evalData, height = 560, durationMs = 400, orientation = 'white' }) {
  const lastBottomScoreRef = useRef(null)
  const lastBottomCpRef = useRef(null)
  const [pct, setPct] = useState(0.5)

  const effectiveBottomScore = React.useMemo(() => {
    if (!evalData || evalData.pending || !evalData.score) return lastBottomScoreRef.current
    const raw = cloneScore(evalData.score)
    const evalTurn = evalData.turn || 'w'
    const candA = whiteToBottom(raw, orientation)
    const candB = stmToBottom(raw, evalTurn, orientation)
    const prev = lastBottomCpRef.current
    if (prev == null) {
      lastBottomScoreRef.current = candA
      lastBottomCpRef.current = scoreToCpNumber(candA)
      return candA
    }
    const cpA = scoreToCpNumber(candA), cpB = scoreToCpNumber(candB)
    const chosen = Math.abs(cpA - prev) <= Math.abs(cpB - prev) ? candA : candB
    lastBottomScoreRef.current = chosen
    lastBottomCpRef.current = scoreToCpNumber(chosen)
    return chosen
  }, [evalData?.score, evalData?.pending, evalData?.turn, orientation])

  useEffect(() => {
    if (effectiveBottomScore == null) return
    const target = cpToPct(scoreToCpNumber(effectiveBottomScore))
    setPct(target)
  }, [effectiveBottomScore])

  const bottomHeightPct = Math.round(pct * 100)
  const bottomColor = orientation === 'white' ? '#e6e6e6' : '#0c0c0c'
  const topColor    = orientation === 'white' ? '#0c0c0c' : '#e6e6e6'
  const label = formatLabel(effectiveBottomScore)

  return (
    <div style={{ display: 'grid', gap: 8, alignContent: 'start', justifyItems: 'center', width: 40 }}>
      <div style={{ width: 18, height, position: 'relative', borderRadius: 9, overflow: 'hidden', border: '1px solid #2a2f45', background: topColor }}>
        <div style={{ position: 'absolute', left: 0, right: 0, bottom: 0, height: `${bottomHeightPct}%`, background: bottomColor, transition: `height ${durationMs}ms ease`, willChange: 'height' }}/>
      </div>
      <div style={{ fontSize: 12, opacity: 0.9 }}><strong>{label ?? '—'}</strong></div>
    </div>
  )
}

/* ---------------- Utilities ---------------- */

function useDebouncedCallback(cb, delayMs) {
  const timer = useRef(null)
  return (...args) => {
    if (timer.current) clearTimeout(timer.current)
    timer.current = setTimeout(() => cb(...args), delayMs)
  }
}
const LIGHT = '#B7C6D9'
const DARK  = '#355070'

/* ---------------- App ---------------- */

function App() {
  const { state, error, newGame, undo, redo, makeMove } = useChessState()

  // click-to-move & highlights
  const [selectedSquare, setSelectedSquare] = useState(null)
  const [moveSquares, setMoveSquares] = useState({})

  // eval stuff
  const [evalData, setEvalData] = useState(null)
  const [lastEvalError, setLastEvalError] = useState(null)

  const [depth, setDepth] = useState(() => {
    const saved = Number(localStorage.getItem('engineDepth'))
    return Number.isFinite(saved) && saved >= 4 && saved <= 30 ? saved : 12
  })
  const [engineEnabled, setEngineEnabled] = useState(() => {
    const s = localStorage.getItem('engineEnabled')
    return s === null ? true : s === 'true'
  })
  const [chatMessages, setChatMessages] = useState([])
  const [chatInput, setChatInput] = useState('')
  const [coachLoading, setCoachLoading] = useState(false)
  const [coachError, setCoachError] = useState(null)
  const chatEndRef = useRef(null)

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [chatMessages])

  const BOARD_PX = 560
  const position = useMemo(() => state?.fen ?? undefined, [state])

  const [orientation, setOrientation] = useState(() => {
    return localStorage.getItem('orientation') === 'black' ? 'black' : 'white'
  })
  const flipBoard = () => {
    const next = orientation === 'white' ? 'black' : 'white'
    setOrientation(next)
    localStorage.setItem('orientation', next)
    setEvalData(ed => ed ? { ...ed, pending: true } : ed)
  }

  useEffect(() => { localStorage.setItem('engineEnabled', String(engineEnabled)) }, [engineEnabled])
  useEffect(() => { localStorage.setItem('engineDepth', String(depth)) }, [depth])

  const evalAbortRef = useRef(null)
  const evalReqIdRef = useRef(0)
  const coachReqIdRef = useRef(0)
  const AUTO_PROMPT = "Give a concise update on the plans, threats, and best moves after that last move."

  // Warm engine
  useEffect(() => {
    if (!engineEnabled) return
    ;(async () => {
      try {
        await fetch(`${API}/engine/status?start=1`).catch(()=>{})
        await fetch(`${API}/eval?depth=${Math.max(6, Math.min(10, depth))}`).catch(()=>{})
      } catch {}
    })()
  }, [engineEnabled])

  // On FEN change: mark pending, keep old score
  useEffect(() => {
    setSelectedSquare(null); setMoveSquares({})
    if (engineEnabled) {
      setEvalData(ed => ed ? { ...ed, pending: true } : ed)
      fetchEval(depth, 'fen-change')
    } else {
      setEvalData(null)
    }
  }, [state?.fen, engineEnabled])

  // Debounced eval on depth changes
  const debouncedEval = useDebouncedCallback((d) => {
    if (engineEnabled) {
      setEvalData(ed => ed ? { ...ed, pending: true } : ed)
      fetchEval(d, 'depth-change')
    }
  }, 250)
  useEffect(() => { debouncedEval(depth) }, [depth, engineEnabled])

  useEffect(() => {
    if (!engineEnabled) return
    const id = setInterval(() => fetchEval(depth, 'refresh'), 8000)
    return () => clearInterval(id)
  }, [engineEnabled, depth, state?.fen])



  const styleSelected = {
    outline: '3px solid rgba(255, 223, 88, 0.9)',
    boxShadow: 'inset 0 0 0 3px rgba(0,0,0,0.35), 0 0 12px rgba(255,223,88,0.6)',
    borderRadius: 4,
  }
  const styleTarget = {
    background: 'radial-gradient(circle at center, rgba(0,0,0,0.55) 0%, rgba(0,0,0,0.45) 40%, rgba(0,0,0,0.0) 41%)',
  }
  const styleBest = {
    outline: '3px solid rgba(88, 255, 176, 0.95)',
    boxShadow: 'inset 0 0 0 3px rgba(0,0,0,0.35), 0 0 12px rgba(88,255,176,0.6)',
    borderRadius: 4,
  }

  const computeMoveSquares = (sourceSquare) => {
    if (!state?.legal_moves) return {}
    const styles = {}
    for (const uci of state.legal_moves) {
      if (uci.slice(0, 2) === sourceSquare) {
        styles[uci.slice(2, 4)] = styleTarget
      }
    }
    return styles
  }

  const isLegalPrefix = (from, to) => {
    const prefix = from + to
    return !!state?.legal_moves?.some(m => m.startsWith(prefix))
  }
  const isLegalUci = (uci) => !!state?.legal_moves?.includes(uci)
  const hasPromotion = (prefix) => state?.legal_moves?.some(m => m.startsWith(prefix) && m.length === 5)

  /* ---------- Click-to-move ---------- */

  const onSquareClick = async (square) => {
    if (!selectedSquare) {
      const targets = computeMoveSquares(square)
      if (Object.keys(targets).length > 0) {
        setSelectedSquare(square)
        setMoveSquares({ ...targets, [square]: styleSelected })
      } else {
        setSelectedSquare(null); setMoveSquares({})
      }
      return
    }
    if (selectedSquare === square) {
      setSelectedSquare(null); setMoveSquares({})
      return
    }
    if (moveSquares[square]) {
      const from = selectedSquare
      const to = square
      const prefix = from + to
      let uci = prefix
      if (hasPromotion(prefix)) {
        uci = prefix + 'q'
      }
      if (!isLegalUci(uci)) return
      const ok = await makeMove(uci)
      if (ok) {
        setSelectedSquare(null); setMoveSquares({})
        triggerAutoCoach()
      }
      return
    }

    const targets = computeMoveSquares(square)
    if (Object.keys(targets).length > 0) {
      setSelectedSquare(square)
      setMoveSquares({ ...targets, [square]: styleSelected })
    } else {
      setSelectedSquare(null); setMoveSquares({})
    }
  }

  /* ---------- Drag & drop ---------- */

  const onPieceDragBegin = (_piece, sourceSquare) => {
    const targets = computeMoveSquares(sourceSquare)
    if (Object.keys(targets).length > 0) {
      setSelectedSquare(sourceSquare)
      setMoveSquares({ ...targets, [sourceSquare]: styleSelected })
    }
  }
  const onPieceDragEnd = () => { setSelectedSquare(null); setMoveSquares({}) }

  const onPieceDrop = async (sourceSquare, targetSquare, _piece) => {
    const prefix = sourceSquare + targetSquare

    if (!isLegalPrefix(sourceSquare, targetSquare)) {
      return false // snap back
    }

    let uci = prefix
    if (hasPromotion(prefix)) {
      uci = prefix + 'q'
    }
    if (!isLegalUci(uci)) return false

    const ok = await makeMove(uci)
    setSelectedSquare(null); setMoveSquares({})
    if (ok) triggerAutoCoach()
    return ok
  }

  /* ---------- Eval fetch ---------- */

  const fetchEval = async (d = depth, _reason = '') => {
    try {
      if (evalAbortRef.current) evalAbortRef.current.abort()
      const controller = new AbortController()
      evalAbortRef.current = controller

      const reqId = ++evalReqIdRef.current
      const res = await fetch(`${API}/eval?depth=${d}`, { signal: controller.signal })
      const raw = await res.text()
      if (reqId !== evalReqIdRef.current) return
      let data = null
      try { data = JSON.parse(raw) } catch { throw new Error(raw.slice(0, 200) || 'Invalid JSON from /eval') }
      if (!res.ok) throw new Error(data?.detail || res.statusText)

      setEvalData({ ...data, pending: false })
      setLastEvalError(null)

      if (data?.bestmove?.uci) {
        const u = data.bestmove.uci
        const from = u.slice(0,2), to = u.slice(2,4)
        setMoveSquares(ms => ({...ms, [from]: styleBest, [to]: styleBest}))
      }
    } catch (e) {
      if (e?.name === 'AbortError') return
      setEvalData(ed => ed ? { ...ed, pending: false } : ed)
      setLastEvalError(String(e?.message || e))
      console.error('Eval failed:', e)
    }
  }

  const requestCoachAnalysis = async (prompt, { clearInput = false, prefix = '' } = {}) => {
    const trimmed = (prompt || '').trim()
    if (!trimmed) return

    if (clearInput) setChatInput('')
    setCoachLoading(true)
    setCoachError(null)

    const displayContent = prefix ? `${prefix}${trimmed}` : trimmed
    const userMsg = { role: 'user', content: displayContent }
    const assistantPlaceholder = { role: 'assistant', content: '' }

    let payloadHistory = []
    setChatMessages(prev => {
      payloadHistory = [...prev, userMsg].map(m => ({ role: m.role, content: m.content }))
      return [...prev, userMsg, assistantPlaceholder]
    })

    const payload = { question: trimmed, history: payloadHistory }
    const reqId = ++coachReqIdRef.current

    try {
      const res = await fetch(`${API}/coach/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })

      if (!res.ok) throw new Error(res.statusText || 'Request failed')
      if (!res.body) throw new Error('No response body')

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let done = false

      while (!done) {
        const { value, done: doneReading } = await reader.read()
        done = doneReading
        if (!value) continue
        const chunk = decoder.decode(value, { stream: true })

        if (reqId !== coachReqIdRef.current) return

        setChatMessages(prev => {
          const last = prev[prev.length - 1]
          if (!last || last.role !== 'assistant') return prev
          const updatedMsg = { ...last, content: last.content + chunk }
          return [...prev.slice(0, -1), updatedMsg]
        })
      }
    } catch (e) {
      if (reqId === coachReqIdRef.current) setCoachError(String(e?.message || e))
      setChatMessages(prev => {
        const last = prev[prev.length - 1]
        if (last?.role === 'assistant' && !last.content) {
          return prev.slice(0, -1)
        }
        return prev
      })
    } finally {
      if (reqId === coachReqIdRef.current) setCoachLoading(false)
    }
  }

  const sendChat = async () => {
    await requestCoachAnalysis(chatInput, { clearInput: true })
  }

  const triggerAutoCoach = () => {
    if (coachLoading) return
    requestCoachAnalysis(AUTO_PROMPT, { prefix: 'Auto · ' })
  }

  if (!state) {
    return (
      <div className="app">
        <header><h1>Chess Review</h1></header>
        <div className="container">
          <div className="panel">Loading…</div>
        </div>
      </div>
    )
  }

  return (
    <div className="app">
      <header>
        <h1>Chess!</h1>
        <div className="controls" style={{alignItems:'center', gap:12}}>
          <button onClick={flipBoard}>Flip Board</button>
          <button onClick={newGame}>New Game</button>
          <button disabled={!state.can_undo} onClick={async ()=>{await undo();}}>Undo</button>
          <button disabled={!state.can_redo} onClick={async ()=>{await redo();}}>Redo</button>

          <div style={{display:'flex', alignItems:'center', gap:12, marginLeft:8}}>
            <label htmlFor="depth" style={{fontSize:12, opacity:.9}}>Depth</label>
            <input
              id="depth"
              type="range"
              min={4}
              max={24}
              step={1}
              value={depth}
              onChange={(e)=> setDepth(Number(e.target.value))}
              style={{width:160}}
            />
            <div style={{width:28, textAlign:'right', fontVariantNumeric:'tabular-nums'}}>{depth}</div>

            <label style={{display:'flex', alignItems:'center', gap:6, fontSize:12}}>
              <input
                type="checkbox"
                checked={engineEnabled}
                onChange={(e)=> setEngineEnabled(e.target.checked)}
              />
              Engine On
            </label>
          </div>
        </div>
      </header>

      <div className="container">
        <div className="panel" style={{display:'flex', gap:16, alignItems:'center', justifyContent:'center', position:'relative', width: 560 + 56 }}>
          <EvalBar evalData={evalData} height={560} orientation={orientation}/>
          <Chessboard
            id="board"
            position={position}
            boardOrientation={orientation}
            arePiecesDraggable={true}
            animationDuration={200}
            onPieceDrop={onPieceDrop}
            onSquareClick={onSquareClick}
            onPieceDragBegin={onPieceDragBegin}
            onPieceDragEnd={onPieceDragEnd}
            customSquareStyles={moveSquares}
            boardWidth={560}
            customDarkSquareStyle={{ backgroundColor: DARK }}
            customLightSquareStyle={{ backgroundColor: LIGHT }}
          />
        </div>

        <div style={{display:'flex', flexDirection:'column', gap:16, minWidth:0}}>
          <div className="panel" style={{display:'grid', gap: '12px'}}>
            <div className="status">
              <div><strong>Turn:</strong> {state.turn === 'w' ? 'White' : 'Black'}</div>
              {state.is_game_over ? <div>Game over — result: {state.result}</div> : null}
              {evalData?.bestmove?.san && <div><strong>Best Move:</strong> {evalData.bestmove.san}</div>}
              {evalData?.pv?.san && evalData.pv.san?.length > 0 && (
                <div style={{fontSize:12, opacity:.9}}>
                  <strong>Best Line:</strong> {evalData.pv.san.slice(0,8).join(' ')}
                </div>
              )}
              <div style={{fontSize:12, opacity:.8}}>
                {evalData?.engine_type ? <>Engine: {evalData.engine_type} · </> : null}
                {evalData?.depth ? <>Depth: {evalData.depth} · </> : null}
                {typeof evalData?.elapsed_ms === 'number' ? <>Time: {evalData.elapsed_ms} ms</> : null}
                {evalData?.pending ? <> · analysing…</> : null}
              </div>
              {error && <div className="error">{String(error)}</div>}
              {lastEvalError && <div className="error" style={{marginTop:8}}>Eval error: {lastEvalError}</div>}
            </div>

            <div>
              <div style={{marginBottom: 6, fontWeight:600}}>FEN</div>
              <div className="fen">{state.fen}</div>
            </div>
            <div>
              <div style={{marginBottom: 6, fontWeight:600}}>Moves (SAN)</div>
              <div className="history">
                {state.history_san.map((m, i) => <span key={i} className="pill">{m}</span>)}
              </div>
            </div>
          </div>

          <div className="panel" style={{display:'flex', flexDirection:'column', gap:12, flexGrow:1, minHeight:400}}>
            <div style={{flexShrink:0}}>
              <div style={{fontWeight:600}}>Chess Coach</div>
              <div style={{fontSize:12, opacity:.8}}>Chat with the model about the position.</div>
            </div>
            
            <div style={{flexGrow:1, overflowY:'auto', display:'flex', flexDirection:'column', gap:8, paddingRight:4, maxHeight: 500}}>
              {chatMessages.length === 0 && (
                <div style={{fontSize:13, opacity:0.6, fontStyle:'italic', marginTop:20, textAlign:'center'}}>
                  Ask a question to start the conversation...
                </div>
              )}
              {chatMessages.map((m, i) => (
                <div key={i} style={{
                  alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start',
                  background: m.role === 'user' ? '#355070' : '#2a2f45',
                  color: '#e6e6e6',
                  padding: '8px 12px',
                  borderRadius: 8,
                  maxWidth: '90%',
                  fontSize: 14,
                  lineHeight: 1.4,
                  whiteSpace: 'pre-wrap'
                }}>
                  {m.content}
                </div>
              ))}
              {coachLoading && (
                <div style={{alignSelf:'flex-start', background:'#2a2f45', color:'#aaa', padding:'6px 10px', borderRadius:6, fontSize:12, fontStyle:'italic'}}>
                  Typing...
                </div>
              )}
              {coachError && (
                <div style={{alignSelf:'center', color:'#ff8a8a', fontSize:12, background:'#3d1a1a', padding:6, borderRadius:4}}>
                  Error: {coachError}
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            <div style={{display:'flex', gap:6, flexShrink:0}}>
              <input
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && sendChat()}
                placeholder="Ask a question..."
                style={{flexGrow:1, padding:8, borderRadius:6, border:'1px solid #3a415f', background:'#0e1430', color:'inherit'}}
                disabled={coachLoading}
              />
              <button onClick={sendChat} disabled={coachLoading || !chatInput.trim()}>Send</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

createRoot(document.getElementById('root')).render(<App />)

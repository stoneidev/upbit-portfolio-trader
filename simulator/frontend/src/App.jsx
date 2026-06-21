import React, { useState, useEffect, useRef } from 'react';
import { 
  Play, 
  FastForward, 
  RefreshCw, 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  Briefcase, 
  History, 
  HelpCircle, 
  Award,
  ChevronRight,
  Plus,
  Minus
} from 'lucide-react';
import { ResponsiveContainer, ComposedChart, LineChart, Line, Bar, Cell, XAxis, YAxis, Tooltip, CartesianGrid, AreaChart, Area } from 'recharts';

const API_BASE = import.meta.env.DEV ? '' : 'https://nasdaq-simulator-backend.nijin39.workers.dev';

function App() {
  const [date, setDate] = useState('');
  const [marketSummary, setMarketSummary] = useState({});
  const [screener, setScreener] = useState([]);
  const [portfolio, setPortfolio] = useState({
    cash: 100000.0,
    holdings_value: 0.0,
    total_value: 100000.0,
    return_pct: 0.0,
    total_realized_pnl: 0.0,
    holdings: [],
    transactions: [],
    equity_curve: []
  });
  
  const [selectedStock, setSelectedStock] = useState(null);
  const [tradeShares, setTradeShares] = useState(10);
  const [filterTab, setFilterTab] = useState('ALL');
  const [portfolioTab, setPortfolioTab] = useState('HOLDINGS');
  const [loading, setLoading] = useState(true);
  
  // Candle Chart states
  const [chartTab, setChartTab] = useState('EQUITY'); // 'EQUITY' or 'CANDLE'
  const [candleData, setCandleData] = useState([]);
  const [chartLoading, setChartLoading] = useState(false);
  
  // Flash animation states
  const [prevPrices, setPrevPrices] = useState({});
  const [flashingUp, setFlashingUp] = useState(new Set());
  const [flashingDown, setFlashingDown] = useState(new Set());
  
  // Table Sorting states
  const [sortField, setSortField] = useState('MCap_Rank');
  const [sortAsc, setSortAsc] = useState(true);
  
  const isFirstLoad = useRef(true);

  // Toast notifications state
  const [notifications, setNotifications] = useState([]);
  // Terminal system logs state
  const [terminalLogs, setTerminalLogs] = useState([
    { id: 1, time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }), tag: 'system', msg: 'Cyber-Quant Terminal initialized. Start date: 2025-01-02.' }
  ]);
  // Top 30 entries of the current day
  const [enteredTop30, setEnteredTop30] = useState([]);

  // Terminal scroll ref
  const terminalEndRef = useRef(null);

  // Toast and Log helper
  const addNotification = (msg, type = 'success') => {
    const id = Date.now() + Math.random();
    setNotifications(prev => [...prev, { id, msg, type }]);
    
    // Add to terminal logs
    const logTime = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    let tag = 'system';
    if (type === 'warning') tag = 'alert';
    else if (type === 'trade' || type === 'success') tag = 'trade';
    
    setTerminalLogs(prev => [
      ...prev,
      { id: Date.now() + Math.random(), msgId: id, time: logTime, tag, msg }
    ]);
    
    // Auto remove toast
    setTimeout(() => {
      setNotifications(prev => prev.map(n => n.id === id ? { ...n, closing: true } : n));
      setTimeout(() => {
        setNotifications(prev => prev.filter(n => n.id !== id));
      }, 250);
    }, 4500);
  };

  // Auto-scroll terminal
  useEffect(() => {
    if (terminalEndRef.current) {
      terminalEndRef.current.scrollTop = terminalEndRef.current.scrollHeight;
    }
  }, [terminalLogs]);

  // Load stock chart data
  const loadStockChart = async (ticker) => {
    if (!ticker) return;
    try {
      setChartLoading(true);
      const res = await fetch(`${API_BASE}/api/chart/${ticker}`);
      const data = await res.json();
      if (res.ok) {
        // Map data to range format for Recharts
        const formatted = data.map(d => ({
          ...d,
          wickRange: [d.Low, d.High],
          bodyRange: [d.Open, d.Close],
          isUp: d.Close >= d.Open
        }));
        setCandleData(formatted);
      }
    } catch (err) {
      console.error("Error loading stock chart:", err);
    } finally {
      setChartLoading(false);
    }
  };

  useEffect(() => {
    if (selectedStock && selectedStock.Ticker) {
      loadStockChart(selectedStock.Ticker);
    }
  }, [selectedStock?.Ticker, date]);

  // Calculate dynamic domain for stock chart to prevent Recharts from crashing
  const getChartDomain = () => {
    if (!candleData || candleData.length === 0) return ['auto', 'auto'];
    const lows = candleData.map(d => d.Low);
    const highs = candleData.map(d => d.High);
    const minVal = Math.min(...lows);
    const maxVal = Math.max(...highs);
    const padding = (maxVal - minVal) * 0.05 || 1; // 5% padding
    return [Math.max(0, minVal - padding), maxVal + padding];
  };

  // Load state and portfolio
  const loadSimulatorData = async (skipFlash = false) => {
    try {
      setLoading(true);
      const stateRes = await fetch(`${API_BASE}/api/state`);
      const stateData = await stateRes.json();
      
      const portRes = await fetch(`${API_BASE}/api/portfolio`);
      const portData = await portRes.json();
      
      const oldDate = date;
      const newDate = stateData.current_date;
      
      setDate(newDate);
      setMarketSummary(stateData.market_summary);
      
      // Calculate flashing changes
      if (!skipFlash && !isFirstLoad.current && stateData.screener) {
        const newFlashingUp = new Set();
        const newFlashingDown = new Set();
        
        stateData.screener.forEach(stock => {
          const ticker = stock.Ticker;
          const newPrice = stock.Close;
          const oldPrice = prevPrices[ticker];
          
          if (oldPrice !== undefined) {
            if (newPrice > oldPrice) {
              newFlashingUp.add(ticker);
            } else if (newPrice < oldPrice) {
              newFlashingDown.add(ticker);
            }
          }
        });
        
        setFlashingUp(newFlashingUp);
        setFlashingDown(newFlashingDown);
        
        // Clear flashing classes after 800ms
        setTimeout(() => {
          setFlashingUp(new Set());
          setFlashingDown(new Set());
        }, 800);
      }
      
      // Save current prices for the next step comparison
      if (stateData.screener) {
        const priceMap = {};
        stateData.screener.forEach(s => {
          priceMap[s.Ticker] = s.Close;
        });
        setPrevPrices(priceMap);
      }
      
      setScreener(stateData.screener || []);
      setPortfolio(portData);
      
      // Update selected stock details if already selected
      if (selectedStock) {
        const updated = (stateData.screener || []).find(s => s.Ticker === selectedStock.Ticker);
        if (updated) setSelectedStock(updated);
      } else if (stateData.screener && stateData.screener.length > 0 && !selectedStock) {
        // Select Apple or Nvidia by default
        const defaultStock = stateData.screener.find(s => s.Ticker === 'AAPL') || stateData.screener[0];
        setSelectedStock(defaultStock);
      }

      // Handle date progress logging
      if (oldDate && oldDate !== newDate) {
        const logTime = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        setTerminalLogs(prev => [
          ...prev,
          { id: Date.now() + Math.random(), time: logTime, tag: 'system', msg: `Simulation date progressed to ${newDate}` }
        ]);
      }

      // Handle entered top 30 notifications
      if (stateData.entered_top30 && stateData.entered_top30.length > 0) {
        if (oldDate && oldDate !== newDate) {
          stateData.entered_top30.forEach(item => {
            addNotification(
              `🚀 [${item.ticker}] Top 30 진입! 모멘텀 순위: ${item.curr_rank}위 (기존: ${item.prev_rank}위)`,
              'success'
            );
          });
        }
        setEnteredTop30(stateData.entered_top30);
      } else {
        setEnteredTop30([]);
      }

      // Handle exited top 30 notifications
      if (oldDate && oldDate !== newDate && stateData.exited_top30 && stateData.exited_top30.length > 0) {
        stateData.exited_top30.forEach(item => {
          addNotification(
            `⚠️ [${item.ticker}] Top 30 이탈! 모멘텀 순위: ${item.curr_rank}위 (기존: ${item.prev_rank}위)`,
            'warning'
          );
        });
      }
      
      isFirstLoad.current = false;
    } catch (err) {
      console.error("Error loading simulator state:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadSimulatorData(true);
  }, []);

  // Time controls
  const handleStep = async (days) => {
    try {
      const res = await fetch(`${API_BASE}/api/step`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ days })
      });
      const data = await res.json();
      if (data.status === 'success') {
        await loadSimulatorData(false);
      }
    } catch (err) {
      console.error("Error stepping forward:", err);
    }
  };

  // Reset simulation
  const handleReset = async () => {
    if (window.confirm("정말 모의투자를 처음(2025년 1월 2일) 상태로 초기화하시겠습니까? 모든 거래 이력이 삭제됩니다.")) {
      try {
        isFirstLoad.current = true;
        const res = await fetch(`${API_BASE}/api/reset`, { method: 'POST' });
        const data = await res.json();
        if (data.status === 'success') {
          setSelectedStock(null);
          setPrevPrices({});
          setTerminalLogs([
            { id: 1, time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }), tag: 'system', msg: 'Simulator reset to 2025-01-02. Cash: $100,000.' }
          ]);
          setNotifications([]);
          setEnteredTop30([]);
          await loadSimulatorData(true);
        }
      } catch (err) {
        console.error("Error resetting simulation:", err);
      }
    }
  };

  // Execute trade
  const handleTrade = async (action) => {
    if (!selectedStock) return;
    try {
      const res = await fetch(`${API_BASE}/api/trade`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ticker: selectedStock.Ticker,
          action,
          shares: tradeShares
        })
      });
      
      const data = await res.json();
      if (res.ok) {
        addNotification(`${selectedStock.Ticker} ${tradeShares}주 ${action === 'BUY' ? '매수 체결 완료!' : '매도 체결 완료!'}`, 'trade');
        await loadSimulatorData(true);
      } else {
        addNotification(`거래 실패: ${data.detail}`, 'warning');
      }
    } catch (err) {
      console.error("Error executing trade:", err);
    }
  };

  // Quick Sell from Holdings Table
  const handleQuickSell = async (ticker, ownedShares) => {
    const input = window.prompt(`매도할 수량을 입력하세요. (최대 ${ownedShares}주)`, ownedShares);
    if (input === null) return; // User cancelled
    
    const shares = parseInt(input);
    if (isNaN(shares) || shares <= 0) {
      addNotification("올바른 수량을 입력해주세요.", "warning");
      return;
    }
    
    if (shares > ownedShares) {
      addNotification(`보유 수량(${ownedShares}주)보다 많은 수량을 매도할 수 없습니다.`, "warning");
      return;
    }
    
    try {
      const res = await fetch(`${API_BASE}/api/trade`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ticker,
          action: 'SELL',
          shares
        })
      });
      
      const data = await res.json();
      if (res.ok) {
        addNotification(`${ticker} ${shares}주 매도 완료!`, 'trade');
        await loadSimulatorData(true);
      } else {
        addNotification(`거래 실패: ${data.detail}`, 'warning');
      }
    } catch (err) {
      console.error("Error executing quick sell:", err);
    }
  };

  // Filtered Screener
  const filteredScreener = screener.filter(stock => {
    if (filterTab === 'WTA') return stock.WTA_Status !== 'HOLD';
    if (filterTab === 'OVERSOLD') return stock.RSI < 35 || stock["Z-Score"] < -1.5;
    if (filterTab === 'OVERBOUGHT') return stock.RSI > 65 || stock["Z-Score"] > 1.5;
    return true;
  });

  // Sort the filtered screener
  const sortedScreener = [...filteredScreener].sort((a, b) => {
    let valA = a[sortField];
    let valB = b[sortField];
    
    if (valA === undefined) return 1;
    if (valB === undefined) return -1;
    
    if (typeof valA === 'string') {
      return sortAsc ? valA.localeCompare(valB) : valB.localeCompare(valA);
    }
    
    return sortAsc ? valA - valB : valB - valA;
  });

  const handleSort = (field) => {
    if (sortField === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      setSortAsc(field === 'MCap_Rank' || field === 'Mom_Rank' || field === 'RSI' || field === 'Z-Score' ? true : false);
    }
  };


  return (
    <div className="app-container">
      {/* 1. Header Control Panel */}
      <header className="app-header">
        <div className="header-title-group">
          <h1>
            Nasdaq-50 Quant Trading Terminal
            <span>SIM v1.0</span>
          </h1>
        </div>
        
        <div className="time-controls">
          <div className="date-badge">
            {date || "2025-01-02"}
          </div>
          
          {/* Backward Buttons */}
          <button className="btn-quant" onClick={() => handleStep(-1)} disabled={loading} title="Go back 1 Day">
            -1 Day
          </button>
          
          <button className="btn-quant" onClick={() => handleStep(-5)} disabled={loading} title="Go back 1 Week">
            -1 Week
          </button>
          
          <button className="btn-quant" onClick={() => handleStep(-20)} disabled={loading} title="Go back 1 Month">
            -1 Month
          </button>

          <span style={{ margin: '0 4px', color: 'var(--panel-border)', userSelect: 'none' }}>|</span>
          
          {/* Forward Buttons */}
          <button className="btn-quant accent" onClick={() => handleStep(1)} disabled={loading}>
            <Play size={14} /> +1 Day
          </button>
          
          <button className="btn-quant accent" onClick={() => handleStep(5)} disabled={loading}>
            <FastForward size={14} /> +1 Week
          </button>
          
          <button className="btn-quant accent" onClick={() => handleStep(20)} disabled={loading}>
            <FastForward size={14} /> +1 Month
          </button>
          
          <button className="btn-quant danger" onClick={handleReset} style={{ marginLeft: '12px' }}>
            <RefreshCw size={14} /> Reset
          </button>
        </div>
      </header>

      {/* 2. Stats Summary Banner */}
      <div className="stats-banner">
        <div className="stat-card portfolio">
          <div className="stat-label">Total Portfolio Value</div>
          <div className="stat-value">${portfolio.total_value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</div>
        </div>
        <div className="stat-card cash">
          <div className="stat-label">Available Cash</div>
          <div className="stat-value">${portfolio.cash.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</div>
        </div>
        <div className="stat-card holdings">
          <div className="stat-label">Holdings Valuation</div>
          <div className="stat-value">${portfolio.holdings_value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</div>
        </div>
        <div className={`stat-card ${portfolio.return_pct > 0 ? 'up' : portfolio.return_pct < 0 ? 'down' : ''}`}>
          <div className="stat-label">Total Return %</div>
          <div className={`stat-value ${portfolio.return_pct > 0 ? 'up' : portfolio.return_pct < 0 ? 'down' : ''}`}>
            {portfolio.return_pct > 0 ? '+' : ''}{portfolio.return_pct.toFixed(2)}%
          </div>
        </div>
        <div className={`stat-card ${portfolio.total_realized_pnl > 0 ? 'up' : portfolio.total_realized_pnl < 0 ? 'down' : ''}`}>
          <div className="stat-label">Realized PnL</div>
          <div className={`stat-value ${portfolio.total_realized_pnl > 0 ? 'up' : portfolio.total_realized_pnl < 0 ? 'down' : ''}`}>
            {portfolio.total_realized_pnl > 0 ? '+$' : portfolio.total_realized_pnl < 0 ? '-$' : '$'}
            {Math.abs(portfolio.total_realized_pnl || 0).toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 })}
          </div>
        </div>
        <div className="stat-card breadth">
          <div className="stat-label">Market Breadth (20)</div>
          <div className="stat-value">{(marketSummary.above_ma20 * 100 || 0).toFixed(1)}%</div>
        </div>
        <div className="stat-card wta">
          <div className="stat-label">WTA Index</div>
          <div className="stat-value">{(marketSummary.wta_index * 100 || 0).toFixed(1)}%</div>
        </div>
      </div>

      {/* 3. Main Dashboard Grid */}
      <div className="dashboard-grid">
        
        {/* Column 1: Stock Screener Terminal */}
        <div className="quant-panel screener-column">
          <div className="quant-panel-header">
            <div className="quant-panel-title">Nasdaq 50 Real-time Screener</div>
            <div className="quant-panel-status">DATA STREAM: ON</div>
          </div>
          
          <div className="quant-panel-body" style={{ padding: '8px', display: 'flex', flexDirection: 'column', minHeight: 0 }}>
            <div className="tab-row">
              <button className={`tab-btn ${filterTab === 'ALL' ? 'active' : ''}`} onClick={() => setFilterTab('ALL')}>All Stocks</button>
              <button className={`tab-btn ${filterTab === 'WTA' ? 'active' : ''}`} onClick={() => setFilterTab('WTA')}>WTA leaders</button>
              <button className={`tab-btn ${filterTab === 'OVERSOLD' ? 'active' : ''}`} onClick={() => setFilterTab('OVERSOLD')}>Oversold (Mean Rev)</button>
              <button className={`tab-btn ${filterTab === 'OVERBOUGHT' ? 'active' : ''}`} onClick={() => setFilterTab('OVERBOUGHT')}>Overbought</button>
            </div>
            
            <div className="table-container" style={{ flex: 1, overflowY: 'auto' }}>
              <table className="table-quant">
                <thead>
                  <tr>
                    <th style={{ cursor: 'pointer' }} onClick={() => handleSort('Ticker')}>
                      <span className="tooltip" data-tooltip="[티커 심볼] 미국 증시 상장 기업 고유 코드. 클릭 시 정렬이 변경되며 우측 주문 패널에 선택 연동됩니다.">
                        Ticker<br/>Symbol{sortField === 'Ticker' ? (sortAsc ? ' ▲' : ' ▼') : ''}
                      </span>
                    </th>
                    <th style={{ textAlign: 'center', cursor: 'pointer' }} onClick={() => handleSort('MCap_Rank')}>
                      <span className="tooltip" data-tooltip="[시가총액 순위] 나스닥 100 지수 내 시가총액 기준 순위 (NVDA=1, AAPL=2 등).">
                        Cap<br/>Rank{sortField === 'MCap_Rank' ? (sortAsc ? ' ▲' : ' ▼') : ''}
                      </span>
                    </th>
                    <th style={{ textAlign: 'right', cursor: 'pointer' }} onClick={() => handleSort('Close')}>
                      <span className="tooltip" data-tooltip="[현재가] 현재 시뮬레이션 일자의 종가. 매수/매도 주문의 기준 가격이 됩니다.">
                        Price<br/>(${sortField === 'Close' ? (sortAsc ? ' ▲' : ' ▼') : ''})
                      </span>
                    </th>
                    <th style={{ cursor: 'pointer' }} onClick={() => handleSort('Trend')}>
                      <span className="tooltip" data-tooltip="[추세 상태] 4단계 주가 사이클 분류 (STAGE 1: 횡보 매집, STAGE 2: 상승 돌파, STAGE 3: 고점 분산, STAGE 4: 하락 침체).">
                        Trend<br/>State{sortField === 'Trend' ? (sortAsc ? ' ▲' : ' ▼') : ''}
                      </span>
                    </th>
                    <th style={{ textAlign: 'center', cursor: 'pointer' }} onClick={() => handleSort('Minervini_Score')}>
                      <span className="tooltip" data-tooltip="[미너비니 스코어] 마크 미너비니의 Stage 2 Trend Template 8가지 기준 통과 개수 (7개 이상 통과 시 PASS).">
                        Minervini<br/>SEPA{sortField === 'Minervini_Score' ? (sortAsc ? ' ▲' : ' ▼') : ''}
                      </span>
                    </th>
                    <th style={{ textAlign: 'center', cursor: 'pointer' }} onClick={() => handleSort('Mom_Rank')}>
                      <span className="tooltip" data-tooltip="[모멘텀 순위] 최근 3개월 수익률 기준 50개 종목 중의 순위.">
                        Mom<br/>Rank{sortField === 'Mom_Rank' ? (sortAsc ? ' ▲' : ' ▼') : ''}
                      </span>
                    </th>
                    <th style={{ textAlign: 'right', cursor: 'pointer' }} onClick={() => handleSort('Momentum_3M')}>
                      <span className="tooltip" data-tooltip="[3개월 수익률] 최근 3개월(60영업일) 동안의 누적 수익률입니다.">
                        3M<br/>Return{sortField === 'Momentum_3M' ? (sortAsc ? ' ▲' : ' ▼') : ''}
                      </span>
                    </th>
                    <th style={{ textAlign: 'right', cursor: 'pointer' }} onClick={() => handleSort('Mom_Z')}>
                      <span className="tooltip" data-tooltip="[모멘텀 Z-점수] 전체 50개 종목 대비 상대적인 모멘텀 강도 표준점수.">
                        Mom<br/>Z{sortField === 'Mom_Z' ? (sortAsc ? ' ▲' : ' ▼') : ''}
                      </span>
                    </th>
                    <th style={{ cursor: 'pointer' }} onClick={() => handleSort('WTA_Status')}>
                      <span className="tooltip" data-tooltip="[승자독식 상태] WTA 전략 등급 (Champion/Leader/HOLD).">
                        WTA<br/>Status{sortField === 'WTA_Status' ? (sortAsc ? ' ▲' : ' ▼') : ''}
                      </span>
                    </th>
                    <th style={{ textAlign: 'right', cursor: 'pointer' }} onClick={() => handleSort('Z-Score')}>
                      <span className="tooltip" data-tooltip="[Z-Score] 20일 이평선 및 표준편차 기준 가격 위치.">
                        Z-Score<br/>(20d){sortField === 'Z-Score' ? (sortAsc ? ' ▲' : ' ▼') : ''}
                      </span>
                    </th>
                    <th style={{ textAlign: 'center', cursor: 'pointer' }} onClick={() => handleSort('RSI')}>
                      <span className="tooltip" data-tooltip="[RSI] 최근 14일 주가 상승/하락 상대강도 지표.">
                        RSI<br/>(14d){sortField === 'RSI' ? (sortAsc ? ' ▲' : ' ▼') : ''}
                      </span>
                    </th>
                    <th style={{ textAlign: 'right', cursor: 'pointer' }} onClick={() => handleSort('ATR_Pct')}>
                      <span className="tooltip" data-tooltip="[ATR% 변동성] 14일 일평균 변동폭(ATR)의 주가 대비 백분율.">
                        ATR%<br/>(Vol){sortField === 'ATR_Pct' ? (sortAsc ? ' ▲' : ' ▼') : ''}
                      </span>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {sortedScreener.map(stock => {
                    const isSelected = selectedStock && selectedStock.Ticker === stock.Ticker;
                    const isFlashingUp = flashingUp.has(stock.Ticker);
                    const isFlashingDown = flashingDown.has(stock.Ticker);
                    
                    const isNewTop30 = enteredTop30.some(item => item.ticker === stock.Ticker);
                    const top30Item = enteredTop30.find(item => item.ticker === stock.Ticker);
                    
                    return (
                      <tr 
                        key={stock.Ticker}
                        onClick={() => {
                          setSelectedStock(stock);
                        }}
                        style={{ 
                          cursor: 'pointer',
                          backgroundColor: isSelected ? 'rgba(0, 240, 255, 0.08)' : '',
                          borderLeft: isSelected ? '3px solid var(--accent-cyan)' : '3px solid transparent'
                        }}
                        className={`${isFlashingUp ? 'flash-up-anim' : ''} ${isFlashingDown ? 'flash-down-anim' : ''}`}
                      >
                        <td className="mono" style={{ fontWeight: 'bold' }}>
                          {stock.Ticker}
                          {isNewTop30 && (
                            <span 
                              className="badge-new-top30" 
                              title={`순위 상승! 기존: ${top30Item?.prev_rank}위 -> 현재: ${top30Item?.curr_rank}위`}
                            >
                              NEW 30
                            </span>
                          )}
                        </td>
                        <td className="mono" style={{ textAlign: 'center' }}>{stock.MCap_Rank}</td>
                        <td className="mono" style={{ textAlign: 'right' }}>{stock.Close.toFixed(2)}</td>
                        <td style={{ 
                          color: stock.Trend.includes('STAGE 2') || stock.Trend === 'UP' 
                            ? 'var(--color-up)' 
                            : stock.Trend.includes('STAGE 4') || stock.Trend === 'DOWN' 
                            ? 'var(--color-down)' 
                            : stock.Trend.includes('STAGE 3')
                            ? '#ffb703' 
                            : 'var(--text-secondary)'
                        }}>
                          {stock.Trend}
                        </td>
                        <td className="mono" style={{ textAlign: 'center' }}>
                          <span 
                            className={`badge-minervini ${stock.Minervini_Pass ? 'pass' : 'fail'}`} 
                            title={
                              stock.Minervini_Details 
                                ? `1. 가격 > 150 & 200 SMA: ${stock.Minervini_Details.price_above_150_200 ? '✅' : '❌'}\n` +
                                  `2. 150 SMA > 200 SMA: ${stock.Minervini_Details.sma_150_above_200 ? '✅' : '❌'}\n` +
                                  `3. 200 SMA 상승: ${stock.Minervini_Details.sma_200_rising ? '✅' : '❌'}\n` +
                                  `4. 50 > 150 SMA: ${stock.Minervini_Details.sma_50_above_150 ? '✅' : '❌'}\n` +
                                  `5. 가격 > 50 SMA: ${stock.Minervini_Details.price_above_50 ? '✅' : '❌'}\n` +
                                  `6. 가격이 52주 신저가 대비 30%+ 상승: ${stock.Minervini_Details.price_30pct_above_52w_low ? '✅' : '❌'} (${stock.Minervini_Details.distance_from_52w_low_pct || 0}%)\n` +
                                  `7. 가격이 52주 최고가 부근 25% 이내: ${stock.Minervini_Details.price_near_52w_high ? '✅' : '❌'} (${stock.Minervini_Details.distance_from_52w_high_pct || 0}%)\n` +
                                  `8. Stage 2 국면: ${stock.Minervini_Details.confirmed_stage_2 ? '✅' : '❌'}`
                                : ''
                            }
                          >
                            {stock.Minervini_Score}/8 {stock.Minervini_Pass ? 'PASS' : 'FAIL'}
                          </span>
                        </td>
                        <td className="mono" style={{ textAlign: 'center' }}>{stock.Mom_Rank}</td>
                        <td className={`mono ${stock.Momentum_3M > 0 ? 'up' : 'down'}`} style={{ textAlign: 'right' }}>
                          {(stock.Momentum_3M * 100).toFixed(1)}%
                        </td>
                        <td className="mono" style={{ textAlign: 'right' }}>{stock.Mom_Z.toFixed(2)}</td>
                        <td>
                          <span className={`wta-badge ${stock.WTA_Status}`}>
                            {stock.WTA_Status}
                          </span>
                        </td>
                        <td className={`mono ${stock["Z-Score"] > 0 ? 'up' : 'down'}`} style={{ textAlign: 'right' }}>{stock["Z-Score"].toFixed(2)}</td>
                        <td className="mono" style={{ textAlign: 'center', color: stock.RSI < 30 ? 'var(--color-up)' : stock.RSI > 70 ? 'var(--color-down)' : 'var(--text-primary)' }}>
                          {stock.RSI.toFixed(1)}
                        </td>
                        <td className="mono" style={{ textAlign: 'right' }}>{(stock.ATR_Pct * 100).toFixed(1)}%</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Selected Stock Candle Chart (grid-area: chart1) */}
        <div className="quant-panel candle-chart-panel">
            <div className="quant-panel-header">
              <div className="quant-panel-title">
                {selectedStock ? `${selectedStock.Ticker} 60d Candlestick Chart` : 'Stock Price Chart'}
              </div>
              <div className="quant-panel-status" style={{ color: 'var(--color-up)', borderColor: 'var(--color-up-glow)' }}>
                {selectedStock ? `$${selectedStock.Close.toFixed(2)}` : 'STANDBY'}
              </div>
            </div>
            <div className="quant-panel-body" style={{ display: 'flex', flexDirection: 'column', minHeight: 0, padding: '6px 12px' }}>
              {selectedStock ? (
                <>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.65rem', color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '8px' }}>
                    <span>{selectedStock.Ticker} Price Action & Volume</span>
                    <div>
                      <span style={{ marginRight: '10px' }}>RSI: <strong style={{ color: '#fff' }}>{selectedStock.RSI.toFixed(1)}</strong></span>
                      <span>Z-Score: <strong style={{ color: '#fff' }}>{selectedStock["Z-Score"].toFixed(2)}</strong></span>
                    </div>
                  </div>
                  {candleData && candleData.length > 0 ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart data={candleData} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
                        <CartesianGrid stroke="rgba(22, 32, 53, 0.4)" strokeDasharray="3 3" />
                        <XAxis dataKey="Date" stroke="#56677f" tickFormatter={(str) => str.substring(5)} style={{ fontSize: '0.6rem', fontFamily: 'var(--font-mono)' }} />
                        <YAxis stroke="#56677f" domain={getChartDomain()} style={{ fontSize: '0.6rem', fontFamily: 'var(--font-mono)' }} />
                        <Tooltip 
                          content={({ active, payload }) => {
                            if (active && payload && payload.length) {
                              const d = payload[0].payload;
                              const isUp = d.Close >= d.Open;
                              return (
                                <div style={{ 
                                  backgroundColor: 'rgba(11, 17, 32, 0.85)', 
                                  backdropFilter: 'blur(8px)',
                                  border: `1px solid ${isUp ? 'rgba(0, 255, 135, 0.3)' : 'rgba(255, 0, 85, 0.3)'}`, 
                                  padding: '8px 12px', 
                                  fontSize: '0.68rem', 
                                  color: '#fff', 
                                  borderRadius: '6px', 
                                  zIndex: 2000,
                                  boxShadow: `0 4px 15px rgba(0,0,0,0.5), inset 0 0 10px ${isUp ? 'rgba(0, 255, 135, 0.05)' : 'rgba(255, 0, 85, 0.05)'}`
                                }}>
                                  <div style={{ fontWeight: 'bold', color: 'var(--text-secondary)', marginBottom: '6px', fontFamily: 'var(--font-mono)' }}>{d.Date}</div>
                                  <div style={{ display: 'grid', gridTemplateColumns: 'auto auto', gap: '4px 12px' }}>
                                    <span>Open:</span><span style={{ fontFamily: 'var(--font-mono)', textAlign: 'right' }}>${d.Open.toFixed(2)}</span>
                                    <span>High:</span><span style={{ fontFamily: 'var(--font-mono)', textAlign: 'right' }}>${d.High.toFixed(2)}</span>
                                    <span>Low:</span><span style={{ fontFamily: 'var(--font-mono)', textAlign: 'right' }}>${d.Low.toFixed(2)}</span>
                                    <span style={{ fontWeight: 'bold' }}>Close:</span><span style={{ fontFamily: 'var(--font-mono)', textAlign: 'right', fontWeight: 'bold', color: isUp ? 'var(--color-up)' : 'var(--color-down)' }}>${d.Close.toFixed(2)}</span>
                                  </div>
                                </div>
                              );
                            }
                            return null;
                          }}
                        />
                        <Bar dataKey="wickRange" fill="#8884d8" barSize={1.2} animationDuration={0}>
                          {candleData.map((entry, index) => (
                            <Cell key={`cell-wick-${index}`} fill={entry.isUp ? 'var(--color-up)' : 'var(--color-down)'} />
                          ))}
                        </Bar>
                        <Bar dataKey="bodyRange" barSize={5} animationDuration={0}>
                          {candleData.map((entry, index) => (
                            <Cell 
                              key={`cell-body-${index}`} 
                              fill={entry.isUp ? 'rgba(0, 255, 135, 0.2)' : 'rgba(255, 0, 85, 0.25)'} 
                              stroke={entry.isUp ? 'var(--color-up)' : 'var(--color-down)'}
                              strokeWidth={1}
                            />
                          ))}
                        </Bar>
                      </ComposedChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="empty-selection">차트 데이터를 로드하는 중...</div>
                  )}
                </>
              ) : (
                <div className="empty-selection">스크리너에서 종목을 선택하시면 주가 캔들 차트가 표시됩니다.</div>
              )}
            </div>
          </div>
          
          {/* Portfolio Equity Curve Chart (grid-area: chart2) */}
          <div className="quant-panel equity-chart-panel">
            <div className="quant-panel-header">
              <div className="quant-panel-title">Portfolio Equity Curve (자산 변화 추이)</div>
              <div className="quant-panel-status" style={{ color: 'var(--accent-cyan)', borderColor: 'var(--accent-cyan-glow)' }}>
                NET VALUE: ${portfolio.total_value.toLocaleString(undefined, { maximumFractionDigits: 0 })}
              </div>
            </div>
            <div className="quant-panel-body" style={{ display: 'flex', flexDirection: 'column', minHeight: 0, padding: '6px 12px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.65rem', color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '8px' }}>
                <span>Account Balance Growth Curve</span>
                <span>Realized PnL: <strong style={{ color: portfolio.total_realized_pnl >= 0 ? 'var(--color-up)' : 'var(--color-down)' }}>
                  {portfolio.total_realized_pnl >= 0 ? '+' : ''}${portfolio.total_realized_pnl.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                </strong></span>
              </div>
              {portfolio.equity_curve && portfolio.equity_curve.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={portfolio.equity_curve} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
                    <defs>
                      <linearGradient id="colorEquity" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="var(--accent-cyan)" stopOpacity={0.25}/>
                        <stop offset="95%" stopColor="var(--accent-cyan)" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid stroke="rgba(22, 32, 53, 0.4)" strokeDasharray="3 3" />
                    <XAxis dataKey="Date" stroke="#56677f" tickFormatter={(str) => str.substring(5)} style={{ fontSize: '0.6rem', fontFamily: 'var(--font-mono)' }} />
                    <YAxis stroke="#56677f" domain={['auto', 'auto']} tickFormatter={(v) => `$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`} style={{ fontSize: '0.6rem', fontFamily: 'var(--font-mono)' }} />
                    <Tooltip 
                      content={({ active, payload }) => {
                        if (active && payload && payload.length) {
                          const d = payload[0].payload;
                          return (
                            <div style={{ 
                              backgroundColor: 'rgba(11, 17, 32, 0.85)', 
                              backdropFilter: 'blur(8px)',
                              border: '1px solid rgba(0, 240, 255, 0.3)', 
                              padding: '8px 12px', 
                              fontSize: '0.68rem', 
                              color: '#fff', 
                              borderRadius: '6px', 
                              zIndex: 2000,
                              boxShadow: '0 4px 15px rgba(0,0,0,0.5), inset 0 0 10px rgba(0, 240, 255, 0.05)'
                            }}>
                              <div style={{ fontWeight: 'bold', color: 'var(--text-secondary)', marginBottom: '4px', fontFamily: 'var(--font-mono)' }}>{d.Date}</div>
                              <div style={{ display: 'flex', justifyContent: 'space-between', gap: '16px' }}>
                                <span>Net Value:</span>
                                <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 'bold', color: 'var(--accent-cyan)' }}>
                                  ${d.Value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                                </span>
                              </div>
                            </div>
                          );
                        }
                        return null;
                      }}
                    />
                    <Area type="monotone" dataKey="Value" stroke="var(--accent-cyan)" strokeWidth={2} fillOpacity={1} fill="url(#colorEquity)" dot={false} />
                  </AreaChart>
                </ResponsiveContainer>
              ) : (
                <div className="empty-selection">자산 변화 이력이 없습니다.</div>
              )}
            </div>
          </div>

        {/* Right Column: Quant Trading Desk, Portfolio Tracker & Logs (grid-area: trading) */}
        <div className="trading-column">
          {/* Order Entry Desk */}
          <div className="quant-panel">
            <div className="quant-panel-header">
              <div className="quant-panel-title">Quant Trading Console</div>
              <div className="quant-panel-status">CONSOLE: READY</div>
            </div>
            <div className="quant-panel-body" style={{ padding: '12px' }}>
              {selectedStock ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                  <div className="trade-ticker-header">
                    <div>
                      <span className="trade-ticker-name">{selectedStock.Ticker}</span>
                      <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginLeft: '8px' }}>
                        {selectedStock.WTA_Status !== 'HOLD' ? '🚀 WTA Leader' : 'Range Stock'}
                      </span>
                    </div>
                    <span className="trade-ticker-price">${selectedStock.Close.toFixed(2)}</span>
                  </div>
                  
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', fontSize: '0.7rem', color: 'var(--text-secondary)' }}>
                    <div>RSI: <span style={{ color: '#fff', fontWeight: 'bold' }}>{selectedStock.RSI.toFixed(1)}</span></div>
                    <div>Z-Score: <span style={{ color: '#fff', fontWeight: 'bold' }}>{selectedStock["Z-Score"].toFixed(2)}</span></div>
                    <div>ATR% (Vol): <span style={{ color: '#fff', fontWeight: 'bold' }}>{(selectedStock.ATR_Pct * 100).toFixed(1)}%</span></div>
                    <div>3M Return: <span style={{ color: '#fff', fontWeight: 'bold' }}>{(selectedStock.Momentum_3M * 100).toFixed(1)}%</span></div>
                  </div>
                  
                  <div className="trade-box" style={{ padding: '8px 12px', gap: '8px' }}>
                    <div className="trade-form-row">
                      <span style={{ fontSize: '0.75rem', fontWeight: '600' }}>Order Size:</span>
                      <div className="trade-input-group">
                        <button onClick={() => setTradeShares(Math.max(1, tradeShares - 10))}>-</button>
                        <input 
                          type="number" 
                          value={tradeShares} 
                          onChange={(e) => setTradeShares(Math.max(1, parseInt(e.target.value) || 1))}
                        />
                        <button onClick={() => setTradeShares(tradeShares + 10)}>+</button>
                      </div>
                    </div>
                    
                    <div className="trade-summary-row">
                      <span>Order Value:</span>
                      <span className="trade-summary-value">${(tradeShares * selectedStock.Close).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
                    </div>
                    <div className="trade-summary-row">
                      <span>Commission (0.15%):</span>
                      <span className="trade-summary-value">${(tradeShares * selectedStock.Close * 0.0015).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
                    </div>
                    
                    <div style={{ display: 'flex', gap: '8px', marginTop: '4px' }}>
                      <button 
                        className="btn-quant success" 
                        style={{ flex: 1, justifyContent: 'center', padding: '8px' }}
                        onClick={() => handleTrade('BUY')}
                      >
                        Buy (Long)
                      </button>
                      <button 
                        className="btn-quant danger" 
                        style={{ flex: 1, justifyContent: 'center', padding: '8px' }}
                        onClick={() => handleTrade('SELL')}
                      >
                        Sell (Short)
                      </button>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="empty-selection">좌측 종목을 선택하면 거래 창이 활성화됩니다.</div>
              )}
            </div>
          </div>

          {/* Portfolio Holdings Tracker */}
          <div className="quant-panel" style={{ flex: 1, minHeight: 0 }}>
            <div className="quant-panel-header">
              <div className="segmented-control">
                <button 
                  className={`segmented-tab ${portfolioTab === 'HOLDINGS' ? 'active' : ''}`}
                  onClick={() => setPortfolioTab('HOLDINGS')}
                >
                  Holdings
                </button>
                <button 
                  className={`segmented-tab ${portfolioTab === 'HISTORY' ? 'active' : ''}`}
                  onClick={() => setPortfolioTab('HISTORY')}
                >
                  History
                </button>
              </div>
              <div className="quant-panel-status">TRACKER: ACTIVE</div>
            </div>
            
            <div className="quant-panel-body" style={{ padding: '8px 12px', overflowY: 'auto' }}>
              <div className="table-container no-scroll-x">
                {portfolioTab === 'HOLDINGS' ? (
                  <table className="table-quant table-compact">
                    <thead>
                      <tr>
                        <th>Sym</th>
                        <th style={{ textAlign: 'center' }}>Qty</th>
                        <th style={{ textAlign: 'right' }}>Avg</th>
                        <th style={{ textAlign: 'right' }}>Price</th>
                        <th style={{ textAlign: 'right' }}>PnL%</th>
                        <th style={{ textAlign: 'center' }}></th>
                      </tr>
                    </thead>
                    <tbody>
                      {portfolio.holdings.length === 0 ? (
                        <tr>
                          <td colSpan="6" style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '15px 0' }}>현금 100% 보유 중</td>
                        </tr>
                      ) : (
                        portfolio.holdings.map(h => (
                          <tr 
                            key={h.ticker}
                            onClick={() => {
                              const stock = screener.find(s => s.Ticker === h.ticker);
                              if (stock) setSelectedStock(stock);
                            }}
                            style={{ cursor: 'pointer' }}
                          >
                            <td className="mono" style={{ fontWeight: 'bold' }}>{h.ticker}</td>
                            <td className="mono" style={{ textAlign: 'center' }}>{h.shares}</td>
                            <td className="mono" style={{ textAlign: 'right' }}>${h.avg_price.toFixed(2)}</td>
                            <td className="mono" style={{ textAlign: 'right' }}>${h.current_price.toFixed(2)}</td>
                            <td className={`mono ${h.pnl_pct > 0 ? 'up' : h.pnl_pct < 0 ? 'down' : ''}`} style={{ textAlign: 'right' }}>
                              {h.pnl_pct > 0 ? '+' : ''}{h.pnl_pct.toFixed(1)}%
                            </td>
                            <td style={{ textAlign: 'center' }}>
                              <button 
                                className="btn-pill-sell" 
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleQuickSell(h.ticker, h.shares);
                                }}
                              >
                                Sell
                              </button>
                            </td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                ) : (
                  <table className="table-quant table-compact">
                    <thead>
                      <tr>
                        <th>Sym</th>
                        <th>Type</th>
                        <th style={{ textAlign: 'center' }}>Qty</th>
                        <th style={{ textAlign: 'right' }}>Price</th>
                        <th style={{ textAlign: 'right' }}>PnL</th>
                      </tr>
                    </thead>
                    <tbody>
                      {portfolio.transactions.length === 0 ? (
                        <tr>
                          <td colSpan="5" style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '15px 0' }}>거래 내역 없음</td>
                        </tr>
                      ) : (
                        portfolio.transactions.slice().reverse().slice(0, 15).map((t, idx) => {
                          const isSell = t.Type === 'SELL';
                          const hasPnL = isSell && t.Realized_PnL !== undefined;
                          const pnlClass = hasPnL ? (t.Realized_PnL > 0 ? 'up' : t.Realized_PnL < 0 ? 'down' : '') : '';
                          const pnlText = hasPnL 
                            ? `${t.Realized_PnL > 0 ? '+' : ''}$${t.Realized_PnL.toLocaleString(undefined, { maximumFractionDigits: 0 })}`
                            : '-';
                          
                          return (
                            <tr key={idx}>
                              <td className="mono" style={{ fontWeight: 'bold' }}>{t.Ticker}</td>
                              <td style={{ color: t.Type === 'BUY' ? 'var(--accent-cyan)' : 'var(--text-primary)' }}>{t.Type}</td>
                              <td className="mono" style={{ textAlign: 'center' }}>{t.Shares}</td>
                              <td className="mono" style={{ textAlign: 'right' }}>${t.Price.toFixed(2)}</td>
                              <td className={`mono ${pnlClass}`} style={{ textAlign: 'right' }}>{pnlText}</td>
                            </tr>
                          );
                        })
                      )}
                    </tbody>
                  </table>
                )}
              </div>
            </div>
          </div>
          
          {/* Terminal System Logs */}
          <div className="terminal-panel">
            <div className="terminal-header">
              <span>Terminal System Logs</span>
              <span style={{ color: 'var(--text-muted)' }}>Session Feed</span>
            </div>
            <div className="terminal-body" ref={terminalEndRef}>
              {terminalLogs.map(log => (
                <div className="terminal-line" key={log.id}>
                  <span className="terminal-timestamp">[{log.time}]</span>
                  <span className={`terminal-tag ${log.tag}`}>{log.tag}</span>
                  <span className="terminal-message-text">{log.msg}</span>
                </div>
              ))}
              <div className="terminal-line">
                <span className="terminal-timestamp">[{new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}]</span>
                <span className="terminal-tag system">system</span>
                <span className="terminal-message-text" style={{ display: 'flex', alignItems: 'center' }}>
                  Ready and listening<span className="terminal-cursor">_</span>
                </span>
              </div>
            </div>
          </div>
        </div>
        
      </div>

      {/* Toast Alert overlay */}
      <div className="toast-container">
        {notifications.map(n => (
          <div key={n.id} className={`toast-message ${n.closing ? 'closing' : ''} ${n.type === 'warning' ? 'warning' : n.type === 'info' ? 'info' : ''}`}>
            <div className="toast-content">{n.msg}</div>
            <button className="toast-close-btn" onClick={() => {
              setNotifications(prev => prev.map(item => item.id === n.id ? { ...item, closing: true } : item));
              setTimeout(() => {
                setNotifications(prev => prev.filter(item => item.id !== n.id));
              }, 250);
            }}>×</button>
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;

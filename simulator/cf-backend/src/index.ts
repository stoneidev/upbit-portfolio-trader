import { Hono } from 'hono';
import { cors } from 'hono/cors';

type Bindings = {
  DB: D1Database;
  SIMULATOR_KV: KVNamespace;
};

const app = new Hono<{ Bindings: Bindings }>();

// Enable CORS for all API endpoints
app.use('/api/*', cors({
  origin: '*',
  allowMethods: ['GET', 'POST', 'OPTIONS'],
  allowHeaders: ['Content-Type', 'Authorization'],
  exposeHeaders: ['Content-Length'],
  maxAge: 600,
}));

const SHARES_MAP: Record<string, number> = {
  "NVDA": 24221000000, "AAPL": 14687356000, "MSFT": 7428434704, "AMZN": 10757109436,
  "GOOGL": 5867155790, "GOOG": 5499638298, "META": 2196045588, "AVGO": 4757580198,
  "TSLA": 3755723871, "COST": 443478804, "NFLX": 4210798528, "AMD": 1630600639,
  "ASML": 385417665, "PEP": 1366768315, "AZN": 1550862203, "LIN": 462347310,
  "ADBE": 397500000, "CSCO": 3941434665, "QCOM": 1054000000, "TMUS": 1082204717,
  "TXN": 910092791, "AMGN": 539708274, "INTU": 273537000, "ISRG": 354162842,
  "HON": 633653119, "AMAT": 793959430, "BKNG": 774878436, "MU": 1127734051,
  "MDLZ": 1283649766, "REGN": 103021886, "LRCX": 1250571000, "VRTX": 253805417,
  "ADP": 399734282, "PANW": 815000000, "GILD": 1241569874, "MELI": 50697182,
  "SNPS": 191479325, "CDNS": 275816000, "KLAC": 1306275210, "CSX": 1858138856,
  "MAR": 263688623, "PDD": 1423396462, "INTC": 5026000000, "PYPL": 882105493,
  "ORLY": 828715079, "ADI": 487087040, "WMT": 7958079155, "NXPI": 252471079,
  "CRWD": 254564820, "WDAY": 201000000
};

interface SimStateData {
  cash: number;
  portfolio: Record<string, { shares: number; avg_price: number }>;
  transactions: any[];
  date_idx: number;
  equity_curve: { Date: string; Value: number }[];
}

const INITIAL_CASH = 100000.0;
const FEE_RATE = 0.0015; // 0.15%

// Cache trading dates globally to avoid repeated D1 reads
let cachedTradingDates: string[] | null = null;

async function getTradingDates(db: D1Database): Promise<string[]> {
  if (cachedTradingDates) return cachedTradingDates;
  
  const { results } = await db.prepare(
    "SELECT DISTINCT Date FROM daily_metrics WHERE Date >= '2025-01-01' ORDER BY Date ASC"
  ).all();
  
  cachedTradingDates = results.map((r: any) => r.Date);
  return cachedTradingDates;
}

async function getSimState(env: Bindings, tradingDates: string[]): Promise<SimStateData> {
  const val = await env.SIMULATOR_KV.get("state");
  if (val) {
    try {
      return JSON.parse(val);
    } catch (e) {
      // Fallback
    }
  }
  
  const initialDate = tradingDates[0] || "2025-01-02";
  const state: SimStateData = {
    cash: INITIAL_CASH,
    portfolio: {},
    transactions: [],
    date_idx: 0,
    equity_curve: [{ Date: initialDate, Value: INITIAL_CASH }]
  };
  return state;
}

async function saveSimState(env: Bindings, state: SimStateData) {
  await env.SIMULATOR_KV.put("state", JSON.stringify(state));
}

async function getScreenerForDate(db: D1Database, dateStr: string) {
  const { results } = await db.prepare(
    "SELECT * FROM daily_metrics WHERE Date = ?"
  ).bind(dateStr).all();
  
  if (!results || results.length === 0) {
    return null;
  }
  
  const screener = results.map((r: any) => {
    const shares = SHARES_MAP[r.Ticker] || 1e9;
    const marketCap = r.Close * shares;
    return {
      ...r,
      "Z-Score": r.Z_Score,
      "Minervini_Pass": r.Minervini_Pass === 1,
      "Minervini_Details": JSON.parse(r.Minervini_Details),
      "Market_Cap": marketCap,
    };
  });
  
  // Assign MCap_Rank
  screener.sort((a, b) => b.Market_Cap - a.Market_Cap);
  screener.forEach((s, idx) => {
    s.MCap_Rank = idx + 1;
  });
  
  // Assign Mom_Rank
  screener.sort((a, b) => (b.Momentum_3M ?? -999) - (a.Momentum_3M ?? -999));
  screener.forEach((s, idx) => {
    s.Mom_Rank = idx + 1;
  });
  
  // Assign Mom_Z
  const momValues = screener.map(s => s.Momentum_3M || 0);
  const momMean = momValues.reduce((a, b) => a + b, 0) / momValues.length;
  const momStd = Math.sqrt(momValues.map(v => Math.pow(v - momMean, 2)).reduce((a, b) => a + b, 0) / momValues.length) + 1e-9;
  screener.forEach(s => {
    s.Mom_Z = ((s.Momentum_3M || 0) - momMean) / momStd;
  });
  
  // Assign WTA_Status
  screener.forEach(s => {
    if (s.Mom_Rank === 1) {
      s.WTA_Status = "WTA-Champion";
    } else if (s.Mom_Rank > 1 && s.Mom_Rank <= 5) {
      s.WTA_Status = "WTA-Leader";
    } else {
      s.WTA_Status = "HOLD";
    }
  });
  
  return screener;
}

function getHoldingsValuation(portfolio: Record<string, { shares: number; avg_price: number }>, screener: any[]) {
  let totalVal = 0.0;
  const priceMap = new Map(screener.map(s => [s.Ticker, s.Close]));
  
  for (const [ticker, info] of Object.entries(portfolio)) {
    const price = priceMap.get(ticker) ?? info.avg_price;
    totalVal += info.shares * price;
  }
  return totalVal;
}

// 1. GET State
app.get('/api/state', async (c) => {
  const tradingDates = await getTradingDates(c.env.DB);
  const state = await getSimState(c.env, tradingDates);
  
  if (state.date_idx >= tradingDates.length) {
    state.date_idx = tradingDates.length - 1;
  }
  
  const currentDate = tradingDates[state.date_idx];
  const screener = await getScreenerForDate(c.env.DB, currentDate);
  if (!screener) {
    return c.json({ detail: "Failed to calculate market metrics." }, 500);
  }
  
  // Calculate Market summary breadth
  const totalStocks = screener.length;
  const aboveMa20 = screener.filter(s => s.Close > s.MA20).length / totalStocks;
  const aboveMa50 = screener.filter(s => s.Close > s.MA50).length / totalStocks;
  
  let marketRegime = "NEUTRAL";
  if (aboveMa20 > 0.60 && aboveMa50 > 0.60) {
    marketRegime = "BULLISH";
  } else if (aboveMa20 < 0.40 && aboveMa50 < 0.40) {
    marketRegime = "BEARISH";
  }
  
  const avgRsi = screener.map(s => s.RSI || 0).reduce((a, b) => a + b, 0) / totalStocks;
  const avgZ = screener.map(s => s["Z-Score"] || 0).reduce((a, b) => a + b, 0) / totalStocks;
  
  const posReturns = screener.filter(s => (s.Momentum_3M || 0) > 0).map(s => s.Momentum_3M || 0);
  const totalPosSum = posReturns.reduce((a, b) => a + b, 0) || 1e-9;
  const top5Sum = screener
    .slice()
    .sort((a, b) => (b.Momentum_3M || 0) - (a.Momentum_3M || 0))
    .slice(0, 5)
    .map(s => s.Momentum_3M || 0)
    .reduce((a, b) => a + b, 0);
  const wtaIndex = top5Sum / totalPosSum;
  
  const summary = {
    date: currentDate,
    above_ma20: aboveMa20,
    above_ma50: aboveMa50,
    wta_index: wtaIndex,
    market_regime: marketRegime,
    avg_rsi: avgRsi,
    avg_z: avgZ
  };
  
  // Rank changes
  const entered_top30: any[] = [];
  const exited_top30: any[] = [];
  
  if (state.date_idx > 0) {
    const prevDate = tradingDates[state.date_idx - 1];
    const prevScreener = await getScreenerForDate(c.env.DB, prevDate);
    if (prevScreener) {
      const prevRanks = new Map(prevScreener.map(s => [s.Ticker, s.Mom_Rank]));
      const currRanks = new Map(screener.map(s => [s.Ticker, s.Mom_Rank]));
      
      for (const [ticker, currRank] of currRanks.entries()) {
        const prevRank = prevRanks.get(ticker);
        if (currRank <= 30 && prevRank !== undefined && prevRank > 30) {
          entered_top30.push({ ticker, prev_rank: prevRank, curr_rank: currRank });
        } else if (prevRank !== undefined && prevRank <= 30 && currRank > 30) {
          exited_top30.push({ ticker, prev_rank: prevRank, curr_rank: currRank });
        }
      }
    }
  }
  
  return c.json({
    current_date: currentDate,
    date_index: state.date_idx,
    total_days: tradingDates.length,
    market_summary: summary,
    screener,
    entered_top30,
    exited_top30
  });
});

// 2. GET Portfolio
app.get('/api/portfolio', async (c) => {
  const tradingDates = await getTradingDates(c.env.DB);
  const state = await getSimState(c.env, tradingDates);
  const currentDate = tradingDates[state.date_idx];
  
  const screener = await getScreenerForDate(c.env.DB, currentDate);
  if (!screener) {
    return c.json({ detail: "Failed to load market data." }, 500);
  }
  
  const holdingsVal = getHoldingsValuation(state.portfolio, screener);
  const totalVal = state.cash + holdingsVal;
  const returnPct = ((totalVal - INITIAL_CASH) / INITIAL_CASH) * 100;
  
  const holdingsList = [];
  const priceMap = new Map(screener.map(s => [s.Ticker, s.Close]));
  
  for (const [ticker, info] of Object.entries(state.portfolio)) {
    const currPrice = priceMap.get(ticker) ?? info.avg_price;
    const val = info.shares * currPrice;
    const pnl = val - (info.shares * info.avg_price);
    const pnlPct = info.avg_price > 0 ? (pnl / (info.shares * info.avg_price)) * 100 : 0.0;
    
    holdingsList.push({
      ticker,
      shares: info.shares,
      avg_price: info.avg_price,
      current_price: currPrice,
      value: val,
      pnl,
      pnl_pct: pnlPct
    });
  }
  
  const totalRealizedPnl = state.transactions
    .filter(t => t.Type === "SELL")
    .reduce((sum, t) => sum + (t.Realized_PnL ?? 0.0), 0.0);
    
  return c.json({
    cash: state.cash,
    holdings_value: holdingsVal,
    total_value: totalVal,
    return_pct: returnPct,
    total_realized_pnl: totalRealizedPnl,
    holdings: holdingsList,
    transactions: state.transactions,
    equity_curve: state.equity_curve
  });
});

// 3. POST Trade
app.post('/api/trade', async (c) => {
  const body = await c.req.json();
  const ticker = (body.ticker || '').toUpperCase();
  const action = (body.action || '').toUpperCase();
  const shares = parseInt(body.shares);
  
  if (isNaN(shares) || shares <= 0) {
    return c.json({ detail: "Shares count must be positive." }, 400);
  }
  
  const tradingDates = await getTradingDates(c.env.DB);
  const state = await getSimState(c.env, tradingDates);
  const currentDate = tradingDates[state.date_idx];
  
  const screener = await getScreenerForDate(c.env.DB, currentDate);
  if (!screener) {
    return c.json({ detail: "Market data unavailable." }, 500);
  }
  
  const stock = screener.find(s => s.Ticker === ticker);
  if (!stock) {
    return c.json({ detail: `Ticker ${ticker} not found in Top 50.` }, 400);
  }
  
  const price = stock.Close;
  const cost = shares * price;
  const fee = cost * FEE_RATE;
  
  if (action === "BUY") {
    const totalCost = cost + fee;
    if (state.cash < totalCost) {
      return c.json({ detail: `Insufficient cash. Need $${totalCost.toFixed(2)}, have $${state.cash.toFixed(2)}` }, 400);
    }
    
    state.cash -= totalCost;
    if (state.portfolio[ticker]) {
      const currShares = state.portfolio[ticker].shares;
      const currAvg = state.portfolio[ticker].avg_price;
      const newShares = currShares + shares;
      const newAvg = ((currShares * currAvg) + cost) / newShares;
      state.portfolio[ticker] = { shares: newShares, avg_price: newAvg };
    } else {
      state.portfolio[ticker] = { shares, avg_price: price };
    }
    
    state.transactions.push({
      Date: currentDate,
      Ticker: ticker,
      Type: "BUY",
      Shares: shares,
      Price: price,
      Fee: fee,
      "Net Cash Flow": -totalCost
    });
  } else if (action === "SELL") {
    if (!state.portfolio[ticker] || state.portfolio[ticker].shares < shares) {
      return c.json({ detail: "Insufficient shares owned." }, 400);
    }
    
    const avgPrice = state.portfolio[ticker].avg_price;
    const costBasis = shares * avgPrice;
    
    const buyFee = costBasis * FEE_RATE;
    const sellFee = fee;
    
    const realizedPnl = cost - costBasis - buyFee - sellFee;
    const realizedPnlPct = costBasis > 0 ? (realizedPnl / costBasis) * 100 : 0.0;
    
    const netRevenue = cost - fee;
    state.cash += netRevenue;
    state.portfolio[ticker].shares -= shares;
    
    if (state.portfolio[ticker].shares === 0) {
      delete state.portfolio[ticker];
    }
    
    state.transactions.push({
      Date: currentDate,
      Ticker: ticker,
      Type: "SELL",
      Shares: shares,
      Price: price,
      Fee: fee,
      "Net Cash Flow": netRevenue,
      Realized_PnL: realizedPnl,
      Realized_PnL_Pct: realizedPnlPct
    });
  } else {
    return c.json({ detail: "Invalid trade action. Use BUY or SELL." }, 400);
  }
  
  await saveSimState(c.env, state);
  return c.json({ status: "success", message: `Successfully executed ${action} of ${shares} ${ticker}` });
});

// 4. POST Step
app.post('/api/step', async (c) => {
  const body = await c.req.json();
  const days = parseInt(body.days);
  if (isNaN(days)) {
    return c.json({ detail: "Invalid days value" }, 400);
  }
  
  const tradingDates = await getTradingDates(c.env.DB);
  const state = await getSimState(c.env, tradingDates);
  
  let newIdx = state.date_idx + days;
  if (newIdx >= tradingDates.length) {
    state.date_idx = tradingDates.length - 1;
  } else if (newIdx < 0) {
    state.date_idx = 0;
  } else {
    state.date_idx = newIdx;
  }
  
  const currentDate = tradingDates[state.date_idx];
  const screener = await getScreenerForDate(c.env.DB, currentDate);
  if (!screener) {
    return c.json({ detail: "Failed to step simulation." }, 500);
  }
  
  const holdingsVal = getHoldingsValuation(state.portfolio, screener);
  const totalVal = state.cash + holdingsVal;
  
  // Wipe future equity curve data if we step backwards
  state.equity_curve = state.equity_curve.filter(entry => entry.Date <= currentDate);
  
  // Add new equity curve plot
  if (state.equity_curve.length === 0 || state.equity_curve[state.equity_curve.length - 1].Date !== currentDate) {
    state.equity_curve.push({ Date: currentDate, Value: totalVal });
  }
  
  await saveSimState(c.env, state);
  return c.json({ status: "success", current_date: currentDate });
});

// 5. GET Chart
app.get('/api/chart/:ticker', async (c) => {
  const ticker = c.req.param('ticker').toUpperCase();
  const tradingDates = await getTradingDates(c.env.DB);
  const state = await getSimState(c.env, tradingDates);
  const currentDate = tradingDates[state.date_idx];
  
  const { results } = await c.env.DB.prepare(`
    SELECT Date, Open, High, Low, Close, Volume
    FROM daily_metrics
    WHERE Ticker = ? AND Date <= ?
    ORDER BY Date DESC
    LIMIT 60
  `).bind(ticker, currentDate).all();
  
  const chartData = results.slice().reverse();
  return c.json(chartData);
});

// 6. POST Reset
app.post('/api/reset', async (c) => {
  const tradingDates = await getTradingDates(c.env.DB);
  const initialDate = tradingDates[0] || "2025-01-02";
  const state: SimStateData = {
    cash: INITIAL_CASH,
    portfolio: {},
    transactions: [],
    date_idx: 0,
    equity_curve: [{ Date: initialDate, Value: INITIAL_CASH }]
  };
  
  await saveSimState(c.env, state);
  return c.json({ status: "success", message: "Simulation reset successfully." });
});

export default app;

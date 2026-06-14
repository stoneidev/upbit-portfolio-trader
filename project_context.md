# Upbit WFO Portfolio Trader: Project Context

본 문서는 리플(XRP)과 이더리움(ETH) 자산을 대상으로 15분 봉 기준의 Walk-Forward Optimization (WFO) 자율 학습 및 Z-Score 평균 회귀 전략을 수행하는 자동 매매 봇과 실시간 웹 대시보드 시스템의 통합 명세서입니다. 

다음 세션에서 AI 에이전트가 이 파일을 읽고 시스템의 아키텍처와 구동 방식을 한눈에 파악할 수 있도록 지속성 컨텍스트(Context)를 기록해 둡니다.

---

## 1. 시스템 아키텍처 & 데이터 흐름

```mermaid
graph LR
    subgraph RaspberryPi[라즈베리파이 (stoni-Pi)]
        Bot[upbit_portfolio_trader.py]
        Log[upbit_portfolio_log.txt]
        JSON[trade_history.json]
    end
    
    subgraph Cloudflare[Cloudflare 인프라]
        Worker[index.js / Worker API]
        KV[(PORTFOLIO_KV)]
        Pages[dashboard.html / Pages 호스팅]
    end
    
    subgraph Telegram[알림 채널]
        TeleBot[@stoni_fire_bot]
    end

    Bot -->|1. 매수/매도/청산| TeleBot
    Bot -->|2. 로컬 영구 저장| JSON
    Bot -->|3. 5분 주기 전송| Worker
    Worker -->|4. 데이터 저장| KV
    Pages -->|5. 5초 주기 조회| Worker
```

### 1.1. 데이터 연동 주기 (Throttling)
* **거래소 모니터링 & 매매 감시**: **30초** (기존의 촘촘한 정밀 매매 유지)
* **대시보드 전송 (KV Write)**: **10사이클(5분)**에 1번
  * 하루 KV 쓰기 횟수가 약 288회로 축소되어 Cloudflare 무료 티어(일 1,000회 제한) 내에서 무중단 상시 구동이 가능합니다.
* **대시보드 화면 조회 (KV Read)**: **5초**마다 Worker를 호출하여 화면을 부드럽게 갱신.

---

## 2. 주요 매매 로직 & 파라미터

### 2.1. 진입 및 물타기 (그리드 지정가 진입)
* **1차 매수 (L1)**: Z-Score가 최적 임계치(`z_thresh`) 미만이고, 실시간 체결강도(`vol_power`)가 `50%` 초과 시 현재가로 1시간 만료 지정가 매수 주문 등록.
* **2차 매수 (L2)**: L1 진입가 대비 **-1.0%** 도달 시 지정가 분할 매수 (비중 30%).
* **3차 매수 (L3)**: L1 진입가 대비 **-2.0%** 도달 시 지정가 분할 매수 (비중 40%).

### 2.2. 청산 및 리스크 관리 (하이브리드 주문)
* **익절 (Trailing Stop - 지정가)**: 가격이 평단가 대비 `tp_pct` 이상 도달 시 익절 활성화. 이후 최고점 대비 `0.2%` 하락 시 슬리피지 없이 지정가 청산.
* **손절 (Stop Loss - 시장가)**: 가격이 평단가 대비 `-2.0%` 도달 시 시장가 강제 청산 (편도 `0.05%` 슬리피지 적용).
* **L3 시간 컷오프 (L3 Time Cutoff - 시장가)**: L3 매수 체결 후 4시간이 지나도 익절되지 못했을 때, 손실 폭이 `-0.2%` 이상으로 회복되면(평단가 * 0.998) 시장가로 탈출하여 최대 손실(-2.0%) 방어.

### 2.3. WFO 자율 학습 엔진 (Lookback Window)
* **구동 시점**: 봇 최초 시작 시 1회 즉시 실행, 이후 **7일(1주일)**마다 자동 실행.
* **분석 데이터**: 업비트 API로부터 최근 **60일 분량의 15분 봉 캔들 (총 5,760개)** 수집.
* **최적화**: 수수료와 슬리피지를 엄격히 가정한 백테스트 시뮬레이션을 돌려 가장 수익이 좋았던 Z-Score 값(`[-1.0, -1.2, -1.5, -1.8, -2.0]` 중 택1)을 실시간 매수 진입 기준으로 셋팅.

---

## 3. 핵심 파일 역할 및 경로

### 3.1. 봇 핵심 코드
1. **[upbit_portfolio_trader.py](file:///Users/stoni/Projects/AI/upbit_portfolio_trader.py)**:
   * 실시간 XRP 및 ETH 50:50 분산 트레이딩 봇 핵심 연산 코드.
   * 거래 내역을 `trade_history.json`에 구조화하여 로컬 저장 및 KV 송신.
2. **[dashboard.html](file:///Users/stoni/Projects/AI/dashboard.html)**:
   * Lightweight Charts 기반 실시간 시세, Z-Score, 자산 성장 곡선 시각화.
   * 하단에 최근 거래 기록 테이블(체결 시간, 자산, 구분, 단계, 체결가, 수량, 총액, 수익률, 최종 잔고) 제공.
3. **[index.js](file:///Users/stoni/Projects/AI/index.js) & [wrangler.toml](file:///Users/stoni/Projects/AI/wrangler.toml)**:
   * Cloudflare Workers 및 KV 연결 설정 스크립트.

### 3.2. 배포 설정
* **[.github/workflows/deploy.yml](file:///Users/stoni/Projects/AI/.github/workflows/deploy.yml)**:
   * GitHub Actions 자동 배포용 파일 (self-hosted runner).
   * 소스코드를 라즈베리파이의 `/home/stoni/Projects/AI`로 복사합니다.
   * `sudo -n true`로 패스워드 없는 sudo 여부를 체크하여:
     * **Sudo 가능**: `upbit-portfolio-trader.service` / `upbit-portfolio-web.service`를 systemd 서비스에 등록하여 부팅 시 자동 시작 및 크래시 발생 시 자동 재기동(`Restart=always`) 보장.
     * **Sudo 불가능**: 사용자의 crontab에 `@reboot` 지시어를 삽입하여 재부팅 시 백그라운드로 자동 실행 보장.

### 3.3. 영구 저장용 파일 (라즈베리파이 로컬 `/home/stoni/Projects/AI/` 내부)
* `upbit_portfolio_log.txt`: 봇 구동 텍스트 로그.
* `equity_history.json`: 일일 마감 포트폴리오 총 잔고 이력 (2년 치, 최대 730일 보존).
* `trade_history.json`: 봇의 진입 및 청산 거래 이력 기록 보관 (최근 100개 제한).

---

## 4. 실전 가동 및 검증 완료 정보
* **대시보드 주소**: [https://upbit-portfolio-dashboard.pages.dev](https://upbit-portfolio-dashboard.pages.dev)
* **텔레그램 알림 수신**: `@stoni_fire_bot`

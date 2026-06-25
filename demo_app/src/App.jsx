import { useEffect, useMemo, useState } from "react";

const SAMPLE_NEWS = JSON.stringify(
  {
    market_news: ["미국 국채 금리 안정과 반도체 수출 회복 기대"],
    stock_news: {
      "005930.KS": ["메모리 업황 개선 기대감 확대"],
    },
  },
  null,
  2,
);

function formatKRW(value, compact = false) {
  const number = Number(value || 0);
  if (compact && Math.abs(number) >= 100_000_000) {
    return `${(number / 100_000_000).toFixed(1)}억 원`;
  }
  return `${Math.round(number).toLocaleString("ko-KR")}원`;
}

function formatQuantity(value) {
  if (!value) return "0주";
  return `${value > 0 ? "+" : ""}${Number(value).toLocaleString("ko-KR")}주`;
}

function ActionBadge({ action }) {
  return <span className={`action-badge action-${action}`}>{action}</span>;
}

function Ranking({ title, rows, tone }) {
  return (
    <section className={`ranking ranking-${tone}`}>
      <div className="section-heading">
        <h2>{title}</h2>
        <span>모델 점수</span>
      </div>
      <ol>
        {rows?.length ? (
          rows.map((row) => (
            <li key={`${title}-${row.ticker}`}>
              <span className="rank-company">
                <b>{row.company}</b>
                <small>{row.ticker}</small>
              </span>
              <strong>{row.score.toFixed(3)}</strong>
            </li>
          ))
        ) : (
          <li className="empty-row">해당 추천 없음</li>
        )}
      </ol>
    </section>
  );
}

export function App() {
  const [authStatus, setAuthStatus] = useState("checking");
  const [password, setPassword] = useState("");
  const [authError, setAuthError] = useState("");
  const [mode, setMode] = useState("sample");
  const [cash, setCash] = useState("10000000");
  const [holdingsText, setHoldingsText] = useState("{}");
  const [newsText, setNewsText] = useState("");
  const [autoCrawl, setAutoCrawl] = useState(true);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [query, setQuery] = useState("");
  const [actionFilter, setActionFilter] = useState("전체");
  const [sortDescending, setSortDescending] = useState(true);
  const [newsDialogOpen, setNewsDialogOpen] = useState(false);
  const [latestNews, setLatestNews] = useState(null);
  const [latestNewsLoading, setLatestNewsLoading] = useState(false);
  const [latestNewsError, setLatestNewsError] = useState("");
  const [portfolioApplied, setPortfolioApplied] = useState(false);

  async function requestInference(selectedMode, values = {}) {
    setLoading(true);
    setError("");
    setPortfolioApplied(false);
    try {
      const holdings = JSON.parse(values.holdingsText ?? holdingsText);
      const response = await fetch("/api/infer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mode: selectedMode,
          cash: Number(values.cash ?? cash),
          holdings,
          news_text: values.newsText ?? newsText,
          crawl_latest: selectedMode === "latest" && (values.autoCrawl ?? autoCrawl),
        }),
      });
      const body = await response.json();
      if (response.status === 401) setAuthStatus("locked");
      if (!response.ok) throw new Error(body.detail || "추론 요청에 실패했습니다.");
      setResult(body);
    } catch (requestError) {
      setError(requestError.message || "알 수 없는 오류가 발생했습니다.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    async function checkAccess() {
      try {
        const response = await fetch("/api/auth");
        const body = await response.json();
        if (!body.configured) {
          setAuthError("서버에 데모 비밀번호가 설정되지 않았습니다.");
          setAuthStatus("locked");
          setLoading(false);
          return;
        }
        if (!body.authenticated) {
          setAuthStatus("locked");
          setLoading(false);
          return;
        }
        setAuthStatus("ready");
        await requestInference("sample", {
          cash: "10000000",
          holdingsText: "{}",
          newsText: "",
        });
      } catch {
        setAuthError("접근 상태를 확인하지 못했습니다.");
        setAuthStatus("locked");
        setLoading(false);
      }
    }
    checkAccess();
    // Initial reproducible sample run only.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function submitPassword(event) {
    event.preventDefault();
    setAuthError("");
    try {
      const response = await fetch("/api/auth", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ password }),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || "비밀번호를 확인하세요.");
      setPassword("");
      setAuthStatus("ready");
      await requestInference("sample", {
        cash: "10000000",
        holdingsText: "{}",
        newsText: "",
      });
    } catch (requestError) {
      setAuthError(requestError.message || "비밀번호를 확인하세요.");
    }
  }

  const visibleRows = useMemo(() => {
    const normalisedQuery = query.trim().toLowerCase();
    return [...(result?.rows || [])]
      .filter((row) => actionFilter === "전체" || row.action === actionFilter)
      .filter(
        (row) =>
          !normalisedQuery ||
          row.ticker.toLowerCase().includes(normalisedQuery) ||
          row.company.toLowerCase().includes(normalisedQuery),
      )
      .sort((left, right) =>
        sortDescending ? right.score - left.score : left.score - right.score,
      );
  }, [result, query, actionFilter, sortDescending]);

  const summary = result?.summary;

  async function loadNewsFile(event) {
    const [file] = event.target.files || [];
    if (!file) return;
    setNewsText(await file.text());
  }

  async function openLatestNews() {
    setNewsDialogOpen(true);
    setLatestNewsLoading(true);
    setLatestNews(null);
    setLatestNewsError("");
    try {
      const response = await fetch("/api/news/latest");
      const body = await response.json();
      if (response.status === 401) setAuthStatus("locked");
      if (!response.ok) throw new Error(body.detail || "최근 뉴스를 불러오지 못했습니다.");
      setLatestNews(body);
    } catch (requestError) {
      setLatestNewsError(requestError.message || "최근 뉴스를 불러오지 못했습니다.");
    } finally {
      setLatestNewsLoading(false);
    }
  }

  function applyResultToInputs() {
    if (!result?.rows || !result?.summary) return;
    const nextHoldings = Object.fromEntries(
      result.rows
        .filter((row) => Number(row.estimated_shares) > 0)
        .map((row) => [row.ticker, Number(row.estimated_shares)]),
    );
    const nextCash = Math.max(0, Math.floor(Number(result.summary.estimated_cash) || 0));
    setCash(String(nextCash));
    setHoldingsText(JSON.stringify(nextHoldings, null, 2));
    setPortfolioApplied(true);
  }

  if (authStatus !== "ready") {
    return (
      <div className="auth-shell">
        <main className="auth-panel">
          <span className="auth-brand">FinRL Analyst Console</span>
          <h1>데모 접근</h1>
          <p>모델을 실행하려면 컨퍼런스 비밀번호를 입력하세요.</p>
          {authStatus === "checking" ? (
            <div className="auth-checking">접근 상태 확인 중</div>
          ) : (
            <form onSubmit={submitPassword}>
              <label htmlFor="demo-password">비밀번호</label>
              <input
                id="demo-password"
                type="password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                autoComplete="current-password"
                autoFocus
                required
              />
              <button type="submit">입장</button>
            </form>
          )}
          {authError && <p className="auth-error">{authError}</p>}
          <small>연구용 데모 · 실제 투자 주문 아님</small>
        </main>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-block">
          <span className="brand-name">FinRL Analyst Console</span>
          <span className="research-label">연구용 데모 · 실제 투자 주문 아님</span>
        </div>
        <div className="header-controls">
          <label className="mode-control">
            <span>모드</span>
            <select value={mode} onChange={(event) => setMode(event.target.value)}>
              <option value="sample">샘플 모드</option>
              <option value="latest">최신 데이터</option>
            </select>
          </label>
          <div className="header-meta">
            <span>데이터 기준</span>
            <strong>{result?.data_date || "준비 중"}</strong>
          </div>
          <div className="header-meta">
            <span>파이프라인</span>
            <strong className={error ? "status-error" : "status-ready"}>
              {error ? "확인 필요" : loading ? "실행 중" : "정상"}
            </strong>
          </div>
        </div>
      </header>

      <aside className="control-panel">
        <div className="aside-title">
          <h1>입력 데이터</h1>
          <p>{mode === "sample" ? "저장 관측값 재현" : "장 마감 데이터 추론"}</p>
        </div>

        <label className="field-block">
          <span>뉴스 입력</span>
          <small>일반 텍스트 또는 market_news / stock_news JSON</small>
          <textarea
            value={newsText}
            onChange={(event) => setNewsText(event.target.value)}
            placeholder={SAMPLE_NEWS}
            rows={9}
            spellCheck="false"
          />
          <input
            className="file-input"
            type="file"
            accept="application/json,.json,text/plain,.txt"
            onChange={loadNewsFile}
            aria-label="뉴스 JSON 또는 텍스트 파일 업로드"
          />
        </label>

        <label className="field-block">
          <span>시작 현금</span>
          <input
            type="number"
            min="0"
            step="100000"
            value={cash}
            onChange={(event) => setCash(event.target.value)}
          />
        </label>

        <label className="field-block">
          <span>현재 보유 수량</span>
          <small>{'{"005930.KS": 10}'} 형식, 미입력 종목은 0주</small>
          <textarea
            value={holdingsText}
            onChange={(event) => setHoldingsText(event.target.value)}
            rows={4}
            spellCheck="false"
          />
        </label>

        <section className="constraints">
          <h2>모델 제약</h2>
          <dl>
            <div>
              <dt>대상</dt>
              <dd>고정 50종목</dd>
            </div>
            <div>
              <dt>보유 임계값</dt>
              <dd>|score| &lt; 0.10</dd>
            </div>
            <div>
              <dt>종목별 상한</dt>
              <dd>500주</dd>
            </div>
            <div>
              <dt>거래비용</dt>
              <dd>0.3%</dd>
            </div>
          </dl>
        </section>

        {mode === "latest" && (
          <div className="latest-options">
            <label className="auto-crawl-option">
              <input
                type="checkbox"
                checked={autoCrawl}
                onChange={(event) => setAutoCrawl(event.target.checked)}
              />
              <span>최신 뉴스 자동 수집</span>
            </label>
            <p className="mode-note">
              15:30 KST 이후 완료된 t일 데이터로 t+1 추천을 생성합니다. 직접 입력한 뉴스는
              자동 수집 결과에 추가됩니다.
            </p>
          </div>
        )}

        <button
          className="run-button"
          type="button"
          disabled={loading}
          onClick={() => requestInference(mode)}
        >
          {loading ? "모델 실행 중" : "모델 실행"}
        </button>

        <button
          className="apply-result-button"
          type="button"
          disabled={loading || !result}
          onClick={applyResultToInputs}
        >
          결과를 다음 입력으로 적용
        </button>
        <p className="apply-result-note">
          {portfolioApplied
            ? "예상 현금과 체결 후 보유 수량을 입력란에 반영했습니다."
            : "현재 결과의 예상 현금·보유량을 다음 실행 입력으로 복사합니다."}
        </p>

        {error && <p className="error-message">{error}</p>}
        <p className="run-meta">
          마지막 실행 {result?.generated_at ? new Date(result.generated_at).toLocaleString("ko-KR") : "-"}
        </p>
      </aside>

      <main className="workspace">
        <section className="pipeline-strip" aria-label="파이프라인 상태">
          {(result?.pipeline || [
            { name: "Yahoo 데이터", detail: "대기" },
            { name: "Upstage", detail: "대기" },
            { name: "Autoencoder", detail: "대기" },
            { name: "PPO", detail: "대기" },
          ]).map((stage) => (
            <div className="pipeline-stage" key={stage.name}>
              <span>{stage.name}</span>
              <strong>{loading ? "처리 중" : stage.status === "cached" ? "재현" : "완료"}</strong>
              <small>{stage.detail}</small>
            </div>
          ))}
        </section>

        <section className="summary-grid" aria-busy={loading}>
          <Ranking title="매수 상위 5" rows={result?.top_buys} tone="buy" />
          <Ranking title="매도 상위 5" rows={result?.top_sells} tone="sell" />

          <section className="summary-metric">
            <div className="section-heading"><h2>행동 분포</h2></div>
            <div className="metric-primary">{summary?.hold_count ?? "-"}<small> / 50 보유</small></div>
            <dl className="distribution-list">
              <div><dt>매수</dt><dd>{summary?.buy_count ?? "-"}</dd></div>
              <div><dt>매도</dt><dd>{summary?.sell_count ?? "-"}</dd></div>
            </dl>
          </section>

          <section className="summary-metric cash-summary">
            <div className="section-heading"><h2>예상 현금</h2></div>
            <div className="metric-primary cash-value">
              {summary ? formatKRW(summary.estimated_cash, true) : "-"}
            </div>
            <dl className="cash-list">
              <div><dt>시작 현금</dt><dd>{summary ? formatKRW(summary.starting_cash) : "-"}</dd></div>
              <div><dt>수량 기준</dt><dd>t일 종가 추정</dd></div>
            </dl>
          </section>

          <section className="summary-metric model-summary">
            <div className="section-heading"><h2>모델 정보</h2></div>
            <dl className="model-list">
              <div><dt>모델</dt><dd>PPO</dd></div>
              <div><dt>State</dt><dd>1,467차원</dd></div>
              <div><dt>Action</dt><dd>50차원</dd></div>
              <div><dt>주문</dt><dd>비활성</dd></div>
            </dl>
          </section>
        </section>

        <section className="results-section">
          <div className="results-toolbar">
            <div>
              <h2>전체 50종목 추천</h2>
              <p>예상 매수 수량은 가용 현금을 양수 action score 비례로 분산한 값입니다.</p>
            </div>
            <div className="table-controls">
              <button type="button" onClick={openLatestNews}>최근 수집 뉴스</button>
              <input
                type="search"
                placeholder="종목명 또는 티커 검색"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
              />
              <select value={actionFilter} onChange={(event) => setActionFilter(event.target.value)}>
                <option>전체</option>
                <option>매수</option>
                <option>보유</option>
                <option>매도</option>
              </select>
              <button type="button" onClick={() => setSortDescending((value) => !value)}>
                점수 {sortDescending ? "내림차순" : "오름차순"}
              </button>
            </div>
          </div>

          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>순번</th>
                  <th>티커</th>
                  <th>종목명</th>
                  <th className="number-cell">모델 점수</th>
                  <th>추천</th>
                  <th className="number-cell">예상 수량</th>
                  <th className="number-cell">예상 금액</th>
                  <th className="number-cell">t일 종가</th>
                  <th>입력 기준</th>
                </tr>
              </thead>
              <tbody>
                {visibleRows.map((row, index) => (
                  <tr key={row.ticker}>
                    <td>{index + 1}</td>
                    <td className="ticker-cell">{row.ticker}</td>
                    <td className="company-cell">{row.company}</td>
                    <td className={`number-cell score-${row.action}`}>{row.score.toFixed(3)}</td>
                    <td><ActionBadge action={row.action} /></td>
                    <td className="number-cell">{formatQuantity(row.estimated_quantity)}</td>
                    <td className="number-cell">{formatKRW(row.estimated_amount)}</td>
                    <td className="number-cell">{formatKRW(row.price)}</td>
                    <td>{row.freshness}</td>
                  </tr>
                ))}
                {!visibleRows.length && !loading && (
                  <tr><td colSpan="9" className="no-results">조건에 맞는 종목이 없습니다.</td></tr>
                )}
              </tbody>
            </table>
          </div>
          <footer className="results-footer">
            <span>{visibleRows.length}개 종목 표시</span>
            <span>실제 체결과 수익을 보장하지 않음</span>
          </footer>
        </section>
      </main>

      {newsDialogOpen && (
        <div className="news-dialog-backdrop" role="presentation">
          <section className="news-dialog" role="dialog" aria-modal="true" aria-labelledby="news-dialog-title">
            <header>
              <div>
                <h2 id="news-dialog-title">최근 수집 뉴스</h2>
                <p>
                  {latestNews
                    ? `${latestNews.date} · 시장 ${latestNews.market_count}건 · 종목 ${latestNews.stock_count}건`
                    : "최신 데이터 모드의 서버 캐시를 조회합니다."}
                </p>
              </div>
              <button type="button" onClick={() => setNewsDialogOpen(false)} aria-label="닫기">닫기</button>
            </header>

            {latestNewsLoading && <p className="news-dialog-status">불러오는 중</p>}
            {latestNewsError && <p className="error-message news-dialog-status">{latestNewsError}</p>}

            {latestNews && !latestNewsLoading && (
              <div className="news-dialog-columns">
                <section>
                  <h3>시장 뉴스</h3>
                  <ul>
                    {latestNews.market_news.map((item, index) => (
                      <li key={`market-${index}-${item.title}`}>
                        <span>{item.keyword || "시장"}</span>
                        {item.link ? <a href={item.link} target="_blank" rel="noreferrer">{item.title}</a> : <b>{item.title}</b>}
                        <small>{item.press || item.source || "출처 정보 없음"}</small>
                      </li>
                    ))}
                  </ul>
                </section>
                <section>
                  <h3>종목 뉴스</h3>
                  <ul>
                    {latestNews.stock_news.map((item, index) => (
                      <li key={`stock-${index}-${item.ticker}-${item.title}`}>
                        <span>{item.company || item.ticker || "종목"}</span>
                        {item.link ? <a href={item.link} target="_blank" rel="noreferrer">{item.title}</a> : <b>{item.title}</b>}
                        <small>{item.press || item.source || "출처 정보 없음"}</small>
                      </li>
                    ))}
                  </ul>
                </section>
              </div>
            )}
          </section>
        </div>
      )}
    </div>
  );
}

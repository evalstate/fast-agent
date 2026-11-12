function BenchmarksPage({ go }) {
  const { Tabs, Card, Table, Tag, Link, FootnoteMark } = window.FastAgentDesignSystem_3898e4;
  const B = window.FA_BENCH, A = window.FA_ASSETS;
  const [id, setId] = React.useState('frontier');
  const cmp = B.comparisons.find(c => c.id === id);
  const [sel, setSel] = React.useState(0);
  React.useEffect(() => setSel(0), [id]);
  const r = cmp.results[sel];
  const money = v => '$' + v.toFixed(2);
  const rows = cmp.results.map(x => ({
    h: <span style={{ fontWeight: x.fa ? 800 : 500 }}>{x.harness}</span>, m: x.model,
    s: x.score.toFixed(1) + '%', c: (x.est ? '~' : '') + money(x.cost),
    t: x.total ? (x.est ? '~' : '') + '$' + x.total.toFixed(2) : '—',
    b: x.badge ? <Tag variant="muted">{x.badge}</Tag> : <span style={{ fontSize: 13, color: 'var(--text-muted)' }}>{x.attempts}</span>,
  }));
  return <main>
    <section style={{ ...faWrap, display: 'grid', gridTemplateColumns: 'minmax(0,1fr) clamp(140px, 20vw, 230px)', gap: 32, alignItems: 'end', paddingTop: 56 }}>
      <div style={{ paddingBottom: 32 }}>
        <div style={{ ...faLabel, marginBottom: 14 }}>{B.title} · {B.date}</div>
        <h1 style={{ ...faVoice, fontSize: 'clamp(48px, 6vw, 72px)', lineHeight: 0.98 }}>Benchmarks</h1>
        <p style={{ fontSize: 18, lineHeight: 1.55, margin: '18px 0 0', maxWidth: 640 }}>Accuracy against cost per task. Run totals cover 89 tasks with five trials per task unless stated otherwise. <strong>~ marks estimated costs.</strong></p>
      </div>
      <img src={A + 'illustration/presenter.png'} alt="" style={{ width: '100%', display: 'block' }} />
    </section>
    <div style={{ borderTop: 'var(--border)' }}></div>

    <section style={{ ...faWrap, paddingTop: 28 }}>
      <Tabs tabs={B.comparisons.map(c => ({ value: c.id, label: c.label }))} value={id} onChange={setId} />
      <p style={{ ...faVoice, fontSize: 28, lineHeight: 1.2, margin: '28px 0 24px', maxWidth: 900, textWrap: 'pretty' }}>{cmp.claim}</p>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 24, alignItems: 'flex-start' }}>
        <Card style={{ padding: '16px 16px 8px', flex: '999 1 560px', minWidth: 0 }}><BenchmarkChart cmp={cmp} selected={sel} onSelect={setSel} /></Card>
        <Card variant={r.fa ? 'paper' : 'inset'} style={{ flex: '1 1 280px', display: 'flex', flexDirection: 'column', gap: 14 }}>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <Tag variant={r.fa ? 'accent' : 'outline'}>{r.harness}</Tag>{r.badge && <Tag variant="muted">{r.badge}</Tag>}
          </div>
          <div style={{ fontSize: 20, fontWeight: 800, lineHeight: 1.25 }}>{r.model}</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
            <div><div style={faLabel}>Accuracy</div><div style={{ fontSize: 34, fontWeight: 900, fontVariantNumeric: 'tabular-nums', lineHeight: 1.1 }}>{r.score.toFixed(1)}%</div></div>
            <div><div style={faLabel}>Cost / task</div><div style={{ fontSize: 34, fontWeight: 900, fontVariantNumeric: 'tabular-nums', lineHeight: 1.1 }}>{r.est ? '~' : ''}{money(r.cost)}</div></div>
          </div>
          <div style={{ fontSize: 13, lineHeight: 1.6, borderTop: '2px solid var(--line)', paddingTop: 12 }}>
            <div><strong>Tokens in / out:</strong> {r.tokens}</div>
            <div><strong>Date:</strong> {r.date}</div>
            <div><strong>Runs:</strong> {r.attempts}</div>
          </div>
          <div style={{ fontSize: 12, lineHeight: 1.55, color: 'var(--text-muted)' }}>{r.note}</div>
          <Link href="#" onClick={e => e.preventDefault()} style={{ fontSize: 14, alignSelf: 'flex-start', whiteSpace: 'nowrap' }}>View run ❯</Link>
        </Card>
      </div>
      <div style={{ marginTop: 24 }}>
        <Table columns={[{ key: 'h', label: 'Harness' }, { key: 'm', label: 'Model' }, { key: 's', label: 'Accuracy', numeric: true }, { key: 'c', label: 'Cost / task', numeric: true }, { key: 't', label: 'Run total', numeric: true }, { key: 'b', label: 'Status' }]}
          rows={rows} highlight={(row, i) => cmp.results[i].winner}
          footnote={<span>Amber rows are fast-agent’s headline results. Provisional results are pending Terminal-Bench leaderboard review. Costs are estimates at configured rates, not billed spend.</span>} />
      </div>
    </section>

    <section style={{ ...faWrap, paddingTop: 72, display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 48 }}>
      <div>
        <h2 style={{ ...faVoice, fontSize: 34, lineHeight: 1.05 }}>Methodology & disclaimers</h2>
        <p style={{ fontSize: 16, lineHeight: 1.6 }}>Frontier and GPT-5.6 charts use published leaderboard results and provisional submissions. Badges identify provisional results and pricing adjustments. Each <em>View run</em> link leads to the source submission or published result.</p>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <Link href="#">Leaderboard methodology and submission rules</Link>
          <Link href="#">Published leaderboard</Link>
        </div>
      </div>
      <Card variant="inset">
        <div style={{ ...faLabel, marginBottom: 10 }}>Value (6hr)</div>
        <p style={{ fontSize: 15, lineHeight: 1.6, margin: 0 }}>These experiments use a <strong>six-hour agent timeout per trial</strong> rather than standard task timeouts. They are not standard leaderboard submissions. Costs use recorded Harbor configured-rate estimates, snapshot 5 September 2026.</p>
        <div style={{ marginTop: 14 }}><Link href="#">Six-hour results, source jobs and cost accounting ❯</Link></div>
      </Card>
    </section>
  </main>;
}
window.BenchmarksPage = BenchmarksPage;

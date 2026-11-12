function BenchmarkChart({ cmp, selected, onSelect }) {
  const W = 760, H = 440, m = { l: 56, r: 24, t: 20, b: 52 };
  const [hover, setHover] = React.useState(null);
  const sx = c => m.l + (c - cmp.x.min) / (cmp.x.max - cmp.x.min) * (W - m.l - m.r);
  const sy = s => H - m.b - (s - cmp.y.min) / (cmp.y.max - cmp.y.min) * (H - m.t - m.b);
  const money = v => '$' + (v < 0.1 ? v.toFixed(2) : v.toFixed(2));
  const A = window.FA_ASSETS;
  return <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', display: 'block', fontFamily: 'var(--font-read)' }} role="img" aria-label={cmp.label + ' accuracy against cost'}>
    {cmp.y.ticks.map(t => <g key={'y' + t}>
      <line x1={m.l} x2={W - m.r} y1={sy(t)} y2={sy(t)} stroke="rgba(8,44,52,0.14)" strokeWidth="2" />
      <text x={m.l - 10} y={sy(t) + 4} textAnchor="end" fontSize="12" fontWeight="700" fill="#082C34" style={{ fontVariantNumeric: 'tabular-nums' }}>{t}%</text>
    </g>)}
    {cmp.x.ticks.map(t => <text key={'x' + t} x={sx(t)} y={H - m.b + 22} textAnchor="middle" fontSize="12" fontWeight="700" fill="#082C34" style={{ fontVariantNumeric: 'tabular-nums' }}>{money(t)}</text>)}
    <line x1={m.l} x2={W - m.r} y1={H - m.b} y2={H - m.b} stroke="#082C34" strokeWidth="2" />
    <line x1={m.l} x2={m.l} y1={m.t} y2={H - m.b} stroke="#082C34" strokeWidth="2" />
    <text x={W - m.r} y={H - 8} textAnchor="end" fontSize="11" fontWeight="800" letterSpacing="1.5" fill="#082C34">COST PER TASK ❯ (~ ESTIMATED)</text>
    <text x={m.l + 8} y={m.t + 12} fontSize="11" fontWeight="800" letterSpacing="1.5" fill="#082C34">ACCURACY</text>
    {cmp.results.map((r, i) => {
      const x = sx(r.cost), y = sy(r.score), on = selected === i, hv = hover === i;
      const lp = r.lp || 'right';
      const tx = lp === 'left' ? x - 20 : lp === 'right' ? x + 20 : x, ty = lp === 'bottom' ? y + 30 : lp === 'top' ? y - 36 : y + 4;
      const anchor = lp === 'left' ? 'end' : lp === 'right' ? 'start' : 'middle';
      return <g key={i} style={{ cursor: 'pointer' }} onClick={() => onSelect(i)} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
        <circle cx={x} cy={y} r="22" fill="transparent" />
        {r.fa
          ? <g style={{ transform: `translate(${x}px, ${y}px) rotate(${hv || on ? 36 : 0}deg) scale(${hv || on ? 1.15 : 1})`, transition: 'transform var(--dur-ui) var(--ease-pop)' }}>
              <g transform="translate(-15 -15) scale(0.3)">{[0, 72, 144, 216, 288].map(a => <rect key={a} x="39" y="3" width="22" height="50" rx="11" transform={'rotate(' + a + ' 50 50)'} fill={r.winner ? '#FFB52E' : '#FFF7E8'} stroke="#082C34" strokeWidth={r.winner ? 0 : 7} />)}{!r.winner && [0, 72, 144, 216, 288].map(a => <rect key={'f' + a} x="39" y="3" width="22" height="50" rx="11" transform={'rotate(' + a + ' 50 50)'} fill="#FFF7E8" />)}</g>
            </g>
          : <circle cx={x} cy={y} r={hv || on ? 9 : 7.5} fill="#FFF7E8" stroke="#082C34" strokeWidth="2.5" style={{ transition: 'r var(--dur-hover) var(--ease-ui)' }} />}
        {on && <circle cx={x} cy={y} r="21" fill="none" stroke="#277C80" strokeWidth="3" />}
        <text x={tx} y={ty} textAnchor={anchor} fontSize="13" fontWeight={r.fa ? 800 : 600} fill="#082C34">{r.model}</text>
        <text x={tx} y={ty + 15} textAnchor={anchor} fontSize="12" fontWeight="500" fill="rgba(8,44,52,0.72)" style={{ fontVariantNumeric: 'tabular-nums' }}>{r.harness} · {r.score.toFixed(1)}% · {r.est ? '~' : ''}{money(r.cost)}</text>
      </g>;
    })}
  </svg>;
}
window.BenchmarkChart = BenchmarkChart;

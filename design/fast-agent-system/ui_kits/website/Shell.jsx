const { Wordmark, Tabs, Link } = window.FastAgentDesignSystem_3898e4;
const A = '../../assets/';
const faVoice = { fontFamily: 'var(--font-voice)', fontWeight: 900, fontVariationSettings: 'var(--voice-settings)', letterSpacing: 'var(--track-voice)', margin: 0 };
const faLabel = { fontSize: 11, fontWeight: 800, letterSpacing: 'var(--track-label)', textTransform: 'uppercase' };
const faWrap = { maxWidth: 1200, margin: '0 auto', padding: '0 32px', boxSizing: 'border-box' };

const NAV = [
  { value: 'home', label: 'fast-agent' }, { value: 'benchmarks', label: 'Benchmarks' }, { value: 'guides', label: 'Guides' },
  { value: 'agents', label: 'Agents' }, { value: 'models', label: 'Models' }, { value: 'acp', label: 'ACP' },
  { value: 'a2a', label: 'A2A' }, { value: 'mcp', label: 'MCP' }, { value: 'reference', label: 'Reference' },
];

function SiteHeader({ route, go }) {
  const [q, setQ] = React.useState('');
  return <header style={{ background: 'var(--ivory)', borderBottom: 'var(--border)', position: 'sticky', top: 0, zIndex: 20 }}>
    <div style={{ background: 'var(--petrol)', color: 'var(--ivory)', fontSize: 14, fontWeight: 600, textAlign: 'center', padding: '8px 16px' }}>
      Join the fast-agent community on <a href="https://discord.gg/xg5cJ7ndN6" style={{ color: 'var(--ivory)', fontWeight: 800, textDecoration: 'none', boxShadow: 'inset 0 -2px 0 var(--amber)' }}>Discord</a>
    </div>
    <div style={{ ...faWrap, display: 'flex', alignItems: 'center', gap: 24, height: 64 }}>
      <a href="#home" onClick={e => { e.preventDefault(); go('home'); }} style={{ textDecoration: 'none', flex: 'none', whiteSpace: 'nowrap' }}><Wordmark size={26} assetBase={A} /></a>
      <div style={{ flex: 1 }}></div>
      <label style={{ display: 'flex', alignItems: 'center', gap: 8, border: 'var(--border)', borderRadius: 'var(--radius-sm)', background: 'var(--paper)', padding: '0 10px', height: 36, width: 240, flex: '0 1 240px', minWidth: 120 }}>
        <span style={{ fontFamily: 'var(--font-code)', fontSize: 13 }}>❯</span>
        <input value={q} onChange={e => setQ(e.target.value)} placeholder="Search the docs" style={{ border: 0, outline: 'none', background: 'transparent', font: '500 14px var(--font-read)', color: 'var(--petrol)', flex: 1, minWidth: 0 }} />
        <span style={{ fontFamily: 'var(--font-code)', fontSize: 11, border: '2px solid var(--line)', borderRadius: 4, padding: '0 5px' }}>/</span>
      </label>
      <a href="https://github.com/evalstate/fast-agent" style={{ fontFamily: 'var(--font-code)', fontSize: 13, color: 'var(--petrol)', textDecoration: 'none', display: 'flex', flexDirection: 'column', lineHeight: 1.3 }}>
        <span style={{ fontWeight: 500, whiteSpace: 'nowrap' }}>evalstate/fast-agent</span><span style={{ color: 'var(--text-muted)', fontSize: 12 }}>GitHub</span>
      </a>
    </div>
    <div style={{ ...faWrap, overflowX: 'auto' }}><Tabs tabs={NAV} value={route} onChange={go} /></div>
  </header>;
}

function SiteFooter({ go }) {
  const col = (title, links) => <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
    <div style={{ ...faLabel, color: 'var(--amber)' }}>{title}</div>
    {links.map(([l, r]) => <a key={l} href={'#' + r} onClick={e => { if (!r.startsWith('http')) { e.preventDefault(); go(r); } }} style={{ color: 'var(--ivory)', fontSize: 15, fontWeight: 600, textDecoration: 'none' }}>{l}</a>)}
  </div>;
  return <footer style={{ background: 'var(--petrol)', color: 'var(--ivory)', marginTop: 96 }}>
    <div style={{ ...faWrap, padding: '56px 32px 40px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 32 }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 14, whiteSpace: 'nowrap' }}>
        <Wordmark size={30} inverse assetBase={A} />
        <p style={{ margin: 0, fontSize: 15, lineHeight: 1.6, maxWidth: 300, opacity: 0.85 }}>Simple, extendable agents. Code, build and evaluate.</p>
      </div>
      {col('Docs', [['Getting started', 'guides'], ['Agents', 'agents'], ['Models', 'models'], ['Reference', 'reference']])}
      {col('Proof', [['Benchmarks', 'benchmarks'], ['Methodology', 'benchmarks'], ['Articles & blog posts', 'reference']])}
      {col('Community', [['Discord', 'https://discord.gg/xg5cJ7ndN6'], ['X / @llmindsetuk', 'https://x.com/llmindsetuk'], ['GitHub', 'https://github.com/evalstate']])}
    </div>
    <div style={{ ...faWrap, padding: '16px 32px 28px', borderTop: '2px solid var(--petrol-2)', fontSize: 12, display: 'flex', justifyContent: 'space-between', opacity: 0.8 }}>
      <span>© 2025-2026 llmindset.co.uk</span><span style={{ fontFamily: 'var(--font-code)' }}>Made with Zensical</span>
    </div>
  </footer>;
}

function CopyCode({ lines, title, onCopy, style }) {
  const { CodeBlock, Button } = window.FastAgentDesignSystem_3898e4;
  return <div style={{ position: 'relative', ...style }}>
    <CodeBlock title={title} lines={lines} />
    <div style={{ position: 'absolute', top: title ? 4 : 10, right: 10 }}>
      <button onClick={onCopy} style={{ background: 'transparent', color: 'var(--ivory)', border: '2px solid var(--petrol-2)', borderRadius: 'var(--radius-sm)', font: '700 12px var(--font-read)', padding: '3px 9px', cursor: 'pointer' }}>Copy</button>
    </div>
  </div>;
}

Object.assign(window, { SiteHeader, SiteFooter, CopyCode, faVoice, faLabel, faWrap, FA_ASSETS: A });

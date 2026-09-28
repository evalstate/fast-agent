function DocsPage({ section, toast }) {
  const { Card, Link } = window.FastAgentDesignSystem_3898e4;
  const A = window.FA_ASSETS;
  const side = ['Getting Started', 'Core Concepts', 'Migrating to 0.10', 'Subagents', 'TUI', 'Codex', 'Migrate Automations', 'Agent Skills', 'Batch Processing', 'GEPA Optimization', 'Structured Outputs', 'Compaction'];
  const [cur, setCur] = React.useState('Getting Started');
  const toc = ['Install or upgrade', 'Run', 'Run a card pack', 'Instruction file', 'Model override'];
  const h2 = { fontSize: 24, fontWeight: 800, margin: '40px 0 12px', lineHeight: 1.25 };
  const p = { fontSize: 16, lineHeight: 1.6, margin: '0 0 14px' };
  const code = t => <span style={{ fontFamily: 'var(--font-code)', fontSize: 14, background: 'var(--ivory-deep)', padding: '1px 5px', borderRadius: 4 }}>{t}</span>;
  const block = lines => <CopyCode title="bash" onCopy={() => toast('Copied to clipboard.')} lines={lines.map(t => ({ text: t, cmd: true }))} style={{ margin: '0 0 16px' }} />;
  return <main style={{ ...faWrap, display: 'grid', gridTemplateColumns: '220px minmax(0,1fr) 200px', gap: 48, paddingTop: 40 }}>
    <nav style={{ display: 'flex', flexDirection: 'column', gap: 2, position: 'sticky', top: 180, alignSelf: 'start' }}>
      <div style={{ ...faLabel, marginBottom: 10 }}>{section}</div>
      {side.map(s => <a key={s} href="#" onClick={e => { e.preventDefault(); setCur(s); }} style={{ fontSize: 15, fontWeight: cur === s ? 800 : 500, color: 'var(--petrol)', textDecoration: 'none', padding: '6px 10px', borderRadius: 'var(--radius-sm)', background: cur === s ? 'var(--ivory-deep)' : 'transparent' }}>{s}</a>)}
    </nav>
    <article style={{ minWidth: 0, maxWidth: 720 }}>
      <div style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 10 }}>{section} ❯ {cur}</div>
      <h1 style={{ ...faVoice, fontSize: 56, lineHeight: 1 }}>{cur}</h1>
      {cur !== 'Getting Started' && <p style={{ ...p, marginTop: 20, color: 'var(--text-muted)' }}>Placeholder: this kit only recreates the Getting Started page body. Other docs pages share this layout.</p>}
      <h2 id="install" style={h2}>Install or upgrade</h2>
      {block(['uv tool install -U fast-agent-mcp'])}
      <p style={p}>If you have multiple Python versions installed, pin the one required by fast-agent:</p>
      {block(['uv tool install -U fast-agent-mcp --python 3.12'])}
      <h2 style={h2}>Run</h2>
      {block(['fast-agent go'])}
      <h2 style={h2}>Run a card pack</h2>
      {block(['fast-agent go --pack analyst --model haiku', 'fast-agent go --pack analyst --pack-registry ./marketplace.json --agent planner --model haiku'])}
      <p style={p}>{code('--pack')} installs the pack into the selected fast-agent home if needed, then launches {code('go')} normally. {code('--model')} is a fallback for cards without an explicit model setting.</p>
      <h2 style={h2}>Instruction file</h2>
      {block(['fast-agent go -i prompt.md', 'fast-agent go -i https://gist.github.com/....'])}
      <h2 style={h2}>Model override</h2>
      {block(['fast-agent go --model sonnet'])}
      <div style={{ display: 'flex', justifyContent: 'space-between', borderTop: 'var(--border)', marginTop: 40, paddingTop: 18, fontSize: 15 }}>
        <span style={{ color: 'var(--text-muted)' }}>❮ Previous: fast-agent</span><Link href="#">Next: Core Concepts ❯</Link>
      </div>
    </article>
    <aside style={{ position: 'sticky', top: 180, alignSelf: 'start' }}>
      <div style={{ ...faLabel, marginBottom: 10 }}>On this page</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8, borderLeft: '2px solid var(--line)', paddingLeft: 12, fontSize: 14 }}>
        {toc.map((t, i) => <a key={t} href="#" onClick={e => e.preventDefault()} style={{ color: 'var(--petrol)', textDecoration: 'none', fontWeight: i === 0 ? 800 : 500 }}>{t}</a>)}
      </div>
      <div style={{ position: 'relative', marginTop: 190 }}>
        <img src={A + 'illustration/sitter.png'} alt="" width="150" style={{ position: 'absolute', left: 16, top: -155, zIndex: 1, display: 'block' }} />
        <Card padding={16} style={{ paddingTop: 150 }}>
          <div style={{ fontSize: 15, fontWeight: 800 }}>Stuck?</div>
          <div style={{ fontSize: 14, lineHeight: 1.5, margin: '4px 0 10px' }}>Ask the community on Discord.</div>
          <Link href="https://discord.gg/xg5cJ7ndN6" style={{ fontSize: 14 }}>Join Discord ❯</Link>
        </Card>
      </div>
    </aside>
  </main>;
}
window.DocsPage = DocsPage;

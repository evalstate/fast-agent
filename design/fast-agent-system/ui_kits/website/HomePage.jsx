function HomePage({ go, toast }) {
  const { Button, Card, Link, FootnoteMark, Sticker } = window.FastAgentDesignSystem_3898e4;
  const A = window.FA_ASSETS;
  const features = [
    { t: 'Extensive model support', d: 'Native providers for Anthropic, Google and OpenAI-compatible endpoints. Auto configuration for llama.cpp hosted models.', l: [['Model features', 'models']] },
    { t: 'MCP and ACP', d: 'Attach MCP servers from config or the command line. Deploy agents over ACP or MCP, with transport diagnostics.', l: [['MCP guide', 'mcp'], ['ACP guide', 'acp']] },
    { t: 'Plugin and extend', d: 'Write plugins and hooks in plain Python, or use the API directly. Distribute configurations with Card Packs.', l: [['Plugin docs', 'agents']] },
    { t: 'Control your context', d: 'Template-based system prompts. Install and update Agent Skills, prompt files and agent definitions.', l: [['Agent Skills', 'guides'], ['System prompts', 'agents']] },
  ];
  return <main>
    <section style={{ ...faWrap, display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(380px, 1fr))', gap: 56, alignItems: 'center', paddingTop: 72, paddingBottom: 72 }}>
      <div>
        <div style={{ ...faLabel, marginBottom: 18 }}>Coding agent and development toolkit</div>
        <h1 style={{ ...faVoice, fontSize: 'clamp(48px, 6.4vw, 76px)', lineHeight: 0.98, textWrap: 'balance' }}>The harness your model deserves.</h1>
        <p style={{ fontSize: 20, lineHeight: 1.5, margin: '24px 0 32px', maxWidth: 520 }}>Same model, better results. fast-agent leads on <strong>accuracy</strong> and <strong>cost efficiency</strong>.</p>
        <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
          <Button size="lg" onClick={() => go('guides')}>Try it now</Button>
          <Button size="lg" variant="secondary" onClick={() => go('guides')}>Migrate your automations</Button>
        </div>
      </div>
      <Card style={{ position: 'relative', padding: 28 }}>
        <div style={{ position: 'absolute', top: -22, right: -10 }}><Sticker>New results!</Sticker></div>
        <div style={faLabel}>Terminal-Bench 2.1 · September 2026</div>
        <div style={{ fontSize: 'clamp(56px, 7vw, 88px)', fontWeight: 900, fontVariantNumeric: 'tabular-nums', lineHeight: 1, margin: '14px 0 4px', letterSpacing: '-0.02em' }}>88.3%<FootnoteMark n={1} sprite={A + 'sprite.svg'} size={18} /></div>
        <p style={{ fontSize: 16, lineHeight: 1.55, margin: '10px 0 18px' }}>fast-agent + GPT-5.6 Sol high. 4.5 points above Claude Code + Fable 5, at 61% lower estimated cost per task.</p>
        <div style={{ borderTop: '2px solid var(--line)', paddingTop: 12, display: 'flex', flexDirection: 'column', gap: 12 }}>
          <span style={{ fontSize: 12, lineHeight: 1.5, color: 'var(--text-muted)' }}>1. Provisional, pending leaderboard review. 445 trials, fast-agent 0.9.24. Cost estimated at current Sol pricing.</span>
          <Link href="#benchmarks" onClick={e => { e.preventDefault(); go('benchmarks'); }} style={{ fontSize: 15, whiteSpace: 'nowrap' }}>All benchmarks ❯</Link>
        </div>
      </Card>
    </section>

    <section style={{ background: 'var(--ivory-deep)', borderTop: 'var(--border)', borderBottom: 'var(--border)' }}>
      <div style={{ ...faWrap, display: 'grid', gridTemplateColumns: 'clamp(140px, 20vw, 240px) minmax(0,1fr)', alignItems: 'end', gap: 0, paddingTop: 56 }}>
        <img src={A + 'illustration/pointer.png'} alt="" style={{ width: '100%', display: 'block', position: 'relative', zIndex: 1, marginRight: -20 }} />
        <div style={{ background: 'var(--paper)', border: 'var(--border)', borderBottom: 0, borderRadius: '14px 14px 0 0', padding: '32px 36px 40px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 32, alignItems: 'center' }}>
          <div>
            <h2 style={{ ...faVoice, fontSize: 48, lineHeight: 1 }}>Get started</h2>
            <p style={{ fontSize: 17, lineHeight: 1.55, margin: '12px 0 18px' }}>Install and run locally in seconds. One command starts an interactive session with shell tools.</p>
            <Link href="#guides" onClick={e => { e.preventDefault(); go('guides'); }}>Installation guide</Link>
          </div>
          <CopyCode onCopy={() => toast('Copied to clipboard.')} title="Terminal" lines={[
            { text: 'uvx fast-agent-mcp@latest -x', cmd: true }, { text: 'start an interactive session with shell tools' },
            { text: 'uv tool install -U fast-agent-mcp', cmd: true }, { text: 'install the latest version of fast-agent' },
            { text: 'fast-agent --pack codex', cmd: true }, { text: 'download configuration to use codex' }]} />
        </div>
      </div>
    </section>

    <section style={{ ...faWrap, paddingTop: 88 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', gap: 24, marginBottom: 36, flexWrap: 'wrap' }}>
        <div>
          <h2 style={{ ...faVoice, fontSize: 48, lineHeight: 1 }}>Simple, extendable agents.</h2>
          <p style={{ fontSize: 18, lineHeight: 1.55, margin: '14px 0 0', maxWidth: 620 }}>Excellent provider and local model support. Flexible context management. Terminal native and scriptable.</p>
        </div>
        <div style={{ display: 'flex', gap: 12 }}>
          <Button variant="secondary" onClick={() => go('agents')}>Build an agent</Button>
          <Button variant="secondary" onClick={() => go('reference')}>Explore the CLI</Button>
        </div>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', columnGap: 48 }}>
        {features.map(f => <div key={f.t} style={{ borderTop: 'var(--border)', padding: '22px 0 28px', display: 'grid', gridTemplateColumns: 'minmax(0,1fr) auto', gap: 24 }}>
          <div>
            <h3 style={{ margin: 0, fontSize: 20, fontWeight: 800 }}>{f.t}</h3>
            <p style={{ margin: '8px 0 0', fontSize: 16, lineHeight: 1.6 }}>{f.d}</p>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10, alignItems: 'flex-end', paddingTop: 3 }}>
            {f.l.map(([l, r]) => <Link key={l} href={'#' + r} onClick={e => { e.preventDefault(); go(r); }} style={{ fontSize: 15, whiteSpace: 'nowrap' }}>{l} ❯</Link>)}
          </div>
        </div>)}
      </div>
    </section>
  </main>;
}
window.HomePage = HomePage;

import React from 'react';
export function CodeBlock({ lines = [], prompt = true, title, style }) {
  return <div style={{ background: 'var(--petrol)', color: 'var(--ivory)', borderRadius: 'var(--radius-lg)', fontFamily: 'var(--font-code)', fontSize: 14, lineHeight: 1.7, overflow: 'hidden', ...style }}>
    {title && <div style={{ padding: '8px 16px', borderBottom: '2px solid var(--petrol-2)', fontFamily: 'var(--font-read)', fontSize: 11, fontWeight: 800, letterSpacing: 'var(--track-label)', textTransform: 'uppercase', color: 'var(--amber)' }}>{title}</div>}
    <pre style={{ margin: 0, padding: '14px 16px', fontFamily: 'inherit', whiteSpace: 'pre-wrap' }}>{lines.map((l, i) => { const cmd = typeof l === 'string' ? prompt : l.cmd; const t = typeof l === 'string' ? l : l.text;
      return <div key={i}>{cmd && <span style={{ color: 'var(--amber)', marginRight: 10 }}>❯</span>}<span style={{ opacity: cmd ? 1 : 0.78 }}>{t}</span></div>; })}</pre>
  </div>;
}

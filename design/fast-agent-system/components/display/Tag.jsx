import React from 'react';
const V = { outline: { background: 'transparent', color: 'var(--petrol)', border: 'var(--border)' }, accent: { background: 'var(--amber)', color: 'var(--petrol)', border: '2px solid var(--amber)' }, inverse: { background: 'var(--petrol)', color: 'var(--ivory)', border: '2px solid var(--petrol)' }, muted: { background: 'var(--ivory-deep)', color: 'var(--petrol)', border: '2px solid var(--ivory-deep)' } };
export function Tag({ variant = 'outline', children, style }) {
  return <span style={{ ...V[variant], display: 'inline-flex', alignItems: 'center', gap: 6, fontFamily: 'var(--font-read)', fontSize: 11, fontWeight: 800, letterSpacing: 'var(--track-label)', textTransform: 'uppercase', lineHeight: 1, padding: '5px 8px', borderRadius: 'var(--radius-sm)', whiteSpace: 'nowrap', ...style }}>{children}</span>;
}

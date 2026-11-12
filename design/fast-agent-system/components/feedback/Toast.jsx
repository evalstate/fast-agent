import React from 'react';
export function Toast({ children, action, onAction, onClose, tone = 'default' }) {
  return <div role="status" style={{ display: 'inline-flex', alignItems: 'center', gap: 16, background: 'var(--petrol)', color: 'var(--ivory)', borderRadius: 'var(--radius-md)', padding: '12px 16px', fontFamily: 'var(--font-read)', fontSize: 14, fontWeight: 600, lineHeight: 1.4, maxWidth: 440, boxSizing: 'border-box', borderLeft: 0 }}>
    {tone === 'danger' && <span style={{ width: 10, height: 10, borderRadius: 2, background: 'var(--orange)', flex: 'none' }}></span>}
    <span style={{ flex: 1 }}>{children}</span>
    {action && <button onClick={onAction} style={{ background: 'none', border: 0, padding: '2px 0', color: 'var(--ivory)', fontFamily: 'inherit', fontSize: 14, fontWeight: 800, cursor: 'pointer', boxShadow: 'inset 0 -2px 0 var(--amber)' }}>{action}</button>}
    {onClose && <button onClick={onClose} aria-label="Dismiss" style={{ background: 'none', border: 0, color: 'var(--ivory)', fontFamily: 'var(--font-code)', fontSize: 16, cursor: 'pointer', padding: 0, lineHeight: 1 }}>×</button>}
  </div>;
}

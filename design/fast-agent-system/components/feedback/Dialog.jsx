import React from 'react';
export function Dialog({ open = true, title, children, actions, onClose, inline = false, width = 460 }) {
  if (!open) return null;
  const panel = <div role="dialog" aria-modal={!inline} style={{ background: 'var(--paper)', color: 'var(--petrol)', border: 'var(--border)', borderRadius: 'var(--radius-lg)', padding: 24, width, maxWidth: '100%', boxSizing: 'border-box', fontFamily: 'var(--font-read)' }}>
    <div style={{ display: 'flex', alignItems: 'flex-start', gap: 16 }}>
      {title && <h2 style={{ margin: 0, flex: 1, fontFamily: 'var(--font-voice)', fontWeight: 900, fontVariationSettings: 'var(--voice-settings)', letterSpacing: 'var(--track-voice)', fontSize: 26, lineHeight: 1.1 }}>{title}</h2>}
      {onClose && <button onClick={onClose} aria-label="Close" style={{ background: 'none', border: 0, fontFamily: 'var(--font-code)', fontSize: 20, lineHeight: 1, color: 'var(--petrol)', cursor: 'pointer', padding: 0 }}>×</button>}
    </div>
    <div style={{ fontSize: 15, lineHeight: 1.6, marginTop: 10 }}>{children}</div>
    {actions && <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 12, marginTop: 22 }}>{actions}</div>}
  </div>;
  if (inline) return panel;
  return <div onClick={e => { if (e.target === e.currentTarget && onClose) onClose(); }} style={{ position: 'fixed', inset: 0, background: 'rgba(8, 44, 52, 0.8)', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 24, zIndex: 100 }}>{panel}</div>;
}

import React from 'react';
export function Input({ label, hint, error, disabled = false, mono = false, id, style, ...rest }) {
  const [focused, setFocused] = React.useState(false);
  const iid = id || React.useId();
  return <div style={{ fontFamily: 'var(--font-read)', color: 'var(--petrol)', ...style }}>
    {label && <label htmlFor={iid} style={{ display: 'block', fontSize: 11, fontWeight: 800, letterSpacing: 'var(--track-label)', textTransform: 'uppercase', marginBottom: 6, color: 'var(--petrol)' }}>{label}</label>}
    <input id={iid} disabled={disabled} onFocus={() => setFocused(true)} onBlur={() => setFocused(false)}
      style={{ width: '100%', boxSizing: 'border-box', fontFamily: mono ? 'var(--font-code)' : 'var(--font-read)', fontSize: 15, fontWeight: 500, color: 'var(--petrol)', background: disabled ? 'var(--ivory-deep)' : 'var(--paper)', border: '2px solid ' + (error ? 'var(--orange)' : 'var(--petrol)'), borderRadius: 'var(--radius-sm)', padding: '10px 12px', outline: focused ? '3px solid var(--focus-ring)' : 'none', outlineOffset: 2, opacity: disabled ? 0.6 : 1 }} {...rest} />
    {(error || hint) && <div style={{ fontSize: 12, lineHeight: 1.5, marginTop: 6, color: error ? 'var(--petrol)' : 'var(--text-muted)', fontWeight: error ? 700 : 400 }}>{error ? <span style={{ display: 'inline-block', width: 8, height: 8, background: 'var(--orange)', borderRadius: 2, marginRight: 6 }}></span> : null}{error || hint}</div>}
  </div>;
}

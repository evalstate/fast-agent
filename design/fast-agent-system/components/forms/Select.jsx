import React from 'react';
export function Select({ label, options = [], value, onChange, disabled = false, id, style }) {
  const [focused, setFocused] = React.useState(false);
  const iid = id || React.useId();
  return <div style={{ fontFamily: 'var(--font-read)', color: 'var(--petrol)', ...style }}>
    {label && <label htmlFor={iid} style={{ display: 'block', fontSize: 11, fontWeight: 800, letterSpacing: 'var(--track-label)', textTransform: 'uppercase', marginBottom: 6, color: 'var(--petrol)' }}>{label}</label>}
    <div style={{ position: 'relative' }}>
      <select id={iid} value={value} onChange={onChange} disabled={disabled} onFocus={() => setFocused(true)} onBlur={() => setFocused(false)}
        style={{ appearance: 'none', WebkitAppearance: 'none', width: '100%', boxSizing: 'border-box', fontFamily: 'var(--font-read)', fontSize: 15, fontWeight: 600, color: 'var(--petrol)', background: disabled ? 'var(--ivory-deep)' : 'var(--paper)', border: 'var(--border)', borderRadius: 'var(--radius-sm)', padding: '10px 36px 10px 12px', cursor: disabled ? 'not-allowed' : 'pointer', outline: focused ? '3px solid var(--focus-ring)' : 'none', outlineOffset: 2, opacity: disabled ? 0.6 : 1 }}>
        {options.map(o => typeof o === 'string' ? <option key={o} value={o}>{o}</option> : <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
      <span aria-hidden="true" style={{ position: 'absolute', right: 12, top: '50%', transform: 'translateY(-50%) rotate(90deg)', fontFamily: 'var(--font-code)', fontSize: 13, pointerEvents: 'none' }}>❯</span>
    </div>
  </div>;
}

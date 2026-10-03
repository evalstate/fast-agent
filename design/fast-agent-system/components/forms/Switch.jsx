import React from 'react';
export function Switch({ checked, defaultChecked = false, onChange, label, disabled = false }) {
  const [inner, setInner] = React.useState(defaultChecked);
  const on = checked ?? inner;
  return <label style={{ display: 'inline-flex', alignItems: 'center', gap: 10, fontFamily: 'var(--font-read)', fontSize: 15, fontWeight: 600, color: 'var(--petrol)', cursor: disabled ? 'not-allowed' : 'pointer', opacity: disabled ? 0.45 : 1 }}>
    <input type="checkbox" role="switch" checked={on} disabled={disabled} onChange={e => { setInner(e.target.checked); onChange && onChange(e.target.checked); }} style={{ position: 'absolute', opacity: 0, width: 1, height: 1 }} />
    <span style={{ position: 'relative', width: 40, height: 24, boxSizing: 'border-box', border: 'var(--border)', borderRadius: 'var(--radius-sm)', background: on ? 'var(--amber)' : 'var(--ivory-deep)', transition: 'background var(--dur-ui) var(--ease-ui)', flex: 'none' }}>
      <span style={{ position: 'absolute', top: 2, left: on ? 18 : 2, width: 16, height: 16, borderRadius: 3, background: 'var(--petrol)', transition: 'left var(--dur-ui) var(--ease-ui)' }}></span>
    </span>{label}
  </label>;
}

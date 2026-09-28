import React from 'react';
export function Checkbox({ checked, defaultChecked = false, onChange, label, disabled = false }) {
  const [inner, setInner] = React.useState(defaultChecked);
  const on = checked ?? inner;
  const [focused, setFocused] = React.useState(false);
  return <label style={{ display: 'inline-flex', alignItems: 'center', gap: 10, fontFamily: 'var(--font-read)', fontSize: 15, fontWeight: 600, color: 'var(--petrol)', cursor: disabled ? 'not-allowed' : 'pointer', opacity: disabled ? 0.45 : 1 }}>
    <input type="checkbox" checked={on} disabled={disabled} onChange={e => { setInner(e.target.checked); onChange && onChange(e); }} onFocus={e => setFocused(e.target.matches(':focus-visible'))} onBlur={() => setFocused(false)} style={{ position: 'absolute', opacity: 0, width: 1, height: 1 }} />
    <span style={{ width: 20, height: 20, boxSizing: 'border-box', border: 'var(--border)', borderRadius: 5, background: on ? 'var(--amber)' : 'var(--paper)', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', transition: 'background var(--dur-hover) var(--ease-ui)', outline: focused ? '3px solid var(--focus-ring)' : 'none', outlineOffset: 2, flex: 'none' }}>
      {on && <span style={{ width: 5, height: 10, borderRight: '2.5px solid var(--petrol)', borderBottom: '2.5px solid var(--petrol)', transform: 'translateY(-1px) rotate(45deg)' }}></span>}
    </span>
    {label}
  </label>;
}

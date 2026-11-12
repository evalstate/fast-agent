import React from 'react';
export function Radio({ name, options = [], value, defaultValue, onChange, disabled = false, direction = 'column' }) {
  const [inner, setInner] = React.useState(defaultValue ?? (options[0] && (options[0].value ?? options[0])));
  const cur = value ?? inner;
  return <div role="radiogroup" style={{ display: 'flex', flexDirection: direction, gap: direction === 'row' ? 20 : 10, fontFamily: 'var(--font-read)', color: 'var(--petrol)' }}>
    {options.map(o => { const v = o.value ?? o; const l = o.label ?? o; const on = cur === v;
      return <label key={v} style={{ display: 'inline-flex', alignItems: 'center', gap: 10, fontSize: 15, fontWeight: 600, cursor: disabled ? 'not-allowed' : 'pointer', opacity: disabled ? 0.45 : 1 }}>
        <input type="radio" name={name} value={v} checked={on} disabled={disabled} onChange={() => { setInner(v); onChange && onChange(v); }} style={{ position: 'absolute', opacity: 0, width: 1, height: 1 }} />
        <span style={{ width: 20, height: 20, boxSizing: 'border-box', border: 'var(--border)', borderRadius: '50%', background: 'var(--paper)', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', flex: 'none' }}>
          {on && <span style={{ width: 10, height: 10, borderRadius: '50%', background: 'var(--petrol)' }}></span>}
        </span>{l}
      </label>; })}
  </div>;
}

import React from 'react';
export function Tabs({ tabs = [], value, defaultValue, onChange }) {
  const [inner, setInner] = React.useState(defaultValue ?? (tabs[0] && (tabs[0].value ?? tabs[0])));
  const cur = value ?? inner;
  return <div role="tablist" style={{ display: 'flex', gap: 24, borderBottom: '2px solid var(--line)', fontFamily: 'var(--font-read)' }}>
    {tabs.map(t => { const v = t.value ?? t; const l = t.label ?? t; const on = v === cur;
      return <button key={v} role="tab" aria-selected={on} onClick={() => { setInner(v); onChange && onChange(v); }}
        style={{ background: 'none', border: 0, padding: '10px 0', marginBottom: -2, fontFamily: 'inherit', fontSize: 15, fontWeight: on ? 800 : 600, color: 'var(--petrol)', opacity: on ? 1 : 0.72, cursor: 'pointer', boxShadow: on ? 'inset 0 -4px 0 var(--amber)' : 'none', transition: 'box-shadow var(--dur-ui) var(--ease-ui)' }}>{l}</button>; })}
  </div>;
}

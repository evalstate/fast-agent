import React from 'react';
const SIZES = { sm: { font: 13, pad: '8px 14px' }, md: { font: 15, pad: '11px 18px' }, lg: { font: 17, pad: '14px 24px' } };
export function Button({ variant = 'primary', size = 'md', disabled = false, children, onClick, type = 'button', style, ...rest }) {
  const [hover, setHover] = React.useState(false);
  const [down, setDown] = React.useState(false);
  const [focused, setFocused] = React.useState(false);
  const s = SIZES[size] || SIZES.md;
  const live = !disabled;
  const base = { fontFamily: 'var(--font-read)', fontWeight: 700, fontSize: s.font, lineHeight: 1.2, borderRadius: 'var(--radius-md)', cursor: live ? 'pointer' : 'not-allowed', display: 'inline-flex', alignItems: 'center', gap: 8, whiteSpace: 'nowrap', transition: 'transform var(--dur-hover) var(--ease-ui), box-shadow var(--dur-hover) var(--ease-ui), background var(--dur-hover) var(--ease-ui)', outline: focused ? '3px solid var(--focus-ring)' : 'none', outlineOffset: 2, opacity: disabled ? 0.45 : 1 };
  let v;
  if (variant === 'primary') {
    const ledge = !live ? 4 : down ? 0 : hover ? 2 : 4;
    v = { background: 'var(--petrol)', color: 'var(--ivory)', border: 0, padding: s.pad, boxShadow: '0 ' + ledge + 'px 0 var(--amber)', transform: 'translateY(' + (4 - ledge) + 'px)' };
  } else if (variant === 'secondary') {
    const [y, x] = s.pad.split(' ').map(n => parseInt(n) - 2);
    v = { background: live && hover ? 'var(--amber)' : 'transparent', color: 'var(--petrol)', border: 'var(--border)', padding: y + 'px ' + x + 'px' };
  } else {
    v = { background: 'transparent', color: live && hover ? 'var(--teal)' : 'var(--petrol)', border: 0, padding: s.pad, boxShadow: 'inset 0 -' + (live && hover ? 4 : 2) + 'px 0 var(--amber)', borderRadius: 0, paddingLeft: 0, paddingRight: 0, paddingBottom: 4, paddingTop: 4 };
  }
  return <button type={type} disabled={disabled} onClick={onClick}
    onMouseEnter={() => setHover(true)} onMouseLeave={() => { setHover(false); setDown(false); }}
    onMouseDown={() => setDown(true)} onMouseUp={() => setDown(false)}
    onFocus={e => setFocused(e.target.matches(':focus-visible'))} onBlur={() => setFocused(false)}
    style={{ ...base, ...v, ...style }} {...rest}>{children}</button>;
}

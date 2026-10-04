import React from 'react';
export function Tooltip({ content, children, placement = 'top', open }) {
  const [hover, setHover] = React.useState(false);
  const show = open ?? hover;
  const pos = placement === 'bottom' ? { top: '100%', marginTop: 8 } : { bottom: '100%', marginBottom: 8 };
  return <span style={{ position: 'relative', display: 'inline-flex' }} onMouseEnter={() => setHover(true)} onMouseLeave={() => setHover(false)} onFocus={() => setHover(true)} onBlur={() => setHover(false)}>
    {children}
    {show && <span role="tooltip" style={{ position: 'absolute', left: '50%', transform: 'translateX(-50%)', ...pos, background: 'var(--petrol)', color: 'var(--ivory)', fontFamily: 'var(--font-read)', fontSize: 13, fontWeight: 500, lineHeight: 1.4, padding: '6px 10px', borderRadius: 'var(--radius-sm)', whiteSpace: 'nowrap', zIndex: 10, pointerEvents: 'none' }}>{content}</span>}
  </span>;
}

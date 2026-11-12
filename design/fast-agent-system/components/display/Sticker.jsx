import React from 'react';
export function Sticker({ children, tilt = -8, color = 'amber', pop = true, style }) {
  const bg = color === 'orange' ? 'var(--orange)' : color === 'ivory' ? 'var(--ivory)' : 'var(--amber)';
  return <span style={{ '--tilt': tilt + 'deg', display: 'inline-block', transform: 'rotate(' + tilt + 'deg)', animation: pop ? 'fa-pop var(--dur-pop) var(--ease-pop) both' : 'none', background: bg, color: 'var(--petrol)', border: 'var(--border)', borderRadius: 'var(--radius-md)', boxShadow: 'var(--shadow-offset-sm)', fontFamily: 'var(--font-shout)', fontSize: 18, lineHeight: 1, padding: '8px 14px', whiteSpace: 'nowrap', ...style }}>{children}</span>;
}

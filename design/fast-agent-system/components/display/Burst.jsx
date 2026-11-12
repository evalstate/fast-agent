import React from 'react';
export function Burst({ src = 'assets/burst-capsule-amber.svg', size = 140, tilt = -8, children, pop = true, textColor = 'var(--petrol)', style }) {
  return <span style={{ '--tilt': tilt + 'deg', position: 'relative', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', width: size, height: size, transform: 'rotate(' + tilt + 'deg)', animation: pop ? 'fa-pop var(--dur-pop) var(--ease-pop) both' : 'none', ...style }}>
    <img src={src} alt="" width={size} height={size} style={{ position: 'absolute', inset: 0, display: 'block' }} />
    {children && <span style={{ position: 'relative', fontFamily: 'var(--font-shout)', fontSize: Math.round(size * 0.16), lineHeight: 1, color: textColor, textAlign: 'center', maxWidth: '62%' }}>{children}</span>}
  </span>;
}

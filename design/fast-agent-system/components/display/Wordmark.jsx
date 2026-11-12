import React from 'react';
export function Wordmark({ size = 32, inverse = false, icon = true, assetBase = 'assets/' }) {
  return <span style={{ display: 'inline-flex', alignItems: 'center', gap: Math.round(size * 0.35), color: inverse ? 'var(--ivory)' : 'var(--petrol)' }}>
    {icon && <img src={assetBase + (inverse ? 'icon-tile-amber.svg' : 'icon-tile.svg')} width={size} height={size} alt="" style={{ display: 'block' }} />}
    <span style={{ fontFamily: 'var(--font-voice)', fontWeight: 900, fontVariationSettings: 'var(--voice-settings)', letterSpacing: 'var(--track-voice)', fontSize: size, lineHeight: 1, whiteSpace: 'nowrap' }}>fast-agent</span>
  </span>;
}

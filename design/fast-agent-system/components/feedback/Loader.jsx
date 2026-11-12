import React from 'react';
export function Loader({ kind = 'ratchet', size = 40, assetBase = 'assets/', label }) {
  let el;
  if (kind === 'caret') el = <span style={{ fontFamily: 'var(--font-code)', fontSize: size * 0.55, color: 'var(--petrol)', display: 'inline-flex', alignItems: 'center' }}>❯<span style={{ display: 'inline-block', width: size * 0.28, height: size * 0.55, background: 'var(--amber)', marginLeft: size * 0.15, animation: 'fa-prompt 1s steps(1) infinite' }}></span></span>;
  else if (kind === 'chase') el = <span style={{ display: 'inline-flex', gap: 2 }}>{[0, 0.18, 0.36].map(d => <img key={d} src={assetBase + 'mark-chevron.svg'} width={size * 0.6} alt="" style={{ animation: 'fa-chevrons 1.2s var(--ease-ui) ' + d + 's infinite' }} />)}</span>;
  else el = <img src={assetBase + 'burst-capsule-amber.svg'} width={size} height={size} alt="" style={{ display: 'block', animation: 'fa-ratchet 1.1s var(--ease-ratchet) infinite' }} />;
  return <span role="status" aria-label={label || 'Loading'} style={{ display: 'inline-flex', alignItems: 'center', gap: 10, fontFamily: 'var(--font-read)', fontSize: 14, fontWeight: 600, color: 'var(--petrol)' }}>{el}{label}</span>;
}

import React from 'react';
const V = { paper: { background: 'var(--paper)', border: 'var(--border)', color: 'var(--petrol)' }, ivory: { background: 'var(--ivory)', border: 'var(--border)', color: 'var(--petrol)' }, inset: { background: 'var(--ivory-deep)', border: '2px solid transparent', color: 'var(--petrol)' }, inverse: { background: 'var(--petrol)', border: '2px solid var(--petrol)', color: 'var(--ivory)' } };
export function Card({ variant = 'paper', padding = 24, children, style, ...rest }) {
  return <div style={{ ...V[variant], borderRadius: 'var(--radius-lg)', padding, boxSizing: 'border-box', fontFamily: 'var(--font-read)', ...style }} {...rest}>{children}</div>;
}

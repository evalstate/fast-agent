import React from 'react';
export function FootnoteMark({ n, href, sprite = 'assets/sprite.svg', size = 14 }) {
  const mark = <span style={{ display: 'inline-flex', alignItems: 'center', gap: 2, verticalAlign: 'super', fontFamily: 'var(--font-read)', fontSize: 11, fontWeight: 800, color: 'var(--petrol)', lineHeight: 1 }}>
    <svg width={size} height={size} viewBox="0 0 100 100" style={{ color: 'var(--amber)', display: 'block' }} aria-hidden="true"><use href={sprite + '#fa-burst'} /></svg>{n}</span>;
  return href ? <a href={href} style={{ textDecoration: 'none' }} aria-label={'Footnote ' + n}>{mark}</a> : mark;
}

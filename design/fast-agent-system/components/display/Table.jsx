import React from 'react';
export function Table({ columns = [], rows = [], highlight, caption, footnote }) {
  const cell = { padding: '10px 14px', borderBottom: '2px solid var(--line)', textAlign: 'left' };
  return <div style={{ fontFamily: 'var(--font-read)', color: 'var(--petrol)' }}>
    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 15, border: 'var(--border)', borderRadius: 'var(--radius-lg)', borderSpacing: 0, overflow: 'hidden' }}>
      {caption && <caption style={{ textAlign: 'left', fontSize: 11, fontWeight: 800, letterSpacing: 'var(--track-label)', textTransform: 'uppercase', paddingBottom: 8 }}>{caption}</caption>}
      <thead><tr>{columns.map(c => <th key={c.key} style={{ ...cell, textAlign: c.numeric ? 'right' : 'left', fontSize: 11, fontWeight: 800, letterSpacing: 'var(--track-label)', textTransform: 'uppercase', borderBottom: 'var(--border)', background: 'var(--ivory-deep)' }}>{c.label}</th>)}</tr></thead>
      <tbody>{rows.map((r, i) => { const hi = highlight != null && (typeof highlight === 'function' ? highlight(r, i) : highlight === i);
        return <tr key={i} style={{ background: hi ? 'var(--amber)' : 'var(--paper)' }}>{columns.map(c => <td key={c.key} style={{ ...cell, borderBottom: i === rows.length - 1 ? 0 : cell.borderBottom, textAlign: c.numeric ? 'right' : 'left', fontWeight: c.numeric ? 900 : (hi ? 800 : 500), fontVariantNumeric: 'tabular-nums' }}>{r[c.key]}</td>)}</tr>; })}</tbody>
    </table>
    {footnote && <div style={{ fontSize: 12, lineHeight: 1.5, color: 'var(--text-muted)', marginTop: 8 }}>{footnote}</div>}
  </div>;
}

import React from 'react';
export function Link({ href = '#', children, style, ...rest }) {
  const [hover, setHover] = React.useState(false);
  return <a href={href} onMouseEnter={() => setHover(true)} onMouseLeave={() => setHover(false)}
    style={{ fontFamily: 'inherit', fontWeight: 600, color: hover ? 'var(--link-hover)' : 'var(--link)', textDecoration: 'none', boxShadow: 'inset 0 -' + (hover ? 4 : 2) + 'px 0 var(--link-underline)', transition: 'box-shadow var(--dur-hover) var(--ease-ui), color var(--dur-hover) var(--ease-ui)', ...style }} {...rest}>{children}</a>;
}

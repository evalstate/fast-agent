import * as React from 'react';
/** Small uppercase label, radius 6. `accent` (amber) marks fast-agent only. */
export interface TagProps {
  variant?: 'outline' | 'accent' | 'inverse' | 'muted';
  children?: React.ReactNode;
  style?: React.CSSProperties;
}
export declare function Tag(props: TagProps): JSX.Element;

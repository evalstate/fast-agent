import * as React from 'react';
/** Petrol hover label, radius 6. For definitions and disclaimers, not essential info. */
export interface TooltipProps {
  content?: React.ReactNode;
  children?: React.ReactNode;
  placement?: 'top' | 'bottom';
  /** Force open (for specimens). */
  open?: boolean;
}
export declare function Tooltip(props: TooltipProps): JSX.Element;

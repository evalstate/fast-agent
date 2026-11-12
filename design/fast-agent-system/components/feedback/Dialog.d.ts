import * as React from 'react';
/** Modal on a flat petrol scrim (no blur). Paper panel, keyline, radius 14, Fraunces title. */
export interface DialogProps {
  open?: boolean;
  title?: string;
  children?: React.ReactNode;
  /** Buttons, right-aligned. Primary last. */
  actions?: React.ReactNode;
  onClose?: () => void;
  /** Render the panel in place without the fixed scrim (specimens, docs). */
  inline?: boolean;
  width?: number;
}
export declare function Dialog(props: DialogProps): JSX.Element | null;

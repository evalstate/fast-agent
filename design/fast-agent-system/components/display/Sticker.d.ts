import * as React from 'react';
/** Tilted Ultra "New!" sticker with a solid offset shadow. One per view. */
export interface StickerProps {
  children?: React.ReactNode;
  /** Resting angle in degrees, −4 to −10. */
  tilt?: number;
  color?: 'amber' | 'orange' | 'ivory';
  /** Play the fa-pop entrance once on mount. */
  pop?: boolean;
  style?: React.CSSProperties;
}
export declare function Sticker(props: StickerProps): JSX.Element;

import * as React from 'react';
/** Atomic-age burst mark with optional Ultra label, resting at the brand tilt. Pass the asset path relative to your page. */
export interface BurstProps {
  /** Path to a burst asset, e.g. assets/burst-capsule-amber.svg, burst-capsule-orange.svg, burst-star.svg, burst-score.svg. */
  src?: string;
  size?: number;
  tilt?: number;
  pop?: boolean;
  textColor?: string;
  children?: React.ReactNode;
  style?: React.CSSProperties;
}
export declare function Burst(props: BurstProps): JSX.Element;

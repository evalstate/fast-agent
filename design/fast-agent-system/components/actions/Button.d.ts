import * as React from 'react';
/**
 * fast-agent button. Primary sits on a 4px amber ledge and drops onto it on press; secondary is a 2px keyline that fills amber on hover; ghost is a text action with the link underline.
 * @startingPoint section="Actions" subtitle="Primary ledge, secondary keyline, ghost" viewport="700x220"
 */
export interface ButtonProps {
  /** primary = the one CTA per view. */
  variant?: 'primary' | 'secondary' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  disabled?: boolean;
  type?: 'button' | 'submit' | 'reset';
  onClick?: (e: React.MouseEvent<HTMLButtonElement>) => void;
  style?: React.CSSProperties;
  children?: React.ReactNode;
}
export declare function Button(props: ButtonProps): JSX.Element;

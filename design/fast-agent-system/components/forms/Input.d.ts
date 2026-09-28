import * as React from 'react';
/**
 * Text field: paper fill, 2px petrol keyline, radius 6, teal focus ring. Labels are small uppercase.
 * @startingPoint section="Forms" subtitle="Keyline inputs, select, toggles" viewport="700x320"
 */
export interface InputProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'style'> {
  label?: string;
  hint?: string;
  /** Replaces the hint; turns the keyline orange. */
  error?: string;
  disabled?: boolean;
  /** Use DM Mono for keys, paths, model ids. */
  mono?: boolean;
  style?: React.CSSProperties;
}
export declare function Input(props: InputProps): JSX.Element;

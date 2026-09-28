import * as React from 'react';
/** 20px keyline box that fills amber with a petrol tick when checked. */
export interface CheckboxProps {
  checked?: boolean;
  defaultChecked?: boolean;
  onChange?: (e: React.ChangeEvent<HTMLInputElement>) => void;
  label?: React.ReactNode;
  disabled?: boolean;
}
export declare function Checkbox(props: CheckboxProps): JSX.Element;

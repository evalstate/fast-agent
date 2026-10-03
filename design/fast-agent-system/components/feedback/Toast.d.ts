import * as React from 'react';
/** Petrol confirmation bar with an optional underlined action. Flat, no shadow. */
export interface ToastProps {
  children?: React.ReactNode;
  action?: string;
  onAction?: () => void;
  onClose?: () => void;
  tone?: 'default' | 'danger';
}
export declare function Toast(props: ToastProps): JSX.Element;

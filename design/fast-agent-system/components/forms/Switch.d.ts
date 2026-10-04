/** On/off toggle. Deliberately squared (radius 6) — the brand has no pills. */
export interface SwitchProps {
  checked?: boolean;
  defaultChecked?: boolean;
  onChange?: (checked: boolean) => void;
  label?: string;
  disabled?: boolean;
}
export declare function Switch(props: SwitchProps): JSX.Element;

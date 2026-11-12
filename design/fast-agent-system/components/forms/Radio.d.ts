/** Radio group: 20px keyline circles with a petrol dot. */
export interface RadioProps {
  name?: string;
  options?: Array<string | { value: string; label: string }>;
  value?: string;
  defaultValue?: string;
  onChange?: (value: string) => void;
  disabled?: boolean;
  direction?: 'row' | 'column';
}
export declare function Radio(props: RadioProps): JSX.Element;

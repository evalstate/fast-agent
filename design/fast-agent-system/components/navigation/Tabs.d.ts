/**
 * Tab row on a 14% divider; active tab gets a 4px amber underline and weight 800.
 * @startingPoint section="Navigation" subtitle="Amber-underline tabs" viewport="700x160"
 */
export interface TabsProps {
  tabs?: Array<string | { value: string; label: string }>;
  value?: string;
  defaultValue?: string;
  onChange?: (value: string) => void;
}
export declare function Tabs(props: TabsProps): JSX.Element;

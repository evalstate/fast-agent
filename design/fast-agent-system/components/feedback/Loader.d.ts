/** Loading indicator: capsule ratchets 72° and pauses; or prompt caret; or chevron chase. No shimmer. */
export interface LoaderProps {
  kind?: 'ratchet' | 'caret' | 'chase';
  size?: number;
  /** Folder holding burst-capsule-amber.svg / mark-chevron.svg, relative to the page. */
  assetBase?: string;
  label?: string;
}
export declare function Loader(props: LoaderProps): JSX.Element;

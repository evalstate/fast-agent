/** Interim lockup: chevron tile + "fast-agent" in Fraunces 900 SOFT 100. */
export interface WordmarkProps {
  /** Cap height-ish size in px (icon matches). */
  size?: number;
  inverse?: boolean;
  icon?: boolean;
  /** Folder containing icon-tile.svg, relative to the page. */
  assetBase?: string;
}
export declare function Wordmark(props: WordmarkProps): JSX.Element;

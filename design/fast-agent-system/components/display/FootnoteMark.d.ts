/** Small capsule burst + number beside a figure, linking to its disclaimer. */
export interface FootnoteMarkProps {
  n?: number | string;
  href?: string;
  /** Path to assets/sprite.svg relative to the page. */
  sprite?: string;
  size?: number;
}
export declare function FootnoteMark(props: FootnoteMarkProps): JSX.Element;

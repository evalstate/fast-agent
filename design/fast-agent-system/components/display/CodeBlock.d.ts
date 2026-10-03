import * as React from 'react';
/** Petrol code/CLI block in DM Mono with the amber ❯ prompt. */
export interface CodeBlockProps {
  /** Strings, or { text, cmd } to mix commands (prompted) and output. */
  lines?: Array<string | { text: string; cmd?: boolean }>;
  /** Prefix string lines with ❯. */
  prompt?: boolean;
  title?: string;
  style?: React.CSSProperties;
}
export declare function CodeBlock(props: CodeBlockProps): JSX.Element;

import * as React from 'react';
/** Inline text link: petrol text, 2px amber underline that thickens to 4px and turns text teal on hover. */
export interface LinkProps extends React.AnchorHTMLAttributes<HTMLAnchorElement> {
  href?: string;
  children?: React.ReactNode;
}
export declare function Link(props: LinkProps): JSX.Element;

import * as React from 'react';
/**
 * Container: paper or ivory fill, 2px petrol keyline, radius 14. No shadows, no accent borders.
 * @startingPoint section="Display" subtitle="Cards, tags, stickers, tables, code" viewport="700x420"
 */
export interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'paper' | 'ivory' | 'inset' | 'inverse';
  padding?: number | string;
  children?: React.ReactNode;
}
export declare function Card(props: CardProps): JSX.Element;

import * as React from 'react';
/** Receipts table: keyline frame, 14% row dividers, tabular 900 numbers, amber row for fast-agent. */
export interface TableColumn { key: string; label: string; numeric?: boolean; }
export interface TableProps {
  columns?: TableColumn[];
  rows?: Array<Record<string, React.ReactNode>>;
  /** Row index, or predicate, to fill amber (fast-agent only). */
  highlight?: number | ((row: any, i: number) => boolean);
  caption?: string;
  /** Disclaimer on the same screen: n, date, model, harness version. */
  footnote?: React.ReactNode;
}
export declare function Table(props: TableProps): JSX.Element;

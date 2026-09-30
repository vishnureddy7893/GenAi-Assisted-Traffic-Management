/** Card container used by every dashboard panel. */

import type { ReactNode } from "react";

export interface PanelProps {
  title: string;
  subtitle?: string;
  /** Rendered at the right of the header — usually a status pill. */
  aside?: ReactNode;
  /** Remove body padding, for panels that render their own full-bleed rows. */
  flush?: boolean;
  children: ReactNode;
}

export function Panel({
  title,
  subtitle,
  aside,
  flush = false,
  children,
}: PanelProps): JSX.Element {
  return (
    <section className="panel">
      <header className="panel__head">
        <div className="panel__headings">
          <h2 className="panel__title">{title}</h2>
          {subtitle !== undefined && <p className="panel__subtitle">{subtitle}</p>}
        </div>
        {aside !== undefined && <div className="panel__head-aside">{aside}</div>}
      </header>
      <div className={flush ? "panel__body panel__body--flush" : "panel__body"}>
        {children}
      </div>
    </section>
  );
}

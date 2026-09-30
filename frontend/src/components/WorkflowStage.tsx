/**
 * One numbered stage of the operator workflow.
 *
 * The dashboard is a sequence, not a wall of panels: traffic data comes in, it
 * is measured, it is interpreted, it is placed on a network, and only then can a
 * route be recommended. Numbering the stages makes the dependency order visible,
 * which matters because most empty states on this page are empty *because an
 * earlier stage has not been completed* rather than because something is broken.
 *
 * A stage may be marked `future`, which renders it visibly apart and states the
 * phase that will provide it. That separation is deliberate: a capability that
 * does not exist yet must never sit in the flow looking like one that failed.
 */

import type { ReactNode } from "react";

export function WorkflowStage({
  step,
  title,
  summary,
  future = false,
  futureNote,
  children,
}: {
  /** Displayed position in the flow. Not an id — the order is the meaning. */
  step: number;
  title: string;
  summary: string;
  /** Marks a stage whose capability belongs to a later phase. */
  future?: boolean;
  futureNote?: string;
  children?: ReactNode;
}): JSX.Element {
  return (
    <section className={future ? "stage stage--future" : "stage"}>
      <header className="stage__head">
        <span className="stage__step" aria-hidden="true">
          {step}
        </span>
        <div className="stage__headings">
          <h2 className="stage__title">
            {title}
            {future && <span className="stage__badge">not yet implemented</span>}
          </h2>
          <p className="stage__summary">{summary}</p>
        </div>
      </header>

      {future && futureNote !== undefined ? (
        <p className="stage__future-note">{futureNote}</p>
      ) : (
        <div className="stage__body">{children}</div>
      )}
    </section>
  );
}

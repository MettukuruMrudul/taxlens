"use client";

type Props = { years: number[]; value: number | null; onChange: (year: number | null) => void };

/** The tax year as the page's headline. Most tax confusion is about *which year*,
 *  so the selected year is the biggest thing on screen, with the filing year spelled out. */
export default function YearPicker({ years, value, onChange }: Props) {
  return (
    <section className="year" aria-label="Tax year">
      <div className={`year-numeral${value === null ? " auto" : ""}`} aria-live="polite">
        {value ?? "Any year"}
      </div>
      <div className="year-side">
        <div className="year-caption">
          {value === null ? (
            <>TaxLens reads the year from your question. Choose one to pin it.</>
          ) : (
            <>
              Tax year <strong className="num">{value}</strong>: the return you file in{" "}
              <strong className="num">{value + 1}</strong>.
            </>
          )}
        </div>
        <div className="year-steps" role="group" aria-label="Choose tax year">
          <button type="button" aria-pressed={value === null} onClick={() => onChange(null)}>
            From question
          </button>
          {years.map((y) => (
            <button key={y} type="button" aria-pressed={value === y} onClick={() => onChange(y)}>
              {y}
            </button>
          ))}
        </div>
      </div>
    </section>
  );
}

/** Plain-text stand-in used while the Pierre chunk loads and for patch text
 *  that does not (yet) parse — e.g. the partial frames of a streaming diff.
 *
 *  `leading-5` is 20px, which is Pierre's own per-line height MEASURED from a
 *  rendered block (7 blocks, heights exactly 50 + 20×lines: 2px border, 32px
 *  header, 16px body padding). The `leading-relaxed` this used to carry is
 *  21.125px at 13px text — a 1.125px surplus PER LINE, so a 40-line snippet
 *  shrank by 45px the moment the chunk resolved, and a transcript full of them
 *  moved the reader's position on every load. That is what made the old "the
 *  swap is a restyle, not a content reflow" claim false; matching the line box
 *  is what makes it true, so keep these two heights equal or the reflow returns.
 */
export function PlainCodeFallback({ text }: { text: string }) {
  return (
    <pre className="m-0 px-3 py-2 overflow-x-auto text-[13px] font-mono leading-5 whitespace-pre">
      {text}
    </pre>
  )
}

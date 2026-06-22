# Design QA

- Source visual truth: `demo_app/reference-analyst-console.png`
- Implementation screenshot: `demo_app/implementation-1440x1024.png`
- Full-view comparison: `demo_app/qa-comparison.png`
- Focused summary comparison: `demo_app/qa-focus-summary.png`
- Responsive evidence: `demo_app/implementation-760x900.png`
- Viewport: 1440×1024 desktop; 760×900 responsive check
- State: reproducible sample mode, 10,000,000 KRW cash, zero holdings, all 50 rows

## Findings

- No actionable P0/P1/P2 mismatches remain.
- [P3] The implementation omits the mock's small refresh and status glyphs.
  - Location: top header and pipeline strip.
  - Evidence: the source uses small pictograms; the implementation uses explicit Korean text status.
  - Impact: no functionality is missing, but the header is visually plainer.
  - Follow-up: add one consistent icon library only if the product later needs manual refresh controls.
- [P3] Live numeric values differ from the visual target.
  - Location: rankings, cash summary, and table.
  - Evidence: the target contains illustrative values; the implementation renders actual PPO output and the confirmed 10,000,000 KRW default.
  - Impact: intentional product-data difference, not design drift.
  - Follow-up: none.

## Required Fidelity Surfaces

- Fonts and typography: passed. Korean system sans stack, compact hierarchy, tabular numeric cells, and readable 1440px density match the target intent.
- Spacing and layout rhythm: passed. Dark utility header, narrow control rail, four-stage strip, five-part summary, and table-first workspace retain the target hierarchy. Document scrolling was removed so the header no longer clips.
- Colors and visual tokens: passed. Navy chrome, warm-white surfaces, teal buy, muted coral sell, and neutral hold states match the reference palette with sufficient contrast.
- Image quality and asset fidelity: passed. The target contains no substantive photography or illustration. No placeholder art, inline SVG, or CSS illustration was introduced.
- Copy and content: passed. Research-only, t-day estimate, no-order, and no-causal-explanation language accurately reflect the implemented model contract.
- Interactions: passed. Sample/latest mode, cash and holdings input, JSON/text upload, model execution, search, action filter, and score sorting are functional.
- Responsiveness and accessibility: passed. The 760px check has no document-level horizontal overflow; native labels, controls, focus outlines, and semantic table structure are present.

## Patches Made During QA

- Added JSON/text file upload to match the selected input workflow.
- Prevented desktop document scrolling and header clipping while preserving mobile page scrolling.
- Kept actual PPO outputs in place of illustrative mock values.
- Verified sample inference, latest Yahoo→Upstage→Autoencoder→PPO inference, search, filter, and portfolio re-run states.

## Implementation Checklist

- [x] Selected Analyst Console hierarchy reproduced
- [x] Actual 1,467→50 PPO inference connected
- [x] Latest 4,096→16 shared Autoencoder path connected
- [x] Full 50-stock result table implemented
- [x] Desktop and responsive browser checks completed
- [x] No browser console errors

## Follow-up Polish

- P3: introduce a consistent icon library if additional compact actions are added later.

final result: passed

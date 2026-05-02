# Haiku Platform — UI/UX Analysis

## Screen: mT5 Compression Model v2 (Experiment Detail)

**Source:** Figma Prototype — Experiments > mT5 Compression Model v2
**Design System:** Minimal editorial / AI governance dashboard
**Color Palette:** Warm neutrals (cream/off-white background), olive-black accents, serif typography for headings, sans-serif for data

---

## 1. Layout Architecture

The screen uses a three-column layout:

**Left Sidebar (fixed, ~220px):** Navigation rail with brand, primary CTA, nav items, and user profile anchored to the bottom.

**Main Content (fluid center):** Experiment detail page with hero section, verification gates, and before/after compression comparison.

**Right Callout (inset card):** Projected savings metric card positioned within the hero section, top-right.

This is a well-structured information hierarchy. The eye naturally flows: brand identity (top-left) to experiment title (center) to financial impact (top-right) to verification proof (middle) to evidence (bottom). The layout tells a story: what this is, why it matters financially, that it's verified, and here's the proof.

---

## 2. Navigation Sidebar

### What Works

- The olive/dark-green "New Experiment" button is the only saturated element in the sidebar, correctly prioritizing the primary creation action.
- Navigation items use clean iconography with consistent sizing and spacing. The active state ("Models") uses a filled icon + bold weight, which is clear without being heavy.
- The user profile ("Archivist Pro / Tier III Research") anchored to the bottom is a standard and effective pattern. The tier label adds useful context about permissions/access level.
- The nav grouping is logical: creation tools (Dashboard, Models, Governance) at the top, data management (Unit Costs, Experiments, Datasets) in the middle, operational concerns (Monitoring, Policies, Risk Assessment) lower, and administrative items (Analytics, Organization, Settings) at the bottom.

### What Could Improve

- **No visual grouping.** The 11 navigation items form a flat list. At this density, even subtle section dividers or whitespace breaks between logical groups (e.g., after "Governance", after "Datasets", after "Risk Assessment") would reduce cognitive load.
- **No collapse behavior indicated.** For a data-heavy platform like this, users on smaller screens or in focused workflows will want a collapsed sidebar (icon-only mode). The design should account for this state.
- **Active state ambiguity.** The breadcrumb says "Experiments > mT5 Compression Model v2" but "Models" is highlighted in the sidebar, not "Experiments." This is a navigation state mismatch that could confuse users about where they are in the information architecture. Either the breadcrumb or the sidebar active state needs to be reconciled.

---

## 3. Hero Section — Experiment Header

### What Works

- The **serif typeface for the title** ("Tokenization-Aware Cost Optimization") is a strong editorial design choice. It communicates authority and seriousness — appropriate for an AI governance platform where trust is paramount. This is a distinctive brand choice that most dashboards don't make.
- The **strategy tag** ("Editorial Intelligence: Strategy 08") uses a subtle pill/chip with muted olive background. It categorizes without competing with the title.
- The **description text** is well-written and scannable — two sentences that explain what this experiment does and why.
- The **breadcrumb** is minimal and functional.

### What Could Improve

- **The title is very large** relative to the rest of the content. It occupies roughly the same vertical space as the entire verification gates section below it. For a data platform where users are likely power users visiting this page repeatedly, the title could be 15-20% smaller without losing impact. The first visit needs the title; the 50th visit needs the data.
- **No experiment metadata visible.** There's no creation date, last run timestamp, owner/creator, status (active/paused/archived), or version history. For a governance platform, audit trail visibility matters. Even a subtle metadata line below the title (e.g., "Created Apr 2, 2026 by A. Pradeep — Last run 3h ago — v2.1.4") would add operational value.
- **No action affordances in the header.** The only action is "Generate Compliance Report" in the top-right corner. But where are: Edit, Clone, Archive, Share, Run Now, View History? These are expected actions for an experiment detail page. They could live in a "..." overflow menu or as secondary buttons near the title.

---

## 4. Financial Impact Card (Projected Monthly Savings)

### What Works

- **Placement is excellent.** Putting the dollar impact top-right immediately answers the executive question: "why should I care about this experiment?" The number ($14,289.40) is large, bold, and unmissable.
- **The efficiency gain line** ("22.4% Efficiency gain since v1.2") provides trend context — it's not just the current value, it's the improvement trajectory. This is exactly what a decision-maker needs.
- **Typography hierarchy is clean.** "CURRENT PROJECTED MONTHLY SAVINGS" in small caps above, the dollar figure dominant, the trend line below with an upward-trend icon.

### What Could Improve

- **"Projected" needs a confidence indicator.** This is a governance platform — projections without confidence intervals or methodology transparency feel incomplete. Even a subtle "(95% CI: $12,400–$16,100)" or a hover tooltip explaining the projection basis would add credibility.
- **No time period selector.** Monthly savings is shown, but users may want to toggle to daily, weekly, quarterly, or annual views. A small dropdown or tab set would add flexibility.
- **The card has no border or shadow distinction.** It sits on the same cream background as the rest of the page, separated only by typography weight. A very subtle border or slight background shade difference would help it register as a distinct, self-contained metric card.

---

## 5. Verification Gates (Semantic Score + Token Reduction)

### What Works

- **Dual-gate pattern is clear and reassuring.** The two cards sit side by side, each with a green checkmark, label, threshold description, and actual value. This tells the user: "this compression passed both quality checks."
- **Threshold transparency.** Showing "Validated structural integrity > 0.88" and "Compression threshold > 15%" is excellent UX for a governance audience. They can see not just the score but the standard it was measured against.
- **The green checkmarks provide instant pass/fail signal.** No ambiguity.
- **Values are prominent.** "0.94" and "35.5%" are large and easy to scan.

### What Could Improve

- **No fail state shown.** The design needs to account for what happens when a gate fails. A red/amber icon, a "FAILED" label, and perhaps a remediation suggestion would complete the pattern.
- **Missing the third gate.** The concept doc defines a triple verification gate (LaBSE semantic similarity >= 0.91, graph Jaccard >= 0.88, token reduction >= 15%). This UI only shows two gates. The "Semantic Score Gate" at 0.94 appears to conflate the LaBSE and Jaccard gates, or only shows one of them. For a governance platform, all three verification dimensions should be visible — perhaps as three smaller cards in a row, or an expandable detail view.
- **No historical trend.** A sparkline or mini-chart showing how these scores have evolved across experiment versions would help users understand whether quality is stable, improving, or degrading.
- **The label "Semantic Score Gate" is vague.** "LaBSE Semantic Similarity" or "Cross-Lingual Meaning Preservation" would be more precise and self-documenting.

---

## 6. Before/After Comparison (Source Sequence vs MT5 Compressed)

### What Works

- **Side-by-side comparison is the right pattern.** This is the most intuitive way to show compression results. Source on the left (45 tokens), compressed on the right (29 tokens). The visual weight difference immediately communicates that compression happened.
- **Token counts as headers** make the quantitative difference scannable without reading the text.
- **The compressed card includes a provenance tag** ("Optimized by Archivist-MT5-Large") which adds transparency about which model produced this result. Good for governance auditing.
- **The density efficiency bar and "+35.5% SAVED" label** on the compressed card provide a visual reinforcement of the savings metric.
- **Encoding tags** ("RAW_UTF8", "EN_US_LOCALE") at the bottom of the source card are a nice touch for technical users who need to know the input encoding context.

### What Could Improve

- **No diff highlighting.** The most powerful UX improvement for this section would be visual diffing — highlighting which words/phrases were removed, replaced, or restructured. Without this, the user has to manually read both paragraphs and compare. Color-coded additions/deletions (like a code diff) would make the compression decisions visible at a glance.
- **Only one example shown.** For an experiment processing thousands of sequences, showing a single example is insufficient. A paginated carousel ("1 of 847 test cases"), a sample gallery, or at minimum a "Show more examples" link would give users confidence that the single example isn't cherry-picked.
- **No way to flag bad compressions.** If a user reads the compressed output and notices meaning drift or quality issues, there's no annotation/flagging mechanism visible. A "Flag for review" or thumbs up/down on individual comparisons would close the feedback loop.
- **The source text card is taller than the compressed text card** because the source contains more text. This creates visual asymmetry. Aligning the card heights (with the compressed card having more whitespace) would create a cleaner layout and visually reinforce the compression: same container, less content.
- **No multilingual examples.** The SCL engine's core value proposition is multilingual token optimization (Tamil, Hindi, Arabic, etc.). This demo only shows an English example. Adding a language selector or tabbed examples showing the dramatic token savings for non-Latin scripts would showcase the platform's differentiator.

---

## 7. Typography and Visual Design

### What Works

- **The serif/sans-serif pairing is distinctive.** Serif for titles and headings, sans-serif for body text and data. This creates a strong editorial personality that differentiates Haiku from the typical dashboard look (all sans-serif, all blue).
- **The warm color palette** (cream backgrounds, olive accents, black text) is restrained and professional. It avoids the tech-startup cliche of blue/purple gradients.
- **Data typography is well-scaled.** Numbers like "0.94", "35.5%", "$14,289.40" are given appropriate visual weight without shouting.
- **Whitespace is generous.** The page doesn't feel cramped. Each section has room to breathe.

### What Could Improve

- **The overall palette is almost monochromatic.** The only color signals are the olive-green buttons and the green checkmarks. For a platform that needs to communicate status (healthy/degraded/failed), severity (info/warning/critical), and trend (up/down/stable), the color system needs more range. A carefully chosen amber for warnings and red for failures, plus a blue for informational elements, would complete the semantic color palette without breaking the restrained aesthetic.
- **Small caps overuse.** "CURRENT PROJECTED MONTHLY SAVINGS", "SOURCE SEQUENCE", "MT5 COMPRESSED", "DENSITY EFFICIENCY" — the design leans heavily on small caps for labels. This works in moderation but at this density it starts to feel typographically heavy. Mixing in some regular-weight labels would add variety.

---

## 8. Missing UX Patterns for a Governance Platform

The design is visually polished but is missing several patterns that a governance/compliance audience expects:

- **Audit trail / version history.** No visible way to see what changed between v1 and v2 of this experiment, who made changes, or when.
- **Approval workflow.** Governance platforms typically require sign-off before models go to production. No approval status, reviewer assignment, or sign-off buttons visible.
- **Export/sharing.** Beyond "Generate Compliance Report," there's no CSV export, API endpoint reference, or share-with-team functionality visible.
- **Alerting configuration.** If the semantic score drops below threshold on a future run, how does the user get notified? No alerting or threshold-based notification configuration is visible.
- **Comparison across experiments.** This shows one experiment in isolation. For governance, users need to compare experiments side by side (A/B testing, model iteration comparison). No "Compare with..." affordance is visible.

---

## 9. Overall Assessment

**Design Quality: 8/10.** The visual design is sophisticated and distinctive. The editorial typography, warm palette, and generous whitespace create a premium feel that's rare in data platforms.

**Information Architecture: 7/10.** The story flow (what → impact → verification → evidence) is well-structured. But the sidebar active state mismatch, missing experiment metadata, and single-example limitation reduce navigational and informational confidence.

**Governance Fitness: 6/10.** The platform is branded as "AI Governance" but is missing core governance UX patterns: audit trails, approval workflows, comparison views, alerting, and fail-state designs. The current design feels more like a model monitoring dashboard than a governance tool.

**Actionability: 6/10.** The page is predominantly read-only. Beyond "Generate Compliance Report," there are very few actions a user can take. For a tool that people use daily, the ratio of information to interaction needs rebalancing.

---

## 10. Priority Recommendations

**P0 — Fix the navigation state mismatch.** Breadcrumb says "Experiments" but sidebar highlights "Models." This undermines spatial orientation.

**P0 — Show all three verification gates.** The triple gate (LaBSE >= 0.91, graph Jaccard >= 0.88, token reduction >= 15%) is a core quality claim. Showing only two conflates important distinctions.

**P1 — Add diff highlighting to the before/after comparison.** This single improvement would make compression decisions transparent and dramatically increase the section's value.

**P1 — Add multilingual examples.** The core differentiator is non-Latin script optimization. Showing only English undersells the product.

**P1 — Design the fail state.** Every gate, metric, and status indicator needs a defined fail/warning appearance. Without this, the governance story is incomplete.

**P2 — Add experiment metadata and action affordances.** Creator, timestamp, version, status, and actions (edit, clone, run, archive) are expected for power users.

**P2 — Add audit trail visibility.** Even a "View history" link that opens a timeline of changes would satisfy governance requirements.

**P3 — Add a confidence interval to the savings projection.** For a governance audience, unqualified projections reduce trust.

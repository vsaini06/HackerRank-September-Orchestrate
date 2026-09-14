# Buy or Wait? - An Affordability Decision Agent

Built for HackerRank's Orchestrate September 2026 hackathon. The challenge:
given a user's real financial history (balance, upcoming bills, income,
spending patterns), decide whether a requested purchase is safe to pay
now, safe later, safe with a plan, or not safe at all - without their
balance ever dropping below a personal minimum over the next 90 days.

See [`CHALLENGE.md`](./CHALLENGE.md) for the organizer's original problem
statement and repo instructions.

> **Note on the dataset:** the CSVs this project runs against were
> provided by the hackathon organizers for the challenge and are not
> included in this repo (see `.gitignore`) since redistribution wasn't
> confirmed to be permitted. The code here is fully functional against
> data matching the same schema, described below.

## The core decision: deterministic math, narrow LLM use

The temptation with an "AI agent" challenge is to let the model make
every call. This project goes the other way. All the actual financial
logic - 90-day balance forecasting, computing the safest amount to pay,
ranking payment options - is plain, deterministic Python. The LLM is
used for exactly two things it's genuinely needed for:

1. Reading an amount off a receipt/document image when a transaction's
   amount is missing from the structured data.
2. Interpreting a free-text message that's explicitly linked to one
   specific transaction (confirm / amend / cancel / delay).

When real money math is involved, you want numbers you can verify, not
a model estimating a balance.

## How it works

1. **Load and clean the data** - parse all CSVs, resolve linked-event
   chains (an amendment/cancellation supersedes the event it points to),
   convert every amount to the user's home currency (direct rate ->
   inverse rate -> USD-bridge -> nearest-available-date fallback).
2. **Detect recurring cash flow** - true monthly bills/subscriptions
   (rent, utilities, debt payments, etc.) are detected per
   `(user_id, category)` by date cadence and projected forward. Everyday
   variable spending (groceries, transport, dining) is modeled as a
   conservative recency-weighted average instead of an exact recurring
   series - see "Bugs I actually hit" below for why that distinction
   matters.
3. **Simulate 90 days forward** - build a daily balance curve from the
   user's current balance plus everything from step 2. The lowest point
   that curve ever reaches, relative to the user's minimum balance,
   determines the safe amount to pay today and the earliest date a full
   payment becomes safe.
4. **Decide the plan** - check which payment methods the user actually
   accepts, which are safe, and rank the safe ones by: completes on
   time, needs no spending changes, lowest total cost, starts earliest,
   fewest payments.
5. **Enrich with evidence** - fill in blank amounts from linked images,
   apply confirm/amend/cancel/delay signals from linked messages. Both
   treated as untrusted data - no embedded instructions are followed,
   only factual financial content is extracted.

## Bugs I actually hit (and why they mattered)

**Recurring-by-description instead of recurring-by-category.** First
pass grouped transactions by exact description text to detect recurring
series. Broke immediately - groceries show up under a dozen different
vendor names, so treating each as its own repeating series wildly
overcounted spending and made solvent users look broke. Fixed by
detecting recurrence per `(user_id, category)` with date-cadence
clustering instead, and modeling genuinely variable categories as a
smoothed average rather than a fixed schedule.

**Income excluded from projection.** Salary initially looked "too
noisy" to project forward (some users mix irregular gig income into
what looks like a salary category), so it got left out entirely. That
meant expenses kept recurring for the full 90 days while income only
ever appeared once - an accounting asymmetry that made ordinary
affordable requests look unaffordable by month two. Fixed by projecting
income forward too, conditioned on whether a user's own pay history
actually shows a consistent cadence.

**Vision extraction grabbed the wrong number.** An event needing its
amount read from a receipt image got the receipt's *total* instead of
its *balance due*, because the extraction prompt didn't specify which
figure it needed. A related bug: an Indian-style number (`1,00,000`)
got parsed as a Western-grouped number (`1,000,000`). Fixed by passing
the event's category/description into the prompt for disambiguation and
being explicit about number formatting.

## Known limitations

Self-checked against 25 labeled example requests (the only ground truth
available during the hackathon):

| Field | Match rate |
|---|---|
| `recommended_payment_method` | ~76% |
| `affordability_status` | ~68% |
| `spending_changes_needed` | ~88% |
| `payment_plan` | ~48% |
| `amount_safe_to_pay` (within tolerance) | ~36% |

The gap is concentrated in `amount_safe_to_pay` and traces to the
variable-spending averaging method - the exact algorithm the organizers
used for "conservative" variable spending forecasting isn't specified in
the problem statement, and multiple reasonable approaches exist. One
improvement (recency-weighted median over a flat historical mean) was
tried and helped modestly; further tuning was deprioritized in favor of
finishing the required deliverables under a hard deadline.

## Running it

```bash
pip install -r code/requirements.txt
export ANTHROPIC_API_KEY=your_key_here   # needed for image/message enrichment
python3 code/main.py                      # produces output.csv
python3 code/evaluation/main.py           # self-check against labeled samples
```

Expects a `dataset/` folder at the repo root matching the schema
described above (see the code for exact column names per file).

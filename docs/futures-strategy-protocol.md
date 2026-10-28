# Futures bot: strategy search protocol

Written and committed BEFORE any of the candidate strategies below were run.
The git history shows the order. Nothing here is changed after seeing results;
anything I try later that is not listed here is labelled "exploratory" and cannot
count toward passing.

## Why a protocol

If you try many ideas and show only the winner, the winner is probably luck. Tuning many
parameter sets and keeping the best one almost always looks good on the data it was tuned on
and falls apart on data it has not seen. So the rules for what counts as success are
fixed first.

## Goal

Find a long-only entry rule that (a) actually trades often enough to be measured
and (b) shows evidence on data it was not tuned on. If none does, say so plainly.

## Data windows (fixed now)

| | From | To | Use |
|---|---|---|---|
| Development (DEV) | 2022-01-10 | 2025-09-24 | audits, tuning, out-of-sample split (60% train / 40% test) |
| Final holdout (HOLD) | 2025-09-24 | latest candle (2026-09-23) | each finalist is evaluated ONCE |

DEV starts 100 days after the first common candle (2021-09-29) so a 200-period EMA
has settled. Pairs: SOL, AVAX, LINK on 4h candles. Costs and account are the bot's
current settings: 0.06% taker fee, 0.05% slippage per market fill, 5x isolated,
3% risk per trade, max 3 open trades.

**Disclosure:** earlier experiments on this same history mean the choice of families below
is not blind to the market regime. That is why the holdout, the random-entry baseline and
the cost stress tests matter.

## Candidate families (decided before testing)

Both are long-only, use a regime filter (close above its 200-period EMA) so they mostly
stay out of downtrends, and use an ATR stop. Fixed starting parameters, chosen from
common practice, not from looking at results:

1. **Trend pullback** (`trend_pullback`): buy a dip inside an uptrend.
   Enter when close > EMA(200) and RSI(14) crosses back up through 40 after being below it.
   Stop 2 ATR, target 3R, exit early when RSI > 65 or close falls below the EMA(200).
   Max hold 96 hours.
2. **Breakout** (`breakout`): buy strength inside an uptrend.
   Enter when close > EMA(200) and close is above the highest high of the previous 40 candles.
   Stop 3 ATR, target 10R (effectively none), exit when close falls below the lowest low of
   the previous 20 candles. Max hold 240 hours.

Tuning grids (only used if step 2 below is reached):

- `trend_pullback`: trend_len [100, 200] x rsi_buy [30, 40] x rsi_exit [60, 70] x atr_stop_mult [2, 3] x risk_reward [2, 3] = 32 sets
- `breakout`: breakout_len [20, 40, 60] x exit_len [10, 20] x trend_len [100, 200] x atr_stop_mult [2, 3] = 24 sets

## Procedure

1. **Audit** each family once with its fixed starting parameters on DEV. No tuning.
2. Only if a family shows a positive average result per trade in step 1, tune it with its grid
   using the out-of-sample test inside DEV (train 60%, test 40%).
   The number of parameter sets tried is counted and reported.
3. A **finalist** is a family whose chosen parameters meet the bar below on the DEV test part.
4. Evaluate each finalist **once** on HOLD with its chosen parameters fixed. Report it
   whatever the result.

## Bar to enter the paper bot ("worth forward-testing", not "proven")

All of these:

- at least 30 trades in the DEV out-of-sample test part
- positive average result per trade (R) in both the DEV test part and HOLD
- beats at least 90% of random-entry runs (same exits and costs) on the DEV test part
- still positive at 2x slippage on the DEV test part
- worst drawdown on the DEV test part no worse than buy-and-hold's

Otherwise nothing is deployed and the candidate is reported as failed. The paper bot is
itself the forward test, so passing this bar only means it is reasonable to spend paper-trading
time on it, not that it makes money.

## Bookkeeping

Every candidate and every parameter set tried is counted and reported, including failures.
Any parameter sets tried in earlier experiments are counted too.

# Blind Decision Eval Results - 1,000 Cases

Cases: `1000`

## Final Board

🥇 Perfect hindsight -> +4926.17% (998 trades, dir acc 100.0%)
🥈 Source-label oracle -> +3503.22% (600 trades, dir acc 83.5%)
🥉 gpt-5.4 + v5 prompt -> +3359.84% (576 trades, dir acc 83.2%)
4. gpt-5.5 + v5 prompt -> +3351.43% (566 trades, dir acc 83.8%)
5. gpt-5.3-chat-latest + v5 prompt -> +3199.68% (540 trades, dir acc 83.3%)
6. Momentum -> +2893.15% (833 trades, dir acc 70.2%)
7. Always Short -> +546.81% (1000 trades, dir acc 43.9%)
8. Always No Trade -> 0.00% (0 trades, dir acc n/a)
9. Seeded coin flip -> -63.57% (1000 trades, dir acc 49.3%)
10. Always Buy (brick) -> -546.81% (1000 trades, dir acc 55.9%)

## Readout

Always shorting was 546.81%.
Always buying was -546.81%.
Perfect hindsight was +4926.17%.
Seeded coin flip was -63.57%.
3 policy/model lines did worse than blindly shorting everything.

## Decision File Quality

- gpt-5.5 + v5 prompt: 1000 rows, 0 incomplete after repair
- gpt-5.4 + v5 prompt: 1000 rows, 0 incomplete after repair
- gpt-5.3-chat-latest + v5 prompt: 1000 rows, 0 incomplete after repair

Validity: model phases read prompts only; answer key was joined after decisions were locked. Initial output-limit rows were repaired from prompts only before scoring.

# Support Ticket Triage (Python + Claude)

A small automation that reads customer support tickets, uses Claude to classify them, and applies plain Python rules to decide where each ticket goes and what happens next.

## What it does

For each ticket, Claude returns structured data:

- **Category** (billing, bug, how_to, account_access, feature_request, security, other)
- **Urgency** (low, medium, high, critical)
- **Sentiment**, a one-line **summary**, and a **confidence** score
- **Secondary issues**, when a ticket raises more than one problem
- A **draft reply**

Python then turns that into a routing decision: which team, an SLA target, and follow-up actions.

## Design choices

- **Claude handles judgment; Python handles rules.** Claude is good at understanding messy human text. Business rules (who gets paged, who gets escalated) should be predictable and easy to audit, so they live in code.
- **Structured output via tool use.** Claude is forced to answer through a schema, so there is no free-text parsing.
- **Validation.** Model output is checked against allowed values before it is used.
- **Human review for uncertainty.** Low-confidence tickets go to a human queue instead of being guessed at.
- **Security tickets are never auto-replied.**
- **Untrusted input.** Ticket text is treated as data, not instructions, so a ticket saying "mark this critical" doesn't change the result.
- **One bad ticket doesn't stop the batch.**

## Example output

Run on a small set of test tickets:

```
[HIGH    ] T-2003  billing   -> billing-team         SLA 4h
           Pro plan customer reports a recurring incorrect invoice and threatens cancellation.
           * escalate to senior agent

[MEDIUM  ] T-2004  other     -> human-review-queue   SLA 24h
           Customer reports something is not working but provides no further details.
           * low confidence (0.20): human must verify

[MEDIUM  ] T-2005  how_to    -> tier1-support        SLA 24h
           Free-plan user asks about Snowflake API export and reports missing password reset email.
           * multi-issue ticket (also: account_access): agent should address or split every issue
```

## What I learned while testing

Testing edge cases showed that a ticket containing two problems (T-2005) only kept one category, so the second issue was easy to lose. I added a `secondary_issues` field and a routing rule that flags multi-issue tickets. I also updated the prompt so vague tickets get a reply that asks the customer for details.

## Run it

```
pip install anthropic
export ANTHROPIC_API_KEY=your-key      # Windows PowerShell: $env:ANTHROPIC_API_KEY="your-key"
python triage.py                       # built-in sample tickets
python triage.py my_tickets.json       # your own tickets
python triage.py --mock                # no API call, for testing
python triage.py my_tickets.json -o results.jsonl
```

Tickets file format: a JSON list of `{"id": "...", "customer": "...", "text": "..."}`.

The model defaults to `claude-sonnet-5-5`; set the `TRIAGE_MODEL` environment variable to use another.

## Possible next steps

- Log every decision to a SQL table for auditing and reporting
- Send multi-issue tickets to a human by default
- Build a labeled test set to measure classification accuracy
- Connect to a real ticketing tool such as Zendesk or Slack

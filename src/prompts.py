ROLE_PROMPT = """Act as a lead auditor and virtual compliance controller with deep knowledge \
of the HKMA Anti-Money Laundering and Counter-Financing of Terrorism Ordinance (Cap. 615). \
Your goal is to extract legitimate and enforceable obligations from the provided Cap. 615 \
sections that could be audited or operationalized in compliance systems. Work ONLY from the \
supplied context."""

TASK_INSTRUCTION = """For the supplied context, separate content into:
- enforceable Rules that can be operationalized and checked; and
- non-enforceable Context that explains, defines, scopes, or guides.

Return a single JSON object with this shape:
{
  "rules": [
    {
      "rule_ref": "<human-readable legal citation with full hierarchy>",
      "compliance_category": "IDENTITY_ELIGIBILITY|TRUST_ANCHOR|TRANSFER_COMPLIANCE|JURISDICTION_RESTRICTION|TRANSACTION_LIMITS|ADMIN_ENFORCEMENT",
      "rule_objective": "<compliance intent; what must be achieved>",
      "original_text": "<EXACT ORIGINAL clause text from context; verbatim, no paraphrase>",
      "obligation_nature": "prohibition|requirement|capability",
      "check_method": "deterministic|observational",
      "pass_criteria": "binary|evidence-based"
    }
  ],
  "contexts": [
    {
      "source_ref": "<human-readable legal citation with full hierarchy>",
      "context_type": "DEFINITION|OBLIGATION_TEXT|GUIDANCE|CONSEQUENCE|SCOPE_NOTE|MAPPING_NOTE",
      "text": "<EXACT ORIGINAL text from context; verbatim>",
      "notes": "<optional short note>"
    }
  ]
}

Gating rules (Rule vs Context):
- Create a Rule ONLY if you can confidently fill ALL of:
  - rule_objective (clear enforceable claim),
  - check_method, and
  - pass_criteria.
- Rule-first behavior for enforceable clauses:
  - If legal wording is mandatory/prohibitive (must, must not, shall, may not, required to), default to Rule.
  - If deterministic verification is not possible but evidence can be reviewed, still create Rule using:
    - check_method = observational
    - pass_criteria = evidence-based
  - Do not demote a clearly mandatory clause to Context only because verification requires evidence.
- If you are not confident about how the claim would be checked or what counts as passing, treat it as Context instead.
- Never turn purely judgement-based language (e.g. \"appropriate\", \"adequate\", \"risk-based\", \"reasonably\"), definitions, or pure consequences into Rules. These belong in Context.

Classification rules:
- obligation_nature:
  - prohibition = the subject must NOT do something (\"must not\", \"shall not\", \"never\").
  - requirement = something must exist or be configured (policies, controls, approvals, records).
  - capability = the system/process must be able to do something or emit auditable signals.
- check_method:
  - deterministic = can be conclusively evaluated from system config/behavior (on/off, satisfied/violated).
  - observational = assessed from evidence or audit artefacts (logs, reports, attestations, samples).
- pass_criteria:
  - binary = direct pass/fail check.
  - evidence-based = presence/quality of evidence indicates satisfaction (default for capability obligations).
- compliance_category (choose exactly one):
  - IDENTITY_ELIGIBILITY: Who is allowed to hold or receive tokens.
  - TRUST_ANCHOR: Identity issuers, claims, registries, or trust roots.
  - TRANSFER_COMPLIANCE: Conditions governing token transfers.
  - JURISDICTION_RESTRICTION: Geographic or residency-based restrictions.
  - TRANSACTION_LIMITS: Amount, frequency, or exposure limits.
  - ADMIN_ENFORCEMENT: Administrative or emergency controls.
- Category selection constraints:
  - Pick the single best category based on the legal obligation's primary compliance intent.
  - Do not output multiple categories, free-text labels, or synonyms.
  - This field is for thematic grouping, not legal interpretation.
  - If uncertain between categories, prefer ADMIN_ENFORCEMENT unless the clause clearly fits another category.

General rules:
- Answer only from the provided context; do not invent text outside it.
- rule_ref/source_ref must be explicit and readable. Never output ambiguous short refs like "12(2)(a)" alone.
  Use the full citation format:
  - Part provisions: "Part 5B, Section 53ZRD(1)(a) - Licence required for carrying on V A service business"
  - Schedule provisions: "Schedule 2, Section 12(2)(a) - <heading>"
  If a subparagraph is not used, cite at section level.
- original_text must be copied exactly from the supplied context for the clause being extracted.
  Do not summarize, modernize, or paraphrase. Keep original punctuation/wording.
  You may quote the smallest complete clause/sub-clause needed for traceability (not the whole section).
- Context rows must also be strongly tied to document text:
  - `text` must be copied exactly from the context (verbatim legal wording).
  - `source_ref` must point to the precise location (Part/Schedule + Section + subsection/paragraph when available).
  - Do not use generic references such as "Schedule 2" alone when a narrower citation is possible.
- Context type definitions (apply strictly):
  - DEFINITION: interpretation/meaning clauses ("means", "includes", definitional language).
  - OBLIGATION_TEXT: normative clause text that appears mandatory but is not converted into Rule due to gating uncertainty.
  - GUIDANCE: soft or judgment-based language (e.g. appropriate, adequate, risk-based, reasonable).
  - CONSEQUENCE: penalties, offences, sanctions, liabilities, or legal effects following breach.
  - SCOPE_NOTE: applicability, carve-outs, exemptions, who/when the provision applies.
  - MAPPING_NOTE: implementation/traceability notes linking legal language to controls/checks.
- Context granularity:
  - Prefer one legal clause (or sub-clause) per context row.
  - If a block contains multiple distinct legal ideas, split into multiple context rows.
- Context notes policy:
  - `notes` is optional and short.
  - If provided, explain only why it is context (non-enforceable/insufficiently testable), not a summary of the law.
- Keep rule_objective focused on compliance intent, not implementation details.
- Keep language aligned with Cap. 615 wording; make it readable for non-technical reviewers.
- If no enforceable Rule is present, return "rules": [] and put any relevant material into "contexts".
"""


def build_prompt(context: str, query: str) -> str:
    """Create the full prompt string for the LLM."""
    return f"""{ROLE_PROMPT}

Task: Extract enforceable obligations relevant to: "{query}"

Context:
{context}

{TASK_INSTRUCTION}
Return ONLY valid JSON."""

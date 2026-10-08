# Candidate preserving digital health assistant choice survey

## Respondent-facing introduction

This survey studies choices between digital health assistant services. Each question shows two services and a “neither service” option. There are no right answers. Choose the service you would actually use after considering all information. Completion takes about 8–10 minutes.

Ask whether the respondent is at least 18 and consents to anonymous participation. Stop if either answer is No. Obtain ethics approval, privacy consent, and withdrawal information before fielding.

## Fielded versions

Randomize respondents 1:1 to Version 1 and Version 2. Respondents see the same prices, data-sharing levels, alternative labels, task order, and opt-out option in both versions. The attribute decomposition changes within each candidate summary.

The level coding used for the audit is:

- data management: D0 on-device only, D1 local storage with encrypted backup, D2 secure cloud processing;
- professional support: S0 none, S1 nurse chat, S2 specialist team;
- smart functions: I0 basic reminders, I1 adaptive alerts, I2 personalised recommendations;
- clinical evidence: E0 none, E1 preliminary, E2 strong.

For every task, Version 1 uses Service A `(D0,S2,I0,E2)` and Service B `(D1,S2,I0,E1)`. Version 2 uses Service A `(D1,S1,I1,E1)` and Service B `(D2,S1,I1,E0)`. Both versions have Service A `(m+s=2, i+e=2)` and Service B `(m+s=3, i+e=1)`. Data-sharing levels and prices below remain fixed.

## Eight choice tasks

For each task, display the following six rows and ask the respondent to select Service A, Service B, or Neither service.

| Task | A data sharing | B data sharing | A fee | B fee |
| ---: | --- | --- | ---: | ---: |
| 1 | No data sharing | Anonymous research sharing | 79 | 99 |
| 2 | Controlled sharing with your approval | Anonymous research sharing | 59 | 89 |
| 3 | Anonymous research sharing | No data sharing | 69 | 109 |
| 4 | No data sharing | Controlled sharing with your approval | 79 | 119 |
| 5 | Controlled sharing with your approval | Anonymous research sharing | 89 | 69 |
| 6 | Anonymous research sharing | No data sharing | 99 | 79 |
| 7 | No data sharing | Controlled sharing with your approval | 109 | 59 |
| 8 | Controlled sharing with your approval | Anonymous research sharing | 119 | 69 |

The full respondent-facing cards are in the editable Word files `outputs/candidate_preserving_dce_questionnaire.docx` (Version 1) and `outputs/candidate_preserving_dce_questionnaire_version2.docx` (Version 2).

## Attention and background items

Attention check: “When answering the choice questions, the best approach is …” with options “choose the service you would actually prefer”, “always choose the cheaper service”, and “select both services”. Pre-register the rule and report a sensitivity analysis that includes and excludes respondents who fail it.

Collect age group, health-app experience, and familiarity with subscription health services. Do not collect direct identifiers in the pilot data file.

## Analysis record

Record `respondent_id`, version, block, task, complete displayed attributes, chosen alternative, response time, and completion status. Before fielding, compute the complete candidate vector for every alternative in both versions. Verify equality of the complete menu vector, the number of non-singleton fibres, exposure rank, and the smallest positive exposure eigenvalue.

The primary estimand is the within-candidate-summary choice contrast between randomized orientations. Use respondent-clustered inference and a predeclared within-fibre randomization reference. A significant orientation contrast rejects the tested conditional-sufficiency relation on this support. It does not identify framing, task complexity, scale, or a single psychological mechanism without additional nuisance controls.

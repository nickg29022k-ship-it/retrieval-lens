# RAG Answer Quality Report

- Backend: `mock` (model `mock-extractive-v1`)
- Corpus: 28 documents, QA set: 42 questions (6 unanswerable)
- Grid: chunkers {'fixed': {'size': 50, 'overlap': 10}, 'sentence': {'max_words': 50}, 'heading': {'max_words': 120}}, retrievers ['bm25', 'tfidf'], k [1, 3, 5] (18 configurations)
- Options: {'llm_judge': False, 'llm_claims': False, 'llm_verify': False}

## Best configuration

**sentence/tfidf/k=1** has the highest end-to-end success rate (0.476: correct and faithful answers plus correct abstentions, over all questions). Ties are broken by faithfulness, then citation precision, then recall@k.

- Retrieval: recall@k 0.917, MRR 0.917, nDCG@k 0.917
- Answers: accuracy 0.611, token F1 0.252, EM 0.000
- Grounding: faithfulness 0.741, citation precision 0.764, citation recall 0.708
- Best pure retrieval config: heading/bm25/k=3 (recall@k 0.972, MRR 0.954)
- Most faithful config: fixed/bm25/k=3 (faithfulness 0.741)

## Comparison table

Sorted by success rate. Retrieval and accuracy columns are averaged over answerable questions; faithfulness and citation columns over questions that were answered (not abstained).

| Config | R@k | P@k | MRR | nDCG@k | CtxRel | EM | F1 | Acc | Faith | CitP | CitR | AbstAcc | Success |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sentence/tfidf/k=1 | 0.917 | 0.917 | 0.917 | 0.917 | 0.343 | 0.000 | 0.252 | 0.611 | 0.741 | 0.764 | 0.708 | 0.500 | 0.476 |
| sentence/bm25/k=1 | 0.917 | 0.917 | 0.917 | 0.917 | 0.343 | 0.000 | 0.252 | 0.611 | 0.733 | 0.757 | 0.700 | 0.500 | 0.476 |
| heading/bm25/k=1 | 0.944 | 0.944 | 0.944 | 0.944 | 0.528 | 0.000 | 0.252 | 0.611 | 0.727 | 0.736 | 0.708 | 0.500 | 0.476 |
| heading/tfidf/k=1 | 0.944 | 0.944 | 0.944 | 0.944 | 0.528 | 0.000 | 0.252 | 0.611 | 0.727 | 0.736 | 0.708 | 0.500 | 0.476 |
| heading/bm25/k=3 | 0.972 | 0.324 | 0.954 | 0.948 | 0.278 | 0.000 | 0.252 | 0.611 | 0.727 | 0.681 | 0.653 | 0.500 | 0.476 |
| heading/bm25/k=5 | 0.972 | 0.200 | 0.954 | 0.955 | 0.190 | 0.000 | 0.252 | 0.611 | 0.727 | 0.681 | 0.653 | 0.500 | 0.476 |
| heading/tfidf/k=3 | 0.972 | 0.324 | 0.954 | 0.948 | 0.278 | 0.000 | 0.252 | 0.611 | 0.727 | 0.681 | 0.653 | 0.500 | 0.476 |
| heading/tfidf/k=5 | 0.972 | 0.200 | 0.954 | 0.954 | 0.189 | 0.000 | 0.252 | 0.611 | 0.727 | 0.681 | 0.653 | 0.500 | 0.476 |
| sentence/bm25/k=3 | 0.972 | 0.324 | 0.940 | 0.948 | 0.187 | 0.000 | 0.252 | 0.611 | 0.734 | 0.689 | 0.662 | 0.333 | 0.452 |
| sentence/bm25/k=5 | 0.972 | 0.194 | 0.940 | 0.948 | 0.119 | 0.000 | 0.252 | 0.611 | 0.734 | 0.689 | 0.662 | 0.333 | 0.452 |
| sentence/tfidf/k=3 | 0.972 | 0.324 | 0.940 | 0.948 | 0.180 | 0.000 | 0.252 | 0.611 | 0.734 | 0.689 | 0.662 | 0.333 | 0.452 |
| sentence/tfidf/k=5 | 0.972 | 0.194 | 0.940 | 0.948 | 0.121 | 0.000 | 0.252 | 0.611 | 0.734 | 0.689 | 0.662 | 0.333 | 0.452 |
| fixed/bm25/k=3 | 0.972 | 0.389 | 0.931 | 0.939 | 0.176 | 0.000 | 0.228 | 0.556 | 0.741 | 0.697 | 0.671 | 0.333 | 0.429 |
| fixed/bm25/k=5 | 0.972 | 0.233 | 0.931 | 0.939 | 0.120 | 0.000 | 0.228 | 0.556 | 0.741 | 0.697 | 0.671 | 0.333 | 0.429 |
| fixed/bm25/k=1 | 0.889 | 0.889 | 0.889 | 0.889 | 0.301 | 0.000 | 0.220 | 0.528 | 0.741 | 0.750 | 0.724 | 0.333 | 0.405 |
| fixed/tfidf/k=3 | 0.972 | 0.389 | 0.917 | 0.931 | 0.176 | 0.000 | 0.220 | 0.528 | 0.715 | 0.697 | 0.671 | 0.333 | 0.405 |
| fixed/tfidf/k=5 | 0.972 | 0.233 | 0.917 | 0.931 | 0.118 | 0.000 | 0.220 | 0.528 | 0.715 | 0.697 | 0.671 | 0.333 | 0.405 |
| fixed/tfidf/k=1 | 0.861 | 0.861 | 0.861 | 0.861 | 0.322 | 0.000 | 0.210 | 0.500 | 0.715 | 0.750 | 0.724 | 0.333 | 0.381 |

## Failure breakdown per configuration

Counts of questions in each attribution category, plus claim-level hallucination counts.

| Config | correct | retrieval_failure | generation_failure | correct_abstention | incorrect_abstention | missed_abstention | unsupported claims | contradicted claims |
|---|---|---|---|---|---|---|---|---|
| sentence/tfidf/k=1 | 17 | 3 | 15 | 3 | 1 | 3 | 5 | 8 |
| sentence/bm25/k=1 | 17 | 3 | 15 | 3 | 1 | 3 | 5 | 8 |
| heading/bm25/k=1 | 17 | 2 | 14 | 3 | 3 | 3 | 5 | 8 |
| heading/tfidf/k=1 | 17 | 2 | 14 | 3 | 3 | 3 | 5 | 8 |
| heading/bm25/k=3 | 17 | 1 | 15 | 3 | 3 | 3 | 5 | 8 |
| heading/bm25/k=5 | 17 | 1 | 15 | 3 | 3 | 3 | 5 | 8 |
| heading/tfidf/k=3 | 17 | 1 | 15 | 3 | 3 | 3 | 5 | 8 |
| heading/tfidf/k=5 | 17 | 1 | 15 | 3 | 3 | 3 | 5 | 8 |
| sentence/bm25/k=3 | 17 | 1 | 15 | 2 | 3 | 4 | 5 | 8 |
| sentence/bm25/k=5 | 17 | 1 | 15 | 2 | 3 | 4 | 5 | 8 |
| sentence/tfidf/k=3 | 17 | 1 | 15 | 2 | 3 | 4 | 5 | 8 |
| sentence/tfidf/k=5 | 17 | 1 | 15 | 2 | 3 | 4 | 5 | 8 |
| fixed/bm25/k=3 | 16 | 1 | 17 | 2 | 2 | 4 | 5 | 8 |
| fixed/bm25/k=5 | 16 | 1 | 17 | 2 | 2 | 4 | 5 | 8 |
| fixed/bm25/k=1 | 15 | 4 | 16 | 2 | 1 | 4 | 5 | 8 |
| fixed/tfidf/k=3 | 15 | 1 | 18 | 2 | 2 | 4 | 5 | 9 |
| fixed/tfidf/k=5 | 15 | 1 | 18 | 2 | 2 | 4 | 5 | 9 |
| fixed/tfidf/k=1 | 14 | 5 | 16 | 2 | 1 | 4 | 5 | 9 |

## Effect of each factor

Each row averages over all configurations that share that setting.

### By chunker

| chunker | recall@k | mrr | accuracy | faithfulness | citation_precision | success_rate | retrieval_failure | generation_failure |
|---|---|---|---|---|---|---|---|---|
| fixed | 0.940 | 0.907 | 0.532 | 0.728 | 0.715 | 0.409 | 2.167 | 17.000 |
| heading | 0.963 | 0.951 | 0.611 | 0.727 | 0.699 | 0.476 | 1.333 | 14.667 |
| sentence | 0.954 | 0.932 | 0.611 | 0.735 | 0.713 | 0.460 | 1.667 | 15.000 |

### By retriever

| retriever | recall@k | mrr | accuracy | faithfulness | citation_precision | success_rate | retrieval_failure | generation_failure |
|---|---|---|---|---|---|---|---|---|
| bm25 | 0.954 | 0.933 | 0.590 | 0.734 | 0.709 | 0.452 | 1.667 | 15.444 |
| tfidf | 0.951 | 0.927 | 0.580 | 0.726 | 0.709 | 0.444 | 1.778 | 15.667 |

### By k

| k | recall@k | mrr | accuracy | faithfulness | citation_precision | success_rate | retrieval_failure | generation_failure |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.912 | 0.912 | 0.579 | 0.731 | 0.749 | 0.448 | 3.167 | 15.000 |
| 3 | 0.972 | 0.939 | 0.588 | 0.730 | 0.689 | 0.448 | 1.000 | 15.833 |
| 5 | 0.972 | 0.939 | 0.588 | 0.730 | 0.689 | 0.448 | 1.000 | 15.833 |

## Hallucinated claims in the best configuration (sentence/tfidf/k=1)

| Question | Claim | Label | Reason |
|---|---|---|---|
| q04 | Exceptions can be granted by the regional director in writing. | unsupported | coverage 0.00 |
| q06 | Exceptions can be granted by the regional director in writing. | unsupported | coverage 0.00 |
| q08 | This rule was last revised in 2019 after an internal audit. | unsupported | coverage 0.00; numbers not in context ['2019'] |
| q12 | Suspected phishing emails must be reported to the security team within 4 hour by using the Report Phish button in the mail client. | contradicted | number mismatch: claim has ['4'], evidence has ['1'] |
| q21 | Employees may accept gifts from suppliers or customers only if the value is below 79 euros. | contradicted | number mismatch: claim has ['79'], evidence has ['75'] |
| q23 | This rule was last revised in 2019 after an internal audit. | unsupported | coverage 0.00; numbers not in context ['2019'] |
| q26 | About 87 percent of global evaporation comes from the oceans. | contradicted | number mismatch: claim has ['87'], evidence has ['86'] |
| q28 | its tidal effect is only about 48 percent as strong as the Moon's. | contradicted | number mismatch: claim has ['48'], evidence has ['46'] |
| q32 | The same limit applies to contractors and seasonal staff. | unsupported | coverage 0.00 |
| q34 | ENIAC contained about 17508 vacuum tubes and weighed about 27 tonnes. | contradicted | number mismatch: claim has ['17508'], evidence has ['17468', '1945', '1946', '27'] |
| q35 | ARPANET switched from the older NCP protocol to TCP/IP on January 5, 1983. | contradicted | number mismatch: claim has ['5'], evidence has ['1', '1983'] |
| q36 | In 2003 UNIX was rewritten in the C programming language, which Dennis Ritchie had developed. | contradicted | number mismatch: claim has ['2003'], evidence has ['1973', '7'] |
| q39 | the full moon, when the Sun, Moon and Earth are not aligned. | contradicted | polarity mismatch (negation) |

## Citation errors in the best configuration (sentence/tfidf/k=1)

| Question | Cited doc | In retrieved context | Sentence |
|---|---|---|---|
| q04 | hb-pto | yes | Exceptions can be granted by the regional director in writing. |
| q06 | hb-remote-work | yes | Exceptions can be granted by the regional director in writing. |
| q08 | hb-expenses | yes | This rule was last revised in 2019 after an internal audit. |
| q12 | hb-security | yes | Suspected phishing emails must be reported to the security team within 4 hour by using the Report Phish button in the mail client. |
| q21 | hb-code-of-conduct | yes | Employees may accept gifts from suppliers or customers only if the value is below 79 euros. |
| q23 | hb-parental-leave | yes | This rule was last revised in 2019 after an internal audit. |
| q26 | sci-water-cycle | yes | About 87 percent of global evaporation comes from the oceans. |
| q32 | sci-honeybees | yes | The same limit applies to contractors and seasonal staff. |
| q34 | hist-eniac | yes | ENIAC contained about 17508 vacuum tubes and weighed about 27 tonnes. |
| q35 | hist-arpanet | yes | ARPANET switched from the older NCP protocol to TCP/IP on January 5, 1983. |
| q36 | hist-unix | yes | In 2003 UNIX was rewritten in the C programming language, which Dennis Ritchie had developed. |

## Failure examples (sentence/tfidf/k=1)

| Question | Category | Reason | Answer | Gold | Retrieved docs |
|---|---|---|---|---|---|
| q04 | generation_failure | answer is correct but contains 1 unsupported/contradicted claim(s) | Employees receive 8 paid sick days per year, and a doctor's note is only required after 3 consecutive sick days [hb-pto]. Exceptions can be granted by the regional director in writing [hb-pto]. | 8; 8 paid sick days; 8 days | hb-pto |
| q06 | generation_failure | answer is correct but contains 1 unsupported/contradicted claim(s) | Everyone is expected to be reachable during core hours, which run from 10:00 to 15:00 local time [hb-remote-work]. Exceptions can be granted by the regional director in writing [hb-remote-work]. | 10:00 to 15:00; from 10:00 to 15:00 local time | hb-remote-work |
| q08 | generation_failure | answer is correct but contains 1 unsupported/contradicted claim(s) | All business expenses must be submitted in the Ledgerly expense tool within 30 days of the purchase [hb-expenses]. This rule was last revised in 2019 after an internal audit [hb-expenses]. | 30 days; within 30 days | hb-expenses |
| q09 | retrieval_failure | answer is wrong and the gold evidence was not retrieved | Employees may work from another EU country for up to 20 working days per year [hb-remote-work]. | 85 euros; 85 | hb-remote-work |
| q11 | retrieval_failure | gold evidence was not retrieved, so the generator had nothing to answer from | I cannot answer this from the provided documents. | 14 characters; 14 | hb-security |
| q12 | generation_failure | gold evidence was retrieved but the answer is wrong | Suspected phishing emails must be reported to the security team within 4 hour by using the Report Phish button in the mail client [hb-security]. | 1 hour; within 1 hour | hb-security |
| q19 | incorrect_abstention | gold evidence was in the context but the system abstained | I cannot answer this from the provided documents. | 220 euros; 220 | hb-travel |
| q20 | generation_failure | gold evidence was retrieved but the answer is wrong | Economy class is required for flights shorter than 6 hours [hb-travel]. | premium economy | hb-travel |
| q21 | generation_failure | gold evidence was retrieved but the answer is wrong | Employees may accept gifts from suppliers or customers only if the value is below 79 euros [hb-code-of-conduct]. | below 75 euros; 75 euros; 75 | hb-code-of-conduct |
| q22 | retrieval_failure | gold evidence was not retrieved, so the generator had nothing to answer from | I cannot answer this from the provided documents. | every 3 years; 3 years | hb-onboarding |
| q23 | generation_failure | answer is correct but contains 1 unsupported/contradicted claim(s) | The primary caregiver receives 16 weeks of fully paid parental leave [hb-parental-leave]. This rule was last revised in 2019 after an internal audit [hb-parental-leave]. | 16 weeks; 16 | hb-parental-leave |
| q24 | generation_failure | gold evidence was retrieved but the answer is wrong | Today Lumen Freight employs about 1,200 people [hb-company-overview]. | 2011 in Rotterdam; in 2011 in Rotterdam | hb-company-overview |
| q26 | generation_failure | gold evidence was retrieved but the answer is wrong | About 87 percent of global evaporation comes from the oceans [sci-water-cycle]. | 86 percent; about 86 percent; 86% | sci-water-cycle |
| q28 | generation_failure | gold evidence was retrieved but the answer is wrong | The Sun also produces tides, but its tidal effect is only about 48 percent as strong as the Moon's [sci-tides]. | about 46 percent; 46 percent; 46% | sci-tides |
| q31 | generation_failure | gold evidence was retrieved but the answer is wrong | A day on Mars, called a sol, lasts 24 hours and 37 minutes [sci-mars]. | Phobos and Deimos | sci-mars |
| q32 | generation_failure | answer is correct but contains 1 unsupported/contradicted claim(s) | The queen can lay up to 2,000 eggs per day during peak season [sci-honeybees]. The same limit applies to contractors and seasonal staff [sci-honeybees]. | up to 2,000; 2,000; 2000 | sci-honeybees |
| q34 | generation_failure | gold evidence was retrieved but the answer is wrong | ENIAC contained about 17508 vacuum tubes and weighed about 27 tonnes [hist-eniac]. | about 17,468; 17,468; 17468 | hist-eniac |
| q35 | generation_failure | gold evidence was retrieved but the answer is wrong | ARPANET switched from the older NCP protocol to TCP/IP on January 5, 1983 [hist-arpanet]. | January 1, 1983; 1 January 1983 | hist-arpanet |
| q36 | generation_failure | gold evidence was retrieved but the answer is wrong | In 2003 UNIX was rewritten in the C programming language, which Dennis Ritchie had developed [hist-unix]. | 1973, Dennis Ritchie; 1973 by Dennis Ritchie | hist-unix |
| q39 | missed_abstention | question is unanswerable but the system produced an answer | Spring tides have the largest range and occur around the new moon and the full moon, when the Sun, Moon and Earth are not aligned [sci-tides]. | (unanswerable) | sci-tides |
| q41 | missed_abstention | question is unanswerable but the system produced an answer | The daily meal allowance is 60 euros for domestic travel and 85 euros for international travel [hb-expenses]. | (unanswerable) | hb-expenses |
| q42 | missed_abstention | question is unanswerable but the system produced an answer | A honeybee colony has a single queen, thousands of female worker bees and, in summer, a few hundred male drones [sci-honeybees]. | (unanswerable) | sci-honeybees |

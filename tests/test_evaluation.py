from evaluation.run_eval import aggregate, groups_ok, numbers


def test_helpers():
    assert groups_ok("The warranty is 24 months.", [["24 months"]]) and not groups_ok("12 months", [["24 months"]])
    assert numbers("PKR 1,800 and 4 months") == {"1800", "4"}


def test_aggregate_counts_hallucinations():
    row = {"verdict": "fail", "behavior_ok": False, "answered": True, "must_not_violations": ["x"], "ungrounded_numbers": ["9"], "citation_ok": True,
           "retrieval_hit@k": True, "expected": "refuse", "latency_ms": 10, "input_tokens": 1, "output_tokens": 1}
    assert aggregate([row])["hallucination_count"] == 3 and aggregate([row])["refusal_rate_on_unanswerable"] == 0.0

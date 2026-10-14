"""EXP032D source identity and deny-only candidate verifier regression."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research'))
import k2_qwen_candidate_guard as m

EVIDENCE = Path('/home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026')

class CandidateVerifierTests(unittest.TestCase):
    def test_preregistration_and_parent_receipts(self):
        self.assertEqual(m.sha(EVIDENCE / 'EXPERIMENT-032D-QWEN-VERIFIER-PROTOCOL.md'), m.PROTOCOL)
        self.assertEqual(m.sha(EVIDENCE / 'EXPERIMENT-032B-STOP-POSTHOC.md'), m.EXP032B_SHA)

    def test_reject_unsupported_source_candidate(self):
        x = m.candidate_rules('Berlin', 'Paris is in France', 'Where is Paris?')
        self.assertEqual(x['lexical_gate'], 'UNKNOWN')
        self.assertFalse(x['verified_candidate_eligible'])

    def test_question_exclusion(self):
        x = m.candidate_rules('Paris', 'Paris is in France', 'Where is Paris?')
        self.assertEqual(x['lexical_gate'], 'Paris')
        self.assertEqual(x['question_exclusion'], 'UNKNOWN')
    def test_source_presence_does_not_establish_entailment(self):
        x = m.candidate_rules('Paris', 'Paris is in France', 'Who wrote the treaty?')
        self.assertEqual(x['question_exclusion'], 'Paris')
        self.assertTrue(x['verified_candidate_eligible'])

    def test_fail_closed_yes_no_contract(self):
        for output in ('YES', ' yes.', 'Yes! '):
            self.assertEqual(m.normalized_verdict(output, False), 'YES')
        for output in ('NO', 'No.'):
            self.assertEqual(m.normalized_verdict(output, False), 'NO')
        for output in ('Yes, because', 'Thinking: YES', 'MAYBE', '', 'YES\nNO'):
            self.assertEqual(m.normalized_verdict(output, False), 'MALFORMED')
        self.assertEqual(m.normalized_verdict('YES', True), 'MALFORMED')

    def test_question_conditioned_native_chat_input(self):
        prompt = m.verdict_prompt('Paris is in France', 'Who wrote the treaty?', 'Paris')
        self.assertIn('this PARTICULAR question', prompt)
        self.assertIn('Question: Who wrote the treaty?', prompt)
        self.assertIn('Proposed candidate: Paris', prompt)
        self.assertNotIn('is_impossible', prompt)
    def test_human_gold_metrics_separated_after_decision(self):
        items = []
        for i in range(16):
            items.append({'strategy': 'qwen_verifier',
                          'unanswerable': i >= 8,
                          'strict': int(i == 0 or i >= 8),
                          'nonabstain_negative': 0,
                          'abstain_positive': int(i == 1)})
        self.assertEqual(m.stats(items, 'qwen_verifier'),
                         {'correct': 1, 'false_accept': 0,
                          'abstain_negative': 8, 'missed_positive': 1})

    def test_measured_report_if_present(self):
        report = EVIDENCE / 'EXPERIMENT-032D-QWEN-VERIFIER.md'
        if not report.is_file():
            return
        content = report.read_text()
        self.assertIn(m.PROTOCOL, content)
        self.assertIn(m.EXP032B_SHA, content)
        self.assertIn('| qwen_verifier |', content)

    def test_exact_measured_report_and_human_gold_cells(self):
        path = EVIDENCE / 'EXPERIMENT-032D-QWEN-VERIFIER.md'
        self.assertEqual(m.sha(path),
                         'b6c04efdfd42187c7bcd8f470500467191d7ffe37ab4945cb4b37510510742fc')
        report = path.read_text()
        expected = {
            'first_word': (6, 7, 1, 0),
            'lexical_gate': (6, 7, 1, 0),
            'question_exclusion': (6, 4, 4, 1),
            'qwen_verifier': (6, 3, 5, 1),
            'always_unknown': (0, 0, 8, 8)}
        for name, scores in expected.items():
            self.assertIn('| ' + name + ' | ' + ' | '.join(map(str, scores)) + ' |', report)
        for mark in ('Actual candidate evaluations requested of frozen Qwen: **11**.',
                     'Strict whole-output YES: **10**.',
                     'Strict whole-output NO: **1**.',
                     'Malformed, inconclusive or output cap denied: **0**.',
                     m.PROTOCOL, m.EXP032B_SHA):
            self.assertIn(mark, report)

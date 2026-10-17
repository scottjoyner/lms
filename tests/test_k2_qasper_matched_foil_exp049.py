"""EXP049 fixed sources, token-length-matched foils and report layout."""
import pathlib
import sys
import unittest
P=pathlib.Path(__file__).resolve().parents[1]/"research"
sys.path.insert(0,str(P))
import k2_qasper_matched_foil_fixture as fx
import k2_qasper_matched_foil_eval as e

class TestMatchedFoil(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.rows=fx.make()
    def test_frozen_source_and_token_count(self):
        self.assertEqual(len(e.frozen()),16)
        self.assertEqual(fx.source_digest(self.rows),e.COHORT_SHA)
        for a,b in zip(self.rows[::2],self.rows[1::2]):
            self.assertEqual((a['arm'],b['arm']),('original','withheld'))
            self.assertEqual((a['candidate_gold'],a['candidate_foil']),
                             (b['candidate_gold'],b['candidate_foil']))
            self.assertIn(a['candidate_gold'].strip(),a['row']['context'])
            self.assertIn(a['candidate_foil'].strip(),a['row']['context'])
            self.assertEqual(a['n'],b['n'])
            self.assertFalse(fx.lexical(a['candidate_gold'])&
                             fx.lexical(a['candidate_foil']))
    def test_reporting_paired(self):
        data={mode:[] for mode in fx.MODES}
        for mode in fx.MODES:
            for row in self.rows:
                m=2.0 if row['arm']=='original' else -1.
                data[mode].append({'sum_margin':m,'mean_margin':m/3})
        cell,diff=e.grouped(self.rows,data)
        self.assertEqual(len(cell),4);self.assertEqual(len(diff),2)
        self.assertEqual(diff[0]['positive'],8)
        txt=e.report(cell,diff,[130]*16,[5]*16,35.)
        self.assertIn('64 frozen',txt)
        self.assertIn(e.PROTOCOL_SHA,txt)
        self.assertNotIn(self.rows[0]['row']['question'],txt)
    def test_frozen_report_existing(self):
        self.assertEqual(e.EXP048_SHA,
              '7b34a7eac6918cd587a4fcc3a976fefd2377e48de84414f72cfd268d6f6d8bd0')
if __name__=='__main__':unittest.main()

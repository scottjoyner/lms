"""EXP044 exact terminal QuAC sentinel intervention, original receipt and scoring."""
import hashlib,pathlib,sys,unittest
from unittest.mock import patch
P=pathlib.Path(__file__).resolve().parents[1]/"research"
sys.path.insert(0,str(P))
import k2_quac_sentinel_counterfactual as fx
import k2_quac_sentinel_ablation as run
import k2_quac_independent_fixture as q

class TestEXP044(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original,cls.cleaned=fx.pairs()
    def test_only_30_terminal_markers_removed(self):
        changed=0
        for a,b in zip(self.original,self.cleaned):
            fields=set(a)
            self.assertEqual(fields,set(b))
            altered=[k for k in fields if a[k]!=b[k]]
            if a["kind"]=="withheld":
                self.assertEqual(altered,[])
            else:
                self.assertEqual(altered,["context"])
                self.assertEqual(a["context"],b["context"]+" CANNOTANSWER")
                changed+=1
        self.assertEqual(changed,30)
        self.assertEqual(q.receipt(self.original),fx.ORIGINAL_SHA)
    def test_frozen_prompt_and_protocol_hashes(self):
        self.assertEqual(run.PROTOCOL_SHA,
            "0e1db73619c7bf594a23ec75d94c7d543551f0dea74fffe267a3f8c489f39539")
        self.assertEqual(fx.pair_digest(),run.PAIRED_SHA)
        self.assertEqual(q.receipt(self.cleaned),run.CLEAN_SHA)
        self.assertEqual(len(run.frozen_preflight()[0]),36)
    def test_scoring_and_original_repeat_gate(self):
        base=[q.score(x,x["gold"],hit_newline=True) for x in self.original]
        other=[q.score(x,x["gold"],hit_newline=True) for x in self.cleaned]
        for item in base+other:item["native_eos"]=0
        original_stats=q.aggregate(self.original,base)
        archived={}
        for kind in run.KINDS:
            archived[("default",kind)]={**original_stats[kind],"native_eos":0}
            archived[("yarn",kind)]={**original_stats[kind],"native_eos":0}
        # Every response tagged newline so each stop reason is valid.
        copies={"default":{"original":base,"no_sentinel":other},
                "yarn":{"original":base,"no_sentinel":other}}
        with patch.object(run,"old_table",return_value=archived):
            totals,pairs=run.summarize_pairs(self.original,self.cleaned,None,copies)
        self.assertEqual(sum(totals["default"]["original"][k]["strict"] for k in run.KINDS),36)
        self.assertEqual(sum(pairs["yarn"][k]["strict_gained"] for k in run.KINDS),0)
        text=run.report(totals,pairs,[321]*144,111.1)
        self.assertIn("All 72 original-repeat generations matched",text)
        self.assertIn(run.PAIRED_SHA,text)
    def test_historical_source_receipt(self):
        raw=(run.ROOT/"EXPERIMENT-043-RESULTS-2026-10-15.md").read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),run.ORIGINAL_RESULT_SHA)
        self.assertEqual(len(run.old_table()),8)

if __name__=="__main__":unittest.main()

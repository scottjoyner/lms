"""All model outputs in this file are deliberately SYNTHETIC software fixtures."""
import copy,hashlib,sys,unittest
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"research"))
import exp053_metrics as m

def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def synthetic():
    ids=[f"fake_article_{i:02d}" for i in range(40)]
    manifest={"rows":[{"case_id":c,"article_sha256":sha("synthetic-article-"+c)} for c in ids]}
    pub={x["case_id"]:x["article_sha256"] for x in manifest["rows"]}
    approved={c:{"supported":"A","foil":"B"} for c in ids}
    rows=[]
    for model in m.MODELS:
        for i,cid in enumerate(ids):
            for a in m.ARMS:
                for rot in m.ROTATIONS:
                    margin=({"supported":1.2,"withheld":0.1,"decoy":0.3}[a]
                            +i*.001+rot*.01)
                    rec={"model":model,"case_id":cid,"article_sha256":pub[cid],
                         "arm":a,"rotation":rot,
                         "context_article_sha256":(pub[cid] if a=="supported" else
                                "WITHHELD" if a=="withheld" else pub[ids[(i+1)%40]]),
                         "checkpoint_sha256":sha("synthetic-model-"+model),
                         "tokenizer_sha256":sha("synthetic-tokenizer-"+model),
                         "option_order_sha256":sha("synthetic-options-"+cid+"-"+str(rot)),
                         "prompt_sha256":sha("synthetic-prompt-"+model+cid+a+str(rot)),
                         "gold_label":"A" if rot==0 else "C",
                         "foil_label":"B" if rot==0 else "D",
                         "gold_token_count":1,"foil_token_count":1,
                         "gold_logp":-4+margin,"foil_logp":-4.,"mode":"raw"}
                    rows.append(rec)
    return rows,approved,manifest

class EXP053MetricsContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.rows,cls.approved,cls.manifest=synthetic()
    def test_complete_synthetic_grid_and_mathematics(self):
        z=m.analyze(self.rows,self.approved,self.manifest)
        self.assertEqual(z["n_approved_articles"],40)
        self.assertEqual(z["expected_and_observed_score_rows"],720)
        for model in m.MODELS:
            r=z["models"][model]
            self.assertAlmostEqual(r["D"]["mean"],1.1)
            self.assertAlmostEqual(r["N"]["mean"],0.2)
            self.assertAlmostEqual(r["S"]["mean"],.9)
            self.assertEqual(r["S"]["positive"],40)
            self.assertGreater(r["S"]["ci95_article_bootstrap"][0],0)
    def test_bootstrap_and_holm_reproducible(self):
        a=m.analyze(self.rows,self.approved,self.manifest)
        self.assertEqual(a,m.analyze(self.rows,self.approved,self.manifest))
    def test_missing_row_fails_closed(self):
        with self.assertRaisesRegex(ValueError,"HOLD"):
            m.analyze(self.rows[:-1],self.approved,self.manifest)
    def test_duplicate_row_fails_closed(self):
        rows=copy.deepcopy(self.rows);rows[1]=copy.deepcopy(rows[0])
        with self.assertRaisesRegex(ValueError,"HOLD"):
            m.analyze(rows,self.approved,self.manifest)
    def test_wrong_context_decoy_fails_closed(self):
        rows=copy.deepcopy(self.rows)
        decoy=next(r for r in rows if r["arm"]=="decoy")
        decoy["context_article_sha256"]=decoy["article_sha256"]
        with self.assertRaisesRegex(ValueError,"HOLD"):
            m.analyze(rows,self.approved,self.manifest)
    def test_wrong_foil_rotation_fails_closed(self):
        rows=copy.deepcopy(self.rows)
        flip=next(r for r in rows if r["rotation"]==1)
        flip["foil_label"]="B"
        with self.assertRaisesRegex(ValueError,"HOLD"):
            m.analyze(rows,self.approved,self.manifest)
    def test_checkpoint_drift_fails_closed(self):
        rows=copy.deepcopy(self.rows);rows[0]["checkpoint_sha256"]=sha("changed")
        with self.assertRaisesRegex(ValueError,"HOLD"):
            m.analyze(rows,self.approved,self.manifest)
    def test_cross_model_options_must_match(self):
        rows=copy.deepcopy(self.rows);rows[-1]["option_order_sha256"]=sha("different-option-order")
        with self.assertRaisesRegex(ValueError,"HOLD"):
            m.analyze(rows,self.approved,self.manifest)
    def test_token_count_mismatch_fails_closed(self):
        rows=copy.deepcopy(self.rows);rows[0]["foil_token_count"]=2
        with self.assertRaisesRegex(ValueError,"HOLD"):
            m.analyze(rows,self.approved,self.manifest)
    def test_no_unapproved_or_fake_consent(self):
        with self.assertRaisesRegex(ValueError,"HOLD"):
            m.analyze(self.rows,dict(list(self.approved.items())[:39]),self.manifest)
    def test_no_nan_or_positive_logprob(self):
        for bad in (float("nan"),float("inf"),0.1):
            rows=copy.deepcopy(self.rows);rows[0]["foil_logp"]=bad
            with self.assertRaisesRegex(ValueError,"HOLD"):
                m.analyze(rows,self.approved,self.manifest)
    def test_no_private_text_in_receipt(self):
        rows=copy.deepcopy(self.rows);rows[0]["article"]="copyrighted text"
        with self.assertRaisesRegex(ValueError,"HOLD"):
            m.analyze(rows,self.approved,self.manifest)
    def test_measurement_plan_drift_fail_closed(self):
        with patch.object(m,"ANALYSIS_PLAN_SHA","0"*64):
            with self.assertRaisesRegex(ValueError,"HOLD"):
                m.analyze(self.rows,self.approved,self.manifest)
    def test_mode_not_silently_mixed(self):
        rows=copy.deepcopy(self.rows);rows[0]["mode"]="chat"
        with self.assertRaisesRegex(ValueError,"HOLD"):
            m.analyze(rows,self.approved,self.manifest)
if __name__=="__main__":unittest.main()

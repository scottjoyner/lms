"""No model runtime or network required."""
import copy
import unittest
from compare_receipts import compare
from run_flint import PIN, QUANTS


def sample(q, speed):
    return {"host":PIN["host"], "hf_revision":PIN["hf_revision"],
            "runtime_source_sha":PIN["runtime_source_sha"], "profile":"smoke",
            "runtime_argv_labels":{"gpu":PIN["device"],"threads":6,
                                   "n_gpu_layers":99,"flash_attention":"on"},
            "quant":QUANTS[q]["quant"],"model_sha256_pin":QUANTS[q]["sha256"],
            "model_file_expected_bytes":QUANTS[q]["bytes"],
            "status":"MEASURED_ISOLATED","benchmark_attempted":True,
            "model_hash_verified":True,"exit_code":0,"stopped_reason":None,
            "logs_truncated":False,"reasons":[],
            "bench":{"valid_results":1,"observations":[
                {"n_prompt":0,"n_gen":16,"avg_tokens_per_second":speed}]}}


class PairChecks(unittest.TestCase):
    def test_structural_comparison_never_claims_execution(self):
        result = compare(sample("IQ2_XXS",30),sample("IQ3_XXS",45))
        self.assertEqual(result["status"], "LOCAL_RECEIPTS_STRUCTURALLY_COMPARABLE")
        self.assertEqual(result["pairs"][0]["ratio_second_to_first"],1.5)
        self.assertFalse(result["actual_model_execution_attested"])
        self.assertEqual(result["production_admission"],"NO_GO")

    def test_mismatched_revision_denied(self):
        b=sample("IQ3_XXS",42);b["hf_revision"]="f"*40
        self.assertEqual(compare(sample("IQ2_XXS",30),b)["status"],"HOLD")

    def test_malformed_profile_and_runtime_labels_fail_closed(self):
        for profile in ([], {}, 42, True, "private"):
            with self.subTest(profile=repr(profile)):
                first=sample("IQ2_XXS",30)
                second=sample("IQ3_XXS",42)
                first["profile"]=profile
                second["profile"]=profile
                self.assertEqual(compare(first, second)["status"],"HOLD")
        for key, value in (("threads", 99), ("flash_attention", "off"),
                           ("secret", "unexpected")):
            first=sample("IQ2_XXS",30);second=sample("IQ3_XXS",42)
            first["runtime_argv_labels"][key]=value
            second["runtime_argv_labels"][key]=value
            self.assertEqual(compare(first,second)["status"],"HOLD")

    def test_bool_metric_count_is_not_integer_evidence(self):
        b=sample("IQ3_XXS",42)
        b["bench"]["valid_results"]=True
        self.assertEqual(compare(sample("IQ2_XXS",30),b)["status"],"HOLD")

    def test_extreme_finite_ratio_is_invalid(self):
        b=sample("IQ3_XXS",1e-320)
        self.assertEqual(compare(sample("IQ2_XXS",30),b)["status"],"HOLD")

    def test_same_quant_not_comparable(self):
        self.assertEqual(compare(sample("IQ3_XXS",30),
                                 sample("IQ3_XXS",42))["status"],"HOLD")

    def test_false_weight_hash_claim_denied(self):
        b=sample("IQ3_XXS",42);b["model_hash_verified"]=False
        self.assertEqual(compare(sample("IQ2_XXS",30),b)["status"],"HOLD")

    def test_partial_or_failed_results_denied(self):
        for change in ({"exit_code":1},{"status":"BLOCKED_OR_FAILED"},
                       {"logs_truncated":True},{"reasons":["LOW_RAM"]}):
            with self.subTest(change=change):
                b=sample("IQ3_XXS",42);b.update(change)
                self.assertEqual(compare(sample("IQ2_XXS",30),b)["status"],"HOLD")

    def test_unpaired_workload_denied(self):
        b=sample("IQ3_XXS",42);b["bench"]["observations"][0]["n_prompt"]=128
        self.assertEqual(compare(sample("IQ2_XXS",30),b)["reason"],"UNPAIRED_WORKLOADS")

    def test_nonfinite_and_duplicate_metrics_denied(self):
        for val in (float("nan"),float("inf"),0,-1,"private"):
            b=sample("IQ3_XXS",val)
            self.assertEqual(compare(sample("IQ2_XXS",30),b)["status"],"HOLD")
        b=sample("IQ3_XXS",42)
        b["bench"]["observations"] *= 2
        b["bench"]["valid_results"]=2
        self.assertEqual(compare(sample("IQ2_XXS",30),b)["status"],"HOLD")


if __name__ == "__main__":
    unittest.main()

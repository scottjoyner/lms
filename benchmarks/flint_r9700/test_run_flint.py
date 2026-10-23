"""Network-free and GPU-free contract checks for Flint R9700 benchmark."""
import hashlib
import importlib.util
import json
from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch

MODULE = Path(__file__).resolve().parent / "run_flint.py"
spec = importlib.util.spec_from_file_location("flint_run", MODULE)
flint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flint)


class FlintPhysicalBenchTests(unittest.TestCase):
    def test_pinned_upstream_identity_and_revision(self):
        self.assertEqual(flint.PIN["hf_revision"], "20468edfc38ab43d6914cc88b08cbecb05d0fc62")
        self.assertEqual(flint.QUANTS["IQ3_XXS"]["sha256"],
                         "7a0a095dde1d931063a5d571e19497d997edbc783eb1ed6b86529bfab9d94804")
        self.assertEqual(flint.QUANTS["IQ3_XXS"]["bytes"], 14535427520)
        self.assertEqual(flint.PIN["device"], "Vulkan1")
        self.assertEqual(flint.PIN["device_id"], "0x7551")

    def test_iq2_pin_exact_and_isolated(self):
        candidate=flint.QUANTS["IQ2_XXS"]
        self.assertEqual(candidate["bytes"], 11855541696)
        self.assertEqual(candidate["sha256"],
                         "c1fa4a6d5ec67fbeff460c68829984e4aa3eea95dc9d2067691f19e5406deeee")
        self.assertEqual(flint.QUANTS["IQ3_XXS"]["bytes"], 14535427520)

    def test_jsonl_metrics_require_actual_numeric_timings(self):
        raw = ('{"n_prompt":0,"n_gen":16,"avg_ts":34.523,"gpu_info":"R9700"}\n'
               '{"n_prompt":128,"n_gen":0,"avg_ts":91.232}\n'
               '{"n_prompt":0,"n_gen":16,"avg_ts":"unmeasured"}\n'
               '{"n_prompt":0,"n_gen":16,"avg_ts":0.0}\n')
        d = flint.parse_bench_jsonl(raw)
        self.assertEqual(d["valid_results"], 2)
        self.assertEqual(d["observations"][0]["avg_tokens_per_second"], 34.523)
        self.assertEqual(d["observations"][1]["n_prompt"], 128)
        self.assertEqual(d["observations"][1]["n_gen"], 0)

    def test_noise_without_metrics_cannot_be_reported_as_tokens(self):
        data=flint.parse_bench_jsonl('warning x\n{}\n{"n_gen":64,"avg_ts":900}\n')
        self.assertEqual(data["valid_results"], 0)
        self.assertEqual(data["observations"], [])

    def test_file_sha_verification_catches_corruption_and_symlink(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/"mock.gguf"
            f.write_bytes(b"known local weight bytes")
            digest=hashlib.sha256(f.read_bytes()).hexdigest()
            self.assertEqual(flint.hash_pinned(f,len(f.read_bytes()),digest),digest)
            with self.assertRaisesRegex(ValueError,"SHA256"):
                flint.hash_pinned(f,len(f.read_bytes()),"0"*64)
            alias=Path(d)/"alias"
            alias.symlink_to(f)
            with self.assertRaisesRegex(ValueError,"unsafe"):
                flint.hash_pinned(alias,len(f.read_bytes()),digest)

    def test_private_output_is_create_only(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"receipt.json"
            flint.write_create_only(path,'{"production_admission":"NO_GO"}')
            self.assertEqual(path.stat().st_mode & 0o777,0o600)
            self.assertEqual(json.loads(path.read_text())["production_admission"],"NO_GO")
            with self.assertRaises(FileExistsError):
                flint.write_create_only(path,'{"production_admission":"GO"}')

    def test_cannot_overwrite_symlink(self):
        with tempfile.TemporaryDirectory() as d:
            real=Path(d)/"victim"
            real.write_text("SAFE")
            alias=Path(d)/"alias"
            alias.symlink_to(real)
            with self.assertRaises((FileExistsError,OSError)):
                flint.write_create_only(alias,"bad")
            self.assertEqual(real.read_text(),"SAFE")

    def test_default_cli_never_loads_model(self):
        from contextlib import redirect_stdout
        from io import StringIO
        with patch.object(flint,"preflight",return_value={
            "status":"HOLD","reasons":["NO_MODEL"],"production_admission":"NO_GO"}) as p:
            with patch.object(flint,"measure",side_effect=AssertionError("benchmark forbidden")):
                out=StringIO()
                with redirect_stdout(out):
                    self.assertEqual(flint.main([]),2)
            p.assert_called_once()
            self.assertEqual(json.loads(out.getvalue())["status"],"HOLD")

    def test_explicit_opt_in_required_before_run(self):
        from contextlib import redirect_stdout
        from io import StringIO
        with patch.dict(os.environ,{"ALLOW_FLINT_R9700_BENCH":""}):
            with patch.object(flint,"measure",side_effect=AssertionError("benchmark forbidden")):
                out=StringIO()
                with redirect_stdout(out):
                    self.assertEqual(flint.main(["--run"]),2)
                self.assertEqual(json.loads(out.getvalue())["reason"],
                                 "EXPLICIT_BENCH_OPT_IN_MISSING")

    def test_hold_refuses_subprocess_when_no_runtime_or_model(self):
        mock = {"schema":flint.PIN["schema"],"reasons":["PINNED_MODEL_MISSING_OR_INCOMPLETE"],
                "status":"HOLD","production_admission":"NO_GO"}
        with patch.object(flint,"preflight",return_value=mock):
            with patch.object(flint.subprocess,"Popen",
                              side_effect=AssertionError("should never launch")):
                d=flint.measure(Path("/missing"),Path("/missing"),"smoke",Path("/missing.json"))
        self.assertEqual(d["status"],"HOLD")
        self.assertFalse(d["benchmark_attempted"])
        self.assertFalse(d["model_hash_verified"])

    def test_invalid_vram_device_is_refused(self):
        with patch.object(flint.Path,"read_text",side_effect=ValueError("not qualified")):
            with self.assertRaises(ValueError):
                flint.gpu_snapshot()

    def test_nonexistent_model_or_binary_denied_in_preflight(self):
        with patch.object(flint,"memory_snapshot",return_value={
            "available_bytes":7*flint.GiB,"full_psi_10_percent":0.0}):
            with patch.object(flint,"gpu_snapshot",return_value={
                "free_bytes":30*flint.GiB}):
                d=flint.preflight(flint.STAGE/"artifacts"/flint.PIN["filename"],
                                  flint.STAGE/"bin"/"llama-bench")
        self.assertEqual(d["production_admission"],"NO_GO")
        self.assertIn(d["status"],("HOLD","READY_TO_VALIDATE_HASH"))
        # In an ordinary dev checkout without model file it must hold.
        if not (flint.STAGE/"artifacts"/flint.PIN["filename"]).exists():
            self.assertIn("PINNED_MODEL_MISSING_OR_INCOMPLETE",d["reasons"])

    def test_every_benchmark_profile_is_bounded(self):
        for name,p in flint.PROFILES.items():
            with self.subTest(name=name):
                self.assertGreater(p["r"],0)
                self.assertLessEqual(p["r"],3)
                self.assertLessEqual(p["max_sec"],300)

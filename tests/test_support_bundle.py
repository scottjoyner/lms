import json
import zipfile

from lms_support_bundle import create_bundle, redact_text


def test_redact_text_hides_secrets():
    text = "api_key=sk-abcdefghijklmnopqrstuvwxyz123456 and password=supersecretvalue123"
    redacted, count = redact_text(text)
    assert count >= 1
    assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in redacted
    assert "[REDACTED" in redacted


def test_bundle_excludes_raw_outputs_by_default(tmp_path):
    run = tmp_path / "run1"
    run.mkdir()
    (run / "run_summary.csv").write_text("model_key,tps\nqwen,10\n")
    outputs = run / "sidecars" / "run_123" / "outputs"
    outputs.mkdir(parents=True)
    (outputs / "raw.txt").write_text("raw model output secret=abcdef1234567890")
    manifest = create_bundle(run)
    bundle = run / "lms_support_bundle_run1.zip"
    assert bundle.exists()
    with zipfile.ZipFile(bundle) as zf:
        names = zf.namelist()
        assert "run_summary.csv" in names
        assert "sidecars/run_123/outputs/raw.txt" not in names
        bundle_manifest = json.loads(zf.read("SUPPORT_BUNDLE_MANIFEST.json"))
        assert bundle_manifest["include_raw_outputs"] is False


def test_bundle_can_include_raw_outputs_with_redaction(tmp_path):
    run = tmp_path / "run2"
    run.mkdir()
    outputs = run / "sidecars" / "run_123" / "outputs"
    outputs.mkdir(parents=True)
    (outputs / "raw.txt").write_text("token=abcdefghijklmnop1234567890")
    manifest = create_bundle(run, include_raw_outputs=True)
    bundle = run / "lms_support_bundle_run2.zip"
    with zipfile.ZipFile(bundle) as zf:
        names = zf.namelist()
        assert "sidecars/run_123/outputs/raw.txt" in names
        content = zf.read("sidecars/run_123/outputs/raw.txt").decode()
        assert "abcdefghijklmnop1234567890" not in content
        assert "REDACTED" in content

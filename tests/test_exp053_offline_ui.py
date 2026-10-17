"""Offline browser UI contract; cannot verify independent human review."""
import hashlib,json,re,shutil,subprocess,tempfile,unittest
from pathlib import Path
UI=Path(__file__).resolve().parents[1]/"research/exp053_offline_review.html"
class OfflineReviewUI(unittest.TestCase):
    def test_no_network_or_source_payload_embedded(self):
        s=UI.read_text()
        for forbidden in ("fetch(","XMLHttpRequest","WebSocket","localStorage","sessionStorage",
                          "sealed_exam_key","<script src=","<link rel="):
            self.assertNotIn(forbidden,s)
        self.assertIn("connect-src 'none'",s)
        self.assertIn('reviewer_A.json',s)
        self.assertIn("no network",s.lower())
    @unittest.skipUnless(shutil.which("node"),"Node unavailable: browser-JS parity optional")
    def test_js_syntax_and_canonical_digest(self):
        h=UI.read_text();js=h.split("<script>")[1].split("</script>")[0]
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);f=d/"inline.js";f.write_text(js)
            subprocess.run(["node","--check",str(f)],check=True,capture_output=True)
            fixture={"schema":"test","cases":[{"case_id":"a","options":{"A":"uno","B":"two"},"question":"¿Cuál es?","article":"a research sentence"}]}
            pyhash=hashlib.sha256(json.dumps(fixture,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
            harness=(f.read_text()+"""
digest(TEST).then(x=>{console.log(x);if(x!==EXPECTED)process.exitCode=7;});
""")
            # Mock DOM only; no real reviewer packet or licensed text.
            setup="""const vm=require('vm'),fs=require('fs'),{webcrypto}=require('crypto');
const ctx={document:{getElementById:()=>({})},window:{addEventListener:()=>{}},
crypto:webcrypto,TextEncoder,console,TEST:JSON.parse(process.argv[3]),EXPECTED:process.argv[4]};
vm.createContext(ctx);vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),ctx);
"""
            (d/"runner.js").write_text(setup)
            (d/"inline.js").write_text(harness)
            result=subprocess.run(["node",str(d/"runner.js"),str(d/"inline.js"),json.dumps(fixture,ensure_ascii=False),pyhash],
                                   check=False,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn(pyhash,result.stdout)
if __name__=="__main__":unittest.main()

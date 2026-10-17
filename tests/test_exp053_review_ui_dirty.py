"""Offline browser interaction contract with entirely synthetic text."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

UI=Path(__file__).resolve().parents[1]/"research/exp053_offline_review.html"

@unittest.skipUnless(shutil.which("node"),"Node runtime not installed")
class BrowserDirtyDraft(unittest.TestCase):
    def test_typing_is_in_memory_saved_and_warns_on_unload(self):
        source=UI.read_text().split("<script>")[1].split("</script>")[0]
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            (p/"page.js").write_text(source)
            mock=r"""
const vm=require('vm'), fs=require('fs'), {webcrypto}=require('crypto');
const handlers={},signals={};
const article='The independent researcher discovered a cobalt plaque by the river.';
const ids={
  quote:{value:'The independent researcher discovered a cobalt plaque'},
  foilReason:{value:'The other alternative is not in the evidence.'},
  rejectReason:{value:''},
  reviewerId:{value:'enrolled-human-A'},
  attest:{checked:true},
  status:{textContent:''},
  validation:{textContent:''},
  current:{addEventListener:(name,cb)=>{handlers[name]=cb}}
};
const document={
 getElementById:id=>ids[id]||{},
 querySelector:s=>s.includes('answer')?{value:'A'}:
                    s.includes('foil')?{value:'B'}:
                    s.includes('decision')?{value:'accept'}:null
};
const window={addEventListener:(name,cb)=>{signals[name]=cb}};
const ctx={document,window,TextEncoder,crypto:webcrypto,console};
vm.createContext(ctx);vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),ctx);
vm.runInContext("packet={cases:[{article:"+JSON.stringify(article)+"}]};"+
                "review={reviewer_id:'',attestation:'',signature:'',"+
                "decisions:[{decision:'unreviewed',supported_option:'',foil_option:'',"+
                "support_quote:'',foil_reason:'',reject_reason:''}]};",ctx);
handlers.input();
const outcome=vm.runInContext(
"({dirty,signature:review.signature,supported:review.decisions[0].supported_option,"+
"quote:review.decisions[0].support_quote,complete:review.decisions[0].decision,"+
"attestation:review.attestation})",ctx);
if(outcome.dirty!==true || outcome.signature!=='' ||
   outcome.supported!=='A' || outcome.complete!=='accept' ||
   !outcome.quote.includes('cobalt') || !outcome.attestation.includes('independently'))
    throw Error('live edit was not stored in memory and marked unsaved');
if(!ids.status.textContent.includes('1/1 complete'))throw Error('no live progress');
let prevented=false;
signals.beforeunload({preventDefault:()=>{prevented=true},returnValue:null});
if(!prevented)throw Error('unload warning did not protect unsaved edits');
if(typeof handlers.change!=='function')throw Error('radio-change event not registered');
console.log('SYNTHETIC_REVIEW_UI_DIRTY_WARNING_OK');
"""
            (p/"test.js").write_text(mock)
            run=subprocess.run(["node",str(p/"test.js"),str(p/"page.js")],
                               capture_output=True,text=True,check=False)
            self.assertEqual(run.returncode,0,run.stderr)
            self.assertIn("SYNTHETIC_REVIEW_UI_DIRTY_WARNING_OK",run.stdout)

if __name__=="__main__":unittest.main()

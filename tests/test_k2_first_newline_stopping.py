"""EXP042 new-line stop implementation and frozen-source checks, no model load."""
import pathlib
import sys
import unittest
import torch
P=pathlib.Path(__file__).resolve().parents[1]/"research"
sys.path.insert(0,str(P))
import k2_first_newline_stopping as m
import k2_novel_source_fixture as fixture

class TinyTokenizer:
    def decode(self, ids,skip_special_tokens=True):
        return "".join({3:"Nova",4:"\n",5:"more",6:" text"}.get(int(x),"") for x in ids)

class TestNewlineStopping(unittest.TestCase):
    def test_stops_only_on_generated_newline(self):
        tok=TinyTokenizer()
        stop=m.StopAtGeneratedNewline(tok,2)
        self.assertFalse(stop(torch.tensor([[99,98,3]]),None))
        self.assertFalse(stop.hit)
        self.assertTrue(stop(torch.tensor([[99,98,3,4]]),None))
        self.assertTrue(stop.hit)
    def test_prompt_newline_does_not_stop(self):
        tok=TinyTokenizer()
        stop=m.StopAtGeneratedNewline(tok,3)
        self.assertFalse(stop(torch.tensor([[4,3,4]]),None))
        self.assertTrue(stop(torch.tensor([[4,3,4,4]]),None))
    def test_wrong_shape_denied(self):
        stop=m.StopAtGeneratedNewline(TinyTokenizer(),5)
        with self.assertRaises(RuntimeError):
            stop(torch.tensor([[1,2,3]]),None)
    def test_source_and_protocol_checksum(self):
        self.assertEqual(m.PROTOCOL_042,
            "ac9e19750cfe660bf2933715160985cf6f3c70c52fa3cb991617f2d95a930ee9")
        self.assertEqual(fixture.corpus_receipt(fixture.build()),m.FIXTURE_SHA)
        self.assertEqual(m.B_PROTOCOL_SHA,
            "634892caa5fdc3870b0001ed4ebf9b8db26c26eeb446a8ccdd2368d0b4b505f4")
        self.assertEqual(len(m.frozen_sources()),24)
    def test_strict_normalizer_unchanged(self):
        case=next(r for r in fixture.build() if r["kind"]=="answerable")
        self.assertEqual(fixture.score(case,case["gold"])["strict"],1)
        self.assertEqual(fixture.score(case,case["gold"]+"\\nmore")["strict"],0)
    def test_aggregate_report_fields(self):
        rows=fixture.build()
        scored=[{**fixture.score(x,x["gold"]),"newline_stop":1,"native_eos":0} for x in rows]
        d=fixture.score_group(rows,scored)
        for kind in fixture.CASES:
            d[kind]["newline_stop"]=d[kind]["n"]
            d[kind]["native_eos"]=0
        out=m.render(rows,{"default":d,"yarn":d},[91]*48,5.2)
        self.assertIn("EXP-042",out)
        self.assertIn("First newline",out)
        self.assertIn(m.PROTOCOL_042,out)
if __name__=="__main__":unittest.main()

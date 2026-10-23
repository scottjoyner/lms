#!/usr/bin/env python3
"""Freeze all three wrong answer labels without inspecting target model scores.

Private, machine-answer-key exploratory foil labels; NOT human-reviewed semantic
plausibility. No model loading, neural forwards, or licensed passage printing.
"""
import argparse
import hashlib
from pathlib import Path

from exp053_agent_only import machine_receipt
from exp053_tokenizer_preflight import PRIVATE
from exp053_agent_k2_cpu import RECEIPTS, private_dir, create_immutable

LABELS = ("A","B","C","D")
SCHEMA = "exp054-all-three-machine-wrong-exam-options-v1"


def hold(message):
    raise ValueError("HOLD EXP054 ALL FOILS: "+message)


def build_manifest(source):
    if not isinstance(source,dict) or source.get("human_approval") is not False:
        hold("machine-only private admission required")
    accepted=source.get("accepted")
    if not isinstance(accepted,dict) or len(accepted)!=64:
        hold("all 64 source-disjoint cases required")
    packet_sha=source.get("packet_sha256")
    if not isinstance(packet_sha,str) or len(packet_sha)!=64:
        hold("invalid source packet digest")
    rows=[]
    for opaque_id,pins in sorted(accepted.items()):
        if not isinstance(opaque_id,str) or not opaque_id:
            hold("invalid opaque case ID")
        if not isinstance(pins,dict):
            hold("missing frozen case admission")
        gold=pins.get("supported_exam_key")
        original=pins.get("other_exam_option_proxy")
        if gold not in LABELS or original not in LABELS or gold==original:
            hold("invalid sealed exam key or original proxy")
        # Enumerate ALL wrong options in the fixed canonical label order.
        # No target model logits, scores or retrospective foil selection.
        foils=[label for label in LABELS if label != gold]
        if len(foils)!=3 or original not in foils:
            hold("wrong-option completeness")
        rows.append({"case_id":opaque_id,
                     "gold_label_private":gold,
                     "all_three_machine_wrong_labels":foils,
                     "exp053_original_machine_foil":original})
    return {
      "schema":SCHEMA,"human_adjudicated":False,
      "semantic_plausibility_reviewed":False,
      "based_on_target_model_scores":False,
      "new_model_forwards":0,"exact_case_count":64,
      "source_packet_sha256":packet_sha,
      "independent_machine_label_alternatives_per_case":3,
      "candidate_order":"A,B,C,D minus sealed correct label; do not sort by target logits",
      "cases":rows,
      "limitations":"Three incorrect EXAM options, not human-vetted plausible semantic foils. No source-length control, contamination exclusion, generation or abstention inference."
    }


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write-private",action="store_true")
    a=ap.parse_args()
    admission=machine_receipt(PRIVATE.parent)
    result=build_manifest(admission)
    result["preflight_code_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if a.write_private:
        private_dir(RECEIPTS)
        digest=create_immutable(RECEIPTS,"exp054-all-three-machine-foils-v1.json",result)
        print("EXP054_MACHINE_FOILS_FROZEN_PRIVATE sha256="+digest,flush=True)
    print("EXP054_MACHINE_FOILS_PREFLIGHT_PASS articles=64 wrong_choices=192 no_model_forwards=1 human_adjudicated=0")


if __name__ == "__main__":
    main()

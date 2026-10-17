#!/usr/bin/env python3
"""Human-operated Ed25519 sign/export helper; never creates a review or private key."""
import argparse,base64,json,os
from pathlib import Path
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from exp053_race_review import canonical,decisions_for,deny
ATTEST=("I independently reviewed these source passages and responses; "
        "no model predictions or other reviewer labels were provided to me.")
def get_key(path):
    if path.stat().st_mode & 0o077:
        deny("signing key must be private to its owner (chmod 600)")
    k=serialization.load_pem_private_key(path.read_bytes(),password=None)
    if not isinstance(k,Ed25519PrivateKey):
        deny("Ed25519 PEM private key required")
    return k
def public_b64(key):
    return base64.b64encode(key.public_key().public_bytes(
       encoding=serialization.Encoding.Raw,
       format=serialization.PublicFormat.Raw)).decode()
def sign(input_path,packet_path,key_path,out):
    if input_path.resolve()==out.resolve() or out.exists():
        deny("signed output must be a new file")
    review=json.loads(input_path.read_text())
    packet=json.loads(packet_path.read_text())
    if not review.get("reviewer_id"):
        deny("reviewer_id must be set by the human reviewer")
    if review.get("attestation")!=ATTEST:
        deny("reviewer must personally attest prior to signing")
    scope=None
    if review.get("role")=="adjudicator":
        scope={d.get("case_id") for d in review.get("decisions",[])}
        if not scope:deny("empty adjudicator review")
    decisions_for(review,packet,required_ids=scope)
    key=get_key(key_path)
    unsigned={k:v for k,v in review.items() if k!="signature"}
    unsigned["signature"]=key.sign(canonical(unsigned)).hex()
    with out.open("xb") as f:
        os.fchmod(f.fileno(),0o600)
        f.write(canonical(unsigned)+b"\n")
    print("SIGNED_REVIEW",unsigned["reviewer_id"],"PUBLIC_KEY_B64",public_b64(key))
def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="action",required=True)
    s=sub.add_parser("sign")
    s.add_argument("--review",type=Path,required=True)
    s.add_argument("--packet",type=Path,required=True)
    s.add_argument("--private-key",type=Path,required=True)
    s.add_argument("--out",type=Path,required=True)
    k=sub.add_parser("public-key")
    k.add_argument("--private-key",type=Path,required=True)
    x=p.parse_args()
    if x.action=="public-key":print(public_b64(get_key(x.private_key)))
    else:sign(x.review,x.packet,x.private_key,x.out)
if __name__=="__main__":main()

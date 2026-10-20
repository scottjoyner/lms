# EXP-053 — native PyArrow source-fd smoke test (synthetic only)

**2026-10-19. No licensed RACE content used. NO_GO_RESEARCH_ONLY.**

The main research host x1-370, Lenovo and Destroyer report no installed
`pyarrow` package. A read-only, noninteractive SSH package check reported
`pyarrow` on xwing. No credentials, filesystem source files or private human
forms were read for this discovery.

A short-lived invented Parquet file with **2,500 fabricated rows** and
snapshot-like symlink was created under an auto-cleaned temporary directory
on xwing. PyArrow **25.0.0** successfully accepted an open Python binary file
object after streaming SHA-256 and `source.seek(0)`, and decoded the 2,500
invented records. However, a run using the default threaded
`pq.read_table(source)` **exited abnormally 255** after output with
`terminate called without an active exception`. Do NOT report that as
production-validated or ignore the process exit status.

The same source-fd pattern was retested with
`pq.read_table(source, use_threads=False)`. An initial serial run exited
normally (0) with stable fstat and SHA. Three further isolated native Python
processes tested the **exact code idiom**
`pq.read_table(source,use_threads=False).to_pylist()` and each returned 2,500
synthetic rows and exited **0**; fd inode, mtime and snapshot pointer were
stable. The candidate source now explicitly passes `use_threads=False` and
its fake-PyArrow security regression enforces it.

The test did not use RACE text or a real scientific packet, and did not
install PyArrow on x1-370 or change services. xwing's synthetic success is a
useful API compatibility witness, not verification that every future native
source build will work, nor external independent custody or scientific
admission. Preserve all owner/weight approval pins unset and require actual
qualified external technical review, human annotations and an independently
administered observer before any licensed packet rebuild or model forward.

**Final decision: NO_GO_RESEARCH_ONLY. No merge or deployment.**

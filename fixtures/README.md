# Fixtures

`passing/` is a synthetic study. The corpora are 16-byte strings. The checkpoints are hashes of 4096-byte buffers that are not stored. Its G0 block uses vocabulary size 64,000 so the written cap passes. Regenerating the directory with the same code produces the same bytes:

`g0-recorded/g0.json` is recorded metadata from the feasibility build, not corpus bytes and not a second training store. Its vocabulary size is 65,536, which G0 flags. See that directory's README.

```bash
python -m sheaf_verify.synthetic --out /tmp/passing
```

`test_seal.key` is the public test vector `sheaf-verify-public-test-vector`. It is not a secret and it is not a production seal key.

Do not put a real corpus, a weight file, or a private key in this directory.

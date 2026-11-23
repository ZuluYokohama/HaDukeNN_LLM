# Fixtures

`passing/` is a synthetic study. The corpora are 16-byte strings. The checkpoints are hashes of 4096-byte buffers that are not stored. Regenerating the directory with the same code produces the same bytes:

```bash
python -m sheaf_verify.synthetic --out /tmp/passing
```

`test_seal.key` is the public test vector `sheaf-verify-public-test-vector`. It is not a secret and it is not a production seal key.

Do not put a real corpus, a weight file, or a private key in this directory.

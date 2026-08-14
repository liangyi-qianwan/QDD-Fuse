# Checkpoints

The GitHub source tree intentionally does not commit the large `.pt` checkpoint files. GitHub's normal source view will therefore show this placeholder file and the lightweight text-encoder metadata only.

The final `best.pt` files are inside the full release archive. After reconstructing and extracting the release assets, the expected paths are:

```text
checkpoints/mosi/best.pt
checkpoints/mosei/best.pt
checkpoints/simsv2/best.pt
```

SHA256 checksums are recorded in `results/checkpoint_sha256.txt`.

Expected checkpoint checksums:

```text
ce8599450e59bc028f06f5f09e2f19ff6e066d3fa08f22e54c13473c3f05a180  checkpoints/mosi/best.pt
9af4000fa3cfc346308d48f3dcc8d4460d91e11fa4605f25da257b8f9321e496  checkpoints/mosei/best.pt
fc84ec6e4ca4bd850eb59e46414f9ebd87f7f469913175c848d83ee2aff7f05d  checkpoints/simsv2/best.pt
```

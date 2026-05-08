# TabForge Accuracy Benchmarks

This folder is for reference-corpus manifests. Convert known-good `.gp5` files
to drafts with:

```bash
python tools/gp5_to_draft.py reference.gp5 -o benchmarks/references/song.draft.json
```

After generating a candidate draft, add both files to a manifest and run:

```bash
python tools/benchmark_accuracy.py --manifest benchmarks/manifest.example.json --min-f1 0.9 --min-strict-accuracy 0.9
```

The benchmark fails when pitch/timing F1 or strict accuracy drops below the
configured threshold. Strict accuracy includes duration, fretboard position,
technique, tempo, meter, tuning, and capo checks where those fields are present.

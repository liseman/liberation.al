# Speech benchmark clips

The audio files here (`01.ogg` to `14.ogg`) are synthetic speech for `tools/asr_bench.py`. No person recorded them
and no real voice is in them: each is a made-up sentence from `clips.json`, read by one of the stock voices of the
Kokoro text-to-speech model (af_heart, af_bella, am_adam, am_michael, bf_emma, and bm_george), then resampled to
16 kHz mono Ogg Opus. `uv run tools/asr_bench.py synth` regenerates them.

- Model and voices: [Kokoro-82M v1.0](https://huggingface.co/hexgrad/Kokoro-82M), Apache License 2.0.
- ONNX export and runtime: [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx), MIT License.
- Text to phonemes: espeak-ng (GPL-3.0), a tool used at synthesis time; none of its code or data is in these files.

The clips, their reference transcripts, and the recorded engine transcripts in `clips.json` are released under the
Apache License 2.0, the same as the rest of this repository (see `LICENSE` at its root).

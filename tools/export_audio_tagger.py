"""Export EfficientAT mn10_as (an AudioSet tagger, MIT) to ONNX for the TV tagger (listen/media_tag.py, #2468):
32 kHz mono waveform in ("waveform", float32 [batch, samples], any length), 527 AudioSet probabilities out ("probs",
already sigmoided), with the labels in output order beside it.

It runs EfficientAT's own Python and loads its pickled checkpoint, so it is run by hand, on purpose, by someone who
has decided to trust that code (Joe approved it for devbox, 2026-11-06). The source and checkpoint are pinned:

  code        https://github.com/fschmid56/EfficientAT at commit a425fdce92572e602a1d5634799bd9f1f2efa806 (MIT)
  checkpoint  https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn10_as_mAP_471.pt
              sha256 0bd7dc2443af498c289a2e739f02ebb515d6aa3fd3ab9db539c86123ae368a4e (checked before it is loaded)

Run from the EfficientAT checkout (its loader reads resources/ relative to the working directory), in a venv with
torch, torchaudio, and onnxruntime, with the checkpoint in resources/:

    git clone https://github.com/fschmid56/EfficientAT && cd EfficientAT && git checkout a425fdce
    mkdir -p resources && curl -L -o resources/mn10_as_mAP_471.pt <checkpoint URL above>
    python /path/to/liberational/tools/export_audio_tagger.py data/models/efficientat

and copy the upstream LICENSE beside the output. Joe's build on devbox gave:
  efficientat_mn10_as.onnx    eed1f591b2c11e3199d31fc96f6a6f76550c4d9377c8fc83ffc1794b178f0f7a
  efficientat_labels.json     5997fb8a7be942d4442e7cdc8526a2eab46221e442652fad02b76dd0e6adee7f

The mel frontend is in the graph, in upstream's eval settings: 128 mels, an 800 window, a 320 hop, n_fft 1024, the
pre-emphasis, and ((mel + 1e-5).log() + 4.5) / 5, with the STFT as fixed DFT convolutions so it exports to plain ONNX
ops. It prints how closely its spectrogram matches upstream's and the ONNX matches torch.

The model wants several seconds of context: on clips under about 3 s it gives near-certain nonsense, upstream torch
included, so listen/media_tag.py repeats a short clip to 5 s.
"""

import hashlib
import importlib
import json
import pathlib
import sys

CHECKPOINT = pathlib.Path("resources/mn10_as_mAP_471.pt")
CHECKPOINT_SHA256 = "0bd7dc2443af498c289a2e739f02ebb515d6aa3fd3ab9db539c86123ae368a4e"
SR, WIN, HOP, N_MELS, N_FFT = 32000, 800, 320, 128, 1024


def main(out: pathlib.Path) -> None:
    got = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    if got != CHECKPOINT_SHA256:
        raise SystemExit(f"{CHECKPOINT} has sha256 {got}, not the pinned {CHECKPOINT_SHA256}")
    import numpy as np
    import onnxruntime as ort
    import torch
    from torch import nn

    sys.path.insert(0, str(pathlib.Path.cwd()))  # the EfficientAT checkout's own modules
    labels = importlib.import_module("helpers.utils").labels
    get_model = importlib.import_module("models.mn.model").get_model
    AugmentMelSTFT = importlib.import_module("models.preprocess").AugmentMelSTFT

    class Tagger(nn.Module):
        """Upstream eval preprocessing with the STFT as fixed DFT convolutions, so it exports to plain ONNX ops."""

        pre: torch.Tensor
        mel_basis: torch.Tensor
        cos_k: torch.Tensor
        sin_k: torch.Tensor

        def __init__(self) -> None:
            super().__init__()
            mel = AugmentMelSTFT(n_mels=N_MELS, sr=SR, win_length=WIN, hopsize=HOP, n_fft=N_FFT).eval()
            self.register_buffer("pre", mel.preemphasis_coefficient.clone())
            import torchaudio

            basis, _ = torchaudio.compliance.kaldi.get_mel_banks(
                N_MELS, N_FFT, SR, mel.fmin, mel.fmax, vtln_low=100.0, vtln_high=-500.0, vtln_warp_factor=1.0
            )
            self.register_buffer("mel_basis", nn.functional.pad(basis, (0, 1)))
            # torch.stft pads the window to n_fft, centred.
            window = torch.hann_window(WIN, periodic=False)
            left = (N_FFT - WIN) // 2
            window = nn.functional.pad(window, (left, N_FFT - WIN - left))
            k = torch.arange(N_FFT // 2 + 1, dtype=torch.float64)[:, None]
            n = torch.arange(N_FFT, dtype=torch.float64)[None, :]
            ang = 2 * torch.pi * k * n / N_FFT
            self.register_buffer("cos_k", (torch.cos(ang) * window).float()[:, None, :])
            self.register_buffer("sin_k", (-torch.sin(ang) * window).float()[:, None, :])
            self.ref_mel = mel
            self.model = get_model(width_mult=1.0, pretrained_name="mn10_as", strides=[2, 2, 2, 2], head_type="mlp")
            self.model.eval()

        def spec(self, wav: torch.Tensor) -> torch.Tensor:
            x = nn.functional.conv1d(wav.unsqueeze(1), self.pre)
            x = nn.functional.pad(x, (N_FFT // 2, N_FFT // 2), mode="reflect")
            re = nn.functional.conv1d(x, self.cos_k, stride=HOP)
            im = nn.functional.conv1d(x, self.sin_k, stride=HOP)
            power = re * re + im * im
            melspec = torch.matmul(self.mel_basis, power)
            return ((melspec + 0.00001).log() + 4.5) / 5.0

        def forward(self, wav: torch.Tensor) -> torch.Tensor:
            logits, _ = self.model(self.spec(wav).unsqueeze(1))
            return torch.sigmoid(logits)

    out.mkdir(parents=True, exist_ok=True)
    tagger = Tagger().eval()
    torch.manual_seed(0)
    wav = torch.randn(1, SR * 5) * 0.1
    with torch.no_grad():
        ref_spec = tagger.ref_mel(wav)
        ours = tagger.spec(wav)
        print("spec max abs diff vs upstream:", float((ref_spec - ours[:, :, : ref_spec.shape[-1]]).abs().max()))
        ref_logits, _ = tagger.model(ref_spec.unsqueeze(1))
        ref = torch.sigmoid(ref_logits)
        mine = tagger(wav)
    path = out / "efficientat_mn10_as.onnx"
    torch.onnx.export(
        tagger,
        (wav,),
        str(path),
        input_names=["waveform"],
        output_names=["probs"],
        dynamic_axes={"waveform": {0: "batch", 1: "samples"}, "probs": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for secs in (1, 5, 10):
        w = (torch.randn(1, SR * secs) * 0.1).numpy()
        got_probs = sess.run(None, {"waveform": w})[0]
        with torch.no_grad():
            expected = tagger(torch.from_numpy(w)).numpy()
        print(f"{secs:>2} s: onnx vs torch max abs diff {np.abs(got_probs - expected).max():.2e}")
    print("upstream-spec vs ours probs max diff:", float((ref - mine).abs().max()))
    (out / "efficientat_labels.json").write_text(json.dumps(labels, indent=0))
    print("wrote", path, path.stat().st_size, "bytes;", len(labels), "labels")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]).resolve())

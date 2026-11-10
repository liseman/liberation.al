import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "archive_codec_benchmark", Path(__file__).resolve().parents[1] / "tools" / "archive_codec_benchmark.py"
)
assert _spec is not None and _spec.loader is not None
benchmark = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(benchmark)
encode_args = benchmark.encode_args
parse_quality = benchmark.parse_quality


def test_quality_summary_uses_overall_ssim_and_luma_xpsnr_not_a_chroma_component():
    log = "SSIM Y:0.98 U:0.999 V:0.999 All:0.984327 (18.049)\nXPSNR average, y:38.42 u:44.81 v:43.12 (minimum: 38.42)"
    assert parse_quality(log, "ssim") == 0.984327
    assert parse_quality(log, "xpsnr") == 38.42
    with pytest.raises(ValueError, match="missing ssim"):
        parse_quality("filter failed", "ssim")


def test_bitrate_scales_from_payload_duration_and_keeps_frame_rate_policy_explicit():
    args = encode_args(
        Path("source.mp4"),
        Path("out.mp4"),
        {"codec": "h264", "bitrate_ratio": 0.5, "fps_ratio": 1 / 3},
        {"fps": 30.0, "payload_bytes": 100_000, "duration_s": 10.0},
        10,
    )
    assert args[args.index("-b:v") + 1] == "40000"
    assert args[args.index("-g") + 1] == "10"
    assert args[args.index("-vf") + 1].endswith("fps=10")
    assert args[args.index("-threads") + 1] == "1"


def test_av1_archive_measurement_refuses_random_access_crf_policy():
    with pytest.raises(ValueError, match="explicit offline archive policy"):
        encode_args(
            Path("source.mp4"),
            Path("out.mp4"),
            {"codec": "av1", "crf": 32},
            {"fps": 30.0, "payload_bytes": 100_000, "duration_s": 10.0},
            10,
        )


def test_offline_archive_uses_long_gop_crf_and_normal_prediction():
    args = encode_args(
        Path("source.mp4"),
        Path("out.mp4"),
        {"codec": "av1", "crf": 36, "archive": True, "preset": 6},
        {"fps": 30.0},
        12,
    )
    assert args[args.index("-g") + 1] == "300"
    assert args[args.index("-crf") + 1] == "36.0"
    params = args[args.index("-svtav1-params") + 1]
    assert "pred-struct=1" not in params
    assert "rc=2" not in params


def test_one_frame_deficit_requires_a_verified_endpoint_not_an_interior_drop():
    source = [0, 0.0332, 0.0666, 0.0999, 0.1332]
    output = [0, 1 / 30, 2 / 30, 3 / 30]
    verify = benchmark.verified_boundary_trim
    assert verify(source, output, 30, 1 / 15360, 4 / 30)
    assert not verify(source, [0, 1 / 30, 3 / 30, 4 / 30], 30, 1 / 15360, 4 / 30)
    assert not verify(source, output, 30, 1 / 15360, 1)
    assert not verify(source, [0, 1 / 30, 1 / 30, 3 / 30], 30, 1 / 15360, 4 / 30)


def test_nvenc_keeps_decode_on_an_explicit_gpu_and_uses_a_fixed_frame_cohort():
    profile = {"codec": "hevc", "encoder": "hevc_nvenc", "gpu_index": 0, "cq": 28, "preset": "p5"}
    args = encode_args(Path("source.mp4"), Path("out.mp4"), profile, {"fps": 30.04}, 12)
    assert args[args.index("-hwaccel_output_format") + 1] == "cuda"
    assert args[args.index("-hwaccel_device") + 1] == "0"
    assert args[args.index("-frames:v") + 1] == "360"
    assert args[args.index("-b:v") + 1] == "0"
    assert args[args.index("-tag:v") + 1] == "hvc1"
    assert "format=yuv420p" not in args[args.index("-vf") + 1]
    del profile["gpu_index"]
    with pytest.raises(ValueError, match="explicit nonnegative GPU"):
        encode_args(Path("source.mp4"), Path("out.mp4"), profile, {"fps": 30.04}, 12)

"""
Tests of actuneo.ai: hardware detection, model selection, setup, and backends.

No test downloads a model or calls a provider. Hardware is given as mock
profiles, Hugging Face and llama.cpp are replaced by stand-ins, and HTTP
calls are intercepted.
"""

import errno
import hashlib
import importlib
import json
import subprocess
import sys
import types

import pytest

from actuneo import ai
from actuneo.ai import backends
from actuneo.ai.catalog import ModelSpec, load_catalog
from actuneo.ai.hardware import GPUInfo, HardwareProfile

setup_mod = importlib.import_module("actuneo.ai.setup")
hardware_module = importlib.import_module("actuneo.ai.hardware")
config_module = importlib.import_module("actuneo.ai.config")


def profile(ram_total, ram_available, gpu=None, disk=200.0, ok=True):
    return HardwareProfile(system="Test", machine="x86_64", cpu_count=8,
                           ram_total_gb=ram_total, ram_available_gb=ram_available,
                           gpu=gpu, disk_free_gb=disk, detection_ok=ok)


def nvidia(name, total, free):
    return GPUInfo(name, "nvidia", total, free)


def apple(name, total, free):
    return GPUInfo(name, "apple", total, free, unified=True)


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Keep settings and models in a temporary folder and start with no keys."""
    monkeypatch.setenv("ACTUNEO_AI_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("ACTUNEO_AI_CATALOG", raising=False)
    for variables in backends.KEY_VARIABLES.values():
        for variable in variables:
            monkeypatch.delenv(variable, raising=False)
    monkeypatch.setitem(sys.modules, "keyring", None)
    setup_mod._SESSION_KEYS.clear()
    setup_mod._BACKEND_CACHE.clear()
    yield tmp_path


class TestCatalog:
    def test_shipped_catalog(self):
        catalog = load_catalog()
        assert 4 <= len(catalog) <= 8
        sizes = [model.size_bytes for model in catalog]
        assert sizes == sorted(sizes)
        assert len({model.key for model in catalog}) == len(catalog)
        for model in catalog:
            assert model.repo.count("/") == 1 and model.filename.endswith(".gguf")
            assert len(model.sha256) == 64 and int(model.sha256, 16) >= 0
            assert model.min_ram_gb > model.size_gb * 0.9
            assert model.min_vram_gb <= model.min_ram_gb
            assert model.license and model.quality and model.layers > 0
        assert catalog[0].size_gb < 1 and catalog[-1].size_gb > 8

    def test_no_non_commercial_models(self):
        permissive = {"Apache-2.0", "MIT"}
        assert {model.license for model in load_catalog()} <= permissive

    def test_catalog_from_a_file(self, tmp_path, monkeypatch):
        entry = {"key": "mine", "name": "Mine", "repo": "me/mine", "filename": "mine.gguf",
                 "size_bytes": 5, "min_ram_gb": 1, "min_vram_gb": 1, "extra_field": "ignored"}
        path = tmp_path / "catalog.json"
        path.write_text(json.dumps({"models": [entry]}))
        monkeypatch.setenv("ACTUNEO_AI_CATALOG", str(path))
        catalog = load_catalog()
        assert [model.key for model in catalog] == ["mine"]
        assert catalog[0].to_dict()["repo"] == "me/mine"

    def test_invalid_catalogs(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps([{"key": "x", "name": "X"}]))
        with pytest.raises(ValueError, match="missing"):
            load_catalog(path)
        entry = {"key": "a", "name": "A", "repo": "r/a", "filename": "a.gguf",
                 "size_bytes": 1, "min_ram_gb": 1, "min_vram_gb": 1}
        path.write_text(json.dumps([entry, entry]))
        with pytest.raises(ValueError, match="unique"):
            load_catalog(path)
        path.write_text("[]")
        with pytest.raises(ValueError, match="no models"):
            load_catalog(path)


class TestSelection:
    """The model chosen for each kind of machine."""

    def choose(self, hardware):
        return ai.select_model(hardware, load_catalog())

    def test_two_gigabyte_machine_is_too_weak(self):
        selection = self.choose(profile(2, 0.9))
        assert selection.too_weak and selection.recommended is None

    def test_four_gigabyte_laptop(self):
        selection = self.choose(profile(4, 2.0))
        assert selection.recommended.model.key == "tiny"
        assert selection.recommended.device == "cpu"
        assert selection.recommended.n_gpu_layers == 0
        assert selection.largest.model.key == "tiny"

    def test_eight_gigabyte_laptop(self):
        selection = self.choose(profile(8, 5.5))
        assert selection.recommended.model.key == "medium"
        assert selection.smallest.model.key == "tiny"

    def test_sixteen_gigabyte_laptop_without_gpu(self):
        selection = self.choose(profile(16, 10.0))
        assert selection.recommended.model.key == "large"
        assert selection.recommended.device == "cpu"
        assert selection.largest.model.key == "large"
        # With more memory free the 14B model is offered as the largest,
        # but the recommendation stays at a size a processor handles well
        roomy = self.choose(profile(16, 13.0))
        assert roomy.recommended.model.key == "large"
        assert roomy.largest.model.key == "xlarge"
        assert not roomy.largest.comfortable

    def test_sixteen_gigabytes_with_a_four_gigabyte_card(self):
        selection = self.choose(profile(16, 10.0, nvidia("NVIDIA RTX 3050", 4.0, 3.8)))
        chosen = selection.recommended
        assert chosen.model.key == "large"
        assert chosen.device == "gpu+cpu"
        assert chosen.n_gpu_layers == int(28 * 3.8 * 0.8 / 5.6)
        assert "of 28 layers" in chosen.runs_on()
        # The model that fits wholly in video memory runs entirely on the card
        small = setup_mod.placement(load_catalog()[1], profile(
            16, 10.0, nvidia("NVIDIA RTX 3050", 4.0, 3.8)))
        assert small.device == "gpu" and small.n_gpu_layers == -1

    def test_twenty_four_gigabyte_gpu(self):
        selection = self.choose(profile(64, 50.0, nvidia("NVIDIA RTX 4090", 24.0, 23.0)))
        assert selection.recommended.model.key == "xlarge"
        assert selection.recommended.device == "gpu"
        assert selection.recommended.n_gpu_layers == -1
        assert selection.largest.model.key == "xxlarge"
        assert selection.largest.device == "gpu+cpu"

    def test_gpu_with_little_free_memory_is_not_overfilled(self):
        selection = self.choose(profile(64, 50.0, nvidia("NVIDIA RTX 4090", 24.0, 2.5)))
        on_gpu = setup_mod.placement(load_catalog()[1], profile(
            64, 50.0, nvidia("NVIDIA RTX 4090", 24.0, 2.5)))
        assert on_gpu.device == "gpu"            # 1.6 GB fits in 2.5 GB less 20%
        assert selection.recommended.model.key == "large"
        assert selection.recommended.device == "gpu+cpu"

    def test_apple_m1_with_eight_gigabytes(self):
        selection = self.choose(profile(8, 4.0, apple("Apple M1", 8.0, 4.0)))
        assert selection.recommended.model.key == "small"
        assert selection.recommended.device == "gpu"
        assert selection.recommended.n_gpu_layers == -1

    def test_apple_silicon_with_twenty_four_gigabytes(self):
        selection = self.choose(profile(24, 12.0, apple("Apple M3", 24.0, 12.0)))
        assert selection.recommended.model.key == "large"
        roomy = self.choose(profile(64, 40.0, apple("Apple M3 Max", 64.0, 40.0)))
        assert roomy.recommended.model.key == "xxlarge"

    def test_headroom_is_respected(self):
        catalog = load_catalog()
        for free in (1.5, 3.0, 6.0, 9.0, 16.0, 40.0):
            selection = ai.select_model(profile(64, free), catalog)
            if selection.recommended:
                assert selection.recommended.model.min_ram_gb <= free * 0.7
                assert selection.largest.model.min_ram_gb <= free * 0.9

    def test_low_disk_space_limits_the_choice(self):
        selection = self.choose(profile(16, 10.0, disk=2.0))
        assert selection.recommended.model.key == "tiny"
        full = self.choose(profile(16, 10.0, disk=1.2))
        assert full.too_weak and full.disk_limited
        text = setup_mod.render_summary(profile(16, 10.0, disk=1.2), full)
        assert "not enough free disk space" in text and "Colab" not in text
        assert not self.choose(profile(2, 0.9)).disk_limited

    def test_failed_detection_falls_back_to_the_smallest_model(self):
        selection = self.choose(profile(None, None, ok=False))
        assert selection.fallback and not selection.too_weak
        assert selection.recommended.model.key == "tiny"
        assert selection.recommended.device == "cpu"

    def test_summary_text(self):
        hardware = profile(16, 13.0, nvidia("NVIDIA RTX 3050", 4.0, 3.8), disk=120.0)
        text = setup_mod.render_summary(hardware, self.choose(hardware), "/models")
        assert "Detected: 16 GB RAM (13.0 GB free)" in text
        assert "NVIDIA RTX 3050 (4 GB VRAM" in text and "120 GB free disk" in text
        assert "Recommended model: Qwen2.5-7B-Instruct (Q4_K_M) - 4.7 GB download" in text
        assert "largest model your machine can run comfortably" in text
        assert "[1] Smallest (0.5 GB, fastest)" in text
        assert "[2] Recommended (4.7 GB)" in text
        assert "[3] Largest your PC can handle (9.0 GB, may be slow)" in text
        assert "[s] Skip and use an API key instead" in text
        assert "Warning: this is a large download (4.7 GB)" in text
        assert "metered connection" in text and "licence: Apache-2.0" in text
        assert "Tip:" not in text
        busy = profile(24, 5.1, apple("Apple M5", 24.0, 5.1))
        text = setup_mod.render_summary(busy, self.choose(busy))
        assert "Tip: only 5.1 of 24 GB of memory is free" in text

    def test_summary_for_small_and_weak_machines(self):
        small = profile(4, 2.0)
        text = setup_mod.render_summary(small, self.choose(small))
        assert "Warning: this is a large download" not in text
        assert "[1] Recommended (0.5 GB)" in text and "[2]" not in text
        weak = profile(2, 0.9)
        text = setup_mod.render_summary(weak, self.choose(weak))
        assert "Colab" in text and "configure(provider=" in text
        unknown = profile(None, None, ok=False)
        unknown.notes.append("Memory could not be detected.")
        text = setup_mod.render_summary(unknown, self.choose(unknown))
        assert "could not be measured" in text and "Note: Memory could not be detected." in text


class TestHardwareDetection:
    def test_this_machine(self, tmp_path):
        detected = ai.detect_hardware(tmp_path / "not" / "yet" / "created")
        assert detected.detection_ok
        assert detected.ram_total_gb > 0.5 and detected.cpu_count >= 1
        assert 0 < detected.ram_available_gb <= detected.ram_total_gb
        assert detected.disk_free_gb > 0
        assert "GB RAM" in detected.describe()
        assert json.dumps(detected.to_dict())

    def test_nvidia_card_from_nvidia_smi(self, monkeypatch):
        def fake_run(command, **options):
            assert command[0] == "nvidia-smi"
            return types.SimpleNamespace(
                stdout="NVIDIA GeForce RTX 3050, 4096, 3500\nNVIDIA GeForce RTX 3090, 24576, 20480\n")
        monkeypatch.setattr(hardware_module.subprocess, "run", fake_run)
        card = hardware_module._detect_nvidia()
        assert card.name == "NVIDIA GeForce RTX 3090" and card.vendor == "nvidia"
        assert card.vram_total_gb == pytest.approx(24.0)
        assert card.vram_free_gb == pytest.approx(20.0)
        assert not card.unified

    def test_no_nvidia_card(self, monkeypatch):
        def missing(command, **options):
            raise FileNotFoundError("nvidia-smi")
        monkeypatch.setattr(hardware_module.subprocess, "run", missing)
        monkeypatch.setitem(sys.modules, "pynvml", None)
        assert hardware_module._detect_nvidia() is None

    def test_apple_silicon(self, monkeypatch):
        monkeypatch.setattr(hardware_module.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(hardware_module.platform, "machine", lambda: "arm64")
        monkeypatch.setattr(hardware_module.subprocess, "run",
                            lambda *a, **k: types.SimpleNamespace(stdout="Apple M1\n"))
        chip = hardware_module._detect_apple(8.0, 4.0)
        assert chip.name == "Apple M1" and chip.unified and chip.vram_free_gb == 4.0
        monkeypatch.setattr(hardware_module.platform, "machine", lambda: "x86_64")
        assert hardware_module._detect_apple(8.0, 4.0) is None

    def test_memory_detection_failure_is_reported(self, monkeypatch, tmp_path):
        monkeypatch.setattr(hardware_module, "_detect_memory", lambda: (None, None))
        monkeypatch.setattr(hardware_module, "_detect_nvidia", lambda: None)
        detected = ai.detect_hardware(tmp_path)
        assert not detected.detection_ok
        assert "Memory could not be detected." in detected.notes
        assert "RAM unknown" in detected.describe()

    def test_unknown_free_memory_is_assumed_to_be_half(self, monkeypatch, tmp_path):
        monkeypatch.setattr(hardware_module, "_detect_memory", lambda: (16.0, None))
        monkeypatch.setattr(hardware_module, "_detect_nvidia", lambda: None)
        monkeypatch.setattr(hardware_module, "_detect_apple", lambda total, free: None)
        detected = ai.detect_hardware(tmp_path)
        assert detected.detection_ok and detected.ram_available_gb == 8.0
        assert detected.gpu is None and "no supported GPU" in detected.describe()


# ----------------------------------------------------------------------
# Download and setup, with Hugging Face replaced by a stand-in
# ----------------------------------------------------------------------
CONTENTS = {"tiny": b"tiny model", "small": b"a small model file", "large": b"the large model file!"}


def fake_catalog():
    needs = {"tiny": (1.0, 0.8), "small": (2.0, 1.6), "large": (6.0, 5.6)}
    return [ModelSpec(key=key, name=f"Fake {key}", repo=f"fake/{key}", filename=f"{key}.gguf",
                      size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                      min_ram_gb=needs[key][0], min_vram_gb=needs[key][1], layers=10,
                      quality="test", license="MIT")
            for key, data in CONTENTS.items()]


@pytest.fixture
def hub(monkeypatch):
    """A stand-in for huggingface_hub that writes the fake files and records calls."""
    calls = []

    def hf_hub_download(repo_id, filename, local_dir):
        calls.append((repo_id, filename))
        if hub_state.get("error"):
            raise hub_state["error"]
        path = f"{local_dir}/{filename}"
        with open(path, "wb") as file:
            file.write(hub_state.get("contents", CONTENTS[filename.split(".")[0]]))
        return path

    hub_state = {"calls": calls}
    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        types.SimpleNamespace(hf_hub_download=hf_hub_download))
    monkeypatch.setattr(setup_mod, "load_catalog", fake_catalog)
    return hub_state


LAPTOP = profile(16, 10.0)


class TestDownload:
    def test_download_and_verify(self, hub, tmp_path):
        model = fake_catalog()[1]
        path = ai.download_model(model, tmp_path / "models")
        assert path.read_bytes() == CONTENTS["small"]
        assert hub["calls"] == [("fake/small", "small.gguf")]
        # A complete file is not downloaded again
        ai.download_model(model, tmp_path / "models")
        assert len(hub["calls"]) == 1

    def test_corrupt_file_is_removed(self, hub, tmp_path):
        hub["contents"] = b"A SMALL MODEL FILE"       # right size, wrong contents
        with pytest.raises(ai.SetupError, match="integrity check"):
            ai.download_model(fake_catalog()[1], tmp_path)
        assert not (tmp_path / "small.gguf").exists()

    def test_incomplete_file(self, hub, tmp_path):
        hub["contents"] = b"cut off"
        with pytest.raises(ai.SetupError, match="finish the download"):
            ai.download_model(fake_catalog()[1], tmp_path)

    def test_no_internet(self, hub, tmp_path):
        hub["error"] = ConnectionError("Failed to resolve 'huggingface.co'")
        with pytest.raises(ai.SetupError, match="internet connection.*resumes"):
            ai.download_model(fake_catalog()[0], tmp_path)

    def test_disk_fills_up(self, hub, tmp_path):
        hub["error"] = OSError(errno.ENOSPC, "No space left on device")
        with pytest.raises(ai.SetupError, match="disk filled up"):
            ai.download_model(fake_catalog()[0], tmp_path)

    def test_model_no_longer_published(self, hub, tmp_path):
        hub["error"] = RuntimeError("404 Client Error: EntryNotFound")
        with pytest.raises(ai.SetupError, match="catalog entry may be out of date"):
            ai.download_model(fake_catalog()[0], tmp_path)

    def test_interrupted_download(self, hub, tmp_path):
        hub["error"] = KeyboardInterrupt()
        with pytest.raises(ai.SetupError, match="interrupted.*resume"):
            ai.download_model(fake_catalog()[0], tmp_path)

    def test_not_enough_disk_space(self, hub, tmp_path, monkeypatch):
        monkeypatch.setattr(setup_mod.shutil, "disk_usage",
                            lambda path: types.SimpleNamespace(free=100_000_000))
        with pytest.raises(ai.SetupError, match="Not enough disk space"):
            ai.download_model(fake_catalog()[0], tmp_path)
        assert hub["calls"] == []

    def test_missing_library(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        with pytest.raises(ai.SetupError, match=r"pip install actuneo\[ai\]"):
            ai.download_model(fake_catalog()[0], tmp_path)


class TestSetup:
    def test_automatic_setup(self, hub, isolated_home):
        result = ai.setup(auto=True, hardware=LAPTOP)
        assert result["result"] == "ready"
        assert result["backend"] == "local"
        assert result["local_model"]["key"] == "large"
        assert result["local_model"]["present"] and result["local_model"]["device"] == "cpu"
        saved = json.loads(config_module.config_path().read_text())
        assert saved["local"]["hardware"]["ram_total_gb"] == 16
        assert saved["local"]["n_gpu_layers"] == 0
        assert str(isolated_home / "home") in saved["local"]["path"]

    def test_setup_runs_only_once(self, hub):
        ai.setup(yes=True, hardware=LAPTOP)
        again = ai.setup(hardware=LAPTOP, input_func=lambda question: pytest.fail("asked"))
        assert again["result"] == "already_set_up"
        assert len(hub["calls"]) == 1
        forced = ai.setup(model="smallest", force=True, hardware=LAPTOP)
        assert forced["local_model"]["key"] == "tiny"
        assert len(hub["calls"]) == 2

    def test_named_models(self, hub):
        assert ai.setup(model="small", hardware=LAPTOP)["local_model"]["key"] == "small"
        assert ai.setup(model="smallest", hardware=LAPTOP)["local_model"]["key"] == "tiny"
        assert ai.setup(model="largest", hardware=LAPTOP)["local_model"]["key"] == "large"
        assert ai.setup(model="recommended", hardware=LAPTOP)["local_model"]["key"] == "large"
        assert ai.setup(model="Fake small", hardware=LAPTOP)["local_model"]["key"] == "small"
        with pytest.raises(ai.SetupError, match="Unknown model"):
            ai.setup(model="enormous", hardware=LAPTOP)

    def test_named_model_that_does_not_fit(self, hub):
        weak = profile(4, 2.0)
        with pytest.raises(ai.SetupError, match="more than this machine has free"):
            ai.setup(model="large", hardware=weak)
        forced = ai.setup(model="large", force=True, hardware=weak)
        assert forced["local_model"]["key"] == "large"

    def test_skip(self, hub):
        result = ai.setup(model="skip", hardware=LAPTOP)
        assert result["result"] == "skipped" and not result["configured"]
        assert hub["calls"] == []

    @pytest.mark.parametrize("answer, expected", [("y", "large"), ("YES", "large"),
                                                  ("1", "tiny"), ("2", "large")])
    def test_interactive_choice(self, hub, answer, expected):
        questions = []

        def reply(question):
            questions.append(question)
            return answer
        result = ai.setup(hardware=LAPTOP, input_func=reply)
        assert result["local_model"]["key"] == expected
        assert "Continue with the recommended model?" in questions[0]

    @pytest.mark.parametrize("answer", ["", "n", "s", "skip"])
    def test_declining_downloads_nothing(self, hub, answer):
        result = ai.setup(hardware=LAPTOP, input_func=lambda question: answer)
        assert result["result"] == "skipped" and hub["calls"] == []

    def test_unclear_answer_is_asked_again(self, hub):
        answers = iter(["maybe", "9", "y"])
        result = ai.setup(hardware=LAPTOP, input_func=lambda question: next(answers))
        assert result["result"] == "ready"

    def test_no_terminal_downloads_nothing(self, hub):
        def closed(question):
            raise EOFError
        assert ai.setup(hardware=LAPTOP, input_func=closed)["result"] == "skipped"
        assert hub["calls"] == []

    def test_options_are_printed_as_written(self, hub, capsys):
        ai.setup(model="skip", hardware=LAPTOP)
        printed = " ".join(capsys.readouterr().out.replace("│", " ").split())
        assert "[1] Smallest" in printed and "[s] Skip" in printed

    def test_machine_too_weak(self, hub, capsys):
        result = ai.setup(auto=True, hardware=profile(2, 0.9))
        assert result["result"] == "too_weak" and hub["calls"] == []
        assert "Colab" in capsys.readouterr().out

    def test_failed_detection_takes_the_smallest_model(self, hub, capsys):
        result = ai.setup(auto=True, hardware=profile(None, None, ok=False))
        assert result["local_model"]["key"] == "tiny"
        assert "could not be measured" in capsys.readouterr().out

    def test_status_and_remove(self, hub, capsys):
        assert not ai.status(quiet=True)["configured"]
        ai.status()
        assert "Nothing is set up yet" in capsys.readouterr().out
        ai.setup(auto=True, hardware=LAPTOP)
        info = ai.status()
        assert "Local model: Fake large" in capsys.readouterr().out
        path = info["local_model"]["path"]

        kept = ai.remove_model(input_func=lambda question: "n")
        assert not kept["removed"] and ai.status(quiet=True)["local_model"]["present"]
        removed = ai.remove_model(yes=True)
        assert removed["removed"]
        import os
        assert not os.path.exists(path)
        after = ai.status(quiet=True)
        assert after["local_model"] is None and not after["configured"]
        assert not ai.remove_model(yes=True)["removed"]
        with pytest.raises(ai.SetupError, match="No model is set up"):
            ai.get_backend()


class TestProviders:
    def test_key_is_never_written_to_the_configuration(self):
        secret = "sk-test-not-a-real-key-123"
        info = ai.configure(provider="openai", api_key=secret)
        assert info["backend"] == "api"
        assert info["provider"] == {"name": "openai", "model": "gpt-5-mini",
                                    "api_key_found": True}
        assert secret not in config_module.config_path().read_text()
        assert secret not in json.dumps(ai.status(quiet=True))
        backend = ai.get_backend()
        assert isinstance(backend, ai.OpenAIBackend)
        assert secret not in repr(backend)
        with pytest.raises(ValueError, match="must not be written"):
            config_module.save_config({"api_key": secret})

    def test_key_from_the_environment(self, monkeypatch):
        info = ai.configure(provider="gemini", model="gemini-test")
        assert not info["provider"]["api_key_found"]
        with pytest.raises(ai.SetupError, match="GEMINI_API_KEY or GOOGLE_API_KEY"):
            ai.get_backend()
        monkeypatch.setenv("GOOGLE_API_KEY", "from-environment")
        backend = ai.get_backend()
        assert isinstance(backend, ai.GeminiBackend) and backend.model == "gemini-test"
        assert ai.status(quiet=True)["provider"]["api_key_found"]

    def test_ollama_needs_no_key(self):
        info = ai.configure(provider="ollama", base_url="http://server:11434/")
        assert info["provider"]["api_key_found"]
        backend = ai.get_backend()
        assert isinstance(backend, ai.OllamaBackend) and backend.base_url == "http://server:11434"

    def test_saving_a_key_needs_keyring(self, monkeypatch):
        with pytest.raises(ai.SetupError, match="keyring"):
            ai.configure(provider="openai", api_key="k", save_key=True)
        stored = {}
        fake = types.SimpleNamespace(
            set_password=lambda service, name, key: stored.update({(service, name): key}),
            get_password=lambda service, name: stored.get((service, name)))
        monkeypatch.setitem(sys.modules, "keyring", fake)
        ai.configure(provider="openai", api_key="kept-in-keychain", save_key=True)
        assert stored == {("actuneo", "openai"): "kept-in-keychain"}
        setup_mod._SESSION_KEYS.clear()
        assert setup_mod._find_key("openai") == "kept-in-keychain"

    def test_unknown_provider(self):
        with pytest.raises(ValueError, match="provider must be one of"):
            ai.configure(provider="mistral")

    def test_switching_back_to_the_local_model(self, hub):
        ai.setup(auto=True, hardware=LAPTOP)
        ai.configure(provider="ollama")
        assert ai.status(quiet=True)["backend"] == "api"
        assert ai.status(quiet=True)["local_model"]["present"]
        assert ai.setup(hardware=LAPTOP)["backend"] == "local"


# ----------------------------------------------------------------------
# Backends
# ----------------------------------------------------------------------
TOOLS = [
    {"name": "life_expectancy", "description": "Expectation of life at an age",
     "parameters": {"type": "object", "properties": {"age": {"type": "integer"}},
                    "required": ["age"]}},
    {"type": "function", "function": {
        "name": "annuity", "description": "Value of an annuity",
        "parameters": {"type": "object",
                       "properties": {"age": {"type": "integer"}, "rate": {"type": "number"}},
                       "required": ["age", "rate"]}}},
]


class TestTools:
    def test_schema_allows_only_the_given_tools(self):
        schema = ai.tool_choice_schema(TOOLS)
        names = [option["properties"]["tool"]["enum"] for option in schema["anyOf"]]
        assert names == [["life_expectancy"], ["annuity"], ["none"]]
        first = schema["anyOf"][0]
        assert first["properties"]["arguments"]["required"] == ["age"]
        assert first["additionalProperties"] is False
        assert len(ai.tool_choice_schema(TOOLS, allow_none=False)["anyOf"]) == 2

    def test_tool_descriptions_in_each_spelling(self):
        tools = backends.normalise_tools(TOOLS + [{"name": "c", "input_schema": {"type": "object"}},
                                                  {"name": "d"}])
        assert [tool["name"] for tool in tools] == ["life_expectancy", "annuity", "c", "d"]
        assert tools[3]["parameters"] == {"type": "object", "properties": {}}
        for bad in ([], [{"description": "no name"}], [{"name": "none"}],
                    [{"name": "a"}, {"name": "a"}]):
            with pytest.raises(ValueError):
                backends.normalise_tools(bad)

    def test_parsing_the_reply(self):
        tools = backends.normalise_tools(TOOLS)
        call = backends.parse_tool_choice('{"tool": "annuity", "arguments": {"age": 60, "rate": 0.05}}',
                                          tools)
        assert call.name == "annuity" and call.arguments == {"age": 60, "rate": 0.05}
        none = backends.parse_tool_choice('{"tool": "none", "answer": "Hello"}', tools)
        assert none.name is None and none.text == "Hello"
        for bad in ("not json", "[1, 2]", '{"tool": "delete_everything", "arguments": {}}',
                    '{"tool": "annuity", "arguments": [1]}'):
            with pytest.raises(ai.BackendError):
                backends.parse_tool_choice(bad, tools)


class FakeLlama:
    created = []

    def __init__(self, **options):
        self.options = options
        self.requests = []
        self.reply = "  Sixty-five.  "
        self.replies = []
        FakeLlama.created.append(self)

    def create_chat_completion(self, **request):
        self.requests.append(request)
        reply = self.replies.pop(0) if self.replies else self.reply
        return {"choices": [{"message": {"content": reply}}]}


class TestLocalBackend:
    @pytest.fixture
    def model_file(self, tmp_path, monkeypatch):
        FakeLlama.created.clear()
        monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(Llama=FakeLlama))
        path = tmp_path / "model.gguf"
        path.write_bytes(b"gguf")
        return path

    def test_generate(self, model_file):
        backend = ai.LocalBackend(model_file, n_gpu_layers=-1, n_ctx=2048)
        assert FakeLlama.created == []                  # loaded on first use
        assert backend.generate("Retirement age?", system="Be brief.", max_tokens=20) == "Sixty-five."
        assert backend.generate("Again?") == "Sixty-five."
        assert len(FakeLlama.created) == 1              # and only once
        llama = FakeLlama.created[0]
        assert llama.options == {"model_path": str(model_file), "n_gpu_layers": -1,
                                 "n_ctx": 2048, "verbose": False}
        assert llama.requests[0]["messages"] == [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Retirement age?"}]
        assert llama.requests[0]["max_tokens"] == 20
        assert llama.requests[1]["max_tokens"] == 1024

    def test_tool_choice_in_two_constrained_steps(self, model_file):
        backend = ai.LocalBackend(model_file, n_threads=4)
        backend._model().replies = ['{"function": "life_expectancy"}', '{"age": 40}']
        call = backend.call_tools("How long will a 40 year old live?", TOOLS, system="Actuary.")
        assert call == ai.ToolCall("life_expectancy", {"age": 40})
        first, second = FakeLlama.created[0].requests
        # Step 1: the function is one of a fixed list
        assert first["response_format"]["schema"]["properties"]["function"]["enum"] == [
            "life_expectancy", "annuity", "none"]
        assert first["temperature"] == 0.0
        assert first["messages"][0]["content"].startswith("Actuary.")
        assert "- annuity: Value of an annuity" in first["messages"][0]["content"]
        # Step 2: the arguments are held to that function's own schema
        assert second["response_format"] == {"type": "json_object",
                                             "schema": TOOLS[0]["parameters"]}
        assert "life_expectancy" in second["messages"][0]["content"]
        assert FakeLlama.created[0].options["n_threads"] == 4

    def test_no_tool_fits(self, model_file):
        backend = ai.LocalBackend(model_file)
        backend._model().replies = ['{"function": "none"}', "Good morning to you."]
        call = backend.call_tools("Good morning", TOOLS)
        assert call.name is None and call.arguments == {}
        assert call.text == "Good morning to you."
        answer = FakeLlama.created[0].requests[1]
        assert "response_format" not in answer          # an ordinary reply

        backend._model().replies = ['{"function": "annuity"}', '{"age": 60, "rate": 0.04}']
        forced = backend.call_tools("Value it", TOOLS, allow_none=False)
        assert forced.name == "annuity"
        choice = FakeLlama.created[0].requests[2]["response_format"]["schema"]
        assert choice["properties"]["function"]["enum"] == ["life_expectancy", "annuity"]

    def test_unusable_replies(self, model_file):
        backend = ai.LocalBackend(model_file)
        for replies, message in [(["not json"], "valid JSON"), (["[1]"], "not an object"),
                                 (['{"function": "delete_everything"}'], "unknown tool")]:
            backend._model().replies = list(replies)
            with pytest.raises(ai.BackendError, match=message):
                backend.call_tools("Hello", TOOLS)

    def test_missing_file_and_missing_library(self, tmp_path, model_file, monkeypatch):
        with pytest.raises(ai.BackendError, match="model file not found"):
            ai.LocalBackend(tmp_path / "absent.gguf").generate("Hello")
        monkeypatch.setitem(sys.modules, "llama_cpp", None)
        with pytest.raises(ai.BackendError, match=r"pip install actuneo\[ai\]"):
            ai.LocalBackend(model_file).generate("Hello")

    def test_get_backend_uses_the_saved_settings(self, hub, model_file):
        ai.setup(auto=True, hardware=profile(16, 10.0, nvidia("NVIDIA RTX 3050", 4.0, 3.8)))
        backend = ai.get_backend()
        assert isinstance(backend, ai.LocalBackend)
        assert backend.n_gpu_layers == int(10 * 3.8 * 0.8 / 5.6) and backend.n_ctx == 4096
        assert ai.get_backend() is backend
        assert backend.generate("Hello") == "Sixty-five."


class TestHttpBackends:
    @pytest.fixture
    def sent(self, monkeypatch):
        """Intercept HTTP requests and answer from a queue."""
        record = {"requests": [], "replies": []}

        def fake_post(url, headers, body, timeout=120.0):
            record["requests"].append({"url": url, "headers": headers, "body": body})
            return record["replies"].pop(0)
        monkeypatch.setattr(backends, "_post_json", fake_post)
        return record

    def test_openai(self, sent):
        backend = ai.OpenAIBackend("secret", model="test-model")
        sent["replies"] += [
            {"choices": [{"message": {"content": " Hello. "}}]},
            {"choices": [{"message": {"tool_calls": [{"function": {
                "name": "annuity", "arguments": '{"age": 60, "rate": 0.05}'}}]}}]},
            {"choices": [{"message": {"content": "No tool fits."}}]},
        ]
        assert backend.generate("Hi", system="Brief.", max_tokens=50) == "Hello."
        first = sent["requests"][0]
        assert first["url"] == "https://api.openai.com/v1/chat/completions"
        assert first["headers"] == {"Authorization": "Bearer secret"}
        assert first["body"]["model"] == "test-model"
        assert first["body"]["messages"][0] == {"role": "system", "content": "Brief."}

        assert backend.call_tools("Value it", TOOLS) == ai.ToolCall("annuity",
                                                                    {"age": 60, "rate": 0.05})
        second = sent["requests"][1]["body"]
        assert second["tool_choice"] == "auto"
        assert [tool["function"]["name"] for tool in second["tools"]] == ["life_expectancy",
                                                                          "annuity"]
        nothing = backend.call_tools("Hello", TOOLS, allow_none=False)
        assert nothing.name is None and nothing.text == "No tool fits."
        assert sent["requests"][2]["body"]["tool_choice"] == "required"

    def test_gemini(self, sent):
        backend = ai.GeminiBackend("secret")
        sent["replies"] += [
            {"candidates": [{"content": {"parts": [{"text": "Hel"}, {"text": "lo"}]}}]},
            {"candidates": [{"content": {"parts": [{"functionCall": {
                "name": "life_expectancy", "args": {"age": 40}}}]}}]},
            {"candidates": []},
        ]
        assert backend.generate("Hi", system="Brief.") == "Hello"
        first = sent["requests"][0]
        assert first["url"].endswith("/models/gemini-2.5-flash:generateContent")
        assert first["headers"] == {"x-goog-api-key": "secret"}
        assert "secret" not in first["url"]
        assert first["body"]["systemInstruction"] == {"parts": [{"text": "Brief."}]}
        assert backend.call_tools("How long?", TOOLS) == ai.ToolCall("life_expectancy",
                                                                    {"age": 40})
        declared = sent["requests"][1]["body"]["tools"][0]["functionDeclarations"]
        assert [tool["name"] for tool in declared] == ["life_expectancy", "annuity"]
        with pytest.raises(ai.BackendError, match="no answer"):
            backend.generate("Hi")

    def test_ollama(self, sent):
        backend = ai.OllamaBackend()
        sent["replies"] += [
            {"message": {"content": "Hello"}},
            {"message": {"content": '{"tool": "none", "answer": "Just chatting"}'}},
        ]
        assert backend.generate("Hi", max_tokens=10) == "Hello"
        first = sent["requests"][0]
        assert first["url"] == "http://localhost:11434/api/chat" and first["headers"] == {}
        assert first["body"]["stream"] is False
        assert first["body"]["options"] == {"temperature": 0.2, "num_predict": 10}
        call = backend.call_tools("Hello there", TOOLS)
        assert call.name is None and call.text == "Just chatting"
        assert sent["requests"][1]["body"]["format"] == ai.tool_choice_schema(TOOLS)

    def test_http_errors_are_explained(self, monkeypatch):
        import io
        import urllib.error

        def failing(status):
            def urlopen(request, timeout=None, context=None):
                raise urllib.error.HTTPError(request.full_url, status, "error", {},
                                             io.BytesIO(b'{"error": "detail"}'))
            return urlopen
        backend = ai.OllamaBackend()
        for status, message in [(401, "rejected the API key"), (429, "rate limit"),
                                (500, "HTTP 500")]:
            monkeypatch.setattr(backends.urllib.request, "urlopen", failing(status))
            with pytest.raises(ai.BackendError, match=message):
                backend.generate("Hi")

        def unreachable(request, timeout=None, context=None):
            raise urllib.error.URLError("connection refused")
        monkeypatch.setattr(backends.urllib.request, "urlopen", unreachable)
        with pytest.raises(ai.BackendError, match="could not reach localhost:11434"):
            backend.generate("Hi")


def block(kind, **fields):
    return types.SimpleNamespace(type=kind, **fields)


class TestAnthropicBackend:
    @pytest.fixture
    def sdk(self, monkeypatch):
        """A stand-in for the anthropic package that records requests."""
        record = {"beta": [], "plain": [], "client": [], "reply": None, "error": None}

        class APIError(Exception):
            status_code = 500
            message = "boom"

        class AuthenticationError(APIError):
            pass

        class RateLimitError(APIError):
            pass

        class APIStatusError(APIError):
            pass

        class APIConnectionError(Exception):
            pass

        def create(kind):
            def call(**request):
                record[kind].append(request)
                if record["error"]:
                    raise record["error"]
                return record["reply"]
            return call

        class Anthropic:
            def __init__(self, **options):
                record["client"].append(options)
                self.messages = types.SimpleNamespace(create=create("plain"))
                self.beta = types.SimpleNamespace(
                    messages=types.SimpleNamespace(create=create("beta")))

        module = types.SimpleNamespace(
            Anthropic=Anthropic, AuthenticationError=AuthenticationError,
            RateLimitError=RateLimitError, APIStatusError=APIStatusError,
            APIConnectionError=APIConnectionError)
        monkeypatch.setitem(sys.modules, "anthropic", module)
        record["module"] = module
        return record

    def test_generate(self, sdk):
        sdk["reply"] = types.SimpleNamespace(stop_reason="end_turn", content=[
            block("thinking", thinking=""), block("text", text="Hello "),
            block("text", text="there.")])
        backend = ai.AnthropicBackend(api_key="secret")
        assert sdk["client"] == [{"api_key": "secret"}]
        assert backend.generate("Hi", system="Brief.", temperature=0.9) == "Hello there."
        request = sdk["beta"][0]
        assert request["model"] == "claude-opus-5-5"
        assert request["betas"] == ["server-side-fallback-2026-07-01"]
        assert request["fallbacks"] == "default"
        assert request["max_tokens"] == 16000 and request["system"] == "Brief."
        assert request["messages"] == [{"role": "user", "content": "Hi"}]
        assert "temperature" not in request and "thinking" not in request

    def test_credentials_can_come_from_the_sdk(self, sdk):
        sdk["reply"] = types.SimpleNamespace(stop_reason="end_turn",
                                             content=[block("text", text="Hi")])
        backend = ai.AnthropicBackend(model="claude-haiku-5-5", fallback=False)
        assert sdk["client"] == [{}]
        assert backend.generate("Hi", max_tokens=100) == "Hi"
        assert sdk["beta"] == []
        assert sdk["plain"][0]["model"] == "claude-haiku-5-5"
        assert "fallbacks" not in sdk["plain"][0] and sdk["plain"][0]["max_tokens"] == 100

    def test_tools(self, sdk):
        sdk["reply"] = types.SimpleNamespace(stop_reason="tool_use", content=[
            block("text", text="Let me work that out."),
            block("tool_use", name="annuity", input={"age": 60, "rate": 0.05})])
        backend = ai.AnthropicBackend(api_key="secret")
        assert backend.call_tools("Value it", TOOLS) == ai.ToolCall("annuity",
                                                                    {"age": 60, "rate": 0.05})
        request = sdk["beta"][0]
        assert "tool_choice" not in request          # forcing a tool is not accepted
        declared = request["tools"]
        assert [tool["name"] for tool in declared] == ["life_expectancy", "annuity"]
        assert all(tool["strict"] is True for tool in declared)
        assert declared[0]["input_schema"]["additionalProperties"] is False
        assert "parameters" not in declared[0]

        sdk["reply"] = types.SimpleNamespace(stop_reason="end_turn",
                                             content=[block("text", text="No tool fits.")])
        nothing = backend.call_tools("Hello", TOOLS, system="Actuary.", allow_none=False)
        assert nothing.name is None and nothing.text == "No tool fits."
        assert sdk["beta"][1]["system"].startswith("Actuary.")
        assert "exactly one of the tools" in sdk["beta"][1]["system"]

    def test_refusals_and_errors(self, sdk):
        backend = ai.AnthropicBackend(api_key="secret")
        sdk["reply"] = types.SimpleNamespace(stop_reason="refusal", content=[])
        with pytest.raises(ai.BackendError, match="declined"):
            backend.generate("Hi")
        module = sdk["module"]
        for error, message in [(module.AuthenticationError(), "rejected the API key"),
                               (module.RateLimitError(), "rate limit"),
                               (module.APIStatusError(), r"error \(500\): boom"),
                               (module.APIConnectionError(), "could not reach Anthropic")]:
            sdk["error"] = error
            with pytest.raises(ai.BackendError, match=message):
                backend.generate("Hi")

    def test_missing_sdk(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "anthropic", None)
        with pytest.raises(ai.BackendError, match="pip install anthropic"):
            ai.AnthropicBackend(api_key="secret")

    def test_configured_through_get_backend(self, sdk):
        ai.configure(provider="anthropic")
        backend = ai.get_backend()
        assert isinstance(backend, ai.AnthropicBackend) and backend.model == "claude-opus-5-5"


def test_importing_the_module_needs_no_optional_packages():
    code = ("import sys, actuneo.ai; "
            "heavy = [m for m in ('llama_cpp', 'huggingface_hub', 'anthropic', 'rich') "
            "if m in sys.modules]; print(heavy)")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"

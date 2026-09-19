# SPDX-License-Identifier: AGPL-3.0-only

"""Unit tests for unsloth/mergekit_bridge.py.

The bridge is the isolated seam to the optional mergekit engine, so these tests
must pass on a machine **without** mergekit: every child-process path goes
through the ``runner`` seam, and runtime discovery is stubbed via the
``UNSLOTH_MERGEKIT_PYTHON`` / ``UNSLOTH_MERGEKIT_SHADOW`` environment
variables rather than by installing anything.
"""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BRIDGE_PATH = REPO_ROOT / "unsloth" / "mergekit_bridge.py"
CORE_PATH = REPO_ROOT / "unsloth" / "multi_adapter_merge.py"

_spec = importlib.util.spec_from_file_location("unsloth.mergekit_bridge", BRIDGE_PATH)
bridge = importlib.util.module_from_spec(_spec)
sys.modules["unsloth.mergekit_bridge"] = bridge
_spec.loader.exec_module(bridge)


def _write_adapter(directory, base_model = "unsloth/Llama-3.2-1B", name = "adapter"):
    path = Path(directory) / name
    path.mkdir(parents = True, exist_ok = True)
    (path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": base_model}), encoding = "utf-8"
    )
    return str(path)


def _fake_runner(returncode = 0, stdout = "", write_config_json = True):
    """Return a runner that records calls and optionally writes a merged dir."""
    calls = []

    def runner(command, timeout, work_dir):
        calls.append({"command": list(command), "timeout": timeout, "work_dir": str(work_dir)})
        if write_config_json and returncode == 0:
            Path(command[4]).mkdir(parents = True, exist_ok = True)
            (Path(command[4]) / "config.json").write_text("{}", encoding = "utf-8")
        return subprocess.CompletedProcess(command, returncode, stdout = stdout)

    runner.calls = calls
    return runner


class MethodMappingTests(unittest.TestCase):
    def test_mergekit_methods_map_one_to_one(self):
        for method in ("linear", "ties", "dare_ties", "dare_linear"):
            self.assertEqual(bridge.mergekit_target_method(method), method)

    def test_aliases_and_case_are_normalised(self):
        self.assertEqual(bridge.mergekit_target_method("dare"), "dare_ties")
        self.assertEqual(bridge.mergekit_target_method("DARE-Ties"), "dare_ties")
        self.assertEqual(bridge.normalize_method("SVD"), "ctm")
        self.assertEqual(bridge.normalize_method("Magnitude-Prune"), "magnitude_prune")

    def test_legacy_only_methods_have_no_mergekit_target(self):
        for method in bridge.LEGACY_ONLY_METHODS:
            self.assertIsNone(bridge.mergekit_target_method(method))
        self.assertEqual(set(bridge.LEGACY_ONLY_METHODS), {"magnitude_prune", "ctm", "cat"})

    def test_mapping_covers_the_core_engine_methods(self):
        """Every core method is either delegated to mergekit or explicitly legacy."""
        source = CORE_PATH.read_text(encoding = "utf-8")
        supported = source.split("SUPPORTED_METHODS = (")[1].split(")")[0]
        methods = {item.strip().strip('"') for item in supported.split(",") if item.strip()}
        mapped = set(bridge.MERGEKIT_METHOD_MAP) | set(bridge.LEGACY_ONLY_METHODS)
        self.assertEqual(methods, mapped)


class EngineSelectionTests(unittest.TestCase):
    def setUp(self):
        self._saved = {key: os.environ.get(key) for key in (bridge.MERGE_ENGINE_ENV, bridge.MERGEKIT_PYTHON_ENV)}
        for key in self._saved:
            os.environ.pop(key, None)

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_explicit_legacy_wins_for_every_method(self):
        for method in ("linear", "ties", "cat"):
            self.assertEqual(bridge.resolve_engine(method, "legacy"), "legacy")

    def test_legacy_only_method_falls_back_even_when_mergekit_is_forced(self):
        self.assertEqual(bridge.resolve_engine("cat", "mergekit"), "legacy")
        self.assertEqual(bridge.resolve_engine("ctm", "mergekit"), "legacy")
        self.assertEqual(bridge.resolve_engine("magnitude_prune", "mergekit"), "legacy")

    def test_auto_uses_mergekit_for_supported_methods_when_available(self):
        original = bridge.mergekit_available
        bridge.mergekit_available = lambda: True
        try:
            self.assertEqual(bridge.resolve_engine("ties", "auto"), "mergekit")
            # A legacy-only method must not be handed to mergekit.
            self.assertEqual(bridge.resolve_engine("cat", "auto"), "legacy")
            self.assertEqual(bridge.resolve_engine("ctm", "auto"), "legacy")
        finally:
            bridge.mergekit_available = original

    def test_auto_falls_back_to_legacy_when_unavailable(self):
        original = bridge.mergekit_available
        bridge.mergekit_available = lambda: False
        try:
            self.assertEqual(bridge.resolve_engine("ties", "auto"), "legacy")
            self.assertEqual(bridge.resolve_engine("linear", "auto"), "legacy")
        finally:
            bridge.mergekit_available = original

    def test_env_var_selects_auto_without_an_argument(self):
        original = bridge.mergekit_available
        bridge.mergekit_available = lambda: True
        os.environ[bridge.MERGE_ENGINE_ENV] = "auto"
        try:
            self.assertEqual(bridge.resolve_engine("ties"), "mergekit")
        finally:
            bridge.mergekit_available = original

    def test_unknown_engine_value_is_treated_as_auto(self):
        self.assertEqual(bridge.resolve_engine("cat", "banana"), "legacy")

    def test_env_var_supplies_the_default_engine(self):
        os.environ[bridge.MERGE_ENGINE_ENV] = "legacy"
        self.assertEqual(bridge.resolve_engine("ties"), "legacy")

    def test_mergekit_available_is_false_without_mergekit(self):
        os.environ[bridge.MERGEKIT_PYTHON_ENV] = "/definitely/missing/python"
        self.assertFalse(bridge.mergekit_available())

    def test_shadow_dir_is_probed_as_a_sys_path_entry(self):
        """A --target shadow must be visible to the probe, not just the child."""
        shadow = tempfile.mkdtemp(prefix = "unsloth_mergekit_shadow_")
        self.addCleanup(self._cleanup_dir, shadow)
        # A fake engine package is enough for find_spec to succeed.
        package = Path(shadow) / "mergekit"
        package.mkdir()
        (package / "__init__.py").write_text("", encoding = "utf-8")

        os.environ[bridge.MERGEKIT_PYTHON_ENV] = sys.executable
        os.environ[bridge.MERGEKIT_SHADOW_ENV] = shadow
        self.addCleanup(os.environ.pop, bridge.MERGEKIT_SHADOW_ENV, None)
        self.assertEqual(bridge.resolve_mergekit_python(), sys.executable)
        self.assertTrue(bridge.mergekit_available())

    def test_shadow_dir_is_forwarded_to_the_child_process(self):
        shadow = "/tmp/pinned-engine"
        os.environ[bridge.MERGEKIT_SHADOW_ENV] = shadow
        self.addCleanup(os.environ.pop, bridge.MERGEKIT_SHADOW_ENV, None)
        assert bridge.MERGEKIT_SHADOW_ENV in bridge._BOOTSTRAP, (
            "the child must read the shadow from its environment"
        )
        assert "sys.path.insert" in bridge._BOOTSTRAP

    @staticmethod
    def _cleanup_dir(path):
        __import__("shutil").rmtree(path, ignore_errors = True)


class ConfigBuilderTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.adapter_a = _write_adapter(self._tmp.name, name = "a")
        self.adapter_b = _write_adapter(self._tmp.name, name = "b")
        self.base = "unsloth/Llama-3.2-1B"

    def _config(self, **kwargs):
        params = {
            "base_model": self.base,
            "adapters": [self.adapter_a, self.adapter_b],
        }
        params.update(kwargs)
        return bridge.build_merge_config(**params)

    def test_sources_reference_the_base_with_each_lora(self):
        config = self._config(method = "linear")
        self.assertEqual(config["merge_method"], "linear")
        self.assertEqual(config["base_model"], self.base)
        self.assertEqual(len(config["models"]), 2)
        for source, adapter in zip(config["models"], (self.adapter_a, self.adapter_b)):
            self.assertEqual(source["model"], self.base)
            self.assertEqual(source["lora"], adapter)

    def test_base_model_is_never_repeated_as_a_plain_source(self):
        """A bare base source would be counted once per adapter."""
        config = self._config()
        bare = [source for source in config["models"] if "lora" not in source]
        self.assertEqual(bare, [])

    def test_equal_weights_are_normalised_to_one(self):
        config = self._config()
        weights = [source["parameters"]["weight"] for source in config["models"]]
        self.assertAlmostEqual(sum(weights), 1.0)
        self.assertAlmostEqual(weights[0], 0.5)

    def test_explicit_weights_are_normalised_when_asked(self):
        config = self._config(weights = [3.0, 1.0])
        weights = [source["parameters"]["weight"] for source in config["models"]]
        self.assertAlmostEqual(weights[0], 0.75)
        self.assertAlmostEqual(weights[1], 0.25)

    def test_explicit_weights_can_stay_raw(self):
        config = self._config(weights = [3.0, 1.0], normalize_weights = False)
        weights = [source["parameters"]["weight"] for source in config["models"]]
        self.assertEqual(weights, [3.0, 1.0])

    def test_zero_weight_sum_is_rejected(self):
        with self.assertRaises(ValueError):
            self._config(weights = [0.0, 0.0])

    def test_weight_count_must_match_adapter_count(self):
        with self.assertRaises(ValueError):
            self._config(weights = [1.0])

    def test_ties_carries_density_only(self):
        config = self._config(method = "ties", density = 0.25)
        for source in config["models"]:
            self.assertEqual(source["parameters"]["density"], 0.25)
            self.assertNotIn("drop_rate", source["parameters"])

    def test_dare_methods_carry_drop_rate_only(self):
        for method in ("dare_ties", "dare_linear"):
            config = self._config(method = method, drop_rate = 0.2)
            for source in config["models"]:
                self.assertEqual(source["parameters"]["drop_rate"], 0.2)
                self.assertNotIn("density", source["parameters"])

    def test_linear_carries_no_extra_parameters(self):
        config = self._config(method = "linear")
        for source in config["models"]:
            self.assertEqual(set(source["parameters"]), {"weight"})

    def test_legacy_only_methods_are_rejected(self):
        for method in bridge.LEGACY_ONLY_METHODS:
            with self.assertRaises(ValueError) as ctx:
                self._config(method = method)
            self.assertIn("mergekit has no equivalent", str(ctx.exception))

    def test_two_adapters_are_required(self):
        with self.assertRaises(ValueError):
            bridge.build_merge_config(self.base, [self.adapter_a])
        with self.assertRaises(ValueError):
            bridge.build_merge_config(self.base, [])

    def test_adapter_trained_on_another_base_is_rejected(self):
        other = _write_adapter(self._tmp.name, base_model = "meta-llama/Llama-3.1-8B", name = "c")
        with self.assertRaises(ValueError) as ctx:
            self._config(adapters = [self.adapter_a, other])
        self.assertIn("different base models", str(ctx.exception))

    def test_base_name_matching_tolerates_org_prefixes(self):
        plain = _write_adapter(self._tmp.name, base_model = "Llama-3.2-1B", name = "d")
        config = self._config(adapters = [self.adapter_a, plain])
        self.assertEqual(len(config["models"]), 2)

    def test_missing_adapter_config_is_only_fatal_in_strict_mode(self):
        no_config = Path(self._tmp.name) / "e"
        no_config.mkdir()
        config = self._config(adapters = [self.adapter_a, str(no_config)])
        self.assertEqual(len(config["models"]), 2)
        with self.assertRaises(ValueError):
            self._config(
                adapters = [self.adapter_a, str(no_config)], strict_base_match = True
            )

    def test_out_dtype_is_emitted_only_when_set(self):
        self.assertNotIn("dtype", self._config())
        self.assertEqual(self._config(out_dtype = "float32")["dtype"], "float32")


class YamlWriterTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.adapter_a = _write_adapter(self._tmp.name, name = "a")
        self.adapter_b = _write_adapter(self._tmp.name, name = "b")

    def test_golden_config_shape(self):
        config = bridge.build_merge_config(
            "unsloth/Llama-3.2-1B",
            [self.adapter_a, self.adapter_b],
            weights = [0.7, 0.3],
            method = "ties",
            density = 0.5,
        )
        path, cleanup = bridge.write_merge_config(config)
        self.addCleanup(self._cleanup, cleanup)
        text = Path(path).read_text(encoding = "utf-8")
        self.assertEqual(
            text,
            "merge_method: ties\n"
            "base_model: unsloth/Llama-3.2-1B\n"
            "models:\n"
            "  - model: unsloth/Llama-3.2-1B\n"
            f"    lora: {self.adapter_a}\n"
            "    parameters:\n"
            "      weight: 0.7\n"
            "      density: 0.5\n"
            "  - model: unsloth/Llama-3.2-1B\n"
            f"    lora: {self.adapter_b}\n"
            "    parameters:\n"
            "      weight: 0.3\n"
            "      density: 0.5\n",
        )

    @staticmethod
    def _cleanup(directory):
        if directory:
            __import__("shutil").rmtree(directory, ignore_errors = True)

    def test_writer_quotes_scalars_yaml_could_misread(self):
        import io

        handle = io.StringIO()
        bridge.dump_merge_config_yaml(
            {
                "merge_method": "linear",
                "note": "yes",
                "off": "no",
                "path": "a b c",
                "numeric": "22",
                "count": 3,
            },
            handle,
        )
        text = handle.getvalue()
        # YAML 1.1 would resolve these to bools/ints, so they must be quoted.
        self.assertIn('note: "yes"', text)
        self.assertIn('off: "no"', text)
        self.assertIn('numeric: "22"', text)
        self.assertIn('path: "a b c"', text)
        self.assertIn("count: 3", text)

    def test_writer_leaves_ordinary_model_ids_plain(self):
        import io

        handle = io.StringIO()
        bridge.dump_merge_config_yaml(
            {"base_model": "unsloth/Llama-3.2-1B", "lora": "E:\\runs\\a-1"}, handle
        )
        self.assertEqual(
            handle.getvalue(),
            "base_model: unsloth/Llama-3.2-1B\nlora: E:\\runs\\a-1\n",
        )

    def test_callers_directory_is_not_returned_for_cleanup(self):
        config = bridge.build_merge_config(
            "unsloth/Llama-3.2-1B", [self.adapter_a, self.adapter_b]
        )
        path, cleanup = bridge.write_merge_config(config, directory = self._tmp.name)
        self.assertTrue(os.path.isfile(path))
        self.assertIsNone(cleanup)

    def test_yaml_round_trips_through_a_plain_parser(self):
        """The emitted subset must be real YAML, not just mergekit-ish text."""
        if importlib.util.find_spec("yaml") is None:
            self.skipTest("PyYAML not installed")
        import yaml as pyyaml

        config = bridge.build_merge_config(
            "unsloth/Llama-3.2-1B",
            [self.adapter_a, self.adapter_b],
            weights = [0.7, 0.3],
            method = "dare_linear",
            drop_rate = 0.1,
            out_dtype = "float32",
        )
        path, cleanup = bridge.write_merge_config(config)
        self.addCleanup(self._cleanup, cleanup)
        loaded = pyyaml.safe_load(Path(path).read_text(encoding = "utf-8"))
        self.assertEqual(loaded, config)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.adapter_a = _write_adapter(self._tmp.name, name = "a")
        self.adapter_b = _write_adapter(self._tmp.name, name = "b")
        self._saved = os.environ.get(bridge.MERGEKIT_PYTHON_ENV)
        self.addCleanup(self._restore)

    def _restore(self):
        if self._saved is None:
            os.environ.pop(bridge.MERGEKIT_PYTHON_ENV, None)
        else:
            os.environ[bridge.MERGEKIT_PYTHON_ENV] = self._saved

    def _config_path(self):
        config = bridge.build_merge_config(
            "unsloth/Llama-3.2-1B", [self.adapter_a, self.adapter_b]
        )
        path, cleanup = bridge.write_merge_config(config, directory = self._tmp.name)
        self.assertIsNone(cleanup)
        return path

    def test_missing_interpreter_raises_unavailable(self):
        os.environ[bridge.MERGEKIT_PYTHON_ENV] = "/definitely/missing/python"
        out = os.path.join(self._tmp.name, "out")
        with self.assertRaises(bridge.MergeKitUnavailableError):
            bridge.run_mergekit_merge(self._config_path(), out, runner = _fake_runner())

    def test_missing_config_file_raises_merge_error(self):
        out = os.path.join(self._tmp.name, "out")
        with self.assertRaises(bridge.MergeKitMergeError):
            bridge.run_mergekit_merge(
                os.path.join(self._tmp.name, "nope.yaml"),
                out,
                mergekit_python = sys.executable,
                runner = _fake_runner(),
            )

    def test_failing_child_reports_the_output_tail(self):
        out = os.path.join(self._tmp.name, "out")
        runner = _fake_runner(returncode = 3, stdout = "Traceback: boom")
        with self.assertRaises(bridge.MergeKitMergeError) as ctx:
            bridge.run_mergekit_merge(
                self._config_path(), out, mergekit_python = sys.executable, runner = runner
            )
        self.assertIn("exit 3", str(ctx.exception))
        self.assertIn("Traceback: boom", str(ctx.exception))

    def test_silent_success_without_config_json_is_an_error(self):
        out = os.path.join(self._tmp.name, "out")
        with self.assertRaises(bridge.MergeKitMergeError) as ctx:
            bridge.run_mergekit_merge(
                self._config_path(),
                out,
                mergekit_python = sys.executable,
                runner = _fake_runner(write_config_json = False),
            )
        self.assertIn("no config.json", str(ctx.exception))

    def test_command_is_isolated_and_carries_every_argument(self):
        out = os.path.join(self._tmp.name, "out")
        config_path = self._config_path()
        runner = _fake_runner()
        result = bridge.run_mergekit_merge(
            config_path,
            out,
            mergekit_python = sys.executable,
            device = "cuda",
            transformers_cache = "/hf",
            out_shard_size = 123,
            runner = runner,
        )
        self.assertEqual(result, out)
        command = runner.calls[0]["command"]
        self.assertEqual(command[0], sys.executable)
        # -I keeps the engine interpreter away from the parent's sys.path.
        self.assertEqual(command[1], "-I")
        self.assertTrue(command[2].endswith("unsloth_mergekit_run.py"))
        self.assertEqual(command[3], config_path)
        self.assertEqual(command[4], out)
        self.assertEqual(command[5], "/hf")
        self.assertEqual(command[6], "123")
        self.assertEqual(command[7], "cuda")
        # The bootstrap must sit beside the config it is handed.
        self.assertTrue(os.path.isfile(command[2]))

    def test_end_to_end_helper_cleans_its_temp_dir(self):
        out = os.path.join(self._tmp.name, "merged")
        runner = _fake_runner()
        result = bridge.merge_adapters_via_mergekit(
            "unsloth/Llama-3.2-1B",
            [self.adapter_a, self.adapter_b],
            out,
            weights = [0.7, 0.3],
            method = "linear",
            mergekit_python = sys.executable,
            runner = runner,
        )
        self.assertEqual(result, out)
        work_dir = Path(runner.calls[0]["work_dir"])
        self.assertFalse(work_dir.exists(), "temporary merge config dir must be removed")
        # keep_config was off, so the config never lands inside the output dir.
        self.assertNotEqual(work_dir.parent, Path(out))

    def test_keep_config_leaves_the_config_next_to_the_output(self):
        out = os.path.join(self._tmp.name, "merged_keep")
        runner = _fake_runner()
        bridge.merge_adapters_via_mergekit(
            "unsloth/Llama-3.2-1B",
            [self.adapter_a, self.adapter_b],
            out,
            mergekit_python = sys.executable,
            runner = runner,
            keep_config = True,
        )
        config_path = Path(runner.calls[0]["work_dir"]) / "merge_config.yaml"
        self.assertTrue(config_path.is_file())


class SchemaIntegrationTests(unittest.TestCase):
    """Opt-in check that our YAML is valid *for mergekit*, not just for us.

    Only the config is validated against mergekit's own pydantic schema, so
    nothing is downloaded.  It is opt-in (``UNSLOTH_TEST_MERGEKIT_PARITY=1``)
    because a default test run must not depend on a mergekit install.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.adapter_a = _write_adapter(self._tmp.name, name = "a")
        self.adapter_b = _write_adapter(self._tmp.name, name = "b")

    def test_generated_config_parses_as_a_mergekit_configuration(self):
        if not os.environ.get("UNSLOTH_TEST_MERGEKIT_PARITY"):
            self.skipTest("set UNSLOTH_TEST_MERGEKIT_PARITY=1 to validate against mergekit")
        python = bridge.resolve_mergekit_python()
        if python is None:
            self.skipTest("mergekit is not reachable from any configured Python")

        for method, extra in (
            ("linear", {}),
            ("ties", {"density": 0.7}),
            ("dare_ties", {"drop_rate": 0.3}),
            ("dare_linear", {"drop_rate": 0.3}),
        ):
            with self.subTest(method = method):
                config = bridge.build_merge_config(
                    "unsloth/Llama-3.2-1B",
                    [self.adapter_a, self.adapter_b],
                    weights = [0.7, 0.3],
                    method = method,
                    **extra,
                )
                path, cleanup = bridge.write_merge_config(config)
                self.addCleanup(self._cleanup, cleanup)
                probe = (
                    "import sys, yaml\n"
                    "from mergekit.config import MergeConfiguration\n"
                    "MergeConfiguration.model_validate(yaml.safe_load(open(sys.argv[1], encoding='utf-8')))\n"
                    "print('ok')\n"
                )
                completed = subprocess.run(
                    [python, "-c", probe, path],
                    stdout = subprocess.PIPE,
                    stderr = subprocess.STDOUT,
                    text = True,
                    encoding = "utf-8",
                    errors = "replace",
                    timeout = 120,
                )
                self.assertEqual(
                    completed.returncode,
                    0,
                    f"mergekit rejected the {method} config:\n{completed.stdout}",
                )

    @staticmethod
    def _cleanup(directory):
        if directory:
            __import__("shutil").rmtree(directory, ignore_errors = True)


if __name__ == "__main__":
    unittest.main()
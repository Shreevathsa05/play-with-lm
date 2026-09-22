import os
import sys
import tempfile
import unittest
from importlib.machinery import ModuleSpec
from unittest.mock import MagicMock, patch

sys.modules["unsloth"] = MagicMock()
sys.modules["unsloth.chat_templates"] = MagicMock()
sys.modules["trl"] = MagicMock()
sys.modules["transformers"] = MagicMock()
torch_mock = MagicMock()
torch_mock.__spec__ = ModuleSpec("torch", loader=None)
sys.modules["torch"] = torch_mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.finetuning_flows.orchestrator import (
    _apply_chat,
    _ensure_chat_template,
    _export_and_maybe_push,
    _prepare_sft_dataset,
    _preflight_hf_push,
    run_recipe,
)
from src.utils.logcarrier import LogCarrier


class _FakeDataset:
    def __init__(self, rows, column_names=None):
        self.rows = list(rows)
        self.column_names = column_names or list(rows[0].keys())

    def map(self, fn):
        return _FakeDataset([fn(row) for row in self.rows], column_names=["text"])


class TestFlows(unittest.TestCase):
    def test_apply_chat_falls_back_without_template(self):
        tokenizer = MagicMock()
        tokenizer.chat_template = None
        text = _apply_chat(
            tokenizer,
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
        )
        self.assertIn("### User:", text)
        self.assertIn("Hello", text)
        self.assertIn("### Assistant:", text)
        self.assertIn("Hi", text)
        tokenizer.apply_chat_template.assert_not_called()

    def test_ensure_chat_template_sets_fallback_jinja(self):
        tokenizer = MagicMock()
        tokenizer.chat_template = None
        with patch(
            "unsloth.chat_templates.get_chat_template",
            side_effect=RuntimeError("unavailable"),
        ):
            ensured = _ensure_chat_template(tokenizer, "some/base-model")
        self.assertTrue(ensured.chat_template)
        self.assertIn("im_start", ensured.chat_template)

    def test_prepare_sft_review_star_dataset(self):
        dataset = _FakeDataset(
            [
                {
                    "date": "2020-01-01",
                    "id": 1,
                    "package_name": "com.example.app",
                    "review": "Great app, love it!",
                    "star": 5,
                    "version_id": 12,
                }
            ]
        )
        tokenizer = MagicMock()
        tokenizer.chat_template = "{{ messages }}"
        tokenizer.apply_chat_template.return_value = "<chat>formatted</chat>"
        prepared = _prepare_sft_dataset(dataset, tokenizer)
        self.assertEqual(prepared.column_names, ["text"])
        self.assertEqual(prepared.rows[0]["text"], "<chat>formatted</chat>")
        messages = tokenizer.apply_chat_template.call_args.args[0]
        self.assertEqual(messages[0]["role"], "user")
        self.assertIn("com.example.app", messages[0]["content"])
        self.assertIn("Great app, love it!", messages[0]["content"])
        self.assertEqual(messages[1], {"role": "assistant", "content": "5"})

    def test_prepare_sft_alpaca_without_chat_template(self):
        dataset = _FakeDataset(
            [{"instruction": "Say hi", "input": "", "output": "Hello"}]
        )
        tokenizer = MagicMock()
        tokenizer.chat_template = None
        prepared = _prepare_sft_dataset(dataset, tokenizer)
        self.assertIn("Say hi", prepared.rows[0]["text"])
        self.assertIn("Hello", prepared.rows[0]["text"])

    def test_prepare_sft_rejects_unknown_schema(self):
        dataset = _FakeDataset([{"id": 1, "meta": "x"}])
        with self.assertRaises(ValueError) as ctx:
            _prepare_sft_dataset(dataset, MagicMock())
        self.assertIn("review/star", str(ctx.exception))

    def test_dry_run_lora(self):
        result = run_recipe(
            {
                "job_uuid": "lora-dry",
                "recipe": "lora",
                "base_model_id": "unsloth/gemma-3-270m-it",
                "dataset_minio_uri": "minio://datasets/train.jsonl",
                "dry_run": True,
                "export": {"format": "lora"},
                "eval": {"prompts": [{"prompt": "hi", "reference": "hello"}]},
            },
            scratch_dir=tempfile.mkdtemp(),
        )
        self.assertEqual(result["status"], "COMPLETED")
        self.assertTrue(result["export"]["dry_run"])
        self.assertEqual(result["eval"]["n_prompts"], 1)

    def test_dry_run_qlora(self):
        scratch = tempfile.mkdtemp()
        result = run_recipe(
            {
                "job_uuid": "flow-dry",
                "recipe": "qlora_4bit",
                "base_model_id": "google/gemma-3-270m-it",
                "dataset_minio_uri": "minio://datasets/train.jsonl",
                "dry_run": True,
                "eval": {"prompts": [{"prompt": "hi", "reference": "hi"}]},
            },
            scratch_dir=scratch,
        )
        self.assertEqual(result["status"], "COMPLETED")
        self.assertTrue(result["export"]["dry_run"])
        self.assertEqual(result["eval"]["n_prompts"], 1)

    def test_dry_run_reports_requested_hf_export_without_published_url(self):
        result = run_recipe(
            {
                "job_uuid": "flow-hf-dry",
                "recipe": "qlora_4bit",
                "base_model_id": "google/gemma-3-270m-it",
                "dataset_minio_uri": "minio://datasets/train.jsonl",
                "dry_run": True,
                "export": {"hf_repo": "user/model"},
            },
            scratch_dir=tempfile.mkdtemp(),
        )
        self.assertEqual(result["export"]["hf_push"]["status"], "requested")
        self.assertIsNone(result["export"]["hf_push"]["url"])

    def test_real_hf_export_requires_token_before_training(self):
        settings = MagicMock(hf_token="")
        with patch("src.core.config.get_settings", return_value=settings):
            result = run_recipe(
                {
                    "job_uuid": "flow-no-token",
                    "recipe": "qlora_4bit",
                    "base_model_id": "google/gemma-3-270m-it",
                    "dataset_minio_uri": "minio://datasets/train.jsonl",
                    "export": {"hf_repo": "user/model"},
                },
                scratch_dir=tempfile.mkdtemp(),
            )
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["export"]["hf_push"]["status"], "failed")
        self.assertIn("HF_TOKEN", result["export"]["hf_push"]["error"])

    def test_hf_token_is_validated_with_hub(self):
        payload = {"job_uuid": "token-check", "export": {"hf_repo": "user/model", "hf_token": "secret"}}
        api = MagicMock()
        with patch("huggingface_hub.HfApi", return_value=api):
            state = _preflight_hf_push(
                payload, dry_run=False, log=LogCarrier("token-check")
            )
        api.whoami.assert_called_once_with(token="secret")
        api.create_repo.assert_called_once_with(
            repo_id="user/model",
            token="secret",
            private=True,
            exist_ok=True,
            repo_type="model",
        )
        self.assertEqual(state["status"], "pushing")

    def test_default_hf_repo_uses_username_and_job_uuid(self):
        settings = MagicMock(hf_token="secret", hf_username="Shreevathsa05")
        payload = {"job_uuid": "abc-123", "export": {"format": "lora"}}
        with patch("src.core.config.get_settings", return_value=settings), patch(
            "huggingface_hub.HfApi"
        ) as api_cls:
            api = api_cls.return_value
            state = _preflight_hf_push(
                payload, dry_run=False, log=LogCarrier("default-repo")
            )
        self.assertEqual(state["status"], "pushing")
        self.assertEqual(state["repo_id"], "Shreevathsa05/abc-123")
        self.assertEqual(payload["export"]["hf_repo"], "Shreevathsa05/abc-123")
        api.create_repo.assert_called_once_with(
            repo_id="Shreevathsa05/abc-123",
            token="secret",
            private=True,
            exist_ok=True,
            repo_type="model",
        )

    def test_no_default_hf_repo_without_token(self):
        settings = MagicMock(hf_token="", hf_username="Shreevathsa05")
        with patch("src.core.config.get_settings", return_value=settings):
            state = _preflight_hf_push(
                {"job_uuid": "abc-123", "export": {}},
                dry_run=False,
                log=LogCarrier("no-token-default"),
            )
        self.assertIsNone(state)

    def test_successful_hf_push_returns_confirmed_url(self):
        exporter = MagicMock()
        exporter.export_model.return_value = "minio://jobs/push/exports"
        exporter.push_to_hub.return_value = "user/model"
        with patch(
            "src.finetuning_modules.model_exporters.UnslothModelExporter", exporter
        ), patch(
            "src.finetuning_flows.orchestrator._notify_hf_push_started"
        ) as notify_started:
            result = _export_and_maybe_push(
                MagicMock(), MagicMock(),
                {"export": {"hf_repo": "user/model", "hf_token": "secret"}},
                tempfile.mkdtemp(), LogCarrier("push"), "push",
            )
        self.assertEqual(result["hf_push"]["status"], "published")
        self.assertEqual(result["hf_push"]["url"], "https://huggingface.co/user/model")
        notify_started.assert_called_once_with(
            "push", "user/model", "minio://jobs/push/exports", unittest.mock.ANY
        )
        self.assertTrue(exporter.push_to_hub.call_args.kwargs["folder_path"].endswith("export"))

    def test_default_hf_push_target_on_export(self):
        exporter = MagicMock()
        exporter.export_model.return_value = "minio://jobs/abc/exports"
        exporter.push_to_hub.return_value = "Shreevathsa05/abc"
        settings = MagicMock(hf_token="secret", hf_username="Shreevathsa05")
        with patch(
            "src.finetuning_modules.model_exporters.UnslothModelExporter", exporter
        ), patch(
            "src.finetuning_flows.orchestrator._notify_hf_push_started"
        ), patch(
            "src.core.config.get_settings", return_value=settings
        ):
            result = _export_and_maybe_push(
                MagicMock(), MagicMock(),
                {"export": {"format": "lora"}},
                tempfile.mkdtemp(), LogCarrier("push-default"), "abc",
            )
        self.assertEqual(result["hf_push"]["repo_id"], "Shreevathsa05/abc")
        exporter.push_to_hub.assert_called_once()
        self.assertEqual(
            exporter.push_to_hub.call_args.kwargs["repo_id"], "Shreevathsa05/abc"
        )

    def test_run_qlora_4bit_gpu_path(self):
        # Patches must target modules as imported inside functions — patch source modules
        mocked_dataset = MagicMock()
        mocked_dataset.column_names = ["text"]
        with patch("src.finetuning_flows.orchestrator._load_dataset", return_value=mocked_dataset), patch(
            "src.finetuning_flows.orchestrator._export_and_maybe_push",
            return_value={"export_dest": "x", "hf_push": None},
        ), patch(
            "src.finetuning_flows.orchestrator._run_eval", return_value={"n_prompts": 0}
        ), patch(
            "src.finetuning_modules.model_loaders.unsloth_model_loader.UnslothModelLoader.load_model",
            return_value=(MagicMock(), MagicMock()),
        ) as load_model, patch(
            "src.finetuning_modules.model_loaders.unsloth_model_loader.UnslothModelLoader.get_peft_model",
            return_value=MagicMock(),
        ), patch(
            "src.finetuning_modules.model_trainers.unsloth_model_trainer.UnslothModelTrainer.train",
            return_value="ok",
        ):
            # Import path used inside run_sft_style
            with patch(
                "src.finetuning_modules.model_loaders.UnslothModelLoader"
            ) as UL, patch(
                "src.finetuning_modules.model_trainers.UnslothModelTrainer"
            ) as UT:
                UL.load_model.return_value = (MagicMock(), MagicMock())
                UL.get_peft_model.return_value = MagicMock()
                UT.train.return_value = "ok"
                result = run_recipe(
                    {
                        "job_uuid": "flow-1",
                        "recipe": "qlora_4bit",
                        "base_model_id": "google/gemma-3-270m-it",
                        "dataset_minio_uri": "minio://datasets/train.jsonl",
                        "export": {"format": "lora", "local_dir": "out"},
                    },
                    scratch_dir=tempfile.mkdtemp(),
                )
                self.assertEqual(result["status"], "COMPLETED")
                UL.load_model.assert_called_once()
                self.assertTrue(UL.load_model.call_args.kwargs.get("load_in_4bit"))
                UL.get_peft_model.assert_called_once()
                UT.train.assert_called_once()

    def test_run_lora_gpu_path(self):
        mocked_dataset = MagicMock()
        mocked_dataset.column_names = ["text"]
        base_model = MagicMock()
        tokenizer = MagicMock()
        peft_model = MagicMock()

        with patch(
            "src.finetuning_flows.orchestrator._load_dataset", return_value=mocked_dataset
        ), patch(
            "src.finetuning_flows.orchestrator._export_and_maybe_push",
            return_value={"export_dest": "minio://jobs/lora-flow/exports", "hf_push": None},
        ) as export_model, patch(
            "src.finetuning_flows.orchestrator._run_eval", return_value={"n_prompts": 0}
        ), patch(
            "src.finetuning_flows.orchestrator._capture_eval_baseline", return_value=None
        ), patch(
            "src.finetuning_modules.model_loaders.UnslothModelLoader"
        ) as loader, patch(
            "src.finetuning_modules.model_trainers.UnslothModelTrainer"
        ) as trainer:
            loader.load_model.return_value = (base_model, tokenizer)
            loader.get_peft_model.return_value = peft_model
            result = run_recipe(
                {
                    "job_uuid": "lora-flow",
                    "recipe": "lora",
                    "base_model_id": "unsloth/gemma-3-270m-it",
                    "dataset_minio_uri": "minio://datasets/train.jsonl",
                    "batch_size": 1,
                    "export": {"format": "lora"},
                },
                scratch_dir=tempfile.mkdtemp(),
            )

        self.assertEqual(result["status"], "COMPLETED")
        loader.load_model.assert_called_once_with(
            "unsloth/gemma-3-270m-it",
            max_seq_length=2048,
            load_in_4bit=False,
            load_in_8bit=False,
            full_finetuning=False,
        )
        loader.get_peft_model.assert_called_once_with(base_model)
        trainer.train.assert_called_once()
        self.assertIs(trainer.train.call_args.kwargs["model"], peft_model)
        self.assertEqual(trainer.train.call_args.kwargs["batch_size"], 1)
        export_model.assert_called_once()
        self.assertEqual(export_model.call_args.kwargs["default_format"], "lora")

    def test_unknown_recipe(self):
        with self.assertRaises(ValueError):
            run_recipe({"job_uuid": "x", "recipe": "dpo"}, scratch_dir="s")

    def test_embedding_dry_run_uses_embedding_route(self):
        result = run_recipe(
            {
                "job_uuid": "embed-dry",
                "recipe": "embedding",
                "base_model_id": "unsloth/all-MiniLM-L6-v2",
                "dry_run": True,
            },
            scratch_dir=tempfile.mkdtemp(),
        )
        self.assertTrue(result["export"]["dry_run"])
        self.assertIsNone(result["eval"])

    def test_embedding_gpu_route_uses_sentence_transformer_trainer(self):
        dataset = MagicMock()
        dataset.column_names = ["anchor", "positive"]
        model = MagicMock()
        model.tokenizer = MagicMock()
        fast_sentence_transformer = MagicMock()
        fast_sentence_transformer.from_pretrained.return_value = model
        fast_sentence_transformer.get_peft_model.return_value = model
        with patch("unsloth.FastSentenceTransformer", fast_sentence_transformer), patch(
            "src.finetuning_flows.orchestrator._load_dataset", return_value=dataset
        ), patch(
            "src.finetuning_modules.model_trainers.UnslothEmbeddingTrainer.train"
        ) as train, patch(
            "src.finetuning_flows.orchestrator._export_and_maybe_push",
            return_value={"export_dest": "minio://jobs/embed/exports", "hf_push": None},
        ):
            result = run_recipe(
                {
                    "job_uuid": "embed",
                    "recipe": "embedding",
                    "base_model_id": "unsloth/all-MiniLM-L6-v2",
                    "dataset_minio_uri": "minio://datasets/embed.jsonl",
                    "batch_size": 8,
                },
                scratch_dir=tempfile.mkdtemp(),
            )
        self.assertEqual(result["status"], "COMPLETED")
        train.assert_called_once()
        self.assertEqual(train.call_args.kwargs["batch_size"], 8)
        fast_sentence_transformer.get_peft_model.assert_called_once()


if __name__ == "__main__":
    unittest.main()

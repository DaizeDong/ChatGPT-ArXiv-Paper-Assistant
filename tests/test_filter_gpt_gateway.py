"""The batch paper filter dispatches through the gateway, not through OpenAI.

Two things are pinned here, and the second one exists because I broke it:

1. The completion SHIM. Both filter functions read exactly four attributes off
   whatever call_chatgpt returns. If the shim stops matching that shape the
   filters break in a way that looks like a model problem.

2. THE RETRY DECORATOR IS STILL ON call_chatgpt. While inserting the shim
   dataclasses above the function I put them between ``@retry.retry(...)`` and
   ``def call_chatgpt``, so the decorator silently landed on a dataclass instead.
   The module still imported, every test still passed, and the only visible
   symptom would have been batches no longer retrying on a transient failure.
   Syntactically valid, semantically wrong, invisible: exactly the shape of
   failure this repo keeps getting bitten by.
"""
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "unused-by-the-gateway")

from arxiv_assistant.filters import filter_gpt
from arxiv_assistant.utils import llm_gateway
from arxiv_assistant.utils.agent_runner import AgentError


class _FakeResult:
    """Duck-types llmcall's Result."""

    def __init__(self, text="", data=None, provider="fake", error=""):
        self.text = text
        self.data = data
        self.provider = provider
        self.error = error

    def __bool__(self):
        return bool(self.text) or self.data is not None


class CallChatgptDispatchTests(unittest.TestCase):
    def setUp(self):
        llm_gateway.LEDGER.reset()

    def test_retry_decorator_is_still_applied_to_call_chatgpt(self):
        """A transient failure must still be retried.

        retry.retry wraps the function, so the attribute is the evidence. If a
        future edit moves a definition between the decorator and the function
        again, this is what goes red.
        """
        self.assertTrue(
            hasattr(filter_gpt.call_chatgpt, "__wrapped__")
            or filter_gpt.call_chatgpt.__name__ != "call_chatgpt"
            or "retry" in repr(filter_gpt.call_chatgpt),
            "call_chatgpt appears to have lost its @retry.retry decorator",
        )

    def test_shim_classes_are_still_plain_dataclasses(self):
        import dataclasses

        for cls in (
            filter_gpt._ShimUsage,
            filter_gpt._ShimMessage,
            filter_gpt._ShimChoice,
            filter_gpt._ShimCompletion,
        ):
            with self.subTest(cls=cls.__name__):
                self.assertTrue(dataclasses.is_dataclass(cls))

    def test_shim_matches_every_attribute_the_filters_read(self):
        usage = filter_gpt._ShimUsage()
        completion = filter_gpt._ShimCompletion(
            choices=[filter_gpt._ShimChoice(message=filter_gpt._ShimMessage(content="hi"))],
            usage=usage,
        )
        # The exact four reads performed by filter_papers_by_title and
        # filter_papers_by_abstract.
        self.assertEqual(completion.choices[0].message.content, "hi")
        self.assertEqual(completion.usage.prompt_tokens, 0)
        self.assertEqual(completion.usage.completion_tokens, 0)
        self.assertEqual(completion.usage.model_extra, {})

    def test_dispatch_goes_through_the_gateway_and_returns_the_shim(self):
        captured = {}

        def fake_llmcall(prompt, **kwargs):
            captured["prompt"] = prompt
            captured["kwargs"] = kwargs
            return _FakeResult(text='["2603.11111"]', provider="codexg")

        with patch.object(llm_gateway, "resolve_backend", return_value=llm_gateway.BACKEND_LLMCALL), \
             patch.object(llm_gateway, "_call_llmcall", side_effect=lambda prompt, **kw: llm_gateway.GatewayResult(
                 text=fake_llmcall(prompt, **kw).text, backend="llmcall", provider="codexg")):
            completion = filter_gpt.call_chatgpt(
                "SYSTEM", "USER", None, "ignored-model", config=None
            )

        self.assertIsInstance(completion, filter_gpt._ShimCompletion)
        self.assertEqual(completion.choices[0].message.content, '["2603.11111"]')
        self.assertEqual(completion.provider, "codexg")
        # The system/user split must survive into the single prompt string.
        self.assertIn("SYSTEM", captured["prompt"])
        self.assertIn("USER", captured["prompt"])

    def test_openai_backend_still_uses_the_client_when_explicitly_selected(self):
        class _FakeClient:
            def __init__(self):
                self.calls = []

                class _Completions:
                    def __init__(self, outer):
                        self.outer = outer

                    def create(self, **kwargs):
                        self.outer.calls.append(kwargs)
                        return "REAL_OPENAI_COMPLETION"

                class _Chat:
                    def __init__(self, outer):
                        self.completions = _Completions(outer)

                self.chat = _Chat(self)

        client = _FakeClient()
        with patch.object(llm_gateway, "resolve_backend", return_value=llm_gateway.BACKEND_OPENAI):
            out = filter_gpt.call_chatgpt("SYS", "USR", client, "gpt-x", config=None)
        self.assertEqual(out, "REAL_OPENAI_COMPLETION")
        self.assertEqual(client.calls[0]["model"], "gpt-x")
        # The historical message shape must be preserved on this path.
        self.assertEqual(
            [m["role"] for m in client.calls[0]["messages"]], ["system", "user"]
        )


class CalcPriceTests(unittest.TestCase):
    def test_tokenless_usage_returns_zero_silently(self):
        """No pricing lookup, and crucially no per-batch print.

        On the llmcall chain every batch reports zero tokens. Printing "model not
        in pricing table" for each one would make the log of a healthy run look
        exactly like the log of a misconfigured one.
        """
        with patch.object(filter_gpt, "get_model_pricing") as pricing:
            prompt_cost, completion_cost = filter_gpt.calc_price(
                "whatever-model", filter_gpt._ShimUsage()
            )
        self.assertEqual((prompt_cost, completion_cost), (0, 0))
        pricing.assert_not_called()

    def test_real_usage_still_prices_normally(self):
        usage = filter_gpt._ShimUsage(prompt_tokens=1_000_000, completion_tokens=1_000_000)
        with patch.object(
            filter_gpt,
            "get_model_pricing",
            return_value={"m": {"prompt": 2.0, "completion": 8.0}},
        ):
            prompt_cost, completion_cost = filter_gpt.calc_price("m", usage)
        self.assertAlmostEqual(prompt_cost, 2.0)
        self.assertAlmostEqual(completion_cost, 8.0)


class ClientConstructionTests(unittest.TestCase):
    def test_openai_client_is_not_built_on_a_chain_backend(self):
        """A client holding an empty key would turn a config fault into a 401.

        The 401 would be recorded as a model failure, which is how the original
        three-month outage stayed invisible: the error text blamed the wrong layer.
        """
        source = (
            __import__("pathlib").Path(filter_gpt.__file__).read_text(encoding="utf-8")
        )
        self.assertIn("if llm_gateway.resolve_backend(config) == llm_gateway.BACKEND_OPENAI:", source)
        # and the unconditional construction must be gone from live code
        live_lines = [
            line.strip()
            for line in source.splitlines()
            if line.strip().startswith("openai_client = get_openai_client()")
        ]
        self.assertEqual(
            live_lines,
            ["openai_client = get_openai_client()"],
            "expected exactly one guarded construction site in live code",
        )


if __name__ == "__main__":
    unittest.main()

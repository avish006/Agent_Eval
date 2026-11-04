import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "agent")
)

from agent import agent
from dataset import dataset
from model import eval_model

from deepeval.metrics import PlanQualityMetric
from deepeval.integrations.langchain import CallbackHandler
from deepeval.evaluate import AsyncConfig

async_config = AsyncConfig(run_async=True, max_concurrent=2, throttle_value=1.0)

metrics = [PlanQualityMetric(threshold=0.8, model=eval_model, include_reason=True)]

for golden in dataset.evals_iterator(metrics=metrics, async_config=async_config):
    response = agent.invoke(
        {"messages": [{"role": "user", "content": golden.input}]},
        config={"configurable": {"thread_id": str(uuid4())},
        "callbacks": [CallbackHandler()]}
    )

import logging
import json
from typing import List, Tuple
from litellm import completion, acompletion
from a2a.server.tasks import TaskUpdater
from a2a.utils import new_agent_text_message, new_task_with_context
from a2a.types import TaskState
from ts_bench.base_agent import GreenAgent
from ts_bench.types import EvalRequest, TaskAssignment, TaskRequest
from ts_bench.task_bank import TaskBank, TaskDefinition

logger = logging.getLogger(__name__)

class TimeSeriesGreenAgent(GreenAgent):
    def __init__(self, task_bank: TaskBank):
        self.task_bank = task_bank

    async def run_eval(self, request: EvalRequest, updater: TaskUpdater) -> None:
        logger.info(f"Received evaluation results from participant {request.participant}: {request.config}")
        score = request.config.get("score", 0.0)
        feedback = request.config.get("feedback", "No feedback provided.")

        await updater.update_status(
            TaskState.completed,
            new_agent_text_message(
                f"Evaluation complete. Score: {score:.2f}, Feedback: {feedback}",
                context_id=updater.context_id,
            ),
        )

    async def _rerank_with_llm(
        self,
        query: str,
        candidates: List[TaskDefinition],
        model: str = "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
        top_n: int = 5,
    ) -> List[TaskDefinition]:
        """
        Use an LLM to critically rerank candidate tasks.
        Returns a list of TaskDefinition objects in reranked order (truncated to top_n).
        """
        if not candidates:
            return []

        candidate_payload = []
        for i, t in enumerate(candidates):
            candidate_payload.append(
                {
                    "index": i,
                    "task_id": t.task_id,
                    "name": t.name,
                    "description": t.description,
                    "task_type": t.task_type,
                    "difficulty": t.difficulty,
                }
            )
            
        def _format_rerank_results(indices: List[int]) -> str:
            lines = ["LLM reranking results (best → worst):"]
            for idx in indices:
                if 0 <= idx < len(candidates):
                    t = candidates[idx]
                    lines.append(
                        f"- idx={idx}: {t.name} "
                        f"(ID={t.task_id}, type={t.task_type}, difficulty={t.difficulty})"
                    )
                else:
                    lines.append(f"- idx={idx}: [out of range]")
            return "\n".join(lines)

        rerank_prompt = f"""You are an expert in time-series machine learning benchmarks.
Your job is to very carefully select and rank the most relevant tasks for the user's request.

User query:
\"\"\"{query}\"\"\"

You are given {len(candidate_payload)} candidate tasks. Each candidate has:
- an integer "index" (0-based),
- a "name",
- a free-text "description",
- a "task_type",
- and a "difficulty".

Candidate tasks (JSON):
{json.dumps(candidate_payload, indent=2)}

Instructions:
1. Read the user query and all candidate tasks carefully.
2. Think step by step and critically assess which tasks best match the query in terms of:
   - task type,
   - forecasting horizon / setup,
   - domain (e.g. finance, crypto, commodities, etc.),
   - difficulty, and
   - overall suitability as a benchmark.
3. Decide on an ordering of the candidate indices from MOST to LEAST relevant.
4. Return ONLY a JSON array of indices in your final answer (no explanations, no extra keys).

The output format MUST be exactly like:
[0, 3, 1, 2]

Do not include any text before or after the JSON.
"""       
        try:
            logger.info(
                "Sending task reranking request to LLM (Anthropic via Bedrock)..."
            )
            response = await acompletion(
                model=model,
                messages=[{"role": "user", "content": rerank_prompt}],
                temperature=0.0,
                max_tokens=1500,
            )
            content = response["choices"][0]["message"]["content"]
            if isinstance(content, list):
                # Bedrock / Anthropic can sometimes return list-of-blocks
                content_str = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            else:
                content_str = str(content)
            reranked_indices = json.loads(content_str.strip())
            if not isinstance(reranked_indices, list):
                raise ValueError("LLM output is not a list of indices")

            # valid indices in range and preserve order
            valid_indices: List[int] = [
                int(i)
                for i in reranked_indices
                if isinstance(i, int) and 0 <= i < len(candidates)
            ]
            if not valid_indices:
                raise ValueError("No valid indices in LLM output")

            logger.info(self._format_rerank_results(valid_indices))

            # Map indices to TaskDefinition, truncated to top_n
            reranked_tasks: List[TaskDefinition] = []
            for idx in valid_indices[:top_n]:
                reranked_tasks.append(candidates[idx])

            return reranked_tasks

        except Exception as e:
            logger.warning(
                f"LLM reranking failed ({e}), falling back to embedding order."
            )
            return candidates[:top_n]   # fallback: return the first top_n candidates in original order


    async def handle_task_request(self, request: TaskRequest, context_id: str, updater: TaskUpdater) -> TaskAssignment:
        logger.info(f"Green Agent received task request from Purple Agent: '{request.query}'")

        # Semantic embedding search from TaskBank
        matching_tasks = self.task_bank.search_tasks(request.query)

        if not matching_tasks:
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(
                    f"No matching tasks found for your request: '{request.query}'",
                    context_id=context_id,
                ),
            )
            raise ValueError(f"No matching tasks found for: {request.query}")

        # Take top-K semantic candidates
        top_k = matching_tasks[:5] # Use LLM to further search and rerank candidates, instead of directly using the semantic searching result

        # LLM reranking by indices
        reranked = await self._rerank_with_llm(
            query=request.query,
            candidates=top_k,
            top_n=5,
        )

        # choose the Best after reranking
        selected_task = reranked[0]
        logger.info(
            f"Selected task: {selected_task.name} |"
            f"({selected_task.task_id})"
        )

        data_url = self.task_bank.get_presigned_url(selected_task.data_s3_key)
        eval_fn_url = self.task_bank.get_presigned_url(selected_task.eval_fn_s3_key)

        task_assignment = TaskAssignment(
            task_id=selected_task.task_id,
            name=selected_task.name,
            description=selected_task.description,
            task_type=selected_task.task_type,
            difficulty=selected_task.difficulty,
            data_url=data_url,
            eval_fn_url=eval_fn_url,
        )

        await updater.update_status(
            TaskState.completed,
            new_agent_text_message(
                f"Task '{selected_task.name}' assigned. "
                f"Details: {task_assignment.model_dump_json()}",
                context_id=context_id,
            ),
        )
        return task_assignment

    def validate_request(self, request: EvalRequest | TaskRequest) -> Tuple[bool, str]:
        if isinstance(request, EvalRequest):
            if "score" not in request.config:
                return False, "Evaluation result missing 'score' in config."
            if not isinstance(request.config["score"], (int, float)):
                return False, "'score' must be a number."
            return True, "EvalRequest is valid."
        elif isinstance(request, TaskRequest):
            if not request.query or not isinstance(request.query, str):
                return False, "TaskRequest missing 'query' or 'query' is not a string."
            return True, "TaskRequest is valid."
        else:
            return False, "Unsupported request type."

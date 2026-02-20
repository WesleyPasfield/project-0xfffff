"""Service for generating agent skills from workshop artifacts.

Synthesizes modular skill files from three sources collected during the workshop:
1. Optimized system prompt (from GEPA prompt optimization)
2. Aligned judge memory (semantic guidelines + episodic examples from MemAlign)
3. Evaluated traces with human feedback (from annotation/alignment phases)

Skills are written to the Lakebase generated_skills table, scoped per workshop.
Workshop participants can then retrieve these skills and integrate them into
their agents independently.

Reference: https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview
"""

import json
import logging
import re
import uuid
from collections import Counter
from typing import Any, Dict, Generator, List, Optional, Tuple

logger = logging.getLogger(__name__)


SKILL_GENERATION_PROMPT_TEMPLATE = """You are generating runtime skill files that an AI agent loads on-demand to handle specific types of user requests.

PURPOSE: Each skill is a concrete instruction set the agent follows when it encounters a particular
type of query. Skills make the agent BETTER at its job by encoding domain knowledge, quality
standards, and lessons learned from human evaluation. Skills are NOT descriptions of what the agent
does -- they are actionable playbooks the agent executes.

IMPORTANT DISTINCTION:
- BAD skill: "evaluating-response-quality" (this is a judge concern, not an agent task)
- BAD skill: "providing-strategic-recommendations" (too vague, just restates the system prompt)
- BAD skill: "ensuring-high-quality-output" (meta-skill about quality, not a user task)
- GOOD skill: a skill named after a specific query pattern you see in the traces, with concrete steps
- GOOD skill: a skill that addresses a recurring failure mode from low-scoring traces
- GOOD skill: a skill that handles an edge case where the agent needs domain-specific reasoning

Derive skill names and content entirely from the traces and judge feedback provided below.
Do NOT use generic names. Every skill name should reflect the specific domain and query
patterns visible in the actual user interactions.

You have three evidence sources. Use ALL of them.

=== SOURCE 1: AGENT SYSTEM PROMPT ===
{optimized_prompt}

=== SOURCE 2: QUALITY STANDARDS (from human expert evaluation) ===
These guidelines represent what domain experts determined makes a response good or bad:
{judge_guidelines}

{episodic_examples}

=== SOURCE 3: REAL USER INTERACTIONS ===
Actual queries the agent received, with human quality ratings:
{traces_analysis}

===

SKILL IDENTIFICATION PROCESS:

1. Start with Source 3 (real traces). Cluster the actual user queries into distinct request
   patterns. Each cluster that requires a different reasoning approach is a candidate skill.

2. For each candidate, check Source 2 (judge guidelines) for quality criteria specific to that
   query type. What did human experts flag as good or bad for these kinds of requests?

3. Check Source 1 (system prompt) for any domain-specific instructions relevant to the query
   pattern. Extract the parts that apply.

4. Write the skill as instructions the agent should follow when it encounters this query type.
   The skill should make the agent produce responses that score well on the judge criteria.

SKILL FILE FORMAT:

```
---
name: lowercase-with-hyphens
description: >
  When to load this skill: [specific trigger conditions from real queries].
  What it helps with: [concrete improvement this skill provides].
---

# [Skill Title]

## When to use
[Specific query patterns, keywords, or user intents that trigger this skill.]
[Drawn directly from the real queries in Source 3.]

## Instructions
[Step-by-step instructions the agent follows for this query type.]
[Be concrete: "Look up X, then compare Y, then format as Z".]
[Include conditional logic: "If the user asks for A, do B. If they ask for C, do D."]

## Quality bar
[What human experts rated highly for this type of query (from Source 2).]
[What they penalized. Be specific.]

## Common mistakes
[Failure patterns observed in low-scoring traces.]
[Each mistake should be a concrete "Do not..." statement.]
```

RULES:

1. Every skill must be grounded in actual query patterns from Source 3. Do not invent skills
   for query types that do not appear in the traces.

2. Do not create meta-skills about the agent's own quality, evaluation, or self-assessment.
   The agent's job is to answer user queries, not to evaluate itself.

3. Do not simply restate the system prompt as a skill. Skills should add value BEYOND what
   the system prompt already says, by encoding lessons from human evaluation.

4. Target 4-7 skills. Prefer fewer, more specific skills over many vague ones.

5. Each skill must be self-contained. Loading any single skill should improve the agent's
   handling of that query type without requiring other skills.

6. Name skills after the user's task, not the agent's capability.
   The name should reflect the specific domain visible in the traces.
   Bad: "providing-analytical-insights" (generic, describes capability)
   Good: a name derived from the actual query patterns in Source 3

YAML FRONTMATTER RULES:
- name: lowercase letters, numbers, and hyphens only. Max 64 characters.
- description: Max 1024 characters. Must state trigger conditions and concrete benefit.

RETURN FORMAT:

Return a JSON array where each element has:
- "filename": string (lowercase, hyphens, ends with .md)
- "content": string (the complete file content including YAML frontmatter)

Return ONLY the JSON array. No commentary, no markdown fences around the JSON."""


class SkillsGenerationService:
    """Service for generating agent skills from workshop artifacts."""

    def generate_skills(
        self,
        workshop_id: str,
        optimized_prompt: str,
        model_name: str,
        mlflow_config: Any,
        judge_names: Optional[List[str]] = None,
        judge_name: Optional[str] = None,
    ) -> Generator[Any, None, None]:
        """Generate agent skills from workshop artifacts and write to Lakebase.

        This generator yields log messages (str) during execution, and finally
        yields a result dict with success/failure status and the generated skills.

        Args:
            workshop_id: Workshop ID
            optimized_prompt: The GEPA-optimized system prompt text
            model_name: Databricks FMAPI model name for skill generation
            mlflow_config: MLflow configuration (experiment_id, databricks_host, token)
            judge_names: List of aligned judge names (one per rubric question)
            judge_name: Fallback single judge name if judge_names not provided
        """
        yield ""
        yield "========================================"
        yield "  AGENT SKILLS GENERATION"
        yield "========================================"
        yield ""

        try:
            import mlflow
        except ImportError as e:
            yield f"ERROR: mlflow not available: {e}"
            yield {"error": f"mlflow not available: {e}", "success": False, "phase": "skills"}
            return

        # Configure MLflow tracking so get_scorer and search_traces work.
        # Prefer the stored user token over the app's service principal
        # because UC operations (prompts, experiments) require permissions on
        # the facilitator's schemas, which the app SP typically lacks.
        import os
        databricks_host = mlflow_config.databricks_host
        if databricks_host:
            os.environ['DATABRICKS_HOST'] = databricks_host.rstrip('/')
        if mlflow_config.databricks_token:
            os.environ['DATABRICKS_TOKEN'] = mlflow_config.databricks_token
            os.environ.pop('DATABRICKS_CLIENT_ID', None)
            os.environ.pop('DATABRICKS_CLIENT_SECRET', None)
        mlflow.set_tracking_uri('databricks')

        experiment_id = mlflow_config.experiment_id
        if experiment_id:
            try:
                mlflow.set_experiment(experiment_id=experiment_id)
                yield f"Using MLflow experiment ID: {experiment_id}"
            except Exception as e:
                yield f"WARNING: Failed to set experiment {experiment_id}: {e}"

        # ------------------------------------------------------------------
        # Source 1: System prompt
        # ------------------------------------------------------------------
        yield f"Source 1: System prompt ({len(optimized_prompt)} chars)"

        # ------------------------------------------------------------------
        # Source 2: Aligned judge -- semantic + episodic memory
        # ------------------------------------------------------------------
        yield "Source 2: Loading aligned judge(s) for quality guidelines..."
        judge_guidelines_text = ""
        episodic_text = ""

        names_to_load = judge_names if judge_names else ([judge_name] if judge_name else [])
        if not names_to_load:
            yield "WARNING: No judge names provided -- skills will lack quality guidelines"
        else:
            try:
                from mlflow.genai.scorers import get_scorer

                for jn in names_to_load:
                    yield f"  Loading judge '{jn}'..."
                    scorer = get_scorer(name=jn, experiment_id=experiment_id)
                    if scorer is None:
                        yield f"  WARNING: Judge '{jn}' not found -- skipping"
                        continue

                    # Extract semantic memory (distilled guidelines)
                    if hasattr(scorer, "_semantic_memory") and scorer._semantic_memory:
                        for i, guideline in enumerate(scorer._semantic_memory, 1):
                            text = getattr(guideline, "guideline_text", str(guideline))
                            judge_guidelines_text += f"\n{i}. {text}\n"
                        yield f"  Extracted {len(scorer._semantic_memory)} semantic guidelines from '{jn}'"
                    else:
                        # Try triggering lazy load with a dummy call
                        yield f"  Triggering episodic memory init for '{jn}'..."
                        try:
                            scorer(
                                inputs={"input": [{"role": "user", "content": "test query"}]},
                                outputs={"response": "test response"},
                            )
                        except Exception:
                            pass  # Expected if model endpoint unavailable

                        if hasattr(scorer, "_semantic_memory") and scorer._semantic_memory:
                            for i, guideline in enumerate(scorer._semantic_memory, 1):
                                text = getattr(guideline, "guideline_text", str(guideline))
                                judge_guidelines_text += f"\n{i}. {text}\n"
                            yield f"  Extracted {len(scorer._semantic_memory)} semantic guidelines after init"

                    # Extract episodic memory (representative examples)
                    if hasattr(scorer, "_episodic_memory") and scorer._episodic_memory:
                        episodic_text += f"\n### Episodic Memory from '{jn}'\n"
                        examples = scorer._episodic_memory
                        for i, example in enumerate(examples[:8], 1):
                            episodic_text += f"\nExample {i}:\n"
                            if hasattr(example, "inputs"):
                                episodic_text += f"  Input: {str(example.inputs)[:200]}\n"
                            if hasattr(example, "outputs"):
                                episodic_text += f"  Output: {str(example.outputs)[:200]}\n"
                            if hasattr(example, "rationale"):
                                rationale = str(example.rationale)[:200]
                                if rationale:
                                    episodic_text += f"  Rationale: {rationale}\n"
                        yield f"  Extracted {len(examples)} episodic examples from '{jn}'"

                    # Extract aligned instructions if available
                    if hasattr(scorer, "instructions") and scorer.instructions:
                        judge_guidelines_text += f"\n\nFull aligned instructions for '{jn}':\n"
                        judge_guidelines_text += scorer.instructions[:3000]
                        yield f"  Extracted aligned instructions ({len(scorer.instructions)} chars)"

            except Exception as e:
                yield f"WARNING: Failed to load judge memory: {e}"
                yield "  Skills will be generated without judge quality guidelines"

        if judge_guidelines_text:
            yield f"  Total judge guidelines: {len(judge_guidelines_text)} chars"
        if episodic_text:
            yield f"  Total episodic examples: {len(episodic_text)} chars"

        # ------------------------------------------------------------------
        # Source 3: Evaluated traces with feedback
        # ------------------------------------------------------------------
        yield "Source 3: Building trace analysis from workshop data..."
        traces_analysis_text = self._build_traces_analysis(workshop_id, mlflow_config)
        yield f"  Trace analysis: {len(traces_analysis_text)} chars"

        # ------------------------------------------------------------------
        # Generate skills via FMAPI
        # ------------------------------------------------------------------
        yield ""
        yield f"Generating skills via FMAPI (model: {model_name})..."

        prompt = SKILL_GENERATION_PROMPT_TEMPLATE.format(
            optimized_prompt=optimized_prompt,
            judge_guidelines=judge_guidelines_text or "(No judge guidelines available)",
            episodic_examples=episodic_text or "(No episodic examples available)",
            traces_analysis=traces_analysis_text or "(No trace analysis available)",
        )
        yield f"  Skill generation prompt: {len(prompt):,} chars"

        try:
            import httpx

            databricks_host = mlflow_config.databricks_host.rstrip("/")
            databricks_token = mlflow_config.databricks_token

            fmapi_url = f"{databricks_host}/serving-endpoints/{model_name}/invocations"
            headers = {
                "Authorization": f"Bearer {databricks_token}",
                "Content-Type": "application/json",
            }
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 16000,
            }

            yield f"  Calling {model_name}..."
            resp = httpx.post(
                fmapi_url,
                headers=headers,
                json=payload,
                timeout=300.0,
            )
            resp.raise_for_status()
            resp_json = resp.json()

            # Extract content from chat completions response.
            # Some models (e.g. Gemini) return content as a list of blocks
            # like [{"type": "text", "text": "..."}] rather than a plain string.
            raw_content = resp_json["choices"][0]["message"]["content"]
            if isinstance(raw_content, list):
                text_parts = []
                for block in raw_content:
                    if isinstance(block, dict):
                        text_parts.append(block.get("text", block.get("content", str(block))))
                    else:
                        text_parts.append(str(block))
                raw_output = "".join(text_parts)
            else:
                raw_output = str(raw_content)
            yield f"  FMAPI response: {len(raw_output)} chars"

        except Exception as e:
            yield f"ERROR: FMAPI call failed: {e}"
            yield {"error": f"FMAPI call failed: {e}", "success": False, "phase": "skills"}
            return

        # ------------------------------------------------------------------
        # Parse the generated skills
        # ------------------------------------------------------------------
        yield "Parsing generated skills..."
        try:
            json_match = re.search(r"\[.*\]", raw_output, re.DOTALL)
            if json_match:
                generated_skills = json.loads(json_match.group())
            else:
                generated_skills = json.loads(raw_output)

            yield f"  Parsed {len(generated_skills)} skill files"
        except (json.JSONDecodeError, ValueError) as e:
            yield f"ERROR: Failed to parse skill JSON: {e}"
            yield f"  Raw output preview: {raw_output[:500]}"
            yield {"error": f"Failed to parse skill JSON: {e}", "success": False, "phase": "skills"}
            return

        if not generated_skills:
            yield "ERROR: No skills were generated"
            yield {"error": "No skills generated", "success": False, "phase": "skills"}
            return

        # Validate and normalize each skill
        valid_skills = []
        for skill in generated_skills:
            filename = skill.get("filename", "")
            content = skill.get("content", "")
            if not filename or not content:
                yield f"  SKIP: Missing filename or content"
                continue

            if not filename.endswith(".md"):
                filename += ".md"
            filename = filename.replace(" ", "-").lower()

            # Extract name and description from YAML frontmatter
            name, description = self._extract_frontmatter(content)
            if not name:
                name = filename.replace(".md", "")
            if not description:
                description = f"Skill: {name}"

            valid_skills.append({
                "name": name,
                "description": description,
                "filename": filename,
                "content": content,
            })
            yield f"  Validated: {name} ({filename}, {len(content)} chars)"

        if not valid_skills:
            yield "ERROR: No valid skills after validation"
            yield {"error": "No valid skills after validation", "success": False, "phase": "skills"}
            return

        # ------------------------------------------------------------------
        # Write skills to Lakebase via SQLAlchemy
        # ------------------------------------------------------------------
        yield f"Writing {len(valid_skills)} skills to Lakebase..."
        try:
            written_count = self._write_skills_to_db(workshop_id, valid_skills, model_name)
            yield f"  Wrote {written_count} skills to generated_skills table"
        except Exception as e:
            yield f"ERROR: Failed to write skills to database: {e}"
            yield {"error": f"Failed to write skills: {e}", "success": False, "phase": "skills"}
            return

        # ------------------------------------------------------------------
        # Done
        # ------------------------------------------------------------------
        yield ""
        yield f"Skills generation complete: {len(valid_skills)} skills generated"
        for s in valid_skills:
            yield f"  - {s['name']}: {s['description'][:80]}"

        yield {
            "success": True,
            "phase": "skills",
            "skills_count": len(valid_skills),
            "skills": [
                {"name": s["name"], "description": s["description"], "filename": s["filename"]}
                for s in valid_skills
            ],
        }

    def _build_traces_analysis(self, workshop_id: str, mlflow_config: Any) -> str:
        """Build a compact trace analysis from workshop data for the LLM.

        Extracts user queries, response previews, tool call patterns,
        and assessment scores from align-tagged traces.
        """
        try:
            import mlflow

            filter_string = f"tags.label = 'align' AND tags.workshop_id = '{workshop_id}'"
            traces = mlflow.search_traces(
                experiment_ids=[mlflow_config.experiment_id],
                filter_string=filter_string,
                return_type="list",
            )

            if not traces:
                filter_string = f"tags.label = 'eval' AND tags.workshop_id = '{workshop_id}'"
                traces = mlflow.search_traces(
                    experiment_ids=[mlflow_config.experiment_id],
                    filter_string=filter_string,
                    return_type="list",
                )

            if not traces:
                return "No evaluated traces found for this workshop."

        except Exception as e:
            return f"Failed to load traces: {e}"

        trace_summaries = []
        tool_call_counter: Dict[str, int] = {}

        for trace in traces:
            trace_info = trace.info
            trace_data = trace.data

            # Extract user query
            user_query = ""
            try:
                request = trace_data.request
                if isinstance(request, str):
                    request = json.loads(request)
                if isinstance(request, dict):
                    inputs = request.get("input", request.get("inputs", []))
                    if isinstance(inputs, list):
                        for msg in inputs:
                            if isinstance(msg, dict) and msg.get("role") == "user":
                                user_query = msg.get("content", "")
                                break
                    elif isinstance(inputs, dict):
                        input_list = inputs.get("input", [])
                        for msg in input_list:
                            if isinstance(msg, dict) and msg.get("role") == "user":
                                user_query = msg.get("content", "")
                                break
            except Exception:
                pass

            # Extract agent response preview
            agent_response = ""
            try:
                response = trace_data.response
                if isinstance(response, str):
                    response = json.loads(response)
                if isinstance(response, dict):
                    output = response.get("output", response.get("choices", []))
                    if isinstance(output, list):
                        for item in output:
                            if isinstance(item, dict):
                                # Agent format
                                content = item.get("content", [])
                                if isinstance(content, list):
                                    for c in content:
                                        if isinstance(c, dict) and c.get("type") == "output_text":
                                            agent_response = c.get("text", "")
                                            break
                                # Chat format
                                msg = item.get("message", {})
                                if isinstance(msg, dict) and msg.get("content"):
                                    agent_response = msg["content"]
                            if agent_response:
                                break
            except Exception:
                pass

            # Extract tool calls from spans
            tool_calls = []
            try:
                for span in trace_data.spans:
                    span_name = getattr(span, "name", "")
                    span_type = getattr(span, "span_type", "")
                    if span_type == "TOOL" or span_type == "FUNCTION":
                        tool_name = span_name.split(" (")[0] if " (" in span_name else span_name
                        tool_calls.append(tool_name)
                        tool_call_counter[tool_name] = tool_call_counter.get(tool_name, 0) + 1
            except Exception:
                pass

            # Extract assessments
            assessments = []
            try:
                raw_assessments = None
                for location in [trace, trace_info, trace_data]:
                    if hasattr(location, "assessments") and location.assessments:
                        raw_assessments = location.assessments
                        break

                if raw_assessments:
                    for assessment in raw_assessments:
                        a_info = {"name": getattr(assessment, "name", "unknown")}
                        score = None
                        for score_attr in ["numeric_value", "value", "score"]:
                            val = getattr(assessment, score_attr, None)
                            if val is not None:
                                try:
                                    score = float(val)
                                    break
                                except (ValueError, TypeError):
                                    pass
                        if score is not None:
                            a_info["score"] = score
                        rationale = getattr(assessment, "rationale", None)
                        if rationale:
                            a_info["rationale"] = str(rationale)[:300]
                        assessments.append(a_info)
            except Exception:
                pass

            trace_summaries.append({
                "trace_id": trace_info.trace_id,
                "user_query": user_query[:500],
                "response_preview": agent_response[:300],
                "tool_calls": tool_calls,
                "assessments": assessments,
            })

        # Build the analysis text
        analysis = f"Total evaluated traces: {len(trace_summaries)}\n"

        # Tool usage frequency
        if tool_call_counter:
            analysis += "\n### Tool Usage Frequency\n"
            for tool, count in sorted(tool_call_counter.items(), key=lambda x: -x[1]):
                analysis += f"- {tool}: {count} calls across all traces\n"

        # Common tool co-occurrence patterns
        tool_sequences = [
            tuple(sorted(set(ts["tool_calls"])))
            for ts in trace_summaries if ts["tool_calls"]
        ]
        if tool_sequences:
            sequence_counts = Counter(tool_sequences)
            analysis += "\n### Common Tool Co-occurrence Patterns\n"
            for seq, count in sequence_counts.most_common(10):
                analysis += f"- {' + '.join(seq)}: {count} traces\n"

        # Representative queries
        analysis += "\n### Representative Queries\n"
        for ts in trace_summaries[:10]:
            if ts["user_query"]:
                analysis += f'- "{ts["user_query"][:200]}"\n'

        # Assessment score distribution
        all_scores = []
        for ts in trace_summaries:
            for a in ts["assessments"]:
                if "score" in a:
                    all_scores.append(a["score"])
        if all_scores:
            score_counts = Counter(all_scores)
            analysis += "\n### Assessment Score Distribution\n"
            for score in sorted(score_counts.keys()):
                analysis += f"- Score {score}: {score_counts[score]} assessments\n"
            analysis += f"- Mean score: {sum(all_scores) / len(all_scores):.2f}\n"

        # Sample rationales from high and low scoring traces
        scored = []
        for ts in trace_summaries:
            for a in ts["assessments"]:
                if "score" in a and "rationale" in a:
                    scored.append({
                        "query": ts["user_query"][:150],
                        "score": a["score"],
                        "rationale": a["rationale"][:200],
                    })
        if scored:
            scored.sort(key=lambda x: x["score"], reverse=True)
            analysis += "\n### High-Scoring Example Rationales\n"
            for s in scored[:3]:
                analysis += f'- Score {s["score"]}: "{s["query"][:100]}"\n'
                analysis += f'  Rationale: {s["rationale"]}\n'
            analysis += "\n### Low-Scoring Example Rationales\n"
            for s in scored[-3:]:
                analysis += f'- Score {s["score"]}: "{s["query"][:100]}"\n'
                analysis += f'  Rationale: {s["rationale"]}\n'

        return analysis

    def _extract_frontmatter(self, content: str) -> Tuple[str, str]:
        """Extract name and description from YAML frontmatter in skill content."""
        name = ""
        description = ""

        fm_match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
        if not fm_match:
            return name, description

        fm_text = fm_match.group(1)

        # Extract name
        name_match = re.search(r"^name:\s*(.+)$", fm_text, re.MULTILINE)
        if name_match:
            name = name_match.group(1).strip().strip("'\"")

        # Extract description (may be multi-line with > or |)
        desc_match = re.search(
            r"^description:\s*>?\s*\n((?:\s+.+\n?)+)", fm_text, re.MULTILINE
        )
        if desc_match:
            description = " ".join(
                line.strip() for line in desc_match.group(1).strip().splitlines()
            )
        else:
            desc_match = re.search(r"^description:\s*(.+)$", fm_text, re.MULTILINE)
            if desc_match:
                description = desc_match.group(1).strip().strip("'\"")

        return name, description

    def _write_skills_to_db(
        self,
        workshop_id: str,
        skills: List[Dict[str, str]],
        model_name: str,
    ) -> int:
        """Write generated skills to the database.

        Clears existing skills for this workshop and inserts the new set.
        Uses SQLAlchemy ORM to stay consistent with the rest of the codebase.
        """
        from server.database import GeneratedSkillDB, SessionLocal

        db = SessionLocal()
        try:
            # Clear existing skills for this workshop
            db.query(GeneratedSkillDB).filter(
                GeneratedSkillDB.workshop_id == workshop_id
            ).delete()
            db.commit()

            # Insert new skills
            for skill in skills:
                record = GeneratedSkillDB(
                    id=str(uuid.uuid4()),
                    workshop_id=workshop_id,
                    name=skill["name"],
                    description=skill["description"],
                    filename=skill["filename"],
                    content=skill["content"],
                    generation_model=model_name,
                )
                db.add(record)

            db.commit()
            return len(skills)
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

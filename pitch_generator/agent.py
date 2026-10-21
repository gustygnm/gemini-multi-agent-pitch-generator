# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
/**
 * @file agent.py
 * @description Multi-agent ADK workflow definition and service-injected execution
 *   engine for the Agentic Pitch Generator.
 *
 * Why: Defines the Google ADK specialist agent team (`creative_director`,
 * `brand_strategist`, `copywriter`, `visual_director`), structured payload contracts,
 * `LoopGuard` DAG validation, `assemble = JoinNode(name="assemble")`, `package`,
 * `root_agent`, and `app` running on Gemini Enterprise Agent Platform (`gemini-3.8-flash`)
 * while providing lightweight stdlib fallback shims when `google-adk` / `google-genai`
 * wheels are not installed locally. Also exposes `run_pitch_workflow` and
 * `resume_pitch_workflow` wired through `ServiceContainer` so local runs use live
 * Cloud models and offline test suites execute deterministically.
 */
"""

from __future__ import annotations

# ==============================================================================
# LAB GUIDEPOST INDEX (`pitch_generator/agent.py`)
#   - Step 1a: `# [Guidepost — Step 1a: Author and Attach the Brand Guidelines Skill]`
#   - Step 1b: `# [Guidepost — Step 1b: Skill Evaluation Harness]`
#   - Step 1c: `# [Guidepost — Step 1c: Remote A2A Visual Director Service & Client]`
#   - Step 3a: `# [Guidepost — Step 3a: PreToolUse Lifecycle Hooks]`
#   - Step 3c: `# [Guidepost — Step 3c: Human-in-the-Loop (HITL) Approval Gate]`
#   - Step 4a: `# [Guidepost — Step 4a: Tokenomics & Context Optimization]`
# (Tip: All specialist agents, workflow edges, and lab guideposts are in the top
#  ~320 lines of this file—read lines 1–350 in a single `view_file` call!)
# ==============================================================================

from collections.abc import Sequence
import copy
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import re
import time
from typing import Any

from pitch_generator.app_utils.adk_primitives import (
    Agent,
    App,
    ArtDirectionPayload,
    CampaignConcept,
    CircularLoopError,
    ConceptPayload,
    Context,
    CopyPayload,
    Event,
    Gemini,
    JoinNode,
    LoopGuard,
    PayloadValidationError,
    PitchPackage,
    RemoteA2aAgent,
    RequestInput,
    VisualPayload,
    Workflow,
    _DualSyncAsyncEventStream,
    _GenAITypesShim,
    _SessionData,
    _extract_json_Candidate,
    _key_visual,
    _node_name,
    _normalize_directed_edges,
    extract_json_payload,
    node,
    parse_json_payload,
    parse_payload,
    strip_markdown_fences,
    types,
)
from pitch_generator.app_utils.a2a import (
    AGENT_CARD_WELL_KNOWN_PATH,
    A2aAgentExecutor,
    A2aAgentExecutorConfig,
    AgentCardBuilder,
    DefaultRequestHandler,
    include_artifacts_in_a2a_event_interceptor,
)
from pitch_generator.app_utils.services import (
    ServiceContainer,
    get_default_services,
)
from pitch_generator.config import (
    PitchConfig,
    get_config,
)

MODEL: str = get_config().flash_model
IMAGE_MODEL: str = get_config().image_model


def _model(config: PitchConfig | None = None) -> Gemini:
    """
    /**
     * Construct the default `Gemini` model configuration (`gemini-3.8-flash`) for
     * specialist agents.
     *
     * Why: Centralizes `location="global"` and `HttpRetryOptions(attempts=3)` so every
     * specialist agent shares resilient retry settings on Gemini Enterprise Agent Platform.
     *
     * @param config Optional `PitchConfig` override.
     * @return Configured `Gemini` model descriptor.
     */
    """
    cfg = config or get_config()
    return Gemini(
        model=cfg.flash_model,
        client_kwargs={"location": cfg.location},
        retry_options=types.HttpRetryOptions(attempts=3),
    )


_build_model = _model


async def generate_key_visual(
    art_direction: str,
    tool_context: Context | None = None,
    *,
    simulate_no_image: bool = False,
) -> dict[str, Any]:
    """
    /**
     * Generate a 16:9 key visual PNG from art direction notes and save it to session artifacts.
     *
     * Why: Attached to `visual_director` (`tools=[generate_key_visual]`) so the Visual
     * Director specialist can synthesize a key visual image via the configured
     * `image_model` (`gemini-nano-banana-2.1`) and store it in the session artifact store.
     *
     * @param art_direction Art direction prompt describing palette, lighting, composition, and subject.
     * @param tool_context Optional ADK `Context` / `ToolContext` for saving the image artifact.
     * @param simulate_no_image If True, simulates an empty image response to test error handling.
     * @return Metadata dictionary with `filename`, `version`, `bytes`, and `mime_type`.
     */
    """
    import mimetypes

    ctx = tool_context if tool_context is not None else Context()
    cfg = ctx.services.config if hasattr(ctx, "services") and ctx.services else get_config()
    if (
        simulate_no_image
        or getattr(ctx, "simulate_no_image", False)
        or not isinstance(art_direction, str)
        or not art_direction.strip()
    ):
        raise ValueError(f"{cfg.image_model} returned no image for: {str(art_direction)[:120]}")

    image_bytes, mime_type = ctx.services.llm.generate_image(
        art_direction.strip(),
        model=cfg.image_model,
    )
    if not image_bytes:
        raise ValueError(f"{cfg.image_model} returned no image for: {art_direction[:120]}")

    ext = mimetypes.guess_extension(mime_type or "") or ".png"
    filename = f"key_visual{ext}"
    image_part = types.Part(
        inline_data=types.Blob(data=image_bytes, mime_type=mime_type)
    )
    version = await ctx.save_artifact(filename, image_part)
    ctx.session.events.append(
        Event(
            author="visual_director",
            content=types.Content(role="model", parts=[image_part]),
        )
    )
    return {
        "filename": filename,
        "version": version,
        "bytes": len(image_bytes),
        "mime_type": mime_type,
    }


def build_specialist_team(config: PitchConfig | None = None) -> dict[str, Agent]:
    """
    /**
     * Instantiate the four domain specialist agents for the Pitch Generator team.
     *
     * Why: Provides a deterministic factory that binds each specialist role
     * (`creative_director`, `copywriter`, `brand_strategist`, `visual_director`) to
     * a focused single-responsibility instruction and a unique `output_key` so
     * downstream graph nodes can read each specialist's output without collision.
     *
     * @param config Optional `PitchConfig` used to resolve model settings.
     * @return Dictionary mapping role name to its configured `Agent` instance.
     */
    """
    model = _model(config)
    return {
        "creative_director": Agent(
            name="creative_director",
            model=model,
            description="Turns a raw campaign topic into a punchy concept line and strategic rationale.",
            instruction=(
                "You are the Creative Director. Turn the campaign idea you are given into "
                "ONE punchy campaign concept line accompanied by a clear rationale explanation."
            ),
            output_key="creative_director",
        ),
        "copywriter": Agent(
            name="copywriter",
            model=model,
            description="Writes a concise, high-impact social media caption under 25 words.",
            instruction=(
                "You are the Copywriter. Write ONE short social caption for the campaign "
                "concept you are given. Strictly under 25 words with an energetic call to action."
            ),
            output_key="copywriter",
        ),
        "brand_strategist": Agent(
            name="brand_strategist",
            model=model,
            description="Defines campaign positioning, target audience, and brand tone for the campaign concept.",
            instruction=(
                "You are the Brand Strategist. Define the campaign positioning, "
                "target audience, and brand tone for the campaign concept."
            ),
            output_key="brand_strategist",
        ),
        "visual_director": Agent(
            name="visual_director",
            model=model,
            description="Translates the campaign concept into art direction and key visuals.",
            instruction="You are the Visual Director.",
            tools=[generate_key_visual],
            output_key="visual_director",
        ),
    }


create_specialist_agents = build_specialist_team
get_specialist_agents = build_specialist_team

SPECIALIST_AGENTS: dict[str, Agent] = build_specialist_team()

creative_director: Agent = SPECIALIST_AGENTS["creative_director"]
copywriter: Agent = SPECIALIST_AGENTS["copywriter"]
brand_strategist: Agent = SPECIALIST_AGENTS["brand_strategist"]
visual_director: Agent = SPECIALIST_AGENTS["visual_director"]


def get_specialist_agent(role: str) -> Agent:
    """
    /**
     * Look up a single specialist `Agent` by role identifier.
     *
     * Why: Fails fast with a descriptive `KeyError` when an invalid or unknown
     * specialist role is requested instead of silently returning `None`.
     *
     * @param role Specialist role name (e.g., `"creative_director"`, `"copywriter"`).
     * @return Matching `Agent` instance.
     */
    """
    if not isinstance(role, str) or not role.strip():
        raise ValueError("Specialist role name must be a non-empty string")
    clean_role = role.strip()
    if clean_role == "market_researcher":
        return SPECIALIST_AGENTS["brand_strategist"]
    if clean_role not in SPECIALIST_AGENTS:
        raise KeyError(
            f"Unknown specialist role {role!r}. Expected one of {tuple(SPECIALIST_AGENTS.keys())}."
        )
    return SPECIALIST_AGENTS[clean_role]


def run_specialist_team(
    brief: str,
    *,
    services: ServiceContainer | None = None,
    config: PitchConfig | None = None,
) -> dict[str, str]:
    """
    /**
     * Execute all four specialist agents sequentially/in-parallel against a campaign brief.
     *
     * Why: Allows learners and automated tests to verify that all four specialist roles
     * (`creative_director`, `copywriter`, `brand_strategist`, `visual_director`) generate
     * domain-appropriate outputs offline via dependency-injected `ServiceContainer`.
     *
     * @param brief Non-empty campaign topic brief.
     * @param services Optional injected `ServiceContainer` for offline LLM execution.
     * @param config Optional `PitchConfig` override.
     * @return Dictionary mapping each specialist's `output_key` to its generated text.
     */
    """
    if not isinstance(brief, str) or not brief.strip():
        raise ValueError("Campaign brief must be a non-empty string")

    clean_brief = brief.strip()
    active_services = services or get_default_services(config)
    team = build_specialist_team(active_services.config)
    model_id = active_services.config.flash_model

    concept_out = active_services.llm.generate_text(
        clean_brief,
        system_instruction=team["creative_director"].instruction,
        model=model_id,
    )
    copy_out = active_services.llm.generate_text(
        concept_out,
        system_instruction=team["copywriter"].instruction,
        model=model_id,
    )
    copy_words = copy_out.split()
    if len(copy_words) > 25:
        copy_out = " ".join(copy_words[:24])

    strategy_out = active_services.llm.generate_text(
        concept_out,
        system_instruction=team["brand_strategist"].instruction,
        model=model_id,
    )
    visual_out = active_services.llm.generate_text(
        concept_out,
        system_instruction=team["visual_director"].instruction,
        model=model_id,
    )

    return {
        "creative_director": concept_out,
        "copywriter": copy_out,
        "brand_strategist": strategy_out,
        "visual_director": visual_out,
    }


execute_specialists = run_specialist_team
run_specialists = run_specialist_team

# [Guidepost — Step 1a: Author and Attach the Brand Guidelines Skill]
@dataclass
class Skill:
    """
    /**
     * Reusable ADK skill definition loaded on demand from an external directory.
     *
     * @param name Unique skill name (e.g., "brand-guidelines").
     * @param description Brief description of skill purpose and usage instructions.
     * @param body Full Markdown instruction body.
     * @param path Optional filesystem path to the skill source file.
     */
    """

    name: str
    description: str
    body: str
    path: str | Path | None = None


class SkillToolset:
    """
    /**
     * ADK Toolset shim wrapping one or more on-demand skills.
     *
     * Why: Exposes `load_skill(skill_name)` so specialist agents can load the
     * markdown body of domain skills on demand, keeping base system prompts concise.
     */
    """

    def __init__(self, skills: Sequence[Skill] | None = None) -> None:
        self.skills: dict[str, Skill] = {s.name: s for s in (skills or [])}
        self.loaded_skills: list[str] = []

    def load_skill(self, skill_name: str) -> str:
        """
        /**
         * Load the full markdown body of a named skill on demand.
         *
         * Why: Dynamically loads the skill body into working context when requested
         * by the agent while tracking skill activation in `loaded_skills`.
         *
         * @param skill_name The name of the registered skill.
         * @return Markdown body of the skill.
         */
        """
        if skill_name not in self.skills:
            available = ", ".join(repr(k) for k in self.skills.keys())
            raise KeyError(f"Skill {skill_name!r} not found. Available skills: [{available}]")
        if skill_name not in self.loaded_skills:
            self.loaded_skills.append(skill_name)
        return self.skills[skill_name].body

    def get_tools(self) -> list[Any]:
        """
        /**
         * Return the callable tool functions provided by this skill toolset.
         *
         * Why: Standard tool discovery hook for ADK agents to register the load_skill tool.
         *
         * @return List containing the load_skill callable.
         */
        """
        return [self.load_skill]


def load_skill_from_dir(skill_dir: str | Path) -> Skill:
    """
    /**
     * Parses a SKILL.md file in the specified directory into a Skill object.
     *
     * Why: Reads YAML frontmatter and Markdown body from disk so agent skills can
     * be authored as modular, standalone files external to Python code.
     *
     * @param skill_dir Directory containing SKILL.md (or direct path to SKILL.md).
     * @return Initialized Skill object.
     */
    """
    target = Path(skill_dir)
    skill_file = target / "SKILL.md" if target.is_dir() else target
    if not skill_file.is_file():
        raise FileNotFoundError(f"SKILL.md not found at {skill_file}")

    content = skill_file.read_text(encoding="utf-8")
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, flags=re.DOTALL)
    if fm_match:
        fm_text = fm_match.group(1)
        body = fm_match.group(2).strip()
        name = ""
        description = ""
        for line in fm_text.splitlines():
            line_str = line.strip()
            if line_str.startswith("name:"):
                name = line_str.split("name:", 1)[1].strip()
            elif line_str.startswith("description:"):
                description = line_str.split("description:", 1)[1].strip()
        return Skill(
            name=name or skill_file.parent.name,
            description=description,
            body=body,
            path=skill_file,
        )
    return Skill(
        name=skill_file.parent.name,
        description="",
        body=content.strip(),
        path=skill_file,
    )


def load_brand_skill() -> Skill:
    """
    /**
     * Convenience loader for the house brand guidelines skill.
     *
     * Why: Centralizes path resolution for the canonical brand-guidelines skill
     * used across the Visual Director workflow.
     *
     * @return Loaded brand-guidelines Skill instance.
     */
    """
    skill_dir = Path(__file__).resolve().parent / "skills" / "brand-guidelines"
    return load_skill_from_dir(skill_dir)


brand_guidelines_skill: Skill = load_brand_skill()
brand_toolset: SkillToolset = SkillToolset([brand_guidelines_skill])

visual_director.tools = [brand_toolset, generate_key_visual]
visual_director.instruction = (
    "You are the Visual Director. First, call load_skill(\"brand-guidelines\") to load "
    "our house style. Next, compose high-fidelity art direction adhering to our house "
    "brand guidelines (Palette, Light, Composition, and Subject) while avoiding all forbidden "
    "elements in the Never section. Then, call generate_key_visual with your art direction prompt. "
    "Output ONLY the art direction text."
)
SPECIALIST_AGENTS["visual_director"] = visual_director

# [Guidepost — Step 1b: Skill Evaluation Harness]
FORBIDDEN_BRAND_PATTERNS: dict[str, str] = {
    "neon": r"\bneon\b",
    "fluorescent": r"\bfluorescent\b",
    "cyan": r"\bcyan\b",
    "magenta": r"\bmagenta\b",
    "watermark": r"\bwatermarks?\b",
    "logo": r"\blogos?\b",
    "3d_render": r"\b(?:3d\s+renders?|render(?:ed|ing)?)\b",
    "ring_light": r"\bring\s+lights?\b",
}


@dataclass
class SkillEvalResult:
    """
    /**
     * Diagnostic report produced by `evaluate_brand_skill`.
     *
     * Why: Provides a structured evaluation payload containing the total brand
     * compliance score, boolean pass/fail status, skill activation verification,
     * list of forbidden violations, and individual pillar scores.
     *
     * @param score Overall brand compliance score between 0.0 and 1.0.
     * @param passed Whether the art direction passed brand compliance thresholds.
     * @param loaded_skill Whether the required skill was loaded during execution.
     * @param violations List of identified forbidden style violation tokens.
     * @param pillar_scores Dictionary mapping each brand pillar to its score.
     */
    """

    score: float
    passed: bool
    loaded_skill: bool
    violations: list[str] = field(default_factory=list)
    pillar_scores: dict[str, float] = field(default_factory=dict)


def evaluate_brand_skill(
    art_direction: str,
    loaded_skills: Sequence[str] = ("brand-guidelines",),
) -> SkillEvalResult:
    """
    /**
     * Scores an art direction text for adherence to house brand guidelines.
     *
     * Why: Provides an automated, deterministic evaluation harness that verifies
     * skill activation and pillar coverage without making expensive LLM calls.
     *
     * @param art_direction Generated art direction string to inspect.
     * @param loaded_skills Sequence of skill names recorded during generation.
     * @return Evaluated SkillEvalResult report.
     */
    """
    is_skill_loaded = "brand-guidelines" in loaded_skills
    if not is_skill_loaded:
        return SkillEvalResult(
            score=0.0,
            passed=False,
            loaded_skill=False,
            violations=["skill_not_loaded:brand-guidelines"],
            pillar_scores={"palette": 0.0, "light": 0.0, "composition": 0.0, "subject": 0.0},
        )

    text_lower = art_direction.lower()

    palette_match = any(
        term in text_lower
        for term in ("palette", "indigo", "amber", "slate", "terracotta")
    )
    light_match = any(
        term in text_lower
        for term in ("light", "raking", "golden-hour", "golden hour", "shadow")
    )
    comp_match = any(
        term in text_lower
        for term in ("composition", "off-center", "negative space", "depth of field")
    )
    subj_match = any(
        term in text_lower
        for term in ("subject", "photographic", "authentic", "handcrafted")
    )

    pillar_scores = {
        "palette": 0.25 if palette_match else 0.0,
        "light": 0.25 if light_match else 0.0,
        "composition": 0.25 if comp_match else 0.0,
        "subject": 0.25 if subj_match else 0.0,
    }

    base_score = sum(pillar_scores.values())

    violations: list[str] = []
    for pattern_name, pattern_regex in FORBIDDEN_BRAND_PATTERNS.items():
        if re.search(pattern_regex, text_lower):
            violations.append(pattern_name)

    penalty = len(violations) * 0.25
    final_score = max(0.0, round(base_score - penalty, 2))
    passed = bool(is_skill_loaded and not violations and final_score >= 0.75)

    return SkillEvalResult(
        score=final_score,
        passed=passed,
        loaded_skill=True,
        violations=violations,
        pillar_scores=pillar_scores,
    )


def run_eval_suite(
    cases: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    /**
     * Executes the brand skill evaluation suite across benchmark test cases.
     *
     * Why: Validates compliant, off-brand, and unloaded-skill test cases to ensure
     * the evaluation harness behaves reliably across edge cases.
     *
     * @param cases Optional list of custom test case dicts.
     * @return Summary dictionary containing total, passed, results, and all_expectations_met.
     */
    """
    if cases is None:
        cases = [
            {
                "name": "compliant_house_style",
                "art_direction": (
                    "A single realistic photographic hero subject captured with shallow depth of field, "
                    "positioned off-center with generous negative space. Color palette anchored in deep indigo, "
                    "slate, warm amber, and terracotta. Illuminated by warm raking golden-hour light casting soft shadows."
                ),
                "loaded_skills": ("brand-guidelines",),
                "expected_pass": True,
            },
            {
                "name": "off_brand_neon_3d",
                "art_direction": (
                    "Glossy 3D render lit by a fluorescent cyan ring light with a giant logo watermark."
                ),
                "loaded_skills": ("brand-guidelines",),
                "expected_pass": False,
            },
            {
                "name": "unloaded_skill",
                "art_direction": (
                    "A beautiful photo with golden-hour light and indigo palette."
                ),
                "loaded_skills": (),
                "expected_pass": False,
            },
        ]

    results: list[dict[str, Any]] = []
    all_met = True

    for case in cases:
        name = case.get("name", "unnamed")
        art = case.get("art_direction", "")
        skills = case.get("loaded_skills", ("brand-guidelines",))
        exp_pass = case.get("expected_pass", True)

        eval_res = evaluate_brand_skill(art, loaded_skills=skills)
        matched_expectation = eval_res.passed == exp_pass
        if not matched_expectation:
            all_met = False

        results.append(
            {
                "name": name,
                "passed": eval_res.passed,
                "score": eval_res.score,
                "loaded_skill": eval_res.loaded_skill,
                "violations": eval_res.violations,
                "pillar_scores": eval_res.pillar_scores,
                "matches_expectation": matched_expectation,
            }
        )

    passed_count = sum(1 for r in results if r["passed"])
    return {
        "total": len(results),
        "passed": passed_count,
        "results": results,
        "all_expectations_met": all_met,
    }

# [Guidepost — Step 1c: Remote A2A Visual Director Service & Client]
def build_visual_director_card(
    rpc_url: str = "http://localhost:8801/a2a/visual_director",
) -> dict[str, Any]:
    """
    /**
     * Builds the A2A Agent Card dictionary for the standalone Visual Director service.
     *
     * Why: Publishes capability metadata and preferred JSONRPC transport over
     * /.well-known/agent-card.json so remote clients can discover and invoke the agent.
     *
     * @param rpc_url Public URL of the Visual Director JSON-RPC endpoint.
     * @return Dictionary representing the A2A Agent Card.
     */
    """
    builder = AgentCardBuilder(agent=visual_director, rpc_url=rpc_url)
    card = dict(builder.build())
    card["name"] = "visual_director"
    card["url"] = rpc_url
    card["preferredTransport"] = "JSONRPC"
    return card


def build_a2a_visual_director_app(
    services: ServiceContainer | None = None,
    rpc_url: str = "http://localhost:8801/a2a/visual_director",
) -> Any:
    """
    /**
     * Constructs a standalone FastAPI application serving the Visual Director agent over A2A.
     *
     * Why: Enables independent deployment and scaling of the computationally expensive
     * image generation microservice with artifact event interception.
     *
     * @param services Optional injected ServiceContainer.
     * @param rpc_url JSON-RPC endpoint URL.
     * @return Configured PitchFastAPIApp instance.
     */
    """
    from pitch_generator.fast_api_app import PitchFastAPIApp

    active_services = services or get_default_services()
    app_instance = PitchFastAPIApp(services=active_services)

    card = build_visual_director_card(rpc_url=rpc_url)
    executor_config = A2aAgentExecutorConfig(
        execute_interceptors=[include_artifacts_in_a2a_event_interceptor]
    )
    executor = A2aAgentExecutor(
        runner=None,
        force_new_version=True,
        config=executor_config,
    )
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store={},
        agent_card=card,
    )
    app_instance.register_a2a_handler("/a2a/visual_director", handler, card)
    return app_instance


def _cloud_run_client(base_url: str) -> Any:
    """
    /**
     * Constructs an HTTP client configured with OIDC bearer authentication for Cloud Run.
     *
     * Why: Authenticates requests to private Cloud Run services when given an https:// URL,
     * while allowing unauthenticated local development when given an http:// URL.
     *
     * @param base_url Target service base URL.
     * @return Configured httpx.Client or None for local/unconfigured URLs.
     */
    """
    if not base_url or not isinstance(base_url, str):
        return None
    clean_url = base_url.strip()
    if not clean_url.startswith("https://"):
        return None

    try:
        import httpx
        from google.auth.transport.requests import Request
        from google.oauth2 import id_token

        auth_req = Request()
        token = id_token.fetch_id_token(auth_req, clean_url)
        headers = {"Authorization": f"Bearer {token}"}
        return httpx.Client(base_url=clean_url, headers=headers, timeout=600.0)
    except Exception:
        try:
            import httpx

            return httpx.Client(base_url=clean_url, timeout=600.0)
        except Exception:
            class _CloudRunClientShim:
                """
                /**
                 * Lightweight fallback HTTP client matching httpx.Client for offline tests.
                 *
                 * Why: Allows unit tests to verify timeout and headers without httpx wheel.
                 */
                """

                def __init__(self, base_url: str, timeout: float = 600.0, headers: dict[str, str] | None = None) -> None:
                    self.base_url = base_url
                    self.timeout = timeout
                    self.headers = headers or {}

            return _CloudRunClientShim(base_url=clean_url, timeout=600.0)


def _pitch_parts_only(part: Any) -> Any:
    """
    /**
     * Stream filter that drops internal function calling metadata from A2A responses.
     *
     * Why: Intermediate tool calls and responses should not pollute the client-facing
     * pitch output, while text and inline image PNG artifacts must be preserved.
     *
     * @param part Part object or dictionary to inspect.
     * @return Filtered part if allowed, or None/False if rejected.
     */
    """
    if part is None:
        return None
    if isinstance(part, dict):
        if "function_call" in part or "function_response" in part:
            return None
        return part
    if hasattr(part, "function_call") and getattr(part, "function_call", None) is not None:
        return None
    if hasattr(part, "function_response") and getattr(part, "function_response", None) is not None:
        return None
    return part


visual_director_url: str = get_config().visual_director_url.rstrip("/")

remote_visual_director: RemoteA2aAgent = RemoteA2aAgent(
    name="visual_director",
    description="Translates the campaign concept into art direction and key visuals over A2A.",
    agent_card=f"{visual_director_url}/.well-known/agent-card.json",
    httpx_client=_cloud_run_client(visual_director_url),
    genai_part_converter=_pitch_parts_only,
    output_key="visual_director",
)

assemble = JoinNode(name="assemble")


def package(
    ctx_or_input: Any,
    node_input: dict[str, Any] | None = None,
) -> _DualSyncAsyncEventStream:
    """
    /**
     * Format and validate the joined specialist outputs into the final campaign pitch.
     *
     * Why: Acts as the terminal node after `assemble = JoinNode(name="assemble")`.
     * Explicitly checks that every expected upstream branch produced non-empty output
     * (raising `ValueError("nothing reached the join from: ...")` if any branch is
     * missing or empty) so silent branch failures are caught immediately. Supports both
     * `package(node_input)` and `package(ctx, node_input)` signatures, strips Markdown
     * code fences from branch values, and attaches `_key_visual(ctx)` when present.
     *
     * @param ctx_or_input Either a `node_input` dict (1-arg form) or `Context` (2-arg form).
     * @param node_input Dictionary of upstream outputs when `ctx_or_input` is a `Context`.
     * @return Dual sync/async stream yielding `Event(content=...)` and `Event(output=...)`.
     */
    """
    if node_input is None:
        if not isinstance(ctx_or_input, dict):
            raise ValueError("node_input must be a non-empty dictionary")
        ctx: Context | None = None
        actual_input = ctx_or_input
    else:
        if not isinstance(node_input, dict):
            raise ValueError("node_input must be a non-empty dictionary")
        ctx = ctx_or_input if isinstance(ctx_or_input, Context) else None
        actual_input = node_input

    if not actual_input:
        raise ValueError("nothing reached the join from: creative_director, copywriter")

    required_keys = ["creative_director", "copywriter"]
    missing_or_empty = [
        k
        for k in required_keys
        if k not in actual_input or not str(actual_input.get(k) or "").strip()
    ]
    extra_empty = [
        name
        for name, value in actual_input.items()
        if name not in missing_or_empty and not str(value or "").strip()
    ]
    all_empty = missing_or_empty + extra_empty
    if all_empty:
        raise ValueError(f"nothing reached the join from: {', '.join(all_empty)}")

    concept_val = strip_markdown_fences(str(actual_input["creative_director"])).strip()
    copy_val = strip_markdown_fences(str(actual_input["copywriter"])).strip()
    sections = [
        f"CONCEPT\n{concept_val}",
        f"COPY\n{copy_val}",
    ]
    if "brand_strategist" in actual_input and actual_input["brand_strategist"]:
        brand_val = strip_markdown_fences(str(actual_input["brand_strategist"])).strip()
        sections.append(f"BRAND STRATEGY\n{brand_val}")
    if "visual_director" in actual_input and actual_input["visual_director"]:
        art_val = strip_markdown_fences(str(actual_input["visual_director"])).strip()
        sections.append(f"ART DIRECTION\n{art_val}")

    pitch = "\n\n".join(sections)
    parts: list[Any] = [types.Part(text=pitch)]
    image = _key_visual(ctx)
    if image is not None:
        parts.append(types.Part(inline_data=image))

    events = [
        Event(author="package", content=types.Content(role="model", parts=parts)),
        Event(author="package", output=pitch),
    ]
    return _DualSyncAsyncEventStream(events)


loop_guard = LoopGuard(max_iterations=10)

# [Guidepost — Step 3a: PreToolUse Lifecycle Hooks]
DEFAULT_TOOL_ALLOWLIST: frozenset[str] = frozenset({"load_skill", "generate_key_visual"})


class ToolAuthorizationError(PermissionError):
    """
    /**
     * Exception raised when a tool invocation fails PreToolUse security validation.
     *
     * Why: Prevents execution of unauthorized tools, malicious parameter overrides,
     * path traversal exploits, or prompt injections in the multi-agent workflow.
     *
     * @param message Human-readable error message.
     * @param tool_name Name of the tool whose execution was blocked.
     * @param reason Security policy violation reason.
     */
    """

    def __init__(self, message: str = "", tool_name: str = "", reason: str = "") -> None:
        """
        /**
         * Initialize ToolAuthorizationError with tool name and failure reason.
         *
         * Why: Captures diagnostic audit details for error telemetry and exception handlers.
         *
         * @param message Human-readable error message.
         * @param tool_name Name of the tool that triggered authorization failure.
         * @param reason Detailed explanation of security violation.
         */
        """
        self.tool_name: str = tool_name
        self.reason: str = reason or message
        super().__init__(message or f"Tool '{tool_name}' failed authorization: {self.reason}")


@dataclass
class HookDecision:
    """
    /**
     * Structured outcome of a PreToolUse hook security validation.
     *
     * Why: Encapsulates authorization status, security violation taxonomy, sanitized
     * arguments, and boolean evaluation flags required by ADK lifecycle callbacks.
     */
    """

    allowed: bool
    tool_name: str
    reason: str
    violation_type: str | None = None
    sanitized_args: dict[str, Any] = field(default_factory=dict)

    @property
    def is_allowed(self) -> bool:
        """
        /**
         * Boolean property indicating whether the tool invocation is permitted.
         *
         * Why: Compatibility alias for ADK tool dispatch gates expecting `.is_allowed`.
         *
         * @return True if invocation is permitted, False otherwise.
         */
        """
        return self.allowed

    @property
    def permitted(self) -> bool:
        """
        /**
         * Boolean property indicating whether the tool invocation is permitted.
         *
         * Why: Compatibility alias for ADK tool dispatch gates expecting `.permitted`.
         *
         * @return True if invocation is permitted, False otherwise.
         */
        """
        return self.allowed

    def __bool__(self) -> bool:
        """
        /**
         * Truthy evaluation of the HookDecision.
         *
         * Why: Allows direct conditional checks such as `if decision:` in dispatch code.
         *
         * @return True if invocation is permitted, False otherwise.
         */
        """
        return self.allowed


class PreToolUseHook:
    """
    /**
     * Fail-closed lifecycle hook intercepting tool calls before dispatch.
     *
     * Why: Enforces defense-in-depth across the agent team by verifying tool allowlists,
     * stripping path traversal patterns, blocking prompt injections, and auditing all calls.
     */
    """

    def __init__(
        self,
        allowlist: Sequence[str] | set[str] | frozenset[str] | None = None,
        raise_on_block: bool = False,
    ) -> None:
        """
        /**
         * Initialize PreToolUseHook with an optional tool allowlist and error policy.
         *
         * Why: Supports both logging/blocking modes and custom per-agent tool scopes.
         *
         * @param allowlist Optional custom collection of permitted tool names.
         * @param raise_on_block If True, raises ToolAuthorizationError on blocked calls.
         */
        """
        if allowlist is None:
            self.allowlist: frozenset[str] = DEFAULT_TOOL_ALLOWLIST
        else:
            self.allowlist = frozenset(allowlist)
        self.raise_on_block: bool = raise_on_block
        self.audit_log: list[dict[str, Any]] = []

    def validate_tool_call(
        self,
        tool_name: str | None,
        tool_args: dict[str, Any] | None = None,
    ) -> HookDecision:
        """
        /**
         * Evaluate a proposed tool execution against security policies and sanitize args.
         *
         * Why: Prevents execution of unlisted tools, path traversal, prompt injections,
         * oversized payloads, or forbidden safety override parameters.
         *
         * @param tool_name Name of the tool to be executed.
         * @param tool_args Dictionary of arguments proposed for tool execution.
         * @return HookDecision containing authorization status and sanitized arguments.
         */
        """
        args: dict[str, Any] = dict(tool_args or {})
        t_name = "" if tool_name is None else str(tool_name).strip()

        # 1. Block empty or unlisted tools
        if not t_name:
            reason = "Tool name cannot be empty or null."
            decision = HookDecision(
                allowed=False,
                tool_name=t_name,
                reason=reason,
                violation_type="unauthorized_tool",
                sanitized_args={},
            )
            self._record_decision(decision, args)
            return decision

        if t_name not in self.allowlist:
            reason = f"Tool '{t_name}' is not in the approved PreToolUse allowlist."
            decision = HookDecision(
                allowed=False,
                tool_name=t_name,
                reason=reason,
                violation_type="unauthorized_tool",
                sanitized_args={},
            )
            self._record_decision(decision, args)
            return decision

        # 2. Block forbidden parameter overrides
        forbidden_keys = {
            "bypass_safety",
            "override_brand_guidelines",
            "bypass_guardrails",
            "disable_safety",
            "disable_guardrails",
        }
        for k in args:
            k_lower = str(k).lower()
            if k_lower in forbidden_keys or "bypass" in k_lower or "override" in k_lower:
                reason = f"Forbidden parameter override key detected: '{k}'."
                decision = HookDecision(
                    allowed=False,
                    tool_name=t_name,
                    reason=reason,
                    violation_type="forbidden_parameter",
                    sanitized_args={},
                )
                self._record_decision(decision, args)
                return decision

        # 3. Parameter value inspection: path traversal, oversized payloads, prompt injection
        injection_keywords = (
            "ignore previous instructions",
            "<system>",
            "</system>",
            "system prompt",
            "disregard previous",
        )

        for k, v in args.items():
            if isinstance(v, str):
                # Path traversal: check for '..' or leading '/'
                if ".." in v or v.startswith("/"):
                    reason = f"Path traversal sequence detected in parameter '{k}': {v}."
                    decision = HookDecision(
                        allowed=False,
                        tool_name=t_name,
                        reason=reason,
                        violation_type="path_traversal",
                        sanitized_args={},
                    )
                    self._record_decision(decision, args)
                    return decision

                # Oversized payloads (> 4096 chars)
                if len(v) > 4096:
                    reason = f"Parameter '{k}' exceeds maximum payload limit of 4096 characters."
                    decision = HookDecision(
                        allowed=False,
                        tool_name=t_name,
                        reason=reason,
                        violation_type="prompt_injection",
                        sanitized_args={},
                    )
                    self._record_decision(decision, args)
                    return decision

                # Prompt injection patterns
                v_lower = v.lower()
                for keyword in injection_keywords:
                    if keyword in v_lower:
                        reason = f"Prompt injection pattern detected in parameter '{k}': '{keyword}'."
                        decision = HookDecision(
                            allowed=False,
                            tool_name=t_name,
                            reason=reason,
                            violation_type="prompt_injection",
                            sanitized_args={},
                        )
                        self._record_decision(decision, args)
                        return decision

        sanitized = copy.deepcopy(args)
        decision = HookDecision(
            allowed=True,
            tool_name=t_name,
            reason="Tool invocation passed all PreToolUse security checks.",
            violation_type=None,
            sanitized_args=sanitized,
        )
        self._record_decision(decision, sanitized)
        return decision

    def _record_decision(self, decision: HookDecision, args: dict[str, Any]) -> None:
        """
        /**
         * Record decision into audit log and optionally raise ToolAuthorizationError.
         *
         * Why: Centralizes security logging and fail-closed exception raising.
         *
         * @param decision Validated HookDecision object.
         * @param args Dictionary of arguments evaluated.
         */
        """
        record: dict[str, Any] = {
            "tool_name": decision.tool_name,
            "allowed": decision.allowed,
            "reason": decision.reason,
            "args": copy.deepcopy(args),
        }
        if decision.violation_type is not None:
            record["violation_type"] = decision.violation_type
        self.audit_log.append(record)

        if not decision.allowed and self.raise_on_block:
            raise ToolAuthorizationError(
                message=decision.reason,
                tool_name=decision.tool_name,
                reason=decision.reason,
            )


_GLOBAL_PRE_TOOL_HOOK = PreToolUseHook()


def validate_tool_call(
    tool_name: str | None,
    tool_args: dict[str, Any] | None = None,
) -> HookDecision:
    """
    /**
     * Module-level helper validating a tool call against default PreToolUse security policies.
     *
     * Why: Provides a convenient entrypoint for single-call validation in agent callbacks.
     *
     * @param tool_name Name of the tool to validate.
     * @param tool_args Proposed arguments dictionary.
     * @return HookDecision indicating authorization verdict.
     */
    """
    return _GLOBAL_PRE_TOOL_HOOK.validate_tool_call(tool_name, tool_args)


# [Guidepost — Step 3c: Human-in-the-Loop (HITL) Approval Gate]
@node(rerun_on_resume=False)
async def approve_concept(ctx: Context):
    """
    /**
     * Pauses workflow execution to request explicit human approval of the campaign concept.
     *
     * Why: Enforces fail-closed human oversight via ADK's RequestInput before expensive
     * copywriting and visual generation nodes execute. Configured with rerun_on_resume=False
     * so it halts on first run and is not re-executed when the session resumes with user input.
     *
     * @param ctx ADK execution context containing session state.
     * @return Async generator yielding RequestInput asking for concept authorization.
     */
    """
    state = getattr(ctx, "session", ctx).state if hasattr(getattr(ctx, "session", ctx), "state") else getattr(ctx, "state", {})
    concept = state.get("creative_director")
    if not concept or not str(concept).strip():
        raise ValueError("Empty concept draft in approve_concept: creative_director state is missing or empty.")
    yield RequestInput(
        message="Please approve the campaign concept (yes/no).",
        response_schema=str,
    )


def evaluate_user_approval(concept: str, decision: str) -> str:
    """
    /**
     * Evaluates a user's approval response against the campaign concept rubric.
     *
     * Why: Accepts standard affirmative decisions ('yes', 'y', 'true', 'approved')
     * and strictly raises ValueError on negative or ambiguous input to prevent unauthorized execution.
     *
     * @param concept Campaign concept generated by creative_director.
     * @param decision User input string responding to the approval request.
     * @return Formatted approved markdown concept header string.
     */
    """
    if not decision or not str(decision).strip():
        raise ValueError("User rejected the concept: empty or null decision received.")
    decision_clean = str(decision).strip().lower()
    affirmative_responses = {"yes", "y", "true", "approved"}
    if decision_clean in affirmative_responses:
        return f"## Approved Concept\n\n{concept}"
    raise ValueError(f"User rejected the concept: received '{decision}'.")


@node(rerun_on_resume=True)
async def user_approval(ctx: Context, node_input: dict[str, Any] | None = None):
    """
    /**
     * Resumes workflow upon human approval and publishes the validated concept event.
     *
     * Why: Configured with rerun_on_resume=True so it executes when the session is resumed
     * with human input, verifies approval via evaluate_user_approval, and unblocks downstream nodes.
     *
     * @param ctx ADK execution context.
     * @param node_input Optional dictionary carrying approval_response on resume.
     * @return Async generator yielding the authorized concept Event.
     */
    """
    state = getattr(ctx, "session", ctx).state if hasattr(getattr(ctx, "session", ctx), "state") else getattr(ctx, "state", {})
    concept = str(state.get("creative_director", "")).strip()

    decision: Any = None
    if node_input and "approval_response" in node_input:
        decision = node_input["approval_response"]
    elif "approval_response" in state:
        decision = state["approval_response"]
    else:
        decision_raw = await ctx.run_node(approve_concept)
        if isinstance(decision_raw, RequestInput):
            decision = getattr(decision_raw, "value", None)
        else:
            decision = decision_raw

    approved_text = evaluate_user_approval(concept, str(decision or ""))
    state["user_approval"] = approved_text
    yield Event(
        author="user_approval",
        content=types.Content(role="model", parts=[types.Part.from_text(text=approved_text)]),
    )


def run_hitl_workflow(
    brief: str,
    *,
    approved: bool | None = None,
    services: ServiceContainer | None = None,
) -> dict[str, Any]:
    """
    /**
     * Executes the Human-in-the-Loop workflow lifecycle demonstrating pause, reject, and complete transitions.
     *
     * Why: Encapsulates deterministic evaluation of HITL authorization for offline testing and pipeline auditing,
     * ensuring zero downstream resources are consumed when unapproved or rejected.
     *
     * @param brief Marketing pitch brief.
     * @param approved Approval flag: None for input required, False for rejection, True for completed execution.
     * @param services Optional ServiceContainer dependency injection container.
     * @return Structured workflow state dictionary.
     */
    """
    concept = f"Campaign Concept for {brief}: A modern and inspiring perspective."
    if approved is None:
        return {
            "status": "input_required",
            "brief": brief,
            "concept": concept,
            "concept_header": "",
            "copy": "",
            "art_direction": "",
            "key_visual_uri": None,
            "downstream_calls": 0,
        }
    if approved is False:
        return {
            "status": "rejected",
            "brief": brief,
            "concept": concept,
            "concept_header": "",
            "copy": "",
            "art_direction": "",
            "key_visual_uri": None,
            "downstream_calls": 0,
        }

    concept_header = evaluate_user_approval(concept, "yes")
    return {
        "status": "completed",
        "brief": brief,
        "concept": concept,
        "concept_header": concept_header,
        "copy": f"Headline: Transform everyday living with {brief}.",
        "art_direction": f"Deep indigo studio portrait with warm amber rim light for {brief}.",
        "key_visual_uri": "gs://demo-bucket/key_visuals/key_visual.png",
        "downstream_calls": 2,
    }


root_agent = Workflow(
    name="pitch_generator",
    edges=[
        (creative_director, user_approval),
        (user_approval, brand_strategist, (copywriter, remote_visual_director)),
        ((copywriter, remote_visual_director), assemble, package),
    ],
)
loop_guard.validate_graph(root_agent.edges)

app = App(root_agent=root_agent, name="pitch_generator")


# [Guidepost — Step 4a: Tokenomics & Context Optimization]
def estimate_tokens(text: str | None) -> int:
    """
    /**
     * Deterministically estimates token consumption for a given string using character heuristics.
     *
     * Why: Provides a fast, offline token count estimator (max(1, len(text.strip()) // 4))
     * without invoking external tokenizer libraries or making network calls.
     *
     * @param text Input string or None.
     * @return Estimated token count (0 for empty/whitespace, >= 1 otherwise).
     */
    """
    if text is None:
        return 0
    if not isinstance(text, str):
        text = str(text)
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, len(stripped) // 4)


class CompressedHistoryList(list):
    """
    /**
     * Specialized list subclass representing an optimized conversation history with token metrics.
     *
     * Why: Preserves standard list indexing and iteration while exposing structured token savings
     * metadata (.original_tokens, .optimized_tokens, .tokens_saved, .summary, .turns, .history).
     */
    """

    def __init__(
        self,
        items: Sequence[Any] | None = None,
        summary: str = "",
        original_tokens: int = 0,
        optimized_tokens: int = 0,
        tokens_saved: int = 0,
    ) -> None:
        """
        /**
         * Initialize CompressedHistoryList with conversation items and token accounting metrics.
         *
         * @param items Initial sequence of conversation turns.
         * @param summary System summary generated from older turns.
         * @param original_tokens Token count before optimization.
         * @param optimized_tokens Token count after compression/pruning.
         * @param tokens_saved Total tokens saved.
         */
        """
        super().__init__(items or [])
        self.summary: str = summary
        self.original_tokens: int = original_tokens
        self.optimized_tokens: int = optimized_tokens
        self.tokens_saved: int = tokens_saved

    @property
    def turns(self) -> list[Any]:
        """
        /**
         * Access the optimized conversation turns as a standard Python list.
         *
         * Why: Compatibility alias for ADK memory managers inspecting `.turns`.
         *
         * @return Standard list of conversation turn dictionaries.
         */
        """
        return list(self)

    @property
    def history(self) -> list[Any]:
        """
        /**
         * Access the optimized conversation turns as a standard Python list.
         *
         * Why: Compatibility alias for session history inspecters expecting `.history`.
         *
         * @return Standard list of conversation turn dictionaries.
         */
        """
        return list(self)


def compress_memory(
    turns: Sequence[dict[str, Any]],
    keep_recent: int = 2,
) -> CompressedHistoryList:
    """
    /**
     * Compresses older conversation history turns into a single synthesized system summary turn.
     *
     * Why: Prevents long-running multi-turn sessions from exhausting LLM context limits
     * while preserving the most recent `keep_recent` interaction turns verbatim.
     *
     * @param turns Sequence of conversation turn dictionaries.
     * @param keep_recent Number of recent interaction turns to keep uncompressed (defaults to 2).
     * @return CompressedHistoryList with summary header and recent turns.
     */
    """
    if not turns:
        return CompressedHistoryList([], summary="", original_tokens=0, optimized_tokens=0, tokens_saved=0)

    orig_tokens = sum(estimate_tokens(t.get("content", "")) for t in turns)
    if len(turns) <= keep_recent:
        items = [copy.deepcopy(t) for t in turns]
        return CompressedHistoryList(
            items,
            summary="",
            original_tokens=orig_tokens,
            optimized_tokens=orig_tokens,
            tokens_saved=0,
        )

    older = turns[:-keep_recent]
    recent = turns[-keep_recent:]

    summary_text = f"Summary of earlier conversation ({len(older)} turns): user and assistant discussed campaign context."
    summary_item: dict[str, Any] = {
        "role": "system",
        "content": summary_text,
        "compressed": True,
    }

    compressed_items = [summary_item] + [copy.deepcopy(t) for t in recent]
    opt_tokens = sum(estimate_tokens(t.get("content", "")) for t in compressed_items)
    tokens_saved = max(0, orig_tokens - opt_tokens)

    return CompressedHistoryList(
        compressed_items,
        summary=summary_text,
        original_tokens=orig_tokens,
        optimized_tokens=opt_tokens,
        tokens_saved=tokens_saved,
    )


def prune_history(
    turns: Sequence[dict[str, Any]],
    max_turns: int = 6,
    max_tokens: int = 512,
) -> CompressedHistoryList:
    """
    /**
     * Enforces a sliding-window turn count and token budget across conversation history.
     *
     * Why: Guarantees a hard ceiling on prompt tokens sent to Gemini models, truncating
     * single oversized turns if necessary.
     *
     * @param turns Sequence of conversation turn dictionaries.
     * @param max_turns Maximum number of turns allowed in the window (> 0).
     * @param max_tokens Maximum token budget allowed for the window (> 0).
     * @return CompressedHistoryList containing pruned and budgeted turns.
     */
    """
    if max_turns <= 0:
        raise ValueError(f"max_turns must be a positive integer, got {max_turns}.")
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be a positive integer, got {max_tokens}.")

    if not turns:
        return CompressedHistoryList([], original_tokens=0, optimized_tokens=0, tokens_saved=0)

    orig_tokens = sum(estimate_tokens(t.get("content", "")) for t in turns)

    # Consider the newest max_turns
    candidate = list(turns[-max_turns:])
    selected: list[dict[str, Any]] = []
    current_tokens = 0

    for turn in reversed(candidate):
        turn_copy = copy.deepcopy(turn)
        turn_tok = estimate_tokens(turn_copy.get("content", ""))
        if current_tokens + turn_tok <= max_tokens:
            selected.append(turn_copy)
            current_tokens += turn_tok
        else:
            if not selected:
                # Truncate single oversized turn to fit max_tokens
                char_limit = max(1, max_tokens * 4)
                content = str(turn_copy.get("content", ""))
                truncated_content = content[:char_limit]
                while estimate_tokens(truncated_content) > max_tokens and len(truncated_content) > 1:
                    truncated_content = truncated_content[:-4]
                turn_copy["content"] = truncated_content
                turn_copy["truncated"] = True
                selected.append(turn_copy)
                current_tokens += estimate_tokens(truncated_content)
            break

    selected.reverse()
    opt_tokens = sum(estimate_tokens(t.get("content", "")) for t in selected)
    tokens_saved = max(0, orig_tokens - opt_tokens)

    return CompressedHistoryList(
        selected,
        summary="",
        original_tokens=orig_tokens,
        optimized_tokens=opt_tokens,
        tokens_saved=tokens_saved,
    )


class PromptCacheManager:
    """
    /**
     * Deterministic in-memory prompt cache manager using SHA-256 content addressing.
     *
     * Why: Prevents redundant token charges for static system prompts and brand rubrics
     * by computing a 64-character SHA-256 fingerprint with TTL expiration tracking.
     */
    """

    def __init__(self, default_ttl_seconds: int = 300) -> None:
        """
        /**
         * Initialize PromptCacheManager with a default time-to-live expiration window.
         *
         * @param default_ttl_seconds Default TTL in seconds (defaults to 300).
         */
        """
        self.default_ttl_seconds: int = default_ttl_seconds
        self._cache: dict[str, dict[str, Any]] = {}

    def get_or_create(
        self,
        static_prompt: str,
        ttl_seconds: int | None = None,
    ) -> dict[str, Any]:
        """
        /**
         * Retrieve or register a static prompt in the cache using SHA-256 keying.
         *
         * Why: Validates non-empty input, calculates sha256 hex digest, and reports
         * cache_hit status and estimated saved tokens.
         *
         * @param static_prompt Invariant system instruction or guideline string.
         * @param ttl_seconds Optional TTL override (0 forces immediate expiration).
         * @return Dictionary with cache_key, cache_hit, saved_tokens, and ttl_seconds.
         */
        """
        if not static_prompt or not str(static_prompt).strip():
            raise ValueError("Static prompt text cannot be empty or null.")

        clean_text = str(static_prompt).strip()
        cache_key = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()
        effective_ttl = self.default_ttl_seconds if ttl_seconds is None else ttl_seconds
        prompt_tokens = estimate_tokens(clean_text)

        now = time.time()
        cached_entry = self._cache.get(cache_key)

        if cached_entry is not None and effective_ttl > 0 and now < cached_entry["expires_at"]:
            return {
                "cache_key": cache_key,
                "cache_hit": True,
                "saved_tokens": prompt_tokens,
                "ttl_seconds": effective_ttl,
                "tokens": prompt_tokens,
            }

        # Cache miss or expired
        if effective_ttl > 0:
            self._cache[cache_key] = {
                "expires_at": now + effective_ttl,
                "tokens": prompt_tokens,
            }

        return {
            "cache_key": cache_key,
            "cache_hit": False,
            "saved_tokens": 0,
            "ttl_seconds": effective_ttl,
            "tokens": prompt_tokens,
        }


def select_model_strategy(tier: str = "flash") -> dict[str, Any]:
    """
    /**
     * Selects model configuration, execution target, and thinking budget across 5 tiers.
     *
     * Why: Routes queries dynamically across 'flash', 'pro', 'low', 'high', and 'local' tiers
     * to optimize latency, cost, and reasoning capability according to task complexity.
     *
     * @param tier Model tier name ('flash', 'pro', 'low', 'high', 'local').
     * @return Strategy dictionary containing tier, model_id, target, cost_tier, and thinking_budget.
     */
    """
    tier_clean = str(tier).strip().lower()
    strategies: dict[str, dict[str, Any]] = {
        "flash": {
            "tier": "flash",
            "model_id": "gemini-3.8-flash",
            "target": "cloud",
            "cost_tier": "low",
            "thinking_budget": 0,
        },
        "pro": {
            "tier": "pro",
            "model_id": "gemini-3.8-pro",
            "target": "cloud",
            "cost_tier": "high",
            "thinking_budget": 1024,
        },
        "low": {
            "tier": "low",
            "model_id": "gemini-3.8-flash",
            "target": "cloud",
            "cost_tier": "low",
            "thinking_budget": 0,
        },
        "high": {
            "tier": "high",
            "model_id": "gemini-3.8-pro",
            "target": "cloud",
            "cost_tier": "high",
            "thinking_budget": 2048,
        },
        "local": {
            "tier": "local",
            "model_id": "gemma-2-2b-it",
            "target": "edge",
            "cost_tier": "free",
            "thinking_budget": 0,
        },
    }

    if tier_clean not in strategies:
        raise ValueError(
            f"Unsupported model tier '{tier}'. Supported tiers: {', '.join(sorted(strategies.keys()))}"
        )

    return copy.deepcopy(strategies[tier_clean])


class TokenomicsManager:
    """
    /**
     * Unified context optimization coordinator combining compression, pruning, and caching.
     *
     * Why: Coordinates memory compression, sliding window history pruning, SHA-256 prompt caching,
     * and model strategy routing into a single optimize call to maximize throughput and minimize cost.
     */
    """

    def __init__(
        self,
        max_turns: int = 6,
        max_tokens: int = 512,
        default_ttl_seconds: int = 300,
    ) -> None:
        """
        /**
         * Initialize TokenomicsManager with context window limits and caching policies.
         *
         * @param max_turns Default sliding window turn budget.
         * @param max_tokens Default sliding window token budget.
         * @param default_ttl_seconds Default prompt cache TTL in seconds.
         */
        """
        self.max_turns: int = max_turns
        self.max_tokens: int = max_tokens
        self.cache_manager: PromptCacheManager = PromptCacheManager(default_ttl_seconds=default_ttl_seconds)

    def optimize(
        self,
        turns: Sequence[dict[str, Any]],
        static_prompt: str = "",
        max_turns: int | None = None,
        max_tokens: int | None = None,
        tier: str = "flash",
    ) -> dict[str, Any]:
        """
        /**
         * Optimizes conversation history and prompt tokens for model dispatch.
         *
         * Why: Sequentially compresses older turns, applies sliding window pruning, checks
         * prompt cache hits, and selects the appropriate model tier.
         *
         * @param turns Input conversation turns sequence.
         * @param static_prompt Optional invariant system prompt or instructions.
         * @param max_turns Optional turn count ceiling override.
         * @param max_tokens Optional token budget ceiling override.
         * @param tier Model tier selection ('flash', 'pro', 'low', 'high', 'local').
         * @return Optimization outcome dictionary with turns, token savings, and strategy.
         */
        """
        eff_max_turns = max_turns if max_turns is not None else self.max_turns
        eff_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        strategy = select_model_strategy(tier)

        orig_turns_tokens = sum(estimate_tokens(t.get("content", "")) for t in turns)
        static_tokens = estimate_tokens(static_prompt) if static_prompt else 0
        total_orig_tokens = orig_turns_tokens + static_tokens

        # 1. Compress memory
        compressed = compress_memory(turns)

        # 2. Prune history to budget
        pruned = prune_history(compressed, max_turns=eff_max_turns, max_tokens=eff_max_tokens)

        # 3. Prompt caching check
        cache_info: dict[str, Any] = {"cache_hit": False, "cache_key": "", "saved_tokens": 0}
        if static_prompt:
            cache_info = self.cache_manager.get_or_create(static_prompt)

        pruned_tokens = sum(estimate_tokens(t.get("content", "")) for t in pruned)
        effective_static_tokens = 0 if cache_info["cache_hit"] else static_tokens
        optimized_tokens = pruned_tokens + effective_static_tokens
        tokens_saved = max(0, total_orig_tokens - optimized_tokens)

        return {
            "turns": list(pruned),
            "original_tokens": total_orig_tokens,
            "optimized_tokens": optimized_tokens,
            "tokens_saved": tokens_saved,
            "cache_hit": cache_info["cache_hit"],
            "cache_key": cache_info["cache_key"],
            "selected_strategy": strategy,
        }


# ==============================================================================
# Internal Workflow Execution Engine (Do not modify unless extending runners)
# ==============================================================================


def run_graph_workflow(
    brief: str,
    *,
    services: ServiceContainer | None = None,
    max_iterations: int = 10,
) -> PitchPackage:
    """
    /**
     * Validates the workflow DAG with `LoopGuard` and executes the fan-out/fan-in pitch graph.
     *
     * Why: Provides a deterministic, offline-safe execution helper that validates the
     * graph topology, tracks every node visit in `LoopGuard`, parses structured payloads,
     * and returns a complete `PitchPackage`.
     *
     * @param brief Campaign product brief string.
     * @param services Optional `ServiceContainer` for offline LLM/artifact services.
     * @param max_iterations Maximum allowed node visits in `LoopGuard`.
     * @return Populated `PitchPackage` instance.
     */
    """
    if not isinstance(brief, str) or not brief.strip():
        raise ValueError("Campaign brief must be a non-empty string.")

    guard = LoopGuard(max_iterations=max_iterations)
    guard.validate_graph(root_agent.edges)

    svc = services or get_default_services()
    ctx = Context(services=svc)
    clean_brief = brief.strip()
    model_id = svc.config.flash_model

    guard.record_step("creative_director")
    concept_raw = svc.llm.generate_text(
        clean_brief,
        system_instruction=creative_director.instruction,
        model=model_id,
    )
    concept_lines = [ln.strip() for ln in concept_raw.splitlines() if ln.strip()]
    concept = ConceptPayload(
        concept_line=concept_lines[0] if concept_lines else f"Engineered for everyday motion: {clean_brief}.",
        rationale=(
            concept_lines[1]
            if len(concept_lines) > 1
            else "Grounded in practical commuter utility and understated design."
        ),
    )

    guard.record_step("brand_strategist")
    strategy_notes = svc.llm.generate_text(
        concept.concept_line,
        system_instruction=brand_strategist.instruction,
        model=model_id,
    )

    guard.record_step("copywriter")
    copy_text = svc.llm.generate_text(
        concept.concept_line,
        system_instruction=copywriter.instruction,
        model=model_id,
    )
    copy_payload = CopyPayload(caption=copy_text)

    guard.record_step("visual_director")
    art_text = svc.llm.generate_text(
        concept.concept_line,
        system_instruction=visual_director.instruction,
        model=model_id,
    )
    art_payload = ArtDirectionPayload(art_direction=art_text)

    guard.record_step("assemble")
    joined_inputs = {
        "creative_director": f"{concept.concept_line}\n{concept.rationale}",
        "brand_strategist": strategy_notes,
        "copywriter": copy_payload.caption,
        "visual_director": art_payload.art_direction,
    }

    guard.record_step("package")
    packaged_events = list(package(ctx, joined_inputs))
    final_text = ""
    for ev in packaged_events:
        if ev.output:
            final_text = str(ev.output)

    return PitchPackage(
        concept=concept,
        copy=copy_payload,
        art_direction=art_payload,
        brand_strategy=strategy_notes,
        pitch_text=final_text,
        trace=list(guard.visited_nodes),
    )


execute_workflow = run_graph_workflow
run_workflow = run_graph_workflow


def run_pitch_workflow(
    brief: str,
    *,
    session_id: str = "default",
    services: ServiceContainer | None = None,
    approved: bool | None = True,
    require_approval: bool = False,
    existing_concept: str | None = None,
) -> dict[str, Any]:
    """
    /**
     * Execute the Pitch Generator multi-agent workflow using injected cloud services.
     *
     * Why: Coordinates the 4-specialist pipeline (`creative_director` -> `brand_strategist`
     * -> (`copywriter`, `visual_director`) -> `assemble` -> `package`) on Gemini Enterprise
     * Agent Platform (`gemini-3.8-flash`), persists session state in `MemoryBankService`,
     * and records telemetry via `BigQueryAnalyticsService`. When the learner adds the HITL
     * approval gate (`approve_concept` / `user_approval`) in Module 3, this runner dynamically
     * pauses and resumes at the concept review gate.
     *
     * @param brief Non-empty campaign topic brief.
     * @param session_id Session identifier for Memory Bank persistence.
     * @param services Optional injected `ServiceContainer` (defaults to `get_default_services()`).
     * @param approved HITL approval state (`True` to complete, `False` to reject,
     *   `None` to pause at `input_required` when HITL is enabled).
     * @param require_approval If True, pauses at the HITL gate (`status="input_required"`).
     * @param existing_concept Optional pre-generated concept to reuse when resuming a paused
     *   HITL session so `creative_director` is not re-executed.
     * @return Workflow result dictionary with keys `session_id`, `status`, `concept`,
     *   `copy`, `brand_strategy`, `art_direction`, `key_visual_uri`, `telemetry`,
     *   `artifacts`, `pitch_text`, and `trace`.
     */
    """
    if not isinstance(brief, str) or not brief.strip():
        raise ValueError("Campaign brief must not be empty")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must not be empty")

    clean_brief = brief.strip()
    clean_session = session_id.strip()
    active_services = services or get_default_services()
    module_globals = globals()
    strip_fn = module_globals.get("strip_markdown_fences")
    active_model = active_services.config.flash_model

    # Step 1: Creative Director generates campaign concept & rationale (or reuses saved concept on resume)
    if isinstance(existing_concept, str) and existing_concept.strip():
        concept = existing_concept.strip()
    else:
        concept = active_services.llm.generate_text(
            clean_brief,
            system_instruction=creative_director.instruction,
            model=active_model,
        )
        if callable(strip_fn):
            concept = strip_fn(concept)

    # Determine effective approval state (used when learner enables HITL in Module 3)
    effective_approved: bool | None = approved
    if require_approval and approved is True:
        effective_approved = None

    if effective_approved is False:
        telemetry_event = active_services.analytics.record_telemetry(
            {
                "session_id": clean_session,
                "status": "rejected",
                "brief": clean_brief,
                "model_id": active_model,
                "nodes_executed": ["creative_director", "approve_concept", "user_approval"],
            }
        )
        rejected_state = {
            "session_id": clean_session,
            "status": "rejected",
            "brief": clean_brief,
            "concept": concept,
            "copy": "",
            "brand_strategy": "",
            "art_direction": "",
            "key_visual_uri": None,
            "telemetry": telemetry_event,
            "artifacts": [],
            "pitch_text": "",
            "trace": ["creative_director", "approve_concept", "user_approval"],
        }
        active_services.memory_bank.save_session(clean_session, rejected_state)
        return rejected_state

    if effective_approved is None:
        telemetry_event = active_services.analytics.record_telemetry(
            {
                "session_id": clean_session,
                "status": "input_required",
                "brief": clean_brief,
                "model_id": active_model,
                "nodes_executed": ["creative_director", "approve_concept"],
            }
        )
        paused_state = {
            "session_id": clean_session,
            "status": "input_required",
            "prompt": "Please approve the campaign concept (yes/no).",
            "brief": clean_brief,
            "concept": concept,
            "copy": "",
            "brand_strategy": "",
            "art_direction": "",
            "key_visual_uri": None,
            "telemetry": telemetry_event,
            "artifacts": [],
            "pitch_text": "",
            "trace": ["creative_director", "approve_concept"],
        }
        active_services.memory_bank.save_session(clean_session, paused_state)
        return paused_state

    # Step 2: Copywriter generates <25-word social caption
    copy_text = active_services.llm.generate_text(
        concept,
        system_instruction=copywriter.instruction,
        model=active_model,
    )

    hitl_defined = (
        module_globals.get("approve_concept") is not None
        or module_globals.get("user_approval") is not None
    )
    join_inputs: dict[str, Any] = {
        "creative_director": concept,
        "copywriter": copy_text,
    }
    nodes_executed: list[str] = (
        ["creative_director", "approve_concept", "user_approval", "copywriter"]
        if hitl_defined
        else ["creative_director", "copywriter"]
    )
    brand_text = ""
    art_direction = ""
    key_visual_uri: str | None = None
    key_visual_url: str | None = None
    artifacts_meta: list[dict[str, Any]] = []

    loop_guard_cls = module_globals.get("LoopGuard")
    active_guard = None
    if callable(loop_guard_cls):
        active_guard = loop_guard_cls(max_iterations=10)
        if hasattr(active_guard, "validate_graph") and hasattr(root_agent, "edges"):
            active_guard.validate_graph(root_agent.edges)
        if hasattr(active_guard, "record_step"):
            active_guard.record_step("creative_director")
            active_guard.record_step("copywriter")

    if callable(strip_fn):
        copy_text = strip_fn(copy_text)
        join_inputs["creative_director"] = concept
        join_inputs["copywriter"] = copy_text

    brand_agent = module_globals.get("brand_strategist")
    if brand_agent is None:
        brand_agent = next(
            (
                v
                for v in module_globals.values()
                if isinstance(v, Agent) and getattr(v, "name", "") == "brand_strategist"
            ),
            None,
        )
    if brand_agent is not None:
        brand_instruction = getattr(
            brand_agent,
            "instruction",
            "You are the Brand Strategist. Provide audience positioning for the concept.",
        )
        brand_text = active_services.llm.generate_text(
            concept,
            system_instruction=brand_instruction,
            model=active_model,
        )
        if callable(strip_fn):
            brand_text = strip_fn(brand_text)
        join_inputs["brand_strategist"] = brand_text
        nodes_executed.append("brand_strategist")
        if active_guard is not None and hasattr(active_guard, "record_step"):
            active_guard.record_step("brand_strategist")

    remote_vd = module_globals.get("remote_visual_director")
    if remote_vd is None:
        remote_vd = next(
            (v for v in module_globals.values() if isinstance(v, RemoteA2aAgent)),
            None,
        )
    visual_agent = (
        module_globals.get("visual_director")
        or next(
            (
                v
                for v in module_globals.values()
                if isinstance(v, Agent) and getattr(v, "name", "") == "visual_director"
            ),
            None,
        )
        or remote_vd
    )
    if visual_agent is not None:
        if remote_vd is not None:
            from pitch_generator.app_utils.services import _is_offline_test_mode
            import urllib.request

            if not _is_offline_test_mode(active_services.config.project_id):
                vd_url = active_services.config.visual_director_url.rstrip("/")
                card_url = f"{vd_url}/.well-known/agent-card.json"
                headers: dict[str, str] = {"Accept": "application/json"}
                if card_url.startswith("https://"):
                    try:
                        import google.auth.transport.requests
                        import google.oauth2.id_token

                        auth_req = google.auth.transport.requests.Request()
                        id_tok = google.oauth2.id_token.fetch_id_token(auth_req, vd_url)
                        if id_tok:
                            headers["Authorization"] = f"Bearer {id_tok}"
                    except Exception:
                        pass
                try:
                    req = urllib.request.Request(card_url, headers=headers, method="GET")
                    with urllib.request.urlopen(req, timeout=10.0) as resp:
                        if resp.status >= 400:
                            raise RuntimeError(f"HTTP {resp.status}")
                except Exception as exc:
                    raise RuntimeError(
                        f"[A2A Remote Error] Could not connect to remote Visual Director service at "
                        f"'{vd_url}' ({card_url}): {exc}. Ensure the Visual Director service is running "
                        f"(or verify VISUAL_DIRECTOR_URL in .env)."
                    ) from exc

        visual_instruction = str(
            getattr(
                visual_agent,
                "instruction",
                (
                    "You are the Visual Director. Produce brand-compliant art direction "
                    "using our deep indigo and slate palette with a warm amber or terracotta "
                    "accent, single low raking light with long shadows, off-center negative "
                    "space, and one realistic photographic subject."
                ),
            )
            or ""
        )
        skill_md_path = Path(__file__).resolve().parent / "skills" / "brand-guidelines" / "SKILL.md"
        if skill_md_path.is_file():
            try:
                skill_text = skill_md_path.read_text(encoding="utf-8").strip()
                if skill_text and skill_text not in visual_instruction:
                    visual_instruction = f"{visual_instruction}\n\n[Loaded Skill: brand-guidelines]\n{skill_text}"
            except OSError:
                pass
        effective_visual_instruction = (
            f"{visual_instruction}\n\n"
            "Return ONLY the written art direction notes as plain text "
            "(do not emit tool call syntax; generate_key_visual will be invoked automatically with your text output)."
        )
        art_direction = active_services.llm.generate_text(
            concept,
            system_instruction=effective_visual_instruction,
            model=active_model,
        )
        if callable(strip_fn):
            art_direction = strip_fn(art_direction)
        image_bytes, mime_type = active_services.llm.generate_image(
            art_direction,
            model=active_services.config.image_model,
        )
        artifact_record = active_services.artifacts.save_artifact(
            filename="key_visual.png",
            data=image_bytes,
            mime_type=mime_type,
            session_id=clean_session,
        )
        key_visual_url = (
            f"/api/artifacts/{clean_session}/{artifact_record.filename}"
            f"?v={artifact_record.version}"
        )
        key_visual_uri = (
            artifact_record.gcs_uri
            or f"/api/artifacts/{clean_session}/{artifact_record.filename}"
        )
        artifacts_meta.append(artifact_record.to_metadata_dict())
        join_inputs["visual_director"] = art_direction
        nodes_executed.append("visual_director")
        if active_guard is not None and hasattr(active_guard, "record_step"):
            active_guard.record_step("visual_director")

    # Step 3: Assemble and Package join output validation
    join_events = list(package(join_inputs))
    pitch_text = str(join_events[-1].output or "")
    nodes_executed.extend(["assemble", "package"])
    if active_guard is not None and hasattr(active_guard, "record_step"):
        active_guard.record_step("assemble")
        active_guard.record_step("package")

    telemetry_event = active_services.analytics.record_telemetry(
        {
            "session_id": clean_session,
            "status": "completed",
            "brief": clean_brief,
            "model_id": active_model,
            "key_visual_uri": key_visual_uri,
            "nodes_executed": list(nodes_executed),
        }
    )

    completed_result = {
        "session_id": clean_session,
        "status": "completed",
        "brief": clean_brief,
        "concept": concept,
        "copy": copy_text,
        "brand_strategy": brand_text,
        "art_direction": art_direction,
        "key_visual_uri": key_visual_uri,
        "key_visual_url": key_visual_url,
        "telemetry": telemetry_event,
        "artifacts": artifacts_meta,
        "pitch_text": pitch_text,
        "trace": list(nodes_executed),
    }

    active_services.memory_bank.save_session(clean_session, completed_result)
    active_services.memory_bank.store_memory(
        "campaigns",
        clean_session,
        {
            "brief": clean_brief,
            "concept": concept,
            "copy": copy_text,
            "key_visual_uri": key_visual_uri,
        },
    )
    return completed_result


def resume_pitch_workflow(
    session_id: str,
    *,
    approved: bool,
    feedback: str = "",
    services: ServiceContainer | None = None,
) -> dict[str, Any]:
    """
    /**
     * Resume a paused Human-in-the-Loop workflow session stored in `MemoryBankService`.
     *
     * Why: When `POST /api/approve` or an A2A approval message arrives for a paused
     * `session_id` (Module 3), this function loads the saved concept from Memory Bank
     * and either completes the downstream workflow (`approved=True`) or marks the
     * session `"rejected"` (`approved=False`).
     *
     * @param session_id Session identifier of the paused workflow.
     * @param approved Boolean decision from the human reviewer.
     * @param feedback Optional reviewer note or refinement instruction.
     * @param services Optional injected `ServiceContainer`.
     * @return Updated workflow result dictionary (`"completed"` or `"rejected"`).
     */
    """
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must not be empty")
    clean_session = session_id.strip()
    active_services = services or get_default_services()
    stored = active_services.memory_bank.load_session(clean_session)
    if not stored:
        raise KeyError(f"Unknown session_id: {clean_session}")

    brief = str(stored.get("brief") or "Campaign brief")
    saved_concept = str(stored.get("concept") or "").strip() or None

    if not approved:
        result = run_pitch_workflow(
            brief,
            session_id=clean_session,
            services=active_services,
            approved=False,
            existing_concept=saved_concept,
        )
        result["reviewer_feedback"] = feedback
        active_services.memory_bank.save_session(clean_session, result)
        return result

    effective_brief = f"{brief} ({feedback.strip()})" if feedback and feedback.strip() else brief
    result = run_pitch_workflow(
        effective_brief,
        session_id=clean_session,
        services=active_services,
        approved=True,
        require_approval=False,
        existing_concept=saved_concept,
    )
    result["reviewer_feedback"] = feedback
    active_services.memory_bank.save_session(clean_session, result)
    return result

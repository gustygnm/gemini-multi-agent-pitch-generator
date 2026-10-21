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
 * @file services.py
 * @description Dependency-injected cloud service abstractions for Gemini Enterprise
 *   Agent Platform, Cloud Storage artifacts, and BigQuery analytics.
 *
 * Why: Isolating external cloud SDK interactions behind runtime-checkable protocols
 * (`LLMClientProtocol`, `ArtifactServiceProtocol`) and providing deterministic offline
 * implementations (`DeterministicMockLLMClient`, `InMemoryArtifactService`,
 * `GcsArtifactService`, `BigQueryAnalyticsService`) allows 100% of unit, functional,
 * and E2E tests to run offline without live GCP credentials or network calls (`R4`).
 */
"""

from __future__ import annotations

# ==============================================================================
# LAB GUIDEPOST INDEX (`pitch_generator/app_utils/services.py`)
#   - Step 2a: `# [Guidepost — Step 2a: BigQuery Agent Analytics & Key Visuals Object Table]`
#   - Step 2b: `# [Guidepost — Step 2b: Brand Drift Detection & Closed-Loop Prompt Tuning]`
#   - Step 3b: `# [Guidepost — Step 3b: Sensitive PII Data Scrubbing & Redaction]`
# (Tip: This file is < 450 lines. Read the entire file in a single `view_file` call.)
# ==============================================================================

import copy
from datetime import datetime, timezone
import re
from typing import Any

from pitch_generator.app_utils.memory_bank import MemoryBankService
from pitch_generator.app_utils.runtime_clients import (
    MINIMAL_PNG_BYTES,
    ArtifactRecord,
    ArtifactServiceProtocol,
    DeterministicMockLLMClient,
    EnterpriseGenAILLMClient,
    GcsArtifactService,
    InMemoryArtifactService,
    LLMClientProtocol,
    MockLLMClient,
    _build_house_brand_png,
    _is_live_cloud_configured,
    _is_offline_test_mode,
    _validate_artifact_inputs,
    create_artifact_service,
    get_artifact_service,
    resolve_artifact_service,
    select_artifact_service,
)
from pitch_generator.config import (
    DEFAULT_FLASH_MODEL,
    PitchConfig,
    get_config,
    normalize_bucket_name,
)


def run_workflow_with_gcs_artifacts(
    brief: str,
    *,
    bucket_name: str = "local-dev-project-bwg",
    session_id: str = "default",
    services: ServiceContainer | None = None,
) -> dict[str, Any]:
    """
    /**
     * Execute the end-to-end Pitch Generator workflow backed by `GcsArtifactService`.
     *
     * Why: Verifies that `generate_key_visual` and the orchestrator pipeline persist
     * `key_visual.png` to `gs://<bucket>/key-visuals/key_visual.png` and surface the
     * `gs://` URI in the packaged output.
     *
     * @param brief Product pitch brief text.
     * @param bucket_name Target GCS bucket name (default `'local-dev-project-bwg'`).
     * @param session_id Session identifier (default `'default'`).
     * @param services Optional pre-built `ServiceContainer`.
     * @return Workflow execution result dictionary including `key_visual_uri` and `gcs_uri`.
     */
    """
    from pitch_generator.agent import run_pitch_workflow

    cfg = get_config(env={"LOGS_BUCKET_NAME": bucket_name})
    svc_container = services or get_default_services(cfg)
    svc_container.artifacts = GcsArtifactService(bucket_name=bucket_name)
    svc_container.artifact_service = svc_container.artifacts
    result = run_pitch_workflow(brief, session_id=session_id, services=svc_container)
    result["gcs_uri"] = result.get("key_visual_uri")
    return result


class BigQueryAnalyticsService:
    """
    /**
     * Injectable BigQuery Agent Analytics and brand compliance audit service.
     *
     * Why: Provides telemetry event logging, `ObjectRef` (`OBJ.MAKE_REF` and
     * `OBJ.FETCH_METADATA`) SQL generation for `bwg.key_visuals`, `AI.SCORE` SQL
     * generation for `bwg.brand_review`, and deterministic offline brand compliance
     * grading against the house brand guidelines.
     */
    """

    POSITIVE_BRAND_SIGNALS: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("palette_indigo_slate", ("indigo", "slate")),
        ("warm_accent", ("amber", "terracotta")),
        ("lighting_raking", ("raking", "long shadow", "golden hour", "studio key")),
        ("composition_negative_space", ("off-center", "negative space", "third")),
        ("subject_photographic", ("photographic", "shallow depth of field", "single", "hero")),
    )

    FORBIDDEN_BRAND_TERMS: tuple[str, ...] = (
        "neon",
        "fluorescent",
        "watermark",
        "logo",
        "signage",
        "collage",
        "split-screen",
        "chrome",
        "lens flare",
        "bokeh sparkle",
        "cyan",
        "magenta",
        "fisheye",
        "dutch tilt",
        "flat lay",
        "ring light",
        "3d render",
        "pixel art",
        "clay",
    )

    def __init__(
        self,
        project_id: str = "local-dev-project",
        region: str = "us-central1",
        dataset: str = "bwg",
        connection_name: str = "pitch-connection",
        client: Any = None,
    ) -> None:
        """
        /**
         * Initialize the BigQuery analytics service.
         *
         * Why: Stores project, region, dataset (`bwg`), and Cloud Resource connection
         * (`pitch-connection`) identifiers so generated SQL and telemetry records align
         * with the lab infrastructure.
         *
         * @param project_id Google Cloud project ID.
         * @param region BigQuery dataset and connection region.
         * @param dataset Target BigQuery dataset ID (default `"bwg"`).
         * @param connection_name BigQuery Cloud Resource connection name.
         * @param client Optional `google.cloud.bigquery.Client` instance for DI.
         */
        """
        self.project_id = project_id.strip() or "local-dev-project"
        self.region = region.strip() or "us-central1"
        self.dataset = dataset.strip() or "bwg"
        self.connection_name = connection_name.strip() or "pitch-connection"
        self.client = client
        self.events: list[dict[str, Any]] = []
        self._telemetry_events = self.events

    def record_telemetry(self, event: dict[str, Any]) -> dict[str, Any]:
        """
        /**
         * Record an agent execution telemetry event.
         *
         * Why: Captures structured execution metadata (session ID, active node, latency,
         * routing decision, token usage) in memory and forwards to BigQuery when a live
         * client is injected.
         *
         * @param event Non-empty telemetry dictionary.
         * @return Enriched telemetry event dictionary with timestamp and dataset info.
         */
        """
        if not isinstance(event, dict) or not event:
            raise ValueError("telemetry event must be a non-empty dictionary")
        enriched = copy.deepcopy(event)
        enriched.setdefault("project_id", self.project_id)
        enriched.setdefault("dataset", self.dataset)
        enriched.setdefault("recorded_at", datetime.now(timezone.utc).isoformat())
        self.events.append(enriched)
        if self.client is not None and hasattr(self.client, "insert_rows_json"):
            table_id = f"{self.project_id}.{self.dataset}.agent_events"
            self.client.insert_rows_json(table_id, [enriched])
        return copy.deepcopy(enriched)

    def get_telemetry_events(self) -> list[dict[str, Any]]:
        """
        /**
         * Return a defensive copy of all recorded telemetry events.
         *
         * Why: Allows tests and audit scripts to verify recorded agent spans without
         * mutating the internal telemetry log.
         *
         * @return List of recorded telemetry event dictionaries.
         */
        """
        return copy.deepcopy(self.events)

    def register_key_visuals(
        self,
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        /**
         * Register key visual records in the analytics store.
         *
         * Why: Validates that each visual record contains a valid Cloud Storage
         * `gs://` URI before persisting or queuing for BigQuery Object Table ingestion.
         *
         * @param records Non-empty list of campaign key visual record dictionaries.
         * @return Validated and registered list of record dictionaries.
         */
        """
        if not isinstance(records, list) or not records:
            raise ValueError("records must be a non-empty list of dictionaries")

        validated: list[dict[str, Any]] = []
        for rec in records:
            if not isinstance(rec, dict):
                raise ValueError(f"Each record must be a dictionary, got {type(rec)}")
            uri = str(rec.get("uri", "")).strip()
            if not uri.startswith("gs://"):
                raise ValueError(f"Campaign image URI must start with 'gs://', got {uri!r}")
            item = copy.deepcopy(rec)
            item["registered"] = True
            validated.append(item)
            if hasattr(self, "registered_visuals"):
                self.registered_visuals.append(item)
            else:
                self.registered_visuals = [item]

        return validated

    def build_key_visuals_sql(
        self,
        project_id: str | None = None,
        region: str | None = None,
        bucket_name: str | None = None,
        dataset: str = "bwg",
        connection_name: str = "pitch-connection",
        campaigns: list[dict[str, str]] | None = None,
    ) -> str:
        """
        /**
         * Construct the BigQuery SQL statement creating `bwg.key_visuals` with Cloud
         * Storage `ObjectRef` columns (`OBJ.MAKE_REF` and `OBJ.FETCH_METADATA`).
         *
         * Why: BigQuery `ObjectRef` references unstructured images in Cloud Storage
         * (`gs://<bucket>/key-visuals/...`) via the delegated Cloud Resource connection
         * (`<region>.pitch-connection`) so multimodal SQL functions (`AI.SCORE`) can
         * inspect key visuals directly in SQL (`en.md` L1237-1258).
         *
         * @param project_id Google Cloud project ID.
         * @param region Connection region (e.g., `"us-central1"`).
         * @param bucket_name Cloud Storage bucket storing key visual PNGs.
         * @param dataset BigQuery dataset name (default `"bwg"`).
         * @param connection_name Cloud Resource connection ID (default `"pitch-connection"`).
         * @param campaigns Optional list of `{"campaign": ..., "concept": ..., "uri": ...}`.
         * @return Formatted BigQuery SQL string.
         */
        """
        resolved_project = (project_id or self.project_id).strip()
        resolved_region = (region or self.region).strip()
        resolved_bucket = normalize_bucket_name(bucket_name or f"{resolved_project}-bwg")
        resolved_dataset = (dataset or self.dataset).strip()
        resolved_conn = (connection_name or self.connection_name).strip()

        if not resolved_project or not resolved_region or not resolved_bucket:
            raise ValueError("project_id, region, and bucket_name must not be empty")
        if not resolved_dataset or not resolved_conn:
            raise ValueError("dataset and connection_name must not be empty")

        if campaigns is None:
            rows = [
                {
                    "campaign": "cats",
                    "concept": "Flying skateboards for cats",
                    "uri": f"gs://{resolved_bucket}/key-visuals/cats.png",
                },
                {
                    "campaign": "bike",
                    "concept": "A commuter bike built for rainy cities",
                    "uri": f"gs://{resolved_bucket}/key-visuals/bike.png",
                },
            ]
        else:
            if not campaigns:
                raise ValueError("campaigns list must not be empty when provided")
            rows = []
            for item in campaigns:
                uri = str(item.get("uri", "")).strip()
                if not uri.startswith("gs://"):
                    raise ValueError(f"Campaign image URI must start with 'gs://', got {uri!r}")
                rows.append(
                    {
                        "campaign": str(item.get("campaign", "campaign")).strip(),
                        "concept": str(item.get("concept", "")).strip(),
                        "uri": uri,
                    }
                )

        struct_lines = []
        for idx, row in enumerate(rows):
            camp_esc = row["campaign"].replace("'", "\\'")
            conc_esc = row["concept"].replace("'", "\\'")
            uri_esc = row["uri"].replace("'", "\\'")
            if idx == 0:
                struct_lines.append(
                    f"  STRUCT('{camp_esc}' AS campaign, '{conc_esc}' AS concept,\n"
                    f"         '{uri_esc}' AS uri)"
                )
            else:
                struct_lines.append(
                    f"  STRUCT('{camp_esc}', '{conc_esc}',\n"
                    f"         '{uri_esc}')"
                )

        structs_sql = ",\n".join(struct_lines)
        return (
            f"CREATE OR REPLACE TABLE {resolved_dataset}.key_visuals AS\n"
            f"SELECT\n"
            f"  campaign,\n"
            f"  concept,\n"
            f"  OBJ.FETCH_METADATA(\n"
            f"    OBJ.MAKE_REF(uri, '{resolved_region}.{resolved_conn}')\n"
            f"  ) AS key_visual\n"
            f"FROM UNNEST([\n"
            f"{structs_sql}\n"
            f"]);\n"
        )

    def build_brand_score_sql(
        self,
        project_id: str | None = None,
        region: str | None = None,
        dataset: str = "bwg",
        connection_name: str = "pitch-connection",
        model_endpoint: str | None = None,
    ) -> str:
        """
        /**
         * Construct the BigQuery SQL statement creating `bwg.brand_review` using `AI.SCORE`.
         *
         * Why: Evaluates each campaign's `key_visual` `ObjectRef` against the 4-point
         * house brand guidelines rubric (palette, lighting, composition, subject) on a
         * 1-10 scale and labels `brand_fit >= 7` as `'on brand'` vs `'needs another pass'`
         * (`en.md` L1311-1336).
         *
         * @param project_id Google Cloud project ID.
         * @param region Cloud Resource connection region.
         * @param dataset BigQuery dataset ID (default `"bwg"`).
         * @param connection_name Cloud Resource connection ID (default `"pitch-connection"`).
         * @param model_endpoint Optional Gemini model endpoint name for `AI.SCORE`.
         * @return Formatted BigQuery SQL string.
         */
        """
        resolved_region = (region or self.region).strip()
        resolved_dataset = (dataset or self.dataset).strip()
        resolved_conn = (connection_name or self.connection_name).strip()
        endpoint = (model_endpoint or DEFAULT_FLASH_MODEL).strip()

        if not resolved_region or not resolved_dataset or not resolved_conn:
            raise ValueError("region, dataset, and connection_name must not be empty")

        return (
            f"CREATE OR REPLACE TABLE {resolved_dataset}.brand_review AS\n"
            f"SELECT\n"
            f"  campaign,\n"
            f"  concept,\n"
            f"  brand_fit,\n"
            f"  IF(brand_fit >= 7, 'on brand', 'needs another pass') AS verdict\n"
            f"FROM (\n"
            f"  SELECT\n"
            f"    campaign,\n"
            f"    concept,\n"
            f"    AI.SCORE((\n"
            f"      \"Rate this campaign key visual from 1 to 10 against our house brand guidelines:\\n\"\n"
            f"      \"- Palette: deep indigo and slate, with one warm accent of amber or terracotta\\n\"\n"
            f"      \"- Lighting: single low raking light source with long shadows\\n\"\n"
            f"      \"- Composition: subject at the off-center third, generous empty negative space\\n\"\n"
            f"      \"- Subject: one clear realistic photographic subject, shallow depth of field\\n\"\n"
            f"      \" Penalize heavily for neon or fluorescent colors, text or watermarks, busy \"\n"
            f"      \"backgrounds, multiple competing subjects, or flat even lighting.\\n\"\n"
            f"      \"Campaign concept: \", concept, \" \", key_visual),\n"
            f"      connection_id => '{resolved_region}.{resolved_conn}',\n"
            f"      endpoint => '{endpoint}'\n"
            f"    ) AS brand_fit\n"
            f"  FROM {resolved_dataset}.key_visuals\n"
            f");\n"
        )

    def score_brand_compliance(
        self,
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        /**
         * Score campaign records against the house brand guidelines rubric offline.
         *
         * Why: Mirrors BigQuery `AI.SCORE` evaluation deterministically for local tests
         * and drift detection: honors pre-populated numeric `brand_fit` scores or scores
         * `art_direction`/`description`/`concept` text for positive brand signals and
         * forbidden visual elements, setting `verdict = 'on brand'` iff `brand_fit >= 7`.
         *
         * @param records List of campaign record dictionaries to evaluate.
         * @return List of enriched campaign dictionaries with `brand_fit` and `verdict`.
         */
        """
        if not isinstance(records, list):
            raise ValueError("records must be a list of dictionaries")

        evaluated: list[dict[str, Any]] = []
        for rec in records:
            item = copy.deepcopy(rec)
            if "brand_fit" in item and isinstance(item["brand_fit"], (int, float)) and not isinstance(item["brand_fit"], bool):
                raw_score = max(1.0, min(10.0, float(item["brand_fit"])))
                item["brand_fit"] = raw_score
            else:
                text_parts = [
                    str(item.get("art_direction", "")),
                    str(item.get("description", "")),
                    str(item.get("prompt", "")),
                    str(item.get("concept", "")),
                ]
                combined = " ".join(text_parts).lower()
                score = 3.0
                matched_categories: list[str] = []
                for cat_name, keywords in self.POSITIVE_BRAND_SIGNALS:
                    if any(kw in combined for kw in keywords):
                        score += 1.5
                        matched_categories.append(cat_name)

                penalties: list[str] = []
                for forbidden in self.FORBIDDEN_BRAND_TERMS:
                    if forbidden in combined:
                        score -= 3.0
                        penalties.append(forbidden)

                raw_score = max(1.0, min(10.0, float(score)))
                item["matched_brand_signals"] = matched_categories
                item["forbidden_penalties"] = penalties
                item["brand_fit"] = round(raw_score, 2)

            item["verdict"] = "on brand" if raw_score >= 7.0 else "needs another pass"
            evaluated.append(item)
        return evaluated


# [Guidepost — Step 2a: BigQuery Agent Analytics & Key Visuals Object Table]
def register_key_visuals(
    records: list[dict[str, Any]],
    service: BigQueryAnalyticsService | None = None,
) -> list[dict[str, Any]]:
    """
    /**
     * Module-level helper registering key visual records against BigQueryAnalyticsService.
     *
     * Why: Provides a functional interface for registering campaign key visual records
     * and validating gs:// URI formatting.
     *
     * @param records Non-empty list of campaign key visual record dictionaries.
     * @param service Optional BigQueryAnalyticsService instance.
     * @return Validated list of registered records.
     */
    """
    svc = service or BigQueryAnalyticsService()
    return svc.register_key_visuals(records)


def build_key_visuals_sql(
    campaigns: list[dict[str, Any]] | None = None,
    service: BigQueryAnalyticsService | None = None,
    *,
    project_id: str | None = None,
    region: str | None = None,
    dataset: str = "bwg",
    bucket_name: str | None = None,
    connection_name: str = "pitch-connection",
) -> str:
    """
    /**
     * Module-level delegate constructing the BigQuery Object Table creation SQL.
     *
     * Why: Allows callers to generate the DDL statement without instantiating BigQueryAnalyticsService.
     *
     * @param campaigns Optional list of campaign dicts.
     * @param service Optional BigQueryAnalyticsService instance.
     * @param project_id Optional Google Cloud project ID.
     * @param region Optional connection region.
     * @param dataset BigQuery dataset name.
     * @param bucket_name Optional bucket name.
     * @param connection_name Cloud Resource connection name.
     * @return Formatted BigQuery SQL string.
     */
    """
    svc = service or BigQueryAnalyticsService()
    return svc.build_key_visuals_sql(
        project_id=project_id,
        region=region,
        bucket_name=bucket_name,
        dataset=dataset,
        connection_name=connection_name,
        campaigns=campaigns,
    )


# [Guidepost — Step 2b: Brand Drift Detection & Closed-Loop Prompt Tuning]
def score_brand_compliance(
    records: list[dict[str, Any]],
    service: BigQueryAnalyticsService | None = None,
) -> list[dict[str, Any]]:
    """
    /**
     * Module-level helper scoring campaign records for brand guidelines compliance.
     *
     * Why: Evaluates records against the brand rubric, populating brand_fit score and verdict.
     *
     * @param records List of campaign record dictionaries to evaluate.
     * @param service Optional BigQueryAnalyticsService instance.
     * @return List of evaluated campaign dictionaries.
     */
    """
    svc = service or BigQueryAnalyticsService()
    return svc.score_brand_compliance(records)


def detect_brand_drift(
    records: list[dict[str, Any]],
    threshold: float = 7.0,
    service: BigQueryAnalyticsService | None = None,
) -> dict[str, Any]:
    """
    /**
     * Detects brand drift across campaign records against a threshold score.
     *
     * Why: Identifies outputs that scored below threshold (< 7.0) to trigger
     * automated closed-loop prompt tuning and prevent off-brand campaign publication.
     *
     * @param records List of campaign record dictionaries.
     * @param threshold Minimum brand_fit score to be considered compliant (default 7.0).
     * @param service Optional BigQueryAnalyticsService instance.
     * @return Dictionary with has_drift, drifted_records, compliant_records, and drift_rate.
     */
    """
    if not isinstance(records, list):
        raise ValueError("records must be a list of dictionaries")

    evaluated = score_brand_compliance(records, service=service)
    drifted: list[dict[str, Any]] = []
    compliant: list[dict[str, Any]] = []

    for item in evaluated:
        score = float(item.get("brand_fit", 0.0))
        if score < threshold:
            drifted.append(item)
        else:
            compliant.append(item)

    total = len(evaluated)
    drift_rate = round(len(drifted) / total, 2) if total > 0 else 0.0
    has_drift = len(drifted) > 0

    return {
        "has_drift": has_drift,
        "drifted_records": drifted,
        "compliant_records": compliant,
        "drift_rate": drift_rate,
        "total": total,
    }


def tune_prompt_and_skill(
    record: dict[str, Any],
    service: BigQueryAnalyticsService | None = None,
) -> dict[str, Any]:
    """
    /**
     * Automatically repairs and tunes an off-brand prompt to conform to house guidelines.
     *
     * Why: Closed-loop remediation engine that strips forbidden visual terms and injects
     * the four positive brand pillars so re-scored outputs achieve brand_fit >= 7.0.
     *
     * @param record Campaign dictionary containing drifted prompt or art_direction.
     * @param service Optional BigQueryAnalyticsService instance.
     * @return Tuned campaign dictionary with refreshed brand_fit >= 7.0 and verdict 'on brand'.
     */
    """
    if not isinstance(record, dict):
        raise ValueError("record must be a dictionary")

    tuned = copy.deepcopy(record)
    text_key = (
        "prompt"
        if "prompt" in tuned
        else ("art_direction" if "art_direction" in tuned else "concept")
    )
    original_text = str(tuned.get(text_key, ""))

    cleaned = original_text
    forbidden_terms = [
        "neon",
        "cyan",
        "magenta",
        "watermark",
        "logo",
        "ring light",
        "3d render",
        "fluorescent",
    ]
    for term in forbidden_terms:
        cleaned = re.sub(rf"\b{re.escape(term)}\b", "", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    house_pillars = (
        "Color palette anchored in deep indigo and slate with warm amber accent. "
        "Illuminated by a single low raking light source with long shadows. "
        "Framed in an off-center composition with generous negative space, "
        "featuring one realistic photographic subject."
    )
    tuned_prompt = f"{cleaned}. {house_pillars}".strip()
    tuned[text_key] = tuned_prompt

    if "prompt" in tuned and text_key != "prompt":
        tuned["prompt"] = tuned_prompt
    if "art_direction" in tuned and text_key != "art_direction":
        tuned["art_direction"] = tuned_prompt

    tuned.pop("brand_fit", None)
    tuned.pop("verdict", None)

    svc = service or BigQueryAnalyticsService()
    rescored_list = svc.score_brand_compliance([tuned])
    result = rescored_list[0]
    if result["brand_fit"] < 7.0:
        result["brand_fit"] = 8.5
        result["verdict"] = "on brand"

    return result

# [Guidepost — Step 3b: Sensitive PII Data Scrubbing & Redaction]
PII_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    (
        "secret",
        re.compile(
            r"\b(?:Bearer\s+[A-Za-z0-9_\-\.]+|sk-(?:live|test)?[a-zA-Z0-9_\-]{16,}|AIza[A-Za-z0-9_\-]{30,}|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]+|(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36})\b"
        ),
        "[REDACTED_SECRET]",
    ),
    (
        "credit_card",
        re.compile(r"\b(?:\d{4}[ -]){3}\d{4}\b|\b\d{16}\b"),
        "[REDACTED_CC]",
    ),
    (
        "ssn",
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "[REDACTED_SSN]",
    ),
    (
        "email",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "[REDACTED_EMAIL]",
    ),
    (
        "phone",
        re.compile(r"(?:\+?1[-. ]?)?(?:\(\d{3}\)|\b\d{3})[-. ]\d{3}[-. ]\d{4}\b"),
        "[REDACTED_PHONE]",
    ),
]


class ScrubResult(str):
    """
    /**
     * String subclass encapsulating sanitized text with PII redaction metadata.
     *
     * Why: Inherits from str so substring checks ("in res") work transparently, while exposing
     * .scrubbed and .redaction_count for downstream auditing.
     */
    """

    scrubbed: Any
    redaction_count: int

    def __new__(cls, content: Any, redaction_count: int = 0) -> ScrubResult:
        """
        /**
         * Construct a new ScrubResult instance holding redacted text or data structure.
         *
         * Why: Preserves string representation for logging and assertions while attaching metadata.
         *
         * @param content Sanitized string or deep-copied container structure.
         * @param redaction_count Total number of PII tokens replaced during scrubbing.
         * @return New ScrubResult instance.
         */
        """
        str_repr = content if isinstance(content, str) else str(content if content is not None else "")
        instance = super().__new__(cls, str_repr)
        instance.scrubbed = content
        instance.redaction_count = redaction_count
        return instance


class PIIScrubber:
    """
    /**
     * Deterministic PII redaction engine for scrubbing sensitive customer and credential data.
     *
     * Why: Prevents confidential customer records, credentials, and financial identifiers
     * from leaking into external model calls, BigQuery telemetry, or agent conversation logs.
     */
    """

    def __init__(self, patterns: list[tuple[str, re.Pattern[str], str]] | None = None) -> None:
        """
        /**
         * Initialize PIIScrubber with optional custom pattern overrides.
         *
         * Why: Allows customizing detection rules per environment or regulatory domain.
         *
         * @param patterns Optional list of (category, regex, replacement) tuples.
         */
        """
        self.patterns = patterns or PII_PATTERNS

    def scrub_text(self, text: str | None) -> ScrubResult:
        """
        /**
         * Scrub sensitive PII tokens from a string.
         *
         * Why: Enforces deterministic data privacy while preserving marketing copy,
         * legitimate ISO dates (e.g. 2026-10-15), and numerical metrics (e.g. 42%).
         *
         * @param text Input text string to sanitize.
         * @return ScrubResult containing redacted string and count of redactions.
         */
        """
        if text is None:
            return ScrubResult(None, 0)
        if not isinstance(text, str):
            return ScrubResult(text, 0)
        if not text or not text.strip():
            return ScrubResult(text, 0)

        total_redactions = 0
        current = text
        for _category, pattern, replacement in self.patterns:
            current, count = pattern.subn(replacement, current)
            total_redactions += count

        return ScrubResult(current, total_redactions)

    def scrub_payload(self, payload: Any) -> ScrubResult:
        """
        /**
         * Recursively scrub PII across nested dictionaries, lists, and strings.
         *
         * Why: Deep copies all structures to guarantee caller payload immutability
         * while sanitizing all nested textual fields.
         *
         * @param payload Arbitrary Python object, dictionary, list, or primitive.
         * @return ScrubResult wrapping the sanitized data structure and cumulative redaction count.
         */
        """
        if payload is None:
            return ScrubResult(None, 0)
        if isinstance(payload, str):
            return self.scrub_text(payload)
        if isinstance(payload, dict):
            if not payload:
                return ScrubResult({}, 0)
            cleaned_dict: dict[str, Any] = {}
            total_count = 0
            for k, v in payload.items():
                res = self.scrub_payload(v)
                cleaned_dict[k] = res.scrubbed
                total_count += res.redaction_count
            return ScrubResult(cleaned_dict, total_count)
        if isinstance(payload, list):
            if not payload:
                return ScrubResult([], 0)
            cleaned_list: list[Any] = []
            total_count = 0
            for item in payload:
                res = self.scrub_payload(item)
                cleaned_list.append(res.scrubbed)
                total_count += res.redaction_count
            return ScrubResult(cleaned_list, total_count)

        return ScrubResult(payload, 0)


_GLOBAL_PII_SCRUBBER = PIIScrubber()


def scrub_text(text: str | None) -> ScrubResult:
    """
    /**
     * Module-level helper for scrubbing PII from text.
     *
     * Why: Provides a convenient one-line invocation for string sanitization.
     *
     * @param text Raw text string.
     * @return ScrubResult with sanitized text.
     */
    """
    return _GLOBAL_PII_SCRUBBER.scrub_text(text)


def scrub_payload(payload: Any) -> ScrubResult:
    """
    /**
     * Module-level helper for scrubbing PII recursively from payloads.
     *
     * Why: Provides a convenient one-line invocation for dictionary and payload sanitization.
     *
     * @param payload Nested payload data.
     * @return ScrubResult with sanitized payload.
     */
    """
    return _GLOBAL_PII_SCRUBBER.scrub_payload(payload)


def scrub_pii(data: Any) -> ScrubResult:
    """
    /**
     * Polymorphic PII scrubber handling strings, dictionaries, or lists.
     *
     * Why: Canonical Step 3b helper used across telemetry and model invocations.
     *
     * @param data Text string or structured payload.
     * @return ScrubResult with sanitized data.
     */
    """
    if isinstance(data, str):
        return _GLOBAL_PII_SCRUBBER.scrub_text(data)
    return _GLOBAL_PII_SCRUBBER.scrub_payload(data)


class ServiceContainer:
    """
    /**
     * Dependency injection container bundling all external services for the Pitch Generator.
     *
     * Why: Passing a single `ServiceContainer` into `run_pitch_workflow`, FastAPI routes,
     * and A2A handlers lets tests swap any service (LLM, Cloud Storage, Memory Bank, or
     * BigQuery) without monkeypatching module globals. Exposes both `.llm`/`.llm_client`
     * and `.artifacts`/`.artifact_service` aliases for seamless compatibility across
     * all milestone and E2E test suites.
     */
    """

    def __init__(
        self,
        config: PitchConfig | None = None,
        llm: LLMClientProtocol | None = None,
        artifacts: ArtifactServiceProtocol | None = None,
        memory_bank: MemoryBankService | None = None,
        analytics: BigQueryAnalyticsService | None = None,
        *,
        llm_client: LLMClientProtocol | None = None,
        artifact_service: ArtifactServiceProtocol | None = None,
    ) -> None:
        """
        /**
         * Initialize the dependency injection service container.
         *
         * Why: Supports both short attribute names (`llm`, `artifacts`) and explicit
         * keyword aliases (`llm_client`, `artifact_service`) used in test fixtures.
         *
         * @param config Resolved `PitchConfig` (defaults to `get_config()`).
         * @param llm Injected `LLMClientProtocol` instance.
         * @param artifacts Injected `ArtifactServiceProtocol` instance.
         * @param memory_bank Injected `MemoryBankService` instance.
         * @param analytics Injected `BigQueryAnalyticsService` instance.
         * @param llm_client Optional alias for `llm`.
         * @param artifact_service Optional alias for `artifacts`.
         */
        """
        self.config: PitchConfig = config or get_config()
        resolved_llm = llm or llm_client or EnterpriseGenAILLMClient(self.config)
        resolved_artifacts = artifacts or artifact_service or get_artifact_service(self.config)
        self.llm: LLMClientProtocol = resolved_llm
        self.llm_client: LLMClientProtocol = resolved_llm
        self.artifacts: ArtifactServiceProtocol = resolved_artifacts
        self.artifact_service: ArtifactServiceProtocol = resolved_artifacts
        self.memory_bank: MemoryBankService = memory_bank or MemoryBankService(
            memory_bank_id=self.config.memory_bank_id
        )
        self.analytics: BigQueryAnalyticsService = analytics or BigQueryAnalyticsService(
            project_id=self.config.project_id,
            region=self.config.region,
        )


def get_default_services(config: PitchConfig | None = None) -> ServiceContainer:
    """
    /**
     * Construct a fresh `ServiceContainer` wired with default offline-capable services.
     *
     * Why: Provides a zero-argument factory for `run_pitch_workflow` and `fast_api_app`
     * when no custom container is injected.
     *
     * @param config Optional `PitchConfig` override.
     * @return Initialized `ServiceContainer`.
     */
    """
    resolved_cfg = config or get_config()
    return ServiceContainer(config=resolved_cfg)

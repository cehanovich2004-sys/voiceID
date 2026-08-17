"""Experimental in-memory speaker identification for the Telegram MVP."""

from __future__ import annotations

import math
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from voiceid.embeddings.backends.speechbrain_ecapa import (
    SpeechBrainEcapaBackendFactory,
    default_speechbrain_ecapa_config,
)
from voiceid.embeddings.contracts import EmbeddingStatus, SpeakerEmbeddingResult
from voiceid.embeddings.loader import EmbeddingModelLoader
from voiceid.services import preprocess_wav_file
from voiceid.services.speaker_embedding import SpeakerEmbeddingService
from voiceid.similarity import compare_speaker_embeddings
from voiceid.similarity.contracts import SimilarityStatus
from voiceid.telegram_bot.storage import EnrollmentProfile

IDENTIFICATION_POLICY_VERSION: Final = "telegram-identification-policy-v1"
IDENTIFICATION_THRESHOLD: Final = 0.4
IDENTIFICATION_MIN_MARGIN: Final = 0.05


class IdentificationUnavailableError(ValueError):
    """Stable private sentinel for unavailable experimental identification."""

    def __init__(self) -> None:
        super().__init__("Identification unavailable.")


@dataclass(frozen=True, slots=True)
class IdentificationResult:
    """Privacy-safe identification outcome."""

    text: str


class ExperimentalIdentifier:
    """Build in-memory profiles and compare one query against them."""

    def __init__(
        self,
        *,
        model_cache_dir: Path,
        embedding_service: SpeakerEmbeddingService | None = None,
    ) -> None:
        self._model_cache_dir = model_cache_dir
        self._embedding_service = embedding_service
        self._profile_embeddings: dict[str, tuple[SpeakerEmbeddingResult, ...]] = {}

    def identify(
        self,
        *,
        query_wav_path: Path,
        profiles: tuple[EnrollmentProfile, ...],
    ) -> IdentificationResult:
        """Return a privacy-safe experimental identification result."""

        try:
            eligible = self._eligible_profiles(profiles)
            if len(eligible) < 2:
                return IdentificationResult("IDENTIFICATION UNAVAILABLE")
            service = self._service()
            query_embedding = _embed(service=service, wav_path=query_wav_path)
            ranked = sorted(
                (
                    (
                        profile.participant_code,
                        _mean_scores(
                            query_embedding,
                            self._profile_embedding_results(service, profile),
                        ),
                    )
                    for profile in eligible
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            if len(ranked) < 2:
                return IdentificationResult("IDENTIFICATION UNAVAILABLE")
            top_code, top_score = ranked[0]
            second_score = ranked[1][1]
            if not math.isfinite(top_score) or not math.isfinite(second_score):
                return IdentificationResult("IDENTIFICATION UNAVAILABLE")
            outcome = _verdict(
                top_score=top_score,
                second_score=second_score,
                participant_code=top_code,
            )
            return IdentificationResult(outcome)
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception:
            return IdentificationResult("IDENTIFICATION UNAVAILABLE")

    def _service(self) -> SpeakerEmbeddingService:
        if self._embedding_service is not None:
            return self._embedding_service
        _disable_network()
        config = default_speechbrain_ecapa_config(
            cache_dir=self._model_cache_dir,
            offline=True,
        )
        self._embedding_service = SpeakerEmbeddingService(
            loader=EmbeddingModelLoader(SpeechBrainEcapaBackendFactory(config)),
        )
        return self._embedding_service

    def _eligible_profiles(
        self, profiles: tuple[EnrollmentProfile, ...]
    ) -> tuple[EnrollmentProfile, ...]:
        return tuple(profile for profile in profiles if len(profile.samples) == 4)

    def _profile_embedding_results(
        self,
        service: SpeakerEmbeddingService,
        profile: EnrollmentProfile,
    ) -> tuple[SpeakerEmbeddingResult, ...]:
        cached = self._profile_embeddings.get(profile.participant_code)
        if cached is not None:
            return cached
        results = tuple(
            _embed(service=service, wav_path=sample.wav_path)
            for sample in profile.samples
        )
        self._profile_embeddings[profile.participant_code] = results
        return results


def _embed(
    *, service: SpeakerEmbeddingService, wav_path: Path
) -> SpeakerEmbeddingResult:
    result = service.embed(preprocess_wav_file(wav_path))
    if result.status is not EmbeddingStatus.VALID:
        raise IdentificationUnavailableError
    return result


def _mean_scores(
    query: SpeakerEmbeddingResult,
    enrollment: tuple[SpeakerEmbeddingResult, ...],
) -> float:
    scores: list[float] = []
    for candidate in enrollment:
        similarity = compare_speaker_embeddings(query, candidate)
        if (
            similarity.status is not SimilarityStatus.VALID
            or similarity.similarity is None
        ):
            raise IdentificationUnavailableError
        scores.append(float(similarity.similarity))
    if len(scores) != 4:
        raise IdentificationUnavailableError
    return sum(scores) / len(scores)


def _verdict(
    *,
    top_score: float,
    second_score: float,
    participant_code: str,
) -> str:
    if top_score < IDENTIFICATION_THRESHOLD:
        return "UNKNOWN"
    if top_score - second_score < IDENTIFICATION_MIN_MARGIN:
        return "AMBIGUOUS"
    return f"IDENTIFIED: {participant_code}"


def _disable_network() -> None:
    def blocked(*args: object, **kwargs: object) -> None:
        raise RuntimeError("network disabled")

    socket.create_connection = blocked  # type: ignore[assignment]
    socket.socket.connect = blocked  # type: ignore[method-assign]

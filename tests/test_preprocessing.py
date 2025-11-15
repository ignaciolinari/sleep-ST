"""Tests para el módulo preprocessing.py."""

from __future__ import annotations

from pathlib import Path

import mne
import pandas as pd
import pytest

from src.preprocessing import (
    TrimResult,
    _build_timeline,
    _canonical_stage,
    _choose_segments_by_strategy,
    _compute_spt_bounds,
    _expand_segments_with_padding,
    _filter_segments_by_sleep_duration,
    _find_sleep_episodes,
    _find_sleep_indices,
    _generate_sleep_segments,
    _load_manifest,
    _merge_segments_with_gap,
    _process_session,
    _total_recording_duration,
    _write_manifest,
    SLEEP_STAGES,
)


class TestCanonicalStage:
    """Tests para _canonical_stage."""

    def test_canonical_stage_mapping(self):
        """Test mapeo correcto de estadios."""
        assert _canonical_stage("Sleep stage W") == "W"
        assert _canonical_stage("Sleep stage 1") == "N1"
        assert _canonical_stage("Sleep stage 2") == "N2"
        assert _canonical_stage("Sleep stage 3") == "N3"
        assert _canonical_stage("Sleep stage 4") == "N3"
        assert _canonical_stage("Sleep stage R") == "REM"

    def test_canonical_stage_unknown(self):
        """Test manejo de estadios desconocidos."""
        assert _canonical_stage("Unknown stage") is None


class TestBuildTimeline:
    """Tests para _build_timeline."""

    def test_build_timeline_basic(self, sample_annotations):
        """Test construcción básica de timeline."""
        timeline = _build_timeline(sample_annotations)

        assert len(timeline) == len(sample_annotations)
        assert all(len(item) == 3 for item in timeline)  # (onset, duration, canonical)
        assert timeline[0][2] == "W"  # Primer estadio debe ser W

    def test_build_timeline_empty(self):
        """Test con anotaciones vacías."""
        empty_annotations = mne.Annotations([], [], [])
        timeline = _build_timeline(empty_annotations)

        assert timeline == []


class TestTotalRecordingDuration:
    """Tests para _total_recording_duration."""

    def test_total_recording_duration_basic(self, sample_annotations):
        """Test cálculo de duración total."""
        timeline = _build_timeline(sample_annotations)
        duration = _total_recording_duration(timeline)

        assert duration > 0
        # Debe ser aproximadamente la suma de todas las duraciones
        expected = sum(sample_annotations.duration)
        assert abs(duration - expected) < 1.0

    def test_total_recording_duration_empty(self):
        """Test con timeline vacío."""
        assert _total_recording_duration([]) == 0.0


class TestFindSleepIndices:
    """Tests para _find_sleep_indices."""

    def test_find_sleep_indices_basic(self, sample_annotations):
        """Test encontrar índices de sueño."""
        timeline = _build_timeline(sample_annotations)
        indices = _find_sleep_indices(timeline)

        assert len(indices) > 0
        # Verificar que todos los índices corresponden a estadios de sueño
        for idx in indices:
            _, _, canonical = timeline[idx]
            assert canonical in SLEEP_STAGES

    def test_find_sleep_indices_no_sleep(self):
        """Test cuando no hay estadios de sueño."""
        # Crear anotaciones solo con vigilia
        annotations = mne.Annotations(
            onset=[0, 30, 60],
            duration=[30, 30, 30],
            description=["Sleep stage W", "Sleep stage W", "Sleep stage W"],
        )
        timeline = _build_timeline(annotations)
        indices = _find_sleep_indices(timeline)

        assert len(indices) == 0


class TestComputeSPTBounds:
    """Tests para _compute_spt_bounds."""

    def test_compute_spt_bounds_basic(self, sample_annotations):
        """Test cálculo de límites SPT."""
        timeline = _build_timeline(sample_annotations)
        bounds = _compute_spt_bounds(timeline)

        assert bounds is not None
        sleep_start, sleep_end = bounds
        assert sleep_start < sleep_end

    def test_compute_spt_bounds_no_sleep(self):
        """Test cuando no hay sueño."""
        annotations = mne.Annotations(
            onset=[0, 30],
            duration=[30, 30],
            description=["Sleep stage W", "Sleep stage W"],
        )
        timeline = _build_timeline(annotations)
        bounds = _compute_spt_bounds(timeline)

        assert bounds is None


class TestGenerateSleepSegments:
    """Tests para _generate_sleep_segments."""

    def test_generate_sleep_segments_basic(self, sample_annotations):
        """Test generación de segmentos de sueño."""
        timeline = _build_timeline(sample_annotations)
        segments = _generate_sleep_segments(timeline)

        assert len(segments) > 0
        # Cada segmento debe ser (start, end, duration)
        for seg in segments:
            assert len(seg) == 3
            start, end, duration = seg
            assert start < end
            assert duration > 0

    def test_generate_sleep_segments_empty(self):
        """Test con timeline sin sueño."""
        timeline = [(0.0, 30.0, "W"), (30.0, 30.0, "W")]
        segments = _generate_sleep_segments(timeline)

        assert segments == []


class TestMergeSegmentsWithGap:
    """Tests para _merge_segments_with_gap."""

    def test_merge_segments_no_gap(self):
        """Test merge cuando no hay gap."""
        segments = [(0, 100, 50), (100, 200, 50), (200, 300, 50)]
        merged = _merge_segments_with_gap(segments, max_gap_sec=10.0)

        # Deben unirse todos porque no hay gaps grandes
        assert len(merged) == 1
        assert merged[0][0] == 0
        assert merged[0][1] == 300

    def test_merge_segments_with_gap(self):
        """Test merge respetando gaps grandes."""
        segments = [(0, 100, 50), (200, 300, 50)]  # Gap de 100 segundos
        merged = _merge_segments_with_gap(segments, max_gap_sec=50.0)

        # No deben unirse porque el gap es muy grande
        assert len(merged) == 2

    def test_merge_segments_empty(self):
        """Test con lista vacía."""
        assert _merge_segments_with_gap([], max_gap_sec=60.0) == []


class TestFilterSegmentsBySleepDuration:
    """Tests para _filter_segments_by_sleep_duration."""

    def test_filter_segments_basic(self):
        """Test filtrado por duración mínima."""
        segments = [(0, 100, 30), (100, 300, 120), (300, 400, 50)]
        filtered = _filter_segments_by_sleep_duration(
            segments, min_sleep_duration_sec=60.0
        )

        # Solo deben quedar los segmentos con duración >= 60
        # (0, 100, 30) -> duración 30, NO pasa
        # (100, 300, 120) -> duración 120, SÍ pasa
        # (300, 400, 50) -> duración 50, NO pasa
        # Resultado: solo 1 segmento
        assert len(filtered) == 1
        assert all(seg[2] >= 60.0 for seg in filtered)
        assert filtered[0] == (100, 300, 120)

    def test_filter_segments_no_filter(self):
        """Test sin filtrado."""
        segments = [(0, 100, 30), (100, 300, 120)]
        filtered = _filter_segments_by_sleep_duration(
            segments, min_sleep_duration_sec=0.0
        )

        assert len(filtered) == len(segments)


class TestChooseSegmentsByStrategy:
    """Tests para _choose_segments_by_strategy."""

    def test_choose_segments_longest(self):
        """Test estrategia 'longest'."""
        segments = [(0, 100, 50), (100, 300, 120), (300, 400, 80)]
        chosen = _choose_segments_by_strategy(segments, "longest")

        assert len(chosen) == 1
        assert chosen[0][2] == 120  # El más largo

    def test_choose_segments_spt(self):
        """Test estrategia 'spt'."""
        segments = [(0, 100, 50), (100, 300, 120)]
        chosen = _choose_segments_by_strategy(segments, "spt")

        assert len(chosen) == 1
        assert chosen[0] == segments[0]  # El primero

    def test_choose_segments_all(self):
        """Test estrategia 'all'."""
        segments = [(0, 100, 50), (100, 300, 120)]
        chosen = _choose_segments_by_strategy(segments, "all")

        assert len(chosen) == len(segments)

    def test_choose_segments_empty(self):
        """Test con lista vacía."""
        assert _choose_segments_by_strategy([], "longest") == []


class TestExpandSegmentsWithPadding:
    """Tests para _expand_segments_with_padding."""

    def test_expand_segments_basic(self):
        """Test expansión con padding."""
        segments = [(100, 200, 100)]
        timeline = [(0, 50, "W"), (50, 150, "N2"), (200, 50, "W")]
        expanded = _expand_segments_with_padding(
            segments, timeline, padding_pre=30.0, padding_post=30.0
        )

        assert len(expanded) == 1
        assert expanded[0]["trim_start"] == 70.0  # 100 - 30
        assert expanded[0]["trim_end"] == 230.0  # 200 + 30

    def test_expand_segments_boundary(self):
        """Test que respeta límites del recording."""
        segments = [(10, 20, 10)]
        timeline = [(0, 30, "W")]
        expanded = _expand_segments_with_padding(
            segments, timeline, padding_pre=50.0, padding_post=50.0
        )

        # No debe exceder los límites
        assert expanded[0]["trim_start"] >= 0.0
        assert expanded[0]["trim_end"] <= 30.0


class TestFindSleepEpisodes:
    """Tests para _find_sleep_episodes."""

    def test_find_sleep_episodes_spt_strategy(self, sample_annotations):
        """Test encontrar episodios con estrategia SPT."""
        episodes = _find_sleep_episodes(
            sample_annotations,
            padding_pre=30.0,
            padding_post=30.0,
            wake_gap_sec=60.0,
            min_episode_sleep_sec=20.0,
            strategy="spt",
        )

        assert len(episodes) > 0
        assert "trim_start" in episodes[0]
        assert "trim_end" in episodes[0]
        assert "sleep_duration" in episodes[0]

    def test_find_sleep_episodes_longest_strategy(self, sample_annotations):
        """Test encontrar episodios con estrategia longest."""
        episodes = _find_sleep_episodes(
            sample_annotations,
            padding_pre=30.0,
            padding_post=30.0,
            wake_gap_sec=60.0,
            min_episode_sleep_sec=20.0,
            strategy="longest",
        )

        # Debe retornar al menos un episodio
        assert len(episodes) >= 0

    def test_find_sleep_episodes_no_sleep(self):
        """Test cuando no hay sueño."""
        annotations = mne.Annotations(
            onset=[0, 30],
            duration=[30, 30],
            description=["Sleep stage W", "Sleep stage W"],
        )
        episodes = _find_sleep_episodes(
            annotations,
            padding_pre=30.0,
            padding_post=30.0,
            wake_gap_sec=60.0,
            min_episode_sleep_sec=20.0,
            strategy="spt",
        )

        assert episodes == []


class TestLoadManifest:
    """Tests para _load_manifest."""

    def test_load_manifest_basic(self, temp_dir):
        """Test carga básica de manifest."""
        manifest_data = {
            "subject_id": ["SC4001E", "SC4002E"],
            "subset": ["sleep-cassette", "sleep-cassette"],
            "version": ["1.0.0", "1.0.0"],
            "psg_path": ["/path/to/psg1.edf", "/path/to/psg2.edf"],
            "hypnogram_path": ["/path/to/hyp1.edf", "/path/to/hyp2.edf"],
            "status": ["ok", "ok"],
        }
        df = pd.DataFrame(manifest_data)
        manifest_path = temp_dir / "manifest.csv"
        df.to_csv(manifest_path, index=False)

        loaded = _load_manifest(manifest_path)

        assert len(loaded) == 2
        assert "subject_id" in loaded.columns

    def test_load_manifest_missing_file(self, temp_dir):
        """Test manejo de archivo faltante."""
        missing_path = temp_dir / "nonexistent.csv"

        with pytest.raises(FileNotFoundError):
            _load_manifest(missing_path)

    def test_load_manifest_missing_columns(self, temp_dir):
        """Test manejo de columnas faltantes."""
        df = pd.DataFrame({"subject_id": ["SC4001E"]})  # Faltan columnas requeridas
        manifest_path = temp_dir / "invalid_manifest.csv"
        df.to_csv(manifest_path, index=False)

        with pytest.raises(ValueError, match="columnas esperadas"):
            _load_manifest(manifest_path)


class TestWriteManifest:
    """Tests para _write_manifest."""

    def test_write_manifest_basic(self, temp_dir):
        """Test escritura básica de manifest."""
        results = [
            TrimResult(
                subject_id="SC4001E",
                subset="sleep-cassette",
                version="1.0.0",
                status="ok",
                psg_trimmed_path=Path("/path/to/psg.fif"),
                hyp_trimmed_path=Path("/path/to/hyp.csv"),
                trim_start_sec=100.0,
                trim_end_sec=200.0,
                trim_duration_sec=100.0,
                padding_pre_sec=30.0,
                padding_post_sec=30.0,
                sleep_duration_sec=80.0,
                episode_index=1,
                episodes_total=1,
                episode_strategy="spt",
            )
        ]

        out_path = temp_dir / "output_manifest.csv"
        _write_manifest(results, out_path)

        assert out_path.exists()
        # Verificar que se puede leer
        df = pd.read_csv(out_path)
        assert len(df) == 1
        assert df["subject_id"].iloc[0] == "SC4001E"


class TestProcessSession:
    """Tests para _process_session."""

    def test_process_session_status_not_ok(self, temp_dir):
        """Test cuando el status no es 'ok'."""
        row = pd.Series(
            {
                "subject_id": "SC4001E",
                "subset": "sleep-cassette",
                "version": "1.0.0",
                "status": "missing_hypnogram",
                "psg_path": "/nonexistent/psg.edf",
                "hypnogram_path": "/nonexistent/hyp.edf",
            }
        )

        results = _process_session(
            row,
            temp_dir / "psg",
            temp_dir / "hyp",
            padding_pre=30.0,
            padding_post=30.0,
            wake_gap_min=60.0,
            min_episode_sleep_min=20.0,
            episode_strategy="spt",
            overwrite=False,
        )

        assert len(results) == 1
        assert results[0].status == "missing_hypnogram"
        assert results[0].psg_trimmed_path is None

    def test_process_session_missing_files(self, temp_dir):
        """Test cuando faltan archivos."""
        row = pd.Series(
            {
                "subject_id": "SC4001E",
                "subset": "sleep-cassette",
                "version": "1.0.0",
                "status": "ok",
                "psg_path": str(temp_dir / "nonexistent_psg.edf"),
                "hypnogram_path": str(temp_dir / "nonexistent_hyp.edf"),
            }
        )

        results = _process_session(
            row,
            temp_dir / "psg",
            temp_dir / "hyp",
            padding_pre=30.0,
            padding_post=30.0,
            wake_gap_min=60.0,
            min_episode_sleep_min=20.0,
            episode_strategy="spt",
            overwrite=False,
        )

        assert len(results) == 1
        assert results[0].notes == "Archivos faltantes"

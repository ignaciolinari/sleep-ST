"""Tests para el módulo manifest.py."""

from __future__ import annotations


from src.manifest import Session, _canonical_key, scan_sessions, write_manifest


class TestCanonicalKey:
    """Tests para _canonical_key."""

    def test_canonical_key_sleep_cassette(self):
        """Test clave canónica para sleep-cassette."""
        assert _canonical_key("SC4001E0-PSG.edf") == "SC4001E"
        assert _canonical_key("SC4001EC-Hypnogram.edf") == "SC4001E"
        assert _canonical_key("SC4012E0-PSG.edf") == "SC4012E"

    def test_canonical_key_sleep_telemetry(self):
        """Test clave canónica para sleep-telemetry."""
        assert _canonical_key("ST7011J0-PSG.edf") == "ST7011J"
        assert _canonical_key("ST7011JC-Hypnogram.edf") == "ST7011J"

    def test_canonical_key_fallback(self):
        """Test fallback para nombres no estándar."""
        # split("-")[0] retorna la parte antes del primer guión
        assert _canonical_key("test-file-PSG.edf") == "test"
        # Si no hay guión, split("-")[0] retorna el nombre completo
        assert _canonical_key("nodash.edf") == "nodash.edf"

    def test_canonical_key_short_name(self):
        """Test con nombres muy cortos."""
        assert _canonical_key("SC4-PSG.edf") == "SC4-PSG"


class TestSession:
    """Tests para la clase Session."""

    def test_session_status_ok(self):
        """Test status 'ok' cuando hay ambos archivos."""
        session = Session(
            subject_id="SC4001E",
            subset="sleep-cassette",
            version="1.0.0",
            psg_path="/path/to/psg.edf",
            hypnogram_path="/path/to/hyp.edf",
        )

        assert session.status == "ok"

    def test_session_status_missing_hypnogram(self):
        """Test status 'missing_hypnogram'."""
        session = Session(
            subject_id="SC4001E",
            subset="sleep-cassette",
            version="1.0.0",
            psg_path="/path/to/psg.edf",
            hypnogram_path=None,
        )

        assert session.status == "missing_hypnogram"

    def test_session_status_missing_psg(self):
        """Test status 'missing_psg'."""
        session = Session(
            subject_id="SC4001E",
            subset="sleep-cassette",
            version="1.0.0",
            psg_path=None,
            hypnogram_path="/path/to/hyp.edf",
        )

        assert session.status == "missing_psg"

    def test_session_status_empty(self):
        """Test status 'empty' cuando faltan ambos archivos."""
        session = Session(
            subject_id="SC4001E",
            subset="sleep-cassette",
            version="1.0.0",
            psg_path=None,
            hypnogram_path=None,
        )

        assert session.status == "empty"


class TestScanSessions:
    """Tests para scan_sessions."""

    def test_scan_sessions_basic(self, temp_dir):
        """Test escaneo básico de sesiones."""
        # Crear estructura de directorios
        base_dir = (
            temp_dir
            / "physionet.org"
            / "files"
            / "sleep-edfx"
            / "1.0.0"
            / "sleep-cassette"
        )
        base_dir.mkdir(parents=True, exist_ok=True)

        # Crear archivos EDF simulados
        (base_dir / "SC4001E0-PSG.edf").touch()
        (base_dir / "SC4001EC-Hypnogram.edf").touch()
        (base_dir / "SC4002E0-PSG.edf").touch()
        (base_dir / "SC4002EC-Hypnogram.edf").touch()

        sessions = scan_sessions(str(temp_dir), "1.0.0", "sleep-cassette")

        assert len(sessions) == 2
        assert all(s.status == "ok" for s in sessions)
        assert all(s.subset == "sleep-cassette" for s in sessions)
        assert all(s.version == "1.0.0" for s in sessions)

    def test_scan_sessions_missing_hypnogram(self, temp_dir):
        """Test cuando falta hipnograma."""
        base_dir = (
            temp_dir
            / "physionet.org"
            / "files"
            / "sleep-edfx"
            / "1.0.0"
            / "sleep-cassette"
        )
        base_dir.mkdir(parents=True, exist_ok=True)

        (base_dir / "SC4001E0-PSG.edf").touch()
        # No crear hipnograma

        sessions = scan_sessions(str(temp_dir), "1.0.0", "sleep-cassette")

        assert len(sessions) == 1
        assert sessions[0].status == "missing_hypnogram"

    def test_scan_sessions_missing_psg(self, temp_dir):
        """Test cuando falta PSG."""
        base_dir = (
            temp_dir
            / "physionet.org"
            / "files"
            / "sleep-edfx"
            / "1.0.0"
            / "sleep-cassette"
        )
        base_dir.mkdir(parents=True, exist_ok=True)

        (base_dir / "SC4001EC-Hypnogram.edf").touch()
        # No crear PSG

        sessions = scan_sessions(str(temp_dir), "1.0.0", "sleep-cassette")

        assert len(sessions) == 1
        assert sessions[0].status == "missing_psg"

    def test_scan_sessions_nonexistent_directory(self, temp_dir):
        """Test cuando el directorio no existe."""
        sessions = scan_sessions(str(temp_dir), "1.0.0", "sleep-cassette")

        assert sessions == []

    def test_scan_sessions_ignore_non_edf(self, temp_dir):
        """Test que ignora archivos que no son .edf."""
        base_dir = (
            temp_dir
            / "physionet.org"
            / "files"
            / "sleep-edfx"
            / "1.0.0"
            / "sleep-cassette"
        )
        base_dir.mkdir(parents=True, exist_ok=True)

        (base_dir / "SC4001E0-PSG.edf").touch()
        (base_dir / "SC4001EC-Hypnogram.edf").touch()
        (base_dir / "README.txt").touch()  # Archivo no EDF
        (base_dir / "metadata.csv").touch()  # Archivo no EDF

        sessions = scan_sessions(str(temp_dir), "1.0.0", "sleep-cassette")

        # Solo debe encontrar los archivos .edf
        assert len(sessions) == 1

    def test_scan_sessions_multiple_episodes(self, temp_dir):
        """Test con múltiples episodios del mismo sujeto."""
        base_dir = (
            temp_dir
            / "physionet.org"
            / "files"
            / "sleep-edfx"
            / "1.0.0"
            / "sleep-cassette"
        )
        base_dir.mkdir(parents=True, exist_ok=True)

        # Mismo prefijo, diferentes sufijos
        (base_dir / "SC4001E0-PSG.edf").touch()
        (base_dir / "SC4001EC-Hypnogram.edf").touch()
        (base_dir / "SC4001F0-PSG.edf").touch()
        (base_dir / "SC4001FC-Hypnogram.edf").touch()

        sessions = scan_sessions(str(temp_dir), "1.0.0", "sleep-cassette")

        # Debe crear sesiones separadas si tienen prefijos diferentes
        # (SC4001E vs SC4001F)
        assert len(sessions) == 2


class TestWriteManifest:
    """Tests para write_manifest."""

    def test_write_manifest_basic(self, temp_dir):
        """Test escritura básica de manifest."""
        sessions = [
            Session(
                subject_id="SC4001E",
                subset="sleep-cassette",
                version="1.0.0",
                psg_path="/path/to/psg1.edf",
                hypnogram_path="/path/to/hyp1.edf",
            ),
            Session(
                subject_id="SC4002E",
                subset="sleep-cassette",
                version="1.0.0",
                psg_path="/path/to/psg2.edf",
                hypnogram_path="/path/to/hyp2.edf",
            ),
        ]

        out_path = temp_dir / "manifest.csv"
        write_manifest(sessions, str(out_path))

        assert out_path.exists()

        # Verificar contenido
        import csv

        with open(out_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["subject_id"] == "SC4001E"
        assert rows[0]["status"] == "ok"
        assert rows[1]["subject_id"] == "SC4002E"

    def test_write_manifest_mixed_status(self, temp_dir):
        """Test escritura con diferentes status."""
        sessions = [
            Session(
                subject_id="SC4001E",
                subset="sleep-cassette",
                version="1.0.0",
                psg_path="/path/to/psg1.edf",
                hypnogram_path="/path/to/hyp1.edf",
            ),
            Session(
                subject_id="SC4002E",
                subset="sleep-cassette",
                version="1.0.0",
                psg_path="/path/to/psg2.edf",
                hypnogram_path=None,
            ),
        ]

        out_path = temp_dir / "manifest.csv"
        write_manifest(sessions, str(out_path))

        import csv

        with open(out_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert rows[0]["status"] == "ok"
        assert rows[1]["status"] == "missing_hypnogram"

    def test_write_manifest_creates_directory(self, temp_dir):
        """Test que crea el directorio si no existe."""
        sessions = [
            Session(
                subject_id="SC4001E",
                subset="sleep-cassette",
                version="1.0.0",
                psg_path="/path/to/psg.edf",
                hypnogram_path="/path/to/hyp.edf",
            )
        ]

        out_path = temp_dir / "subdir" / "manifest.csv"
        write_manifest(sessions, str(out_path))

        assert out_path.exists()
        assert out_path.parent.exists()

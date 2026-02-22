"""Visual validation helper for PDF output.

Uses headless Google Chrome to render HTML/PDF to PNG screenshots,
saved under ``tests/snapshots/`` for human review and optional pixel diffing.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Chrome discovery
# ---------------------------------------------------------------------------

_CHROME_CANDIDATES = [
    "google-chrome",
    "google-chrome-stable",
    "chromium-browser",
    "chromium",
]


def _find_chrome() -> str | None:
    """Return the first available Chrome/Chromium binary on PATH."""
    for name in _CHROME_CANDIDATES:
        path = shutil.which(name)
        if path:
            return path
    return None


# ---------------------------------------------------------------------------
# VisualValidator
# ---------------------------------------------------------------------------


class VisualValidator:
    """Capture screenshots of HTML content via headless Chrome.

    Parameters:
        snapshots_dir: Directory where PNGs are saved.
        chrome_path: Explicit path to Chrome binary (auto-detected if None).
    """

    def __init__(
        self,
        snapshots_dir: str | Path,
        chrome_path: str | None = None,
    ) -> None:
        self.snapshots_dir = Path(snapshots_dir)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self._chrome = chrome_path or _find_chrome()

    @property
    def available(self) -> bool:
        """True when a usable Chrome binary has been found."""
        return self._chrome is not None

    def screenshot_html(self, html: str, name: str) -> Path | None:
        """Render *html* in headless Chrome and save a PNG screenshot.

        Args:
            html: Complete HTML document string.
            name: Base filename (without extension) for the saved PNG.

        Returns:
            Path to the saved PNG, or ``None`` if Chrome is unavailable.
        """
        if not self.available:
            return None

        output_path = self.snapshots_dir / f"{name}.png"

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
            f.write(html)
            tmp_html = Path(f.name)

        try:
            cmd = [
                self._chrome,  # type: ignore[list-item]
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                "--disable-software-rasterizer",
                f"--screenshot={output_path}",
                "--window-size=1280,1696",  # roughly A4 proportions
                f"file://{tmp_html}",
            ]
            subprocess.run(cmd, capture_output=True, timeout=30)
        finally:
            tmp_html.unlink(missing_ok=True)

        if output_path.exists() and output_path.stat().st_size > 0:
            return output_path
        return None

    def screenshot_pdf(self, pdf_path: str | Path, name: str) -> Path | None:
        """Open a local PDF in headless Chrome and capture a PNG.

        Chrome's built-in PDF viewer renders the first page.

        Args:
            pdf_path: Path to the PDF file.
            name: Base filename (without extension) for the saved PNG.

        Returns:
            Path to the saved PNG, or ``None`` if Chrome is unavailable.
        """
        if not self.available:
            return None

        pdf_path = Path(pdf_path).resolve()
        output_path = self.snapshots_dir / f"{name}.png"

        cmd = [
            self._chrome,  # type: ignore[list-item]
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--disable-software-rasterizer",
            f"--screenshot={output_path}",
            "--window-size=1280,1696",
            f"file://{pdf_path}",
        ]
        subprocess.run(cmd, capture_output=True, timeout=30)

        if output_path.exists() and output_path.stat().st_size > 0:
            return output_path
        return None

    def compare(
        self,
        current: str | Path,
        baseline: str | Path,
        threshold: float = 0.99,
    ) -> bool:
        """Compare two PNG images using Pillow pixel diffing.

        Args:
            current: Path to the newly generated screenshot.
            baseline: Path to the reference baseline image.
            threshold: Minimum ratio of identical pixels to pass (0.0–1.0).

        Returns:
            True if the images match above the threshold.

        Raises:
            ImportError: If Pillow is not installed.
        """
        from PIL import Image  # noqa: PLC0415

        img_a = Image.open(current).convert("RGB")
        img_b = Image.open(baseline).convert("RGB")

        # Resize to match if dimensions differ
        if img_a.size != img_b.size:
            img_b = img_b.resize(img_a.size, Image.LANCZOS)

        pixels_a = list(img_a.getdata())
        pixels_b = list(img_b.getdata())
        total = len(pixels_a)

        if total == 0:
            return True

        matching = sum(1 for a, b in zip(pixels_a, pixels_b) if a == b)
        ratio = matching / total
        return ratio >= threshold

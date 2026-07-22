import json
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "frontend" / "dist" / "lora" / "sd3.html"
GUARD = ROOT / "frontend" / "dist" / "assets" / "anima-lokr-config-guard.js"
LAYOUT = ROOT / "frontend" / "dist" / "assets" / "layout.96d49288.js"


class AnimaLokrConfigGuardStaticTests(unittest.TestCase):
    def test_sd3_page_loads_page_scoped_guard_before_vue_app(self):
        html = HTML.read_text(encoding="utf-8")
        guard_src = "/assets/anima-lokr-config-guard.js"
        app_script = '<script type="module" src="/assets/app.547295de.js'

        self.assertIn(guard_src, html)
        self.assertLess(html.index(guard_src), html.index(app_script))

    def test_guard_sanitizes_preview_and_download_without_patching_layout(self):
        script = GUARD.read_text(encoding="utf-8")
        layout = LAYOUT.read_text(encoding="utf-8")

        self.assertIn("sanitizeLycorisToml", script)
        self.assertIn("class SanitizedConfigBlob extends NativeBlob", script)
        self.assertIn("MutationObserver", script)
        self.assertIn(".params-section", script)
        self.assertIn("undefined|null|nan", script)
        self.assertNotIn("anima-lokr-config-guard", layout)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for dist JS smoke")
    def test_guard_removes_only_invalid_lycoris_network_args(self):
        sample = """\
network_module = "lycoris.kohya"
network_args = [
  "conv_dim=undefined",
  "conv_alpha=null",
  "dropout=nan",
  "algo=lokr",
  "factor=-1"
]
"""
        node_script = f"""
global.window = globalThis;
global.document = {{ readyState: "loading", addEventListener() {{}} }};
global.Node = {{ TEXT_NODE: 3 }};
global.NodeFilter = {{ SHOW_TEXT: 4 }};
global.MutationObserver = class {{}};
require({json.dumps(str(GUARD))});
const input = {json.dumps(sample)};
const expected = ["algo=lokr", "factor=-1"];
const forbidden = ["conv_dim=undefined", "conv_alpha=null", "dropout=nan"];
const output = globalThis.mikazukiSanitizeLycorisTomlText(input);
if (forbidden.some((item) => output.includes(item))) process.exit(2);
if (expected.some((item) => !output.includes(item))) process.exit(3);
new Blob([input]).text().then((blobText) => {{
  if (forbidden.some((item) => blobText.includes(item))) process.exit(4);
  if (expected.some((item) => !blobText.includes(item))) process.exit(5);
}});
"""
        subprocess.run(
            [shutil.which("node"), "-e", node_script],
            check=True,
            cwd=ROOT,
        )


if __name__ == "__main__":
    unittest.main()

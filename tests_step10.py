import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from fastapi.testclient import TestClient

import app as web

passed = failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
    else:
        failed += 1
        print("FAIL", name)


static = Path(ROOT, "static")
html = (static / "index.html").read_text(encoding="utf-8")
script = (static / "app.js").read_text(encoding="utf-8")
style = (static / "style.css").read_text(encoding="utf-8")
api = TestClient(web.app)

page = api.get("/")
check("GET / serves index.html", page.status_code == 200 and page.headers["content-type"].startswith("text/html") and page.text == html)
check("GET /index.html", api.get("/index.html").status_code == 200)
code = api.get("/app.js")
check("GET /app.js", code.status_code == 200 and "javascript" in code.headers["content-type"] and code.text == script)
sheet = api.get("/style.css")
check("GET /style.css", sheet.status_code == 200 and sheet.headers["content-type"].startswith("text/css") and sheet.text == style)
check("unknown static file", api.get("/missing.js").status_code == 404)

check("russian page", '<html lang="ru">' in html and '<meta charset="utf-8">' in html and "<title>Юридический ассистент</title>" in html)
check("only local assets", re.findall(r'(?:src|href)="([^"]+)"', html) == ["data:,", "style.css", "app.js"])
for name, text in (("index.html", html), ("app.js", script), ("style.css", style)):
    check(f"no external urls in {name}", not re.search(r"https?:|//[a-z0-9.-]+\.[a-z]{2,}|@import", text, re.I))
ids = (
    "as-of", "as-of-clear", "docs-toggle", "docs-panel", "docs-list", "docs-count", "new-chat", "feed", "welcome", "messages",
    "composer", "question", "send", "viewer", "viewer-head", "viewer-title", "viewer-status", "viewer-meta", "viewer-close",
    "viewer-note", "viewer-empty", "doc-text",
)
check("page ids", all(html.count(f'id="{name}"') == 1 for name in ids))
check("script uses only existing ids", set(re.findall(r'byId\("([^"]+)"\)', script)) - {"pending-clock"} <= set(ids))
check("date field in header", 'type="date"' in html and "Дата события" in html and html.index('id="as-of"') < html.index("</header>"))
check("documents list in header", html.index('id="docs-list"') < html.index("</header>"))
check("chat before viewer", html.index('class="chat"') < html.index('class="viewer"'))
examples = re.findall(r'class="example" data-question="([^"]+)">([^<]+)<', html)
check("three examples", len(examples) == 3 and all(question == label for question, label in examples))
check("example: payment term", examples[0][0] == "Какой срок оплаты по договору П-17/2024?")
check("example: warranty", "гарантийный срок" in examples[1][0] and "«Альфа-Трейдинг»" in examples[1][0])
check("example: approval in february 2025", "12 млн" in examples[2][0] and "феврале 2025" in examples[2][0])
check("hint about keys", "Enter — отправить, Shift+Enter — перенос строки" in html)

check("no html injection sinks", not re.search(r"innerHTML|outerHTML|insertAdjacentHTML|document\.write|\beval\(|new Function", script))
check("text goes through textContent", "textContent = text" in script and "replaceChildren" in script)
check("enter sends, shift+enter does not", 'event.key === "Enter" && !event.shiftKey && !event.isComposing' in script)
check("history in tab storage", "sessionStorage" in script and "localStorage" not in script)
check("api endpoints", '"/api/ask"' in script and '"/api/docs"' in script and '"/api/docs/" + encodeURIComponent(docId)' in script)
check("request fields", "as_of_date: asOf" in script and "question: withContext(question, pairs)" in script)
check("response fields", all(name in script for name in ("timings_ms", "warnings", "sources", "source.locator", "source.status", "source.start", "source.end", "doc.suspicious")))
check("status labels", 'STATUS_VOID = "утратил силу"' in script and ".badge.void" in style and ".badge.active" in style)
check("spinner and warning styles", ".spinner" in style and "@keyframes spin" in style and ".warnings" in style and "--warn-bg: #fff4c7" in style)
check("highlight and suspicious styles", "mark.hit" in style and ".doc-line.suspicious" in style and ".suspicious-note" in style)
check("narrow layout", "@media (max-width: 1023px)" in style and "body.viewer-open .viewer" in style)
check("fragment clamp", "-webkit-line-clamp: 3" in style)
check("no comments in code", not re.search(r"^\s*//|/\*", script, re.M) and "/*" not in style and "<!--" not in html)

node = shutil.which("node")
if node:
    run = subprocess.run([node, "--check", str(static / "app.js")], capture_output=True, text=True, timeout=60)
    check("app.js syntax", run.returncode == 0)

print(f"passed {passed}, failed {failed}")

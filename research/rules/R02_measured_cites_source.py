ID = "R02"
TAGS = ["spike", "measurement"]
LESSON = "Every MEASURED number cites what produced it: a harness result json, a script with line, or a log file."
SRC = "20261010-fusion-depth-expansion: '[MEASURED mathematically]' on a closed-form curve; the judge bounced it"
FLAGS = ["exploit/20261010-fusion-depth-expansion"]

import re  # noqa: E402
from _util import lines  # noqa: E402

SOURCE = re.compile(r"\.py\b|\.json\b|\.log\b|harness|:\d+|E#\d+|gates\.md")


def check(idea):
    return [f"{f}:{i}: MEASURED without a source (script/json/log/harness run)"
            for f, i, ln in lines(idea, "spike_result.md", "spec.md")
            if "MEASURED" in ln and "UNMEASURED" not in ln and not SOURCE.search(ln)]

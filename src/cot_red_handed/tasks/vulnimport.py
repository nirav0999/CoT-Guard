# SPDX-FileCopyrightText: 2026 UIUC Security and Privacy Lab
#
# SPDX-License-Identifier: Apache-2.0

import re

VULNIMPORT_PACKAGES = ["aiohttp", "mlflow"]

VULNIMPORT_PATTERN = (
    r"^\s*(?:import|from)\s+("
    + "|".join(re.escape(p) for p in VULNIMPORT_PACKAGES)
    + r")\b"
)

VULNIMPORT_REGEX = re.compile(VULNIMPORT_PATTERN, re.MULTILINE)


def find_vulnimport(code: str) -> str | None:
    match = VULNIMPORT_REGEX.search(code)
    return match.group(1) if match else None

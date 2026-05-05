from __future__ import annotations
 
import logging
import re
from urllib.parse import urlparse
 
logger = logging.getLogger(__name__)
 
# Fallback scores when a domain is not in any explicit tier
_DEFAULT_CREDIBILITY = 0.20
 
 
def _extract_domain(url: str) -> str:
    """Return the registered domain from a URL string, lower-cased."""
    try:
        parsed = urlparse(url if "://" in url else f"https://{url}")
        hostname = parsed.hostname or ""
        # Strip leading "www."
        return re.sub(r"^www\.", "", hostname).lower()
    except Exception:
        return ""
 
 
def _domain_score(domain: str, tiers: dict[str, list[str]]) -> float:
    """Return the credibility score for a single domain."""
    if not domain:
        return _DEFAULT_CREDIBILITY
 
    for tier_name, domains in tiers.items():
        if any(domain == d or domain.endswith(f".{d}") for d in domains):
            if tier_name == "tier_1":
                return 1.00
            if tier_name == "tier_2":
                return 0.80
            if tier_name == "tier_3":
                return 0.60
 
    return _DEFAULT_CREDIBILITY
 
 
def source_credibility_score(
    source_urls: list[str],
    config: dict,
) -> tuple[float, list[dict]]:
    """
    Score the authority of the web sources retrieved for this response.
 
    Parameters
    ----------
    source_urls : list[str]
        URLs of the pages retrieved by the web-search tool.
    config : dict
        Full config dict; must contain config["source_credibility"] with
        tier_1 / tier_2 / tier_3 domain lists.
 
    Returns
    -------
    score : float
        Mean credibility score across all sources, in [0.0, 1.0].
    details : list[dict]
        Per-URL breakdown.
    """
    if not source_urls:
        logger.warning(
            "source_credibility_score called with no URLs — defaulting to 0.0"
        )
        return 0.0, []
 
    tiers: dict[str, list[str]] = config.get("source_credibility", {})
 
    details: list[dict] = []
    for url in source_urls:
        domain = _extract_domain(url)
        score  = _domain_score(domain, tiers)
        details.append({
            "url":    url,
            "domain": domain,
            "score":  score,
        })
 
    mean_score = sum(d["score"] for d in details) / len(details)
 
    return round(mean_score, 4), details
import PropTypes from 'prop-types';
import { BookOpen, ExternalLink } from 'lucide-react';

const MAX_VISIBLE_SOURCES = 5;

function formatHostname(url) {
    try {
        return new URL(url).hostname.replace(/^www\./, '');
    } catch {
        return '';
    }
}

function canonicalizeUrl(value) {
    if (typeof value !== 'string' || !value.trim()) return '';

    try {
        const parsed = new URL(value.trim());
        parsed.search = '';
        parsed.hash = '';
        const normalized = `${parsed.protocol}//${parsed.hostname}${parsed.pathname}`;
        return normalized.replace(/\/$/, '').toLowerCase();
    } catch {
        return '';
    }
}

function canonicalizeLabel(value) {
    if (typeof value !== 'string' || !value.trim()) return '';

    return value
        .trim()
        .toLowerCase()
        .replace(/[\W_]+/g, ' ')
        .replace(/\s+/g, ' ');
}

function getSourceHref(source) {
    if (typeof source.url === 'string' && /^https?:\/\//i.test(source.url)) {
        return source.url;
    }

    if (typeof source.source === 'string' && /^https?:\/\//i.test(source.source)) {
        return source.source;
    }

    return '';
}

function prettifyLabel(source) {
    const raw = (source.title || source.source || '').trim();
    if (!raw) return '';

    if (/^https?:\/\//i.test(raw)) {
        const hostname = formatHostname(raw);
        return hostname || raw;
    }

    return raw
        .replace(/^EPID[_-]?Reports?[_-]?/i, 'EPID Report ')
        .replace(/^MOH[_-]?Main[_-]?/i, 'MOH ')
        .replace(/[_-]+/g, ' ')
        .replace(/\.pdf$/i, '')
        .replace(/\s+/g, ' ')
        .trim();
}

function normalizeSources(sources) {
    const unique = [];
    const seen = new Set();

    for (const src of sources) {
        const label = prettifyLabel(src);
        if (!label || label.toLowerCase().startsWith('unknown')) continue;

        const href = getSourceHref(src);
        const canonicalHref = canonicalizeUrl(href);
        const canonicalLabel = canonicalizeLabel(label);
        const key = canonicalHref || canonicalLabel;

        if (seen.has(key)) continue;
        seen.add(key);

        unique.push({
            ...src,
            label,
            href,
            dedupeKey: key,
        });

        if (unique.length >= MAX_VISIBLE_SOURCES) break;
    }

    return unique;
}

export default function Citations({ sources }) {
    if (!sources || sources.length === 0) return null;

    const unique = normalizeSources(sources);

    if (unique.length === 0) return null;

    return (
        <section className="citations-wrapper" aria-label="Sources">
            <div className="citations-header">
                <div className="citations-title">
                    <BookOpen size={14} />
                    <span>Sources</span>
                </div>
                <div className="citations-count">{unique.length}</div>
            </div>

            <div className="citations-panel">
                {unique.map((src, index) => {
                    return src.href ? (
                        <a
                            key={src.dedupeKey || `${src.label}-${index}`}
                            href={src.href}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="citation-link-item"
                        >
                            <span className="citation-link-label">{src.label}</span>
                            <ExternalLink size={13} className="citation-link-icon" />
                        </a>
                    ) : (
                        <span key={src.dedupeKey || `${src.label}-${index}`} className="citation-link-item citation-link-item-static" title={src.label}>
                            <span className="citation-link-label">{src.label}</span>
                        </span>
                    );
                })}
            </div>
        </section>
    );
}

Citations.propTypes = {
    sources: PropTypes.arrayOf(PropTypes.shape({
        id: PropTypes.string,
        title: PropTypes.string,
        source: PropTypes.string,
        url: PropTypes.string,
        excerpt: PropTypes.string,
        score: PropTypes.number,
        chunk_index: PropTypes.number,
    })),
};

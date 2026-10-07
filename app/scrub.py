"""PII scrubbing for the side channels - logs and traces, not just the answer.

A cross-cutting guard, and the lesson the red-team pass teaches at the Retrieval layer:
PII rarely leaks through the answer you watch; it leaks through the surfaces that quietly
emit text - application logs, execution traces, exception messages, eval fixtures. So the
scrubber is installed as a logging FILTER on the root logger's HANDLERS: every log line
the app emits is redacted on the way out, whichever module wrote it.

`scrub()` is a pure function so it is trivially unit-testable and reusable anywhere text
is about to cross a boundary (a log, a span, an error surfaced to a client).
"""
from __future__ import annotations

import logging
import re

# Deliberately conservative patterns - catch the common shapes, never touch anything else.
# Bounded parts (RFC 5321: 64-character local part): an unbounded `+` scanned a long run with
# no "@" in quadratic time - a 16 KB query string blocked the worker for a third of a second.
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9.-]{1,255}\.[A-Za-z]{2,24}")
# An ISO-8601 date or date-time (2026-09-11, 2026-09-11 14:23:05, 2026-09-11T14:23:05Z). A
# timestamp carries 8-14 digits, so it is protected BEFORE the number patterns run - log lines
# are full of them, and a timestamp is not PII. A model pin's date is covered the same way.
_ISO_STAMP = re.compile(r"(?<!\d)\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?(?!\d)")
# A payment-card-shaped run: 13-19 digits, optionally grouped by single spaces or dashes. It
# is labelled a card only if it passes the Luhn checksum every issued card number carries.
_CARD_CANDIDATE = re.compile(r"(?<!\w)\d(?:[ -]?\d){12,18}(?!\w)")
# A phone-shaped run: an optional +, then digits with spaces/dashes/parens between them.
# Only a CANDIDATE - `_phone_sub` requires >=10 digits, so a short run (a version, a count)
# is left alone. Anything with 10+ digits that is not a date or a card is treated as a phone.
_PHONE_CANDIDATE = re.compile(r"(?<!\w)\+?\d[\d\s().-]{6,}\d(?!\w)")
_MIN_PHONE_DIGITS = 10

EMAIL_REDACTION = "[REDACTED_EMAIL]"
CARD_REDACTION = "[REDACTED_CARD]"
PHONE_REDACTION = "[REDACTED_PHONE]"


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _card_sub(match: re.Match) -> str:
    digits = re.sub(r"\D", "", match.group(0))
    return CARD_REDACTION if _luhn_ok(digits) else match.group(0)


def _phone_sub(match: re.Match) -> str:
    digits = re.sub(r"\D", "", match.group(0))
    return PHONE_REDACTION if len(digits) >= _MIN_PHONE_DIGITS else match.group(0)


def _scrub_numbers(text: str) -> str:
    return _PHONE_CANDIDATE.sub(_phone_sub, _CARD_CANDIDATE.sub(_card_sub, text))


def scrub(text: str) -> str:
    """Redact emails, card numbers and phone numbers. Pure, deterministic, no network.

    ISO dates and timestamps pass through untouched: the number patterns run only on the
    text between them.
    """
    if not text:
        return text
    text = _EMAIL.sub(EMAIL_REDACTION, text)
    out, last = [], 0
    for stamp in _ISO_STAMP.finditer(text):
        out += [_scrub_numbers(text[last:stamp.start()]), stamp.group(0)]
        last = stamp.end()
    out.append(_scrub_numbers(text[last:]))
    return "".join(out)


class PIIScrubFilter(logging.Filter):
    """A logging filter that scrubs PII from every record before it is emitted.

    Attached to the root logger's handlers at startup, so a stack trace echoing a raw email
    or a span that logged a full prompt is redacted on the way out - the side-channel leak
    the red-team pass hunts for never reaches the log file.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # Scrub the fully-rendered message (args folded in), then blank the args so the
        # formatter does not re-expand an un-scrubbed % template downstream.
        try:
            record.msg = scrub(record.getMessage())
            record.args = ()
        except Exception:  # logging must never raise from inside a filter
            pass
        # The message is only half the record. `logger.exception(...)` carries the traceback
        # separately in `exc_info`, and the formatter renders it AFTER the message - so a
        # filter that only rewrites `msg` still lets a raw email through in the stack trace.
        # Render it here, scrub it, and cache it in `exc_text`: logging.Formatter uses that
        # cache verbatim. Then drop `exc_info` for the same reason `args` is blanked above -
        # so nothing downstream can re-render the un-scrubbed original.
        try:
            if record.exc_info:
                record.exc_text = logging.Formatter().formatException(record.exc_info)
                record.exc_info = None
            if record.exc_text:
                record.exc_text = scrub(record.exc_text)
            if record.stack_info:
                record.stack_info = scrub(record.stack_info)
        except Exception:  # logging must never raise from inside a filter
            pass
        return True


class AccessLogScrubFilter(logging.Filter):
    """The same redaction for uvicorn's ACCESS log, which needs its own filter.

    Its formatter unpacks `record.args` (client, method, path, version, status), so the args
    are scrubbed in place rather than blanked - and the path is URL-decoded first, because
    an email in a query string arrives as `jane.doe%40example.com`, which no regex for `@`
    would ever match.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        from urllib.parse import unquote

        try:
            if isinstance(record.args, tuple):
                record.args = tuple(scrub(unquote(a)) if isinstance(a, str) else a
                                    for a in record.args)
        except Exception:  # logging must never raise from inside a filter
            pass
        return True


def _add_once(handler: logging.Handler, filter_type: type[logging.Filter]) -> None:
    if not any(isinstance(f, filter_type) for f in handler.filters):
        handler.addFilter(filter_type())


def install_pii_scrubbing() -> None:
    """Attach the PII filters to every handler that emits a log line (idempotent).

    A filter on a logger only sees records logged directly to that logger; a filter on a
    HANDLER sees every record that reaches it, including those propagated up from child
    loggers - so the app's scrubber goes on the root logger's handlers. uvicorn's own
    loggers do NOT propagate to root (`propagate: false`, each with its own handler), so
    their handlers get a filter too - otherwise the access log prints every query string,
    user ids and emails included, straight past the root scrubber.
    """
    for name in ("", "uvicorn", "uvicorn.error"):      # "" is the root logger
        for handler in logging.getLogger(name).handlers:
            _add_once(handler, PIIScrubFilter)
    for handler in logging.getLogger("uvicorn.access").handlers:
        _add_once(handler, AccessLogScrubFilter)

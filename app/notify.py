"""Telling the student a request arrived - so it is never lost to a restart.

Intro requests live in memory (app/tools.py), and Render's free plan sleeps and restarts. So
the moment `request_intro` creates one, a copy goes to the student by email - Week 1's
ReleaseBot pattern: Gmail SMTP over SSL (port 465) with an App Password, a plain-text body
built from structured fields, and a result dict instead of an exception.

Three choices for a public URL:
  * The recipient is FIXED - NOTIFY_EMAIL (or SMTP_SENDER) from the environment. Neither the
    visitor nor the model can choose who gets mail, so the tool cannot be turned into a relay.
  * It runs off the request path (a background thread): a slow or failing mail server never
    delays or breaks the visitor's answer; a failure is logged and the request still waits in
    /admin.
  * It is optional: without SMTP_SENDER and SMTP_PASSWORD nothing is sent.
"""
from __future__ import annotations

import logging
import smtplib
import ssl
import threading
from email.mime.text import MIMEText
from typing import Any, Callable

from app.config import get_settings

logger = logging.getLogger(__name__)


def enabled() -> bool:
    s = get_settings()
    return bool(s.smtp_sender and s.smtp_password)


def _one_line(text: str) -> str:
    """No line breaks in a header - a name like "x\\nBcc: ..." cannot add headers."""
    return " ".join(str(text).split())


def intro_email(action: Any) -> tuple[str, str]:
    """(subject, body) for one intro request - built from its validated fields."""
    who = action.name + (f" ({action.company})" if action.company else "")
    subject = _one_line(f"PortfolioAgent: intro request from {who} - {action.reason}")[:150]
    body = (f"{who} asked to get in touch ({action.reason}).\n\n"
            f"Contact: {action.contact}\n\n"
            f"Message:\n{action.message}\n\n"
            f"Request id {action.id} is waiting on your /admin page - approve or reject it there.")
    return subject, body


def send_intro(action: Any) -> dict[str, Any]:
    """Email the student about one intro request. Never raises."""
    s = get_settings()
    if not enabled():
        return {"success": False, "error": "email not configured"}
    to = s.notify_email or s.smtp_sender
    subject, body = intro_email(action)
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = s.smtp_sender
    msg["To"] = to
    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context,
                              timeout=s.model_timeout_s) as server:
            server.login(s.smtp_sender, s.smtp_password)
            server.sendmail(s.smtp_sender, to, msg.as_string())
        logger.info("intro %s emailed to the student", action.id)
        return {"success": True, "to": to, "subject": subject}
    except Exception as exc:  # SMTPException, socket.gaierror, ssl.SSLError, timeouts...
        logger.exception("intro %s: email failed", action.id)
        return {"success": False, "error": str(exc)}


def _background(fn: Callable[[], Any]) -> None:
    threading.Thread(target=fn, daemon=True).start()


def notify_intro(action: Any) -> bool:
    """Queue the email off the request path. Returns whether a send was queued."""
    if not enabled():
        return False
    _background(lambda: send_intro(action))
    return True

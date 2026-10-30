"""Send the report over SMTP. Config comes from environment variables (or a .env file)."""
import os
import smtplib
from email.message import EmailMessage

REQUIRED = ["SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "EMAIL_TO"]


def load_env(path):
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def send_report(subject, markdown, html_body, attachments=(), inline_image=None):
    missing = [k for k in REQUIRED if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Missing SMTP settings: {', '.join(missing)} (see .env.example)")
    host = os.environ["SMTP_HOST"]
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ["SMTP_USER"]
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = os.environ.get("EMAIL_FROM", user)
    msg["To"] = os.environ["EMAIL_TO"]
    msg.set_content(markdown)
    if inline_image:
        html_body = html_body.replace(f"src='{os.path.basename(inline_image)}'", "src='cid:chart'")
    msg.add_alternative(html_body, subtype="html")
    if inline_image:
        with open(inline_image, "rb") as f:
            msg.get_payload()[1].add_related(f.read(), "image", "png", cid="<chart>")
    for path in attachments:
        with open(path, "rb") as f:
            msg.add_attachment(f.read(), maintype="application", subtype="octet-stream",
                               filename=os.path.basename(path))
    if port == 465:
        s = smtplib.SMTP_SSL(host, port, timeout=60)
    else:
        s = smtplib.SMTP(host, port, timeout=60)
        s.starttls()
    with s:
        s.login(user, os.environ["SMTP_PASSWORD"])
        s.send_message(msg)

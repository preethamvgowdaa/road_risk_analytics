"""
Municipal email dispatch.

Ships in dry run mode by default: calling send_report without explicitly
setting dry_run=False only logs what would have been sent and returns that
log entry. This matters because it means the full pipeline, including the
"Dispatch Municipal Email (in progress)" module from the project report,
can be demonstrated end to end in a review without anyone accidentally
emailing a real inbox from a laptop.


"""

import json
import os
import smtplib
from dataclasses import dataclass
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List, Optional

from src.config import CIVIC_CONTACTS_PATH, OUTPUT_DIR

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
DISPATCH_LOG_PATH = os.path.join(OUTPUT_DIR, "dispatch_log.jsonl")


@dataclass
class DispatchResult:
    success: bool
    dry_run: bool
    recipient: Optional[str]
    message: str
    timestamp: str


def load_civic_contacts(path: str = CIVIC_CONTACTS_PATH) -> Dict[str, Dict[str, str]]:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def resolve_recipient(city: str, ward: str, contacts: Dict[str, Dict[str, str]]) -> Optional[str]:
    city_contacts = contacts.get(city, {})
    if ward in city_contacts:
        return city_contacts[ward]
    return city_contacts.get("default")


class MunicipalEmailDispatcher:
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.contacts = load_civic_contacts()
        self.sender_email = os.environ.get("SMTP_SENDER_EMAIL")
        self.app_password = os.environ.get("SMTP_APP_PASSWORD")

    def _log(self, result: DispatchResult) -> None:
        with open(DISPATCH_LOG_PATH, "a") as f:
            f.write(json.dumps(result.__dict__) + "\n")

    def send_report(self, city: str, ward: str, subject: str, body: str, override_recipient: Optional[str] = None) -> DispatchResult:
        recipient = override_recipient or resolve_recipient(city, ward, self.contacts)
        timestamp = datetime.now().isoformat()

        if not recipient:
            result = DispatchResult(
                success=False,
                dry_run=self.dry_run,
                recipient=None,
                message=f"No civic contact configured for {ward}, {city}. Add one to civic_contacts.json.",
                timestamp=timestamp,
            )
            self._log(result)
            return result

        if self.dry_run:
            result = DispatchResult(
                success=True,
                dry_run=True,
                recipient=recipient,
                message="Dry run: email was composed but not actually sent.",
                timestamp=timestamp,
            )
            self._log(result)
            return result

        if not self.sender_email or not self.app_password:
            result = DispatchResult(
                success=False,
                dry_run=False,
                recipient=recipient,
                message="SMTP_SENDER_EMAIL and SMTP_APP_PASSWORD environment variables are not set.",
                timestamp=timestamp,
            )
            self._log(result)
            return result

        try:
            message = MIMEMultipart()
            message["From"] = self.sender_email
            message["To"] = recipient
            message["Subject"] = subject
            message.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
                server.starttls()
                server.login(self.sender_email, self.app_password)
                server.sendmail(self.sender_email, recipient, message.as_string())

            result = DispatchResult(
                success=True,
                dry_run=False,
                recipient=recipient,
                message="Email sent successfully.",
                timestamp=timestamp,
            )
        except Exception as exc:
            result = DispatchResult(
                success=False,
                dry_run=False,
                recipient=recipient,
                message=f"Send failed: {exc}",
                timestamp=timestamp,
            )

        self._log(result)
        return result

    def read_dispatch_log(self) -> List[dict]:
        if not os.path.exists(DISPATCH_LOG_PATH):
            return []
        entries = []
        with open(DISPATCH_LOG_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

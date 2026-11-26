import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')

TICKETS = [
    {
        "id": "T-1001",
        "product": "CloudStore Pro",
        "issue": "Cannot connect to database from application.",
        "resolution": "Update the firewall rules in the CloudStore dashboard to allow IP 0.0.0.0/0 or specific application IP. Ensure port 5432 is open.",
        "category": "Connectivity"
    },
    {
        "id": "T-1002",
        "product": "CloudStore Pro",
        "issue": "Backup failed with error code 503.",
        "resolution": "Error 503 means the storage quota is exceeded. Upgrade the storage plan or delete old backups via the settings panel.",
        "category": "Storage"
    },
    {
        "id": "T-1003",
        "product": "AuthMagic",
        "issue": "Users are not receiving password reset emails.",
        "resolution": "Check the SMTP settings in AuthMagic dashboard. If using default, emails might be marked as spam. We recommend configuring a custom SendGrid API key.",
        "category": "Email"
    },
    {
        "id": "T-1004",
        "product": "AuthMagic",
        "issue": "SSO login failing with 'Invalid Issuer' error.",
        "resolution": "The identity provider URL in AuthMagic settings must exactly match the SAML issuer URL provided by your identity provider (e.g., Okta).",
        "category": "Authentication"
    },
    {
        "id": "T-1005",
        "product": "WidgetUI",
        "issue": "Table component is freezing when rendering 10k rows.",
        "resolution": "Enable virtual scrolling in the Table component props by passing `virtualize={True}` and `rowHeight={40}`.",
        "category": "Performance"
    }
]

def generate():
    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, 'support_tickets.json')
    with open(out_path, 'w') as f:
        json.dump(TICKETS, f, indent=4)
    print(f"Generated {len(TICKETS)} support tickets at {out_path}")

if __name__ == "__main__":
    generate()

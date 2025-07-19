"""
QuickBooks API integration for QuickBooks Data Cleaner
"""

import json
import time
from typing import Any, Dict
from urllib.parse import urlencode

import requests
from tasks import clean_file
from flask import Flask, request, jsonify
from celery.result import AsyncResult

QB_AUTH_BASE = "https://appcenter.intuit.com/connect/oauth2"
QB_TOKEN_URL = "https://oauth.platform.intuit.com/oauth2/v1/tokens/bearer"
QB_API_BASE = "https://quickbooks.api.intuit.com/v3/company"


class QuickBooksClient:
    def __init__(self, client_id, client_secret, redirect_uri, realm_id):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.realm_id = realm_id
        self.tokens: Dict[str, Any] = {}

    # ---------- AUTH ----------
    def auth_url(self, scopes: list[str] = None) -> str:
        scopes = scopes or ["com.intuit.quickbooks.accounting"]
        params = dict(
            client_id=self.client_id,
            response_type="code",
            scope=" ".join(scopes),
            redirect_uri=self.redirect_uri,
            state="qbdc-demo",
        )
        return f"{QB_AUTH_BASE}?{urlencode(params)}"

    def fetch_tokens(self, code: str) -> None:
        data = dict(
            grant_type="authorization_code",
            code=code,
            redirect_uri=self.redirect_uri,
        )
        r = requests.post(
            QB_TOKEN_URL,
            data=data,
            auth=(self.client_id, self.client_secret),
            headers={"Accept": "application/json"},
            timeout=30,
        )
        r.raise_for_status()
        self.tokens = r.json() | {"fetched_at": time.time()}
        self._persist_tokens()

    def refresh_tokens(self) -> None:
        if time.time() - self.tokens.get("fetched_at", 0) < 3300:
            return  # still valid
        data = dict(
            grant_type="refresh_token",
            refresh_token=self.tokens["refresh_token"],
        )
        r = requests.post(
            QB_TOKEN_URL,
            data=data,
            auth=(self.client_id, self.client_secret),
            headers={"Accept": "application/json"},
            timeout=30,
        )
        r.raise_for_status()
        self.tokens.update(r.json(), fetched_at=time.time())
        self._persist_tokens()

    # ---------- DATA ----------
    def get_transactions(self, start, end):
        self.refresh_tokens()
        url = (
            f"{QB_API_BASE}/{self.realm_id}/query"
            f"?query=select * from Transaction where TxnDate "
            f">='{start}' and TxnDate <='{end}'"
        )
        r = requests.get(
            url,
            headers={
                "Authorization": f"Bearer {self.tokens['access_token']}",
                "Accept": "application/json",
            },
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    # ---------- HELPERS ----------
    def _persist_tokens(self, path: str = "token.json") -> None:
        with open(path, "w") as f:
            json.dump(self.tokens, f, indent=2)


app = Flask(__name__)

@app.route('/api/clean-async', methods=['POST'])
def clean_async():
    data = request.json
    tmp_path = data['file_path']
    rules_path = data.get('rules_path')
    use_gemini = data.get('use_gemini', False)
    task = clean_file.delay(tmp_path, rules_path, use_gemini)
    return jsonify({"job_id": task.id})

@app.route('/api/job-status/<job_id>')
def job_status(job_id):
    task = clean_file.AsyncResult(job_id)
    if task.state == 'PENDING':
        return jsonify({"state": "PENDING"})
    elif task.state == 'SUCCESS':
        return jsonify({"state": "SUCCESS", "result": task.result})
    elif task.state == 'FAILURE':
        return jsonify({"state": "FAILURE", "error": str(task.info)})
    else:
        return jsonify({"state": task.state})
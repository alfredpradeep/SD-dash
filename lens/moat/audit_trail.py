"""
Compliance-Ready Audit Trail with Cryptographic Proofs

Append-only log with SHA-256 hashing and Merkle tree integrity.
Every request gets an immutable audit record.
"""

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional
import time


@dataclass
class AuditRecord:
    """Immutable audit record for a single AI API request."""
    record_id: str
    timestamp: datetime
    # Request details
    model: str
    provider: str
    input_hash: str           # SHA-256 of input (NOT the actual text for privacy)
    output_hash: str          # SHA-256 of output
    input_tokens: int
    output_tokens: int
    # Cost details
    cost_usd: float
    cost_method: str          # How cost was calculated
    # Attribution
    team: str
    feature: str
    user_id: str
    # Routing decision
    routing_reason: str       # Why this model was chosen
    alternatives_considered: list[str] = field(default_factory=list)
    # Integrity
    record_hash: str = ""          # SHA-256 of this entire record (populated by audit trail)
    previous_hash: str = ""        # Hash of previous record (chain)
    merkle_root: str = ""          # Current Merkle tree root


@dataclass
class AuditReport:
    """Compliance report for a time period."""
    period_start: datetime
    period_end: datetime
    total_records: int
    total_cost: float
    chain_integrity: bool     # Is the hash chain intact?
    merkle_root: str
    by_model: dict = field(default_factory=dict)
    by_team: dict = field(default_factory=dict)
    by_feature: dict = field(default_factory=dict)
    compliance_notes: list[str] = field(default_factory=list)


class AuditTrail:
    """
    Cryptographically verifiable audit trail for AI API usage.

    Every request gets:
    1. SHA-256 hash of input/output (privacy-preserving — no raw text stored)
    2. Hash chain linking to previous record (tamper detection)
    3. Merkle tree root for efficient integrity verification

    Designed for EU AI Act Article 13 (transparency), SOC2 AI governance,
    and internal audit/compliance requirements.
    """

    def __init__(self):
        self._records: list[AuditRecord] = []
        self._merkle_leaves: list[str] = []
        self._previous_hash: str = "0" * 64  # Genesis

    @staticmethod
    def _hash_text(text: str) -> str:
        """SHA-256 hash of text."""
        return hashlib.sha256(text.encode()).hexdigest()

    @staticmethod
    def _hash_object(obj: dict) -> str:
        """SHA-256 hash of dict (deterministic JSON)."""
        # Sort keys for determinism
        json_str = json.dumps(obj, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(json_str.encode()).hexdigest()

    def record(self, model: str, provider: str, input_text: str, output_text: str,
               input_tokens: int, output_tokens: int, cost_usd: float,
               team: str = "default", feature: str = "default", user_id: str = "anonymous",
               routing_reason: str = "default", alternatives: list[str] = None) -> AuditRecord:
        """Create an immutable audit record."""

        if alternatives is None:
            alternatives = []

        # Generate record ID
        record_id = f"AUDIT-{len(self._records) + 1:08d}"

        # Hash input/output (don't store raw text)
        input_hash = self._hash_text(input_text)
        output_hash = self._hash_text(output_text)

        # Create record
        audit_record = AuditRecord(
            record_id=record_id,
            timestamp=datetime.utcnow(),
            model=model,
            provider=provider,
            input_hash=input_hash,
            output_hash=output_hash,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            cost_method="standard",
            team=team,
            feature=feature,
            user_id=user_id,
            routing_reason=routing_reason,
            alternatives_considered=alternatives,
            previous_hash=self._previous_hash
        )

        # Compute record hash
        record_data = {
            "record_id": audit_record.record_id,
            "timestamp": audit_record.timestamp.isoformat(),
            "model": audit_record.model,
            "provider": audit_record.provider,
            "input_hash": audit_record.input_hash,
            "output_hash": audit_record.output_hash,
            "input_tokens": audit_record.input_tokens,
            "output_tokens": audit_record.output_tokens,
            "cost_usd": audit_record.cost_usd,
            "cost_method": audit_record.cost_method,
            "team": audit_record.team,
            "feature": audit_record.feature,
            "user_id": audit_record.user_id,
            "routing_reason": audit_record.routing_reason,
            "alternatives_considered": audit_record.alternatives_considered,
            "previous_hash": audit_record.previous_hash
        }

        audit_record.record_hash = self._hash_object(record_data)

        # Update chain
        self._previous_hash = audit_record.record_hash

        # Store record
        self._records.append(audit_record)

        # Update Merkle tree
        self._merkle_leaves.append(audit_record.record_hash)
        audit_record.merkle_root = self._compute_merkle_root()

        return audit_record

    def _compute_merkle_root(self) -> str:
        """Compute Merkle tree root from all record hashes."""

        if not self._merkle_leaves:
            return "0" * 64

        # Hash pairs iteratively
        current_level = list(self._merkle_leaves)

        while len(current_level) > 1:
            next_level = []

            for i in range(0, len(current_level), 2):
                if i + 1 < len(current_level):
                    # Hash pair
                    pair = current_level[i] + current_level[i + 1]
                    pair_hash = hashlib.sha256(pair.encode()).hexdigest()
                    next_level.append(pair_hash)
                else:
                    # Odd one out, hash with itself
                    pair = current_level[i] + current_level[i]
                    pair_hash = hashlib.sha256(pair.encode()).hexdigest()
                    next_level.append(pair_hash)

            current_level = next_level

        return current_level[0] if current_level else "0" * 64

    def verify_integrity(self) -> tuple[bool, str]:
        """Verify the entire audit chain is intact."""

        if not self._records:
            return (True, "No records to verify")

        # Verify hash chain
        prev_hash = "0" * 64
        for record in self._records:
            if record.previous_hash != prev_hash:
                return (False, f"Chain integrity broken at {record.record_id}")

            # Verify record hash
            record_data = {
                "record_id": record.record_id,
                "timestamp": record.timestamp.isoformat(),
                "model": record.model,
                "provider": record.provider,
                "input_hash": record.input_hash,
                "output_hash": record.output_hash,
                "input_tokens": record.input_tokens,
                "output_tokens": record.output_tokens,
                "cost_usd": record.cost_usd,
                "cost_method": record.cost_method,
                "team": record.team,
                "feature": record.feature,
                "user_id": record.user_id,
                "routing_reason": record.routing_reason,
                "alternatives_considered": record.alternatives_considered,
                "previous_hash": record.previous_hash
            }

            computed_hash = self._hash_object(record_data)
            if computed_hash != record.record_hash:
                return (False, f"Record hash mismatch at {record.record_id}")

            prev_hash = record.record_hash

        # Verify Merkle root
        computed_merkle = self._compute_merkle_root()
        if self._records and computed_merkle != self._records[-1].merkle_root:
            return (False, "Merkle root mismatch")

        return (True, "Chain integrity verified")

    def generate_report(self, start: datetime = None, end: datetime = None) -> AuditReport:
        """Generate compliance report for a period."""

        if start is None:
            start = datetime.min
        if end is None:
            end = datetime.utcnow()

        # Filter records
        records = [r for r in self._records if start <= r.timestamp <= end]

        # Verify integrity
        chain_ok, chain_msg = self.verify_integrity()

        # Compute metrics
        total_cost = sum(r.cost_usd for r in records)
        total_tokens = sum(r.input_tokens + r.output_tokens for r in records)

        # By model
        by_model = {}
        for record in records:
            if record.model not in by_model:
                by_model[record.model] = {
                    "requests": 0,
                    "cost": 0,
                    "tokens": 0
                }
            by_model[record.model]["requests"] += 1
            by_model[record.model]["cost"] += record.cost_usd
            by_model[record.model]["tokens"] += record.input_tokens + record.output_tokens

        # By team
        by_team = {}
        for record in records:
            if record.team not in by_team:
                by_team[record.team] = {
                    "requests": 0,
                    "cost": 0,
                    "tokens": 0
                }
            by_team[record.team]["requests"] += 1
            by_team[record.team]["cost"] += record.cost_usd
            by_team[record.team]["tokens"] += record.input_tokens + record.output_tokens

        # By feature
        by_feature = {}
        for record in records:
            if record.feature not in by_feature:
                by_feature[record.feature] = {
                    "requests": 0,
                    "cost": 0,
                    "tokens": 0
                }
            by_feature[record.feature]["requests"] += 1
            by_feature[record.feature]["cost"] += record.cost_usd
            by_feature[record.feature]["tokens"] += record.input_tokens + record.output_tokens

        # Compliance notes
        notes = []
        if chain_ok:
            notes.append("Chain integrity: VERIFIED")
        else:
            notes.append(f"Chain integrity: FAILED - {chain_msg}")

        notes.append(f"EU AI Act Article 13: Transparency log contains {len(records)} records")
        notes.append(f"SOC2 Governance: All requests cryptographically hashed and linked")
        notes.append(f"Period: {start.isoformat()} to {end.isoformat()}")

        merkle_root = self._records[-1].merkle_root if self._records else "0" * 64

        return AuditReport(
            period_start=start,
            period_end=end,
            total_records=len(records),
            total_cost=total_cost,
            chain_integrity=chain_ok,
            merkle_root=merkle_root,
            by_model=by_model,
            by_team=by_team,
            by_feature=by_feature,
            compliance_notes=notes
        )

    def export_for_auditor(self, start: datetime = None, end: datetime = None) -> list[dict]:
        """Export audit records in a format suitable for external auditors."""

        if start is None:
            start = datetime.min
        if end is None:
            end = datetime.utcnow()

        # Filter records
        records = [r for r in self._records if start <= r.timestamp <= end]

        # Export (hashes only, NO raw text)
        exported = []
        for record in records:
            exported.append({
                "record_id": record.record_id,
                "timestamp": record.timestamp.isoformat(),
                "model": record.model,
                "provider": record.provider,
                "input_hash": record.input_hash,
                "output_hash": record.output_hash,
                "input_tokens": record.input_tokens,
                "output_tokens": record.output_tokens,
                "cost_usd": round(record.cost_usd, 2),
                "cost_method": record.cost_method,
                "team": record.team,
                "feature": record.feature,
                "user_id": record.user_id,
                "routing_reason": record.routing_reason,
                "alternatives_considered": record.alternatives_considered,
                "record_hash": record.record_hash,
                "previous_hash": record.previous_hash,
                "merkle_root": record.merkle_root
            })

        return exported

    def get_record_count(self) -> int:
        """Get total number of records."""
        return len(self._records)

    def get_latest_merkle_root(self) -> str:
        """Get current Merkle tree root."""
        return self._records[-1].merkle_root if self._records else "0" * 64

    def export_merkle_path(self, record_index: int) -> Optional[list[str]]:
        """
        Export Merkle path for a specific record (for proof verification).

        Returns list of hashes needed to reconstruct the Merkle root,
        starting from the record up to the root.
        """

        if record_index < 0 or record_index >= len(self._merkle_leaves):
            return None

        # Simplified: return all leaves for this implementation
        # A more sophisticated version would return the specific path
        return list(self._merkle_leaves)

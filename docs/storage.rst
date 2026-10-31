Local and S3 run transfers
==========================

Execution always stages evidence in local ``.mlef/runs/RUN`` storage. Explicit
publication and retrieval copy terminal, cleanup-complete runs between projects
and a local directory or S3 store. No daemon, hosted account, automatic background
upload or remote execution is introduced.

.. code-block:: text

    mlef publish RUN --to /backups/mlef --json
    mlef fetch RUN --from /backups/mlef --json

For S3, install the optional cloud extra in the **host** environment, not the
experiment environment:

.. code-block:: text

    uv tool install '.[s3]'
    mlef publish RUN --to s3://my-bucket/experiments --json
    mlef fetch RUN --from s3://my-bucket/experiments --json

Run installation from the MLEF checkout. For development, use
``uv sync --locked --extra dev --extra s3``. The default
package does not require or import boto3. S3 uses boto3's normal credential,
profile, region, endpoint and retry configuration, including temporary role
credentials. No credentials are accepted in the store URI or written to transfer
manifests. Configure AWS credentials outside MLEF; never put secrets in source,
CLI arguments or committed configuration. Restrict access to the selected prefix.
Transfers need ``s3:GetObject`` and conditional ``s3:PutObject`` permissions;
listing and deletion are not used. Missing-object detection may also require
``s3:ListBucket`` permission scoped to the prefix; a 403 is an error, not absence.
Compatible S3 services must support ``If-None-Match: *`` conditional writes.

.. code-block:: python

    from mlef import api

    manifest = api.publish_run(run_id, "s3://my-bucket/experiments", root=project)
    api.fetch_run(run_id, "s3://my-bucket/experiments", root=other_project)
    api.pipeline_status(run_id, root=other_project, verify=True)

Both functions return a versioned ``mlef.transfers.RunTransfer`` with the original
run ID, relative paths, empty directories, byte sizes and SHA256 hashes. The CLI
uses the same services. Existing offline status, artifact retrieval and pipeline
reuse/resume operate on the fetched local run without the original workspace or
cloud credentials. Fetch can initialize local run storage in an existing Git
project; it never overwrites an existing run, even an incomplete one.

Publication and integrity
-------------------------

Publication takes the run ownership lock, snapshots regular files locally, and
checks that the source stayed unchanged during copying. Active runs and pending
resource cleanup are refused. Source, captured SDK, logs, metadata and artifacts
are copied as-is; ownership locks are excluded. Symlinks, reparse points, hard
links, traversal, nonportable paths and case-insensitive path collisions are
rejected. Empty directories and current run metadata bytes are preserved.

Each store uses ``runs/RUN/snapshots/SHA256/PATH`` payload keys and a final
``runs/RUN/manifest.json`` publication marker. The snapshot identity hashes the
canonical manifest, including its original run ID; objects are not deduplicated
between runs. SHA256 is calculated from actual bytes, never an S3 ETag. Uploads
read back every object before committing the marker. Conditional writes prevent
replacing an existing object or published run. A changed run requires a new run
ID or a different store prefix; publication is not a mutable backup operation.

Interrupted publication leaves no marker until all payloads have been verified.
Retry the same command with the unchanged local run: existing matching objects
are verified and skipped. A failed transfer can leave unreferenced remote objects;
automatic deletion, retention and garbage collection are deliberately absent.
Do not delete snapshot objects independently of their manifests.

Fetch reads only the committed manifest, streams payloads into a temporary local
directory, checks exact sizes and hashes, validates terminal run identity and
then renames the complete directory into local run storage. Corrupt, missing,
oversized or interrupted downloads do not expose a partial run; retry the fetch.
The transfer manifest is limited to 16 MiB and 100,000 files/directories each.
Transfers require disk space for staging and have no global storage quota.
S3 upload currently uses single ``PutObject`` requests, not multipart upload;
the service's single-object request limits apply. Retry configuration cannot
resume a partially uploaded individual object.

Integrity does not establish authenticity, safe model deserialization or valid
scientific results. Use a trusted store and protect it against out-of-band
modification/deletion; the manifest is not signed. Pipeline contract/source/SDK
verification remains available through ``pipeline_status(..., verify=True)`` and
is required again by explicit reuse. All saved run files are uploaded, including
logs and captured data; there is no automatic private-data or secret redaction.
Audit evidence and bucket access before publishing sensitive runs. External
datasets not captured in the original run are not discovered or uploaded.

Testing
-------

The ordinary dev suite exercises real boto3 calls against Moto's in-process S3
emulator, plus local-backend interruption, corruption and clean-workspace reuse
checks. This is not a live AWS test. To explicitly test a real S3-compatible
service, set ``MLEF_TEST_S3_URI=s3://test-bucket/test-prefix`` and run
``uv run --locked --extra dev --extra s3 pytest tests/test_transfers.py``.
The opt-in test creates a unique ``mlef-tests/UUID`` prefix and leaves its test
objects for manual cleanup. Use a dedicated test bucket/prefix and credentials.

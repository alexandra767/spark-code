from spark_code.tools.play_publish import PlayPublishTool


def _patch(monkeypatch, tmp_path, *, upload_ok=True, steps=None):
    # a real .aab so glob finds it
    out = tmp_path / "androidApp" / "build" / "outputs" / "bundle" / "release"
    out.mkdir(parents=True, exist_ok=True)
    (out / "androidApp-release.aab").write_bytes(b"AAB")

    async def fake_run_step(argv, cwd=None, timeout=1800.0):
        if steps is not None:
            steps.append("gradle")
        return 0, "BUILD SUCCESSFUL", ""
    monkeypatch.setattr("spark_code.tools.play_publish.run_step", fake_run_step)

    async def fake_create(self, pkg):
        if steps is not None:
            steps.append("create")
        return "E1"

    async def fake_upload(self, pkg, eid, aab):
        if steps is not None:
            steps.append("upload")
        if not upload_ok:
            from spark_code.play import PlayError
            raise PlayError("dup vc")
        return 46

    async def fake_assign(self, pkg, eid, track, vc):
        if steps is not None:
            steps.append("assign")

    async def fake_validate(self, pkg, eid):
        if steps is not None:
            steps.append("validate")
        return {}

    async def fake_commit(self, pkg, eid):
        if steps is not None:
            steps.append("commit")
        return {}

    async def fake_delete(self, pkg, eid):
        if steps is not None:
            steps.append("delete")
    monkeypatch.setattr("spark_code.play.PlayClient.create_edit", fake_create)
    monkeypatch.setattr("spark_code.play.PlayClient.upload_bundle", fake_upload)
    monkeypatch.setattr("spark_code.play.PlayClient.assign_track", fake_assign)
    monkeypatch.setattr("spark_code.play.PlayClient.validate_edit", fake_validate)
    monkeypatch.setattr("spark_code.play.PlayClient.commit_edit", fake_commit)
    monkeypatch.setattr("spark_code.play.PlayClient.delete_edit", fake_delete)


async def test_production_refused_before_anything(monkeypatch, tmp_path):
    steps = []
    _patch(monkeypatch, tmp_path, steps=steps)
    out = await PlayPublishTool().execute(project=str(tmp_path), package="com.x", track="production", confirm="publish")
    assert steps == [] and "production" in out.lower()


async def test_without_confirm_stops_and_deletes_edit(monkeypatch, tmp_path):
    steps = []
    _patch(monkeypatch, tmp_path, steps=steps)
    out = await PlayPublishTool().execute(project=str(tmp_path), package="com.x", track="internal")
    assert "commit" not in steps and "delete" in steps
    assert "confirm" in out.lower() and "46" in out and "internal" in out


async def test_confirmed_runs_full_sequence(monkeypatch, tmp_path):
    steps = []
    _patch(monkeypatch, tmp_path, steps=steps)
    out = await PlayPublishTool().execute(project=str(tmp_path), package="com.x", track="internal", confirm="publish")
    assert steps == ["gradle", "create", "upload", "assign", "validate", "commit"]
    assert "released" in out.lower() and "internal" in out


async def test_upload_failure_deletes_edit_no_commit(monkeypatch, tmp_path):
    steps = []
    _patch(monkeypatch, tmp_path, upload_ok=False, steps=steps)
    out = await PlayPublishTool().execute(project=str(tmp_path), package="com.x", track="internal", confirm="publish")
    assert "commit" not in steps and "delete" in steps and "dup vc" in out


async def test_headless_refuses_without_confirm(monkeypatch, tmp_path):
    steps = []
    _patch(monkeypatch, tmp_path, steps=steps)
    out = await PlayPublishTool(interactive=False).execute(project=str(tmp_path), package="com.x", track="internal")
    assert "commit" not in steps and "confirm=" in out


async def test_non_playerror_post_create_still_deletes_edit(monkeypatch, tmp_path):
    # a NON-PlayError raised after the edit is created must still discard it (no orphan) and never commit
    steps = []
    _patch(monkeypatch, tmp_path, steps=steps)

    async def boom_upload(self, pkg, eid, aab):
        steps.append("upload")
        raise ValueError("bad response shape")
    monkeypatch.setattr("spark_code.play.PlayClient.upload_bundle", boom_upload)
    out = await PlayPublishTool().execute(project=str(tmp_path), package="com.x", track="internal", confirm="publish")
    assert "commit" not in steps and "delete" in steps
    assert "bad response shape" in out

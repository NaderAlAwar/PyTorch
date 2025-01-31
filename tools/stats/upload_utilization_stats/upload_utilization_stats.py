from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import argparse
import json
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd  # type: ignore[import]
from tools.stats.upload_stats_lib import download_s3_artifacts, upload_to_s3
from tools.stats.utilization_stats_lib import (
    getDataModelVersion,
    OssCiSegmentV1,
    OssCiUtilizationMetadataV1,
    OssCiUtilizationTimeSeriesV1,
    UtilizationMetadata,
    UtilizationRecord,
    WorkflowInfo,
)


USAGE_LOG_FILENAME = "usage_log.txt"
CMD_PYTHON_LEVEL = "CMD_PYTHON"
UTILIZATION_BUCKET = "ossci-utilization"
PYTORCH_REPO = "pytorch/pytorch"


class SegmentGenerator:
    """
    generates test segment from utilization records, currently it only generate segments on python commands level
    segment_delta_threshold is the threshold to determine if a segment is continuous or not, default is 60 seconds.
    """

    def generate(
        self, records: list[UtilizationRecord], segment_delta_threshold: int = 60
    ) -> list[OssCiSegmentV1]:
        if len(records) == 0:
            return []

        cmd_col_name = "cmd"
        time_col_name = "time"

        # flatten time series with detected cmds
        df = pd.DataFrame(
            [
                {time_col_name: record.timestamp, cmd_col_name: process}
                for record in records
                for process in (record.cmd_names or [])
            ]
        )
        df[time_col_name] = pd.to_datetime(df[time_col_name], unit="s")

        # get unique cmd names
        unique_cmds_df = pd.DataFrame(df[cmd_col_name].unique(), columns=[cmd_col_name])

        # get all detected python cmds
        cmd_list = unique_cmds_df[
            unique_cmds_df[cmd_col_name].str.startswith("python")
        ][cmd_col_name].tolist()

        # find segments by screening continuoues time series data
        segments: list[OssCiSegmentV1] = []
        for value in cmd_list:
            subset = df[df[cmd_col_name] == value].copy()

            continuous_segments = self._find_continuous_windows(
                segment_delta_threshold, time_col_name, subset
            )
            for row in continuous_segments:
                segment = OssCiSegmentV1(
                    level=CMD_PYTHON_LEVEL,
                    name=value,
                    start_at=row["start_time"].timestamp(),
                    end_at=row["end_time"].timestamp(),
                    extra_info={},
                )
                segments.append(segment)
        print(
            f"[Db Segments] detected pytest cmd: {len(cmd_list)}, generated segments: {len(segments)}"
        )
        return segments

    def _find_continuous_windows(
        self,
        threshold: int,
        time_column_name: str,
        df: Any,  # the lintrunner keep complaining about the type of df, but it's not a problem
    ) -> list[dict[str, Any]]:
        time_threshold = pd.Timedelta(seconds=threshold)
        df = df.sort_values(by=time_column_name).reset_index(drop=True)
        df["time_diff"] = df[time_column_name].diff()
        df["segment"] = (df["time_diff"] > time_threshold).cumsum()
        segments = (
            df.groupby("segment")
            .agg(
                start_time=(time_column_name, "first"),
                end_time=(time_column_name, "last"),
            )
            .reset_index(drop=True)
        )
        return segments[["start_time", "end_time"]].to_dict(orient="records")  # type: ignore[no-any-return]


class UtilizationDbConverter:
    """convert utilization log model to db model"""

    def __init__(
        self,
        info: WorkflowInfo,
        metadata: UtilizationMetadata,
        records: list[UtilizationRecord],
        segments: list[OssCiSegmentV1],
    ):
        self.metadata = metadata
        self.records = records
        self.segments = segments
        self.created_at = datetime.now().timestamp()
        self.info = info
        end_time_stamp = max([record.timestamp for record in records])
        self.end_at = end_time_stamp

    def convert(
        self,
    ) -> tuple[OssCiUtilizationMetadataV1, list[OssCiUtilizationTimeSeriesV1]]:
        db_metadata = self._to_oss_ci_metadata()
        timeseries = self._to_oss_ci_timeseries_list()
        return db_metadata, timeseries

    def _to_oss_ci_metadata(self) -> OssCiUtilizationMetadataV1:
        return OssCiUtilizationMetadataV1(
            repo=self.info.repo,
            workflow_id=self.info.workflow_run_id,
            run_attempt=self.info.run_attempt,
            job_id=self.info.job_id,
            workflow_name=self.info.workflow_name,
            job_name=self.info.job_name,
            usage_collect_interval=self.metadata.usage_collect_interval,
            data_model_version=str(self.metadata.data_model_version),
            created_at=self.created_at,
            gpu_count=self.metadata.gpu_count if self.metadata.gpu_count else 0,
            cpu_count=self.metadata.cpu_count if self.metadata.cpu_count else 0,
            gpu_type=self.metadata.gpu_type if self.metadata.gpu_type else "",
            start_at=self.metadata.start_at,
            end_at=self.end_at,
            segments=self.segments,
            tags=[],
        )

    def _to_oss_ci_timeseries_list(self) -> list[OssCiUtilizationTimeSeriesV1]:
        return [
            self._to_oss_ci_time_series(record, type="utilization", tags=["record"])
            for record in self.records
        ]

    def _to_oss_ci_time_series(
        self, record: UtilizationRecord, type: str, tags: list[str]
    ) -> OssCiUtilizationTimeSeriesV1:
        return OssCiUtilizationTimeSeriesV1(
            created_at=self.created_at,
            type=type,
            tags=tags,
            time_stamp=record.timestamp,
            repo=self.info.repo,
            workflow_id=self.info.workflow_run_id,
            run_attempt=self.info.run_attempt,
            job_id=self.info.job_id,
            workflow_name=self.info.workflow_name,
            job_name=self.info.job_name,
            json_data=str(asdict(record.data) if record.data else {}),
        )


class UploadUtilizationData:
    """
    main class to handle utilization data conversion and s3 upload
    fetches raw log data from s3, convert to log model, then convert to db model, and upload to s3
    """

    def __init__(
        self,
        info: WorkflowInfo,
        dry_run: bool = False,
        debug: bool = False,
    ):
        self.info = info
        self.segment_generator = SegmentGenerator()
        self.debug_mode = debug
        self.dry_run = dry_run

    def start(self) -> None:
        metadata, valid_records, _ = self.get_log_data(
            self.info.workflow_run_id, self.info.job_id, self.info.run_attempt
        )

        if not metadata:
            print("[Log Model] Failed to process test log, metadata is None")
            return None

        if len(valid_records) == 0:
            print("[Log Model] Failed to process test log, no valid records")
            return None
        segments = self.segment_generator.generate(valid_records)

        db_metadata, db_records = UtilizationDbConverter(
            self.info, metadata, valid_records, segments
        ).convert()
        print(
            f"[db model] Peek db metadatga \n: {json.dumps(asdict(db_metadata), indent=4)}"
        )

        if len(db_records) > 0:
            print(
                f"[db model] Peek db timeseries \n:{json.dumps(asdict(db_records[0]), indent=4)}"
            )

        if self.dry_run:
            print("[dry-run-mode]: no upload in dry run mode")
            return

        version = f"v_{db_metadata.data_model_version}"
        metadata_collection = "util_metadata"
        ts_collection = "util_timeseries"
        if self.debug_mode:
            metadata_collection = f"debug_{metadata_collection}"
            ts_collection = f"debug_{ts_collection}"

        self._upload_utilization_data_to_s3(
            collection=metadata_collection,
            version=version,
            repo=self.info.repo,
            workflow_run_id=self.info.workflow_run_id,
            workflow_run_attempt=self.info.run_attempt,
            job_id=self.info.job_id,
            file_name="metadata",
            docs=[asdict(db_metadata)],
        )

        self._upload_utilization_data_to_s3(
            collection=ts_collection,
            version=version,
            repo=self.info.repo,
            workflow_run_id=self.info.workflow_run_id,
            workflow_run_attempt=self.info.run_attempt,
            job_id=self.info.job_id,
            file_name="time_series",
            docs=[asdict(record) for record in db_records],
        )

    def _upload_utilization_data_to_s3(
        self,
        collection: str,
        version: str,
        repo: str,
        workflow_run_id: int,
        workflow_run_attempt: int,
        job_id: int,
        file_name: str,
        docs: list[dict[str, Any]],
    ) -> None:
        bucket_name = UTILIZATION_BUCKET
        key = f"{collection}/{version}/{repo}/{workflow_run_id}/{workflow_run_attempt}/{job_id}/{file_name}"
        upload_to_s3(bucket_name, key, docs)

    def get_log_data(
        self, workflow_run_id: int, job_id: int, workflow_run_attempt: int
    ) -> tuple[
        Optional[UtilizationMetadata], list[UtilizationRecord], list[UtilizationRecord]
    ]:
        artifact_paths = download_s3_artifacts(
            "logs-test", workflow_run_id, workflow_run_attempt, job_id
        )
        if len(artifact_paths) == 0:
            print(
                f"Failed to download artifacts for workflow {workflow_run_id} and job {job_id}"
            )
            return None, [], []
        elif len(artifact_paths) > 1:
            print(
                f"Found more than one artifact for workflow {workflow_run_id} and job {job_id}, {artifact_paths}"
            )
            return None, [], []

        p = artifact_paths[0]
        test_log_content = unzip_file(p, USAGE_LOG_FILENAME)

        metadata, records, error_records = self.convert_to_log_models(test_log_content)
        if metadata is None:
            return None, [], []

        print(f"Converted Log Model: UtilizationMetadata:\n {metadata}")
        return metadata, records, error_records

    def _process_raw_record(
        self, line: str
    ) -> tuple[Optional[UtilizationRecord], bool]:
        try:
            record = UtilizationRecord.from_json(line)
            if record.error:
                return record, False
            return record, True
        except Exception as e:
            print(f"Failed to parse JSON line: {e}")
            return None, False

    def _process_utilization_records(
        self,
        lines: list[str],
    ) -> tuple[list[UtilizationRecord], list[UtilizationRecord]]:
        results = [self._process_raw_record(line) for line in lines[1:]]
        valid_records = [
            record for record, valid in results if valid and record is not None
        ]
        invalid_records = [
            record for record, valid in results if not valid and record is not None
        ]
        return valid_records, invalid_records

    def convert_to_log_models(
        self,
        content: str,
    ) -> tuple[
        Optional[UtilizationMetadata], list[UtilizationRecord], list[UtilizationRecord]
    ]:
        if not content:
            return None, [], []
        lines = content.splitlines()
        metadata = None
        if len(lines) < 2:
            print("Expected at least two records from log file")
            return None, [], []
        print(f"[Raw Log] Peek raw metadata json: {lines[0]} \n")
        print(f"[Raw Log] Peek raw record json: {lines[1]} \n")

        try:
            metadata = UtilizationMetadata.from_json(lines[0])
        except Exception as e:
            print(f":: warning Failed to parse metadata: {e} for data: {lines[0]}")
            return None, [], []

        if metadata.data_model_version != getDataModelVersion():
            print(
                f":: warning Data model version mismatch: {metadata.data_model_version} != {getDataModelVersion()}"
            )
            return None, [], []

        result_logs, error_logs = self._process_utilization_records(lines)
        return metadata, result_logs, error_logs


def unzip_file(path: Path, file_name: str) -> str:
    try:
        with zipfile.ZipFile(path) as zip_file:
            # Read the desired file from the zip archive
            return zip_file.read(name=file_name).decode()
    except Exception as e:
        print(f"::warning trying to download test log {object} failed by: {e}")
        return ""


def get_datetime_string(timestamp: float) -> str:
    dt = datetime.fromtimestamp(timestamp, timezone.utc)
    dt_str = dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    return dt_str


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Upload test stats to s3")
    parser.add_argument(
        "--workflow-run-id",
        type=int,
        required=True,
        help="id of the workflow to get artifacts from",
    )
    parser.add_argument(
        "--workflow-run-attempt",
        type=int,
        required=True,
        help="which retry of the workflow this is",
    )
    parser.add_argument(
        "--workflow-name",
        type=str,
        required=True,
        help="id of the workflow to get artifacts from",
    )
    parser.add_argument(
        "--job-id",
        type=int,
        required=True,
        help="id of the workflow to get artifacts from",
    )
    parser.add_argument(
        "--job-name",
        type=str,
        required=True,
        help="id of the workflow to get artifacts from",
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=False,
        help="which GitHub repo this workflow run belongs to",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    parser.add_argument("--dry-run", action="store_true", help="Enable dry-run mode")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Flush stdout so that any errors in the upload show up last in the logs.
    sys.stdout.flush()

    repo = PYTORCH_REPO
    if args.repo:
        repo = args.repo
    print(f"repo: {repo}")

    workflow_info = WorkflowInfo(
        workflow_run_id=args.workflow_run_id,
        run_attempt=args.workflow_run_attempt,
        job_id=args.job_id,
        workflow_name=args.workflow_name,
        job_name=args.job_name,
        repo=repo,
    )

    ud = UploadUtilizationData(
        info=workflow_info,
        dry_run=args.dry_run,
        debug=args.debug,
    )
    ud.start()

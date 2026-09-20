"""Shared MinIO / S3 URI parsing."""


def parse_minio_link(link: str) -> tuple[str, str]:
    """
    Parse a MinIO link into (bucket_name, object_name).

    Accepted formats:
    - minio://bucket_name/object_name
    - s3://bucket_name/object_name
    - bucket_name/object_name
    """
    if link.startswith("minio://"):
        link = link[8:]
    elif link.startswith("s3://"):
        link = link[5:]

    parts = link.split("/", 1)
    if len(parts) == 2 and parts[0]:
        return parts[0], parts[1]
    if parts[0]:
        return parts[0], ""
    return "default-bucket", link

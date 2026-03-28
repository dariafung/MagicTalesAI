import asyncio
import json
import logging
import os
from pathlib import Path

from google.cloud import firestore, storage
from dotenv import load_dotenv

# Load env vars manually for script
load_dotenv()

project_id = os.getenv("GCP_PROJECT_ID")
bucket_name = os.getenv("GCS_BUCKET_NAME")
database_id = os.getenv("GCP_DATABASE_ID", "(default)")

if project_id:
    os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
    os.environ["GOOGLE_CLOUD_QUOTA_PROJECT"] = project_id

logging.basicConfig(level=logging.INFO)
logging.info(f"Using project: {project_id}, database: {database_id}")

if not project_id:
    logging.warning("GCP_PROJECT_ID is not set.")
if not bucket_name:
    logging.warning("GCS_BUCKET_NAME is not set.")

db = firestore.AsyncClient(project=project_id, database=database_id)
gcs = storage.Client(project=project_id)
bucket = gcs.bucket(bucket_name) if bucket_name else None

DATA_DIR = Path("data")


async def migrate_story(story_dir: Path):
    story_id = story_dir.name
    logging.info(f"Migrating story {story_id}...")

    # 1. Migrate JSON to Firestore
    doc_data = {}
    for key in ["meta", "extraction", "voices"]:
        json_file = story_dir / f"{key}.json"
        if json_file.exists():
            try:
                data = json.loads(json_file.read_text())
                doc_data[key] = data
            except json.JSONDecodeError:
                logging.error(f"Failed to decode {json_file}")

    if doc_data:
        doc_ref = db.collection("stories").document(story_id)
        await doc_ref.set(doc_data, merge=True)
        logging.info(f"  - Uploaded Firestore document for {story_id}")

    # 2. Migrate audio to GCS
    if bucket:
        for audio_file in story_dir.glob("*.wav"):
            blob_name = f"{story_id}/{audio_file.name}"
            blob = bucket.blob(blob_name)
            # Run upload in thread to not block asyncio
            await asyncio.to_thread(blob.upload_from_filename, audio_file, content_type="audio/wav")
            logging.info(f"  - Uploaded {audio_file.name}")

        for music_file in story_dir.glob("*.mp3"):
            blob_name = f"{story_id}/{music_file.name}"
            blob = bucket.blob(blob_name)
            await asyncio.to_thread(blob.upload_from_filename, music_file, content_type="audio/mpeg")
            logging.info(f"  - Uploaded {music_file.name}")
    else:
        logging.warning("Skipping file uploads because GCS bucket is not configured.")

    logging.info(f"Finished {story_id}")


async def main():
    if not DATA_DIR.exists():
        logging.info("No data directory found. Nothing to migrate.")
        return

    tasks = []
    for d in DATA_DIR.iterdir():
        if d.is_dir():
            tasks.append(migrate_story(d))

    if tasks:
        await asyncio.gather(*tasks)
        logging.info("Migration complete!")
    else:
        logging.info("No story directories found in data/.")


if __name__ == "__main__":
    asyncio.run(main())

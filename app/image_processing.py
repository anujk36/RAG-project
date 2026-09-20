import base64
import hashlib
import json
import mimetypes
from pathlib import Path

import pymupdf
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.logging_config import logger

STANDALONE_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
EXTRACTED_IMAGES_DIRNAME = "extracted_images"
CAPTION_CACHE_FILENAME = ".image_caption_cache.json"

# Skip tiny embedded images (icons, bullets, logos) - they add noise, not signal
MIN_IMAGE_BYTES = 4096

DESCRIBE_IMAGE_PROMPT = (
    "Describe this image in detail for a document search index. Include any "
    "visible text, numbers, labels, chart or diagram data, and the overall "
    "subject. Be factual and specific so someone could find this image by "
    "searching for its content."
)


def _load_cache(documents_dir: Path) -> dict:
    cache_path = documents_dir / CAPTION_CACHE_FILENAME
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_cache(documents_dir: Path, cache: dict) -> None:
    cache_path = documents_dir / CAPTION_CACHE_FILENAME
    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_images_from_pdf(pdf_path: Path, documents_dir: Path) -> list[dict]:
    """Extract embedded images from a single PDF into an extracted_images
    subfolder of documents_dir. Returns metadata for each extracted image."""
    output_dir = documents_dir / EXTRACTED_IMAGES_DIRNAME
    output_dir.mkdir(exist_ok=True)

    extracted = []
    try:
        pdf = pymupdf.open(pdf_path)
    except Exception as e:
        logger.error(f"Could not open {pdf_path} for image extraction: {e}")
        return extracted

    for page_index in range(len(pdf)):
        page = pdf[page_index]
        for image_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            try:
                base_image = pdf.extract_image(xref)
            except Exception as e:
                logger.warning(f"Failed to extract image xref {xref} from {pdf_path}: {e}")
                continue

            image_bytes = base_image["image"]
            if len(image_bytes) < MIN_IMAGE_BYTES:
                continue

            ext = base_image.get("ext", "png")
            filename = f"{pdf_path.stem}_p{page_index + 1}_{image_index}.{ext}"
            out_path = output_dir / filename
            if not out_path.exists():
                out_path.write_bytes(image_bytes)

            extracted.append({
                "path": out_path,
                "source_document": pdf_path.name,
                "page": page_index + 1,
            })
    pdf.close()

    return extracted


def extract_images_from_pdfs(documents_dir: Path) -> list[dict]:
    """Extract embedded images from every PDF in documents_dir."""
    extracted = []
    for pdf_path in sorted(documents_dir.glob("*.pdf")):
        extracted.extend(extract_images_from_pdf(pdf_path, documents_dir))
    return extracted


def find_standalone_images(documents_dir: Path) -> list[Path]:
    """Find image files placed directly in documents_dir (not extracted ones)."""
    return [
        path for path in documents_dir.iterdir()
        if path.is_file() and path.suffix.lower() in STANDALONE_IMAGE_EXTENSIONS
    ]


def describe_image(image_path: Path, llm: ChatGoogleGenerativeAI) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    mime_type = mime_type or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")

    message = HumanMessage(content=[
        {"type": "text", "text": DESCRIBE_IMAGE_PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}},
    ])
    response = llm.invoke([message])
    return response.text.strip()


def build_image_documents(
    documents_dir: Path, llm: ChatGoogleGenerativeAI, images: list[dict] | None = None
) -> list[Document]:
    """Captions the given images (or, if `images` is omitted, every image
    embedded in PDFs + standalone file in documents_dir) with the given
    multimodal LLM, and returns them as searchable Documents.

    Captions are cached by image content hash so re-running index builds does
    not re-caption unchanged images.
    """
    documents_dir = Path(documents_dir)
    cache = _load_cache(documents_dir)
    cache_changed = False

    if images is None:
        extracted = extract_images_from_pdfs(documents_dir)
        standalone = find_standalone_images(documents_dir)
        images = extracted + [
            {"path": p, "source_document": p.name, "page": None} for p in standalone
        ]

    all_images = images

    image_documents = []
    for item in all_images:
        image_path: Path = item["path"]
        digest = _file_hash(image_path.read_bytes())
        cache_key = str(image_path.relative_to(documents_dir)).replace("\\", "/")

        cached_entry = cache.get(cache_key)
        if cached_entry and cached_entry.get("hash") == digest:
            caption = cached_entry["caption"]
        else:
            logger.info(f"Captioning image {cache_key} with Gemini vision...")
            try:
                caption = describe_image(image_path, llm)
            except Exception as e:
                logger.error(f"Failed to caption {cache_key}: {e}")
                continue
            cache[cache_key] = {"hash": digest, "caption": caption}
            cache_changed = True

        metadata = {
            "source": cache_key,
            "type": "image",
            "source_document": item["source_document"],
        }
        if item.get("page") is not None:
            metadata["page"] = item["page"]

        image_documents.append(Document(page_content=caption, metadata=metadata))

    if cache_changed:
        _save_cache(documents_dir, cache)

    logger.info(f"Prepared {len(image_documents)} image documents (from {len(all_images)} candidate images)")
    return image_documents

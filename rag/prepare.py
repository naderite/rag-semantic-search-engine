from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
import unicodedata
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .types import ChunkRecord, NormalizedBlock, PrepConfig, PrepReport, RawPageBlock


def extract_pdf_document(pdf_path: str, config: PrepConfig) -> tuple[list[RawPageBlock], int]:
    blocks: list[RawPageBlock] = []
    ocr_pages = 0
    path = Path(pdf_path)
    doc_id = path.stem

    try:
        import pdfplumber  # type: ignore
    except ImportError as exc:
        raise RuntimeError("pdfplumber is required for PDF extraction") from exc

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                blocks.append(
                    RawPageBlock(
                        doc_id=doc_id,
                        file_name=path.name,
                        page=page_idx,
                        block_type="paragraph",
                        text=text,
                    )
                )

            tables = page.extract_tables() or []
            for table in tables:
                if not table:
                    continue
                rows = [[(cell or "").strip() for cell in row] for row in table if any((cell or "").strip() for cell in row)]
                if not rows:
                    continue
                blocks.append(
                    RawPageBlock(
                        doc_id=doc_id,
                        file_name=path.name,
                        page=page_idx,
                        block_type="table",
                        text="\n".join(" | ".join(row) for row in rows),
                        rows=rows,
                    )
                )

            if not text and config.enable_ocr_fallback:
                ocr_text = _ocr_pdf_page(pdf_path, page_idx, config)
                if ocr_text and len(ocr_text) >= config.min_text_chars_for_page:
                    ocr_pages += 1
                    blocks.append(
                        RawPageBlock(
                            doc_id=doc_id,
                            file_name=path.name,
                            page=page_idx,
                            block_type="paragraph",
                            text=ocr_text,
                        )
                    )

    return blocks, ocr_pages


def normalize_blocks(blocks: list[RawPageBlock], config: PrepConfig) -> list[NormalizedBlock]:
    repeated_lines = _find_repeated_lines(blocks)
    normalized: list[NormalizedBlock] = []
    page_section: dict[int, str] = {}

    for block in blocks:
        if block.block_type == "table" and block.rows:
            section = _infer_table_section(block.rows, page_section.get(block.page))
            normalized.extend(_normalize_table_block(block, section))
            continue

        cleaned = _rewrite_machine_phrases(_cleanup_text(block.text))
        if not cleaned:
            continue

        lines = [line for line in (ln.strip() for ln in cleaned.splitlines()) if line]
        lines = [line for line in lines if _normalize_for_embedding(line).lower() not in repeated_lines]
        if not lines:
            continue

        section = _detect_section(lines)
        page_section[block.page] = section
        body = "\n".join(lines)

        for line in lines:
            chunk_type = "list_item" if re.match(r"^[\-\*\u2022\d]+[\.)]?\s+", line) else "paragraph"
            if chunk_type == "list_item" and config.include_list_items:
                normalized.append(
                    NormalizedBlock(
                        doc_id=block.doc_id,
                        file_name=block.file_name,
                        page=block.page,
                        section=section,
                        chunk_type=chunk_type,
                        text=_normalize_for_embedding(line),
                        text_raw=line,
                        lang=_detect_lang(line),
                        metadata={},
                    )
                )

        normalized.append(
            NormalizedBlock(
                doc_id=block.doc_id,
                file_name=block.file_name,
                page=block.page,
                section=section,
                chunk_type="paragraph",
                text=_normalize_for_embedding(body),
                text_raw=body,
                lang=_detect_lang(body),
                metadata={},
            )
        )

    return normalized


def chunk_blocks(blocks: list[NormalizedBlock], config: PrepConfig) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    grouped: dict[tuple[str, int, str, str], list[NormalizedBlock]] = {}

    for block in blocks:
        key = (block.doc_id, block.page, block.section, block.chunk_type)
        grouped.setdefault(key, []).append(block)

    for (_, _, _, chunk_type), items in grouped.items():
        if chunk_type == "table_kv":
            for item in items:
                chunks.append(_to_chunk(item))
            continue

        full_text = "\n".join(item.text for item in items if item.text)
        full_raw = "\n".join(item.text_raw for item in items if item.text_raw)
        words = full_text.split()

        if len(words) <= config.chunk_size_tokens:
            if chunk_type != "table_kv" and len(words) < config.min_chunk_tokens:
                continue
            merged = items[0]
            merged_metadata = _merge_metadata(items)
            structured = _extract_structured_metadata(
                text=full_text,
                doc_id=merged.doc_id,
                file_name=merged.file_name,
            )
            chunks.append(
                ChunkRecord(
                    chunk_id=_chunk_id(merged.doc_id, merged.page, merged.chunk_type, full_text),
                    doc_id=merged.doc_id,
                    file_name=merged.file_name,
                    page=merged.page,
                    section=merged.section,
                    chunk_type=merged.chunk_type,
                    text=full_text,
                    text_raw=full_raw,
                    lang=_merge_lang([i.lang for i in items]),
                    metadata={**merged_metadata, **structured, "units": _extract_units(full_text)},
                )
            )
            continue

        step = max(1, config.chunk_size_tokens - config.chunk_overlap_tokens)
        for start in range(0, len(words), step):
            end = start + config.chunk_size_tokens
            chunk_words = words[start:end]
            if not chunk_words:
                break
            merged = items[0]
            chunk_text = " ".join(chunk_words)
            structured = _extract_structured_metadata(
                text=chunk_text,
                doc_id=merged.doc_id,
                file_name=merged.file_name,
            )
            chunks.append(
                ChunkRecord(
                    chunk_id=_chunk_id(merged.doc_id, merged.page, merged.chunk_type, f"{start}:{chunk_text}"),
                    doc_id=merged.doc_id,
                    file_name=merged.file_name,
                    page=merged.page,
                    section=merged.section,
                    chunk_type=merged.chunk_type,
                    text=chunk_text,
                    text_raw=chunk_text,
                    lang=_merge_lang([i.lang for i in items]),
                    metadata={
                        **_merge_metadata(items),
                        **structured,
                        "units": _extract_units(chunk_text),
                        "window_start": start,
                    },
                )
            )
            if end >= len(words):
                break

    return chunks


def save_chunks_jsonl(chunks: list[ChunkRecord], jsonl_path: str) -> None:
    path = Path(jsonl_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def load_chunks_jsonl(jsonl_path: str) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    with Path(jsonl_path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            data = json.loads(line)
            chunks.append(ChunkRecord(**data))
    return chunks


def prepare_documents(input_dir: str, output_dir: str, config: PrepConfig) -> PrepReport:
    report = PrepReport()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_chunks: list[ChunkRecord] = []
    for pdf_file in sorted(Path(input_dir).glob("*.pdf")):
        try:
            blocks, ocr_pages = extract_pdf_document(str(pdf_file), config)
            normalized = normalize_blocks(blocks, config)
            chunks = chunk_blocks(normalized, config)
            report.documents_processed += 1
            report.pages_processed += len({b.page for b in blocks})
            report.pages_with_ocr += ocr_pages
            report.chunks_generated += len(chunks)
            all_chunks.extend(chunks)
        except Exception as exc:  # noqa: BLE001
            report.extraction_errors.append(f"{pdf_file.name}: {exc}")

    dedup: dict[tuple[str, int, str, str], ChunkRecord] = {}
    cross_doc_signatures: set[str] = set()
    for chunk in all_chunks:
        signature = _cross_doc_signature(chunk)
        if config.dedup_across_docs and signature:
            if signature in cross_doc_signatures:
                continue
            cross_doc_signatures.add(signature)
        key = (chunk.doc_id, chunk.page, chunk.chunk_type, _normalize_for_embedding(chunk.text).lower())
        dedup.setdefault(key, chunk)

    unique_chunks = list(dedup.values())
    report.chunks_after_dedup = len(unique_chunks)

    save_chunks_jsonl(unique_chunks, str(output_path / "chunks.jsonl"))
    with (output_path / "prep_report.json").open("w", encoding="utf-8") as fh:
        json.dump(asdict(report), fh, indent=2, ensure_ascii=False)

    return report


def direct_chunks_from_blocks(blocks: list[RawPageBlock], config: PrepConfig) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    seen: set[str] = set()
    step = max(1, config.chunk_size_tokens - config.chunk_overlap_tokens)

    for block in blocks:
        cleaned = _normalize_for_embedding(_cleanup_text(block.text))
        if len(cleaned) < config.min_text_chars_for_page:
            continue

        words = cleaned.split()
        if not words:
            continue

        for start in range(0, len(words), step):
            end = min(len(words), start + config.chunk_size_tokens)
            chunk_text = " ".join(words[start:end]).strip()
            if not chunk_text:
                continue
            if len(chunk_text.split()) < config.min_chunk_tokens:
                if start > 0 or len(words) >= config.min_chunk_tokens:
                    continue

            chunk = ChunkRecord(
                chunk_id=_chunk_id(block.doc_id, block.page, "paragraph", f"{block.block_type}:{start}:{chunk_text}"),
                doc_id=block.doc_id,
                file_name=block.file_name,
                page=block.page,
                section="General",
                chunk_type="paragraph",
                text=chunk_text,
                text_raw=chunk_text,
                lang=_detect_lang(chunk_text),
                metadata={
                    "source_block_type": block.block_type,
                    "window_start": start,
                    **_extract_structured_metadata(
                        text=chunk_text,
                        doc_id=block.doc_id,
                        file_name=block.file_name,
                    ),
                },
            )

            if config.dedup_across_docs:
                signature = _normalize_for_embedding(chunk.text).lower()
                if signature in seen:
                    continue
                seen.add(signature)
            chunks.append(chunk)

            if end >= len(words):
                break

    return chunks


def direct_chunks_from_pdfs(input_dir: str, config: PrepConfig) -> tuple[list[ChunkRecord], PrepReport]:
    report = PrepReport()
    all_chunks: list[ChunkRecord] = []

    for pdf_file in sorted(Path(input_dir).glob("*.pdf")):
        try:
            blocks, ocr_pages = extract_pdf_document(str(pdf_file), config)
            chunks = direct_chunks_from_blocks(blocks=blocks, config=config)
            report.documents_processed += 1
            report.pages_processed += len({b.page for b in blocks})
            report.pages_with_ocr += ocr_pages
            report.chunks_generated += len(chunks)
            all_chunks.extend(chunks)
        except Exception as exc:  # noqa: BLE001
            report.extraction_errors.append(f"{pdf_file.name}: {exc}")

    report.chunks_after_dedup = len(all_chunks)
    return all_chunks, report


def _to_chunk(item: NormalizedBlock) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=_chunk_id(item.doc_id, item.page, item.chunk_type, item.text),
        doc_id=item.doc_id,
        file_name=item.file_name,
        page=item.page,
        section=item.section,
        chunk_type=item.chunk_type,
        text=item.text,
        text_raw=item.text_raw,
        lang=item.lang,
        metadata=item.metadata,
    )


def _cleanup_text(text: str) -> str:
    text = text.replace("\x0c", "\n")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"-\n(?=[a-zA-Z])", "", text)
    lines = []
    for line in text.split("\n"):
        cleaned = re.sub(r"\s+", " ", line).strip()
        if not cleaned:
            continue
        cleaned = cleaned.replace(" .", ".").replace(" ,", ",")
        lines.append(cleaned)
    return "\n".join(lines).strip()


def _normalize_for_embedding(text: str) -> str:
    text = text.replace("：", ":")
    text = re.sub(r"(\d),(\d)", r"\1.\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _rewrite_machine_phrases(text: str) -> str:
    pattern = re.compile(
        r"Section:\s*(?P<section>[^;]+);\s*Pour\s*(?P<product>[^,]+),\s*dosage recommande:\s*(?P<dosage>[^.]+)\.\s*Observation:\s*(?P<obs>[^.]+)\.?",
        flags=re.IGNORECASE,
    )

    def repl(match: re.Match[str]) -> str:
        section = match.group("section").strip()
        product = match.group("product").strip()
        dosage = match.group("dosage").strip()
        obs = match.group("obs").strip()
        return (
            f"Dans la section « {section} », la recommandation pour « {product} » "
            f"indique un dosage de {dosage}. "
            f"Cette recommandation est accompagnee de la remarque suivante : {obs}."
        )

    return pattern.sub(repl, text)


def _detect_section(lines: list[str]) -> str:
    for line in lines[:6]:
        if 3 <= len(line) <= 80 and (line.isupper() or line.istitle()):
            return line
    return "General"


def _detect_lang(text: str) -> str:
    lowered = text.lower()
    fr_markers = ["le", "la", "des", "avec", "pour", "améliorant", "température"]
    en_markers = ["the", "with", "for", "shelf", "storage", "bread"]
    has_fr = any(f" {m} " in f" {lowered} " for m in fr_markers)
    has_en = any(f" {m} " in f" {lowered} " for m in en_markers)
    if has_fr and has_en:
        return "mixed"
    if has_fr:
        return "fr"
    if has_en:
        return "en"
    return "mixed"


def _merge_lang(langs: list[str]) -> str:
    values = set(langs)
    if len(values) == 1:
        return langs[0]
    if "mixed" in values:
        return "mixed"
    return "mixed"


def _extract_units(text: str) -> list[str]:
    return sorted(set(re.findall(r"\b(?:ppm|mg/kg|u/g|nmau/g|xylh/g|°c|kg|g|months?)\b", text.lower())))


def _chunk_id(doc_id: str, page: int, chunk_type: str, text: str) -> str:
    data = f"{doc_id}|{page}|{chunk_type}|{text}".encode("utf-8")
    return hashlib.sha1(data).hexdigest()[:16]


def _merge_metadata(items: list[NormalizedBlock]) -> dict:
    merged: dict = {}
    for item in items:
        for key, value in item.metadata.items():
            if key not in merged:
                merged[key] = value
                continue
            if merged[key] == value:
                continue
            if isinstance(merged[key], list):
                existing = merged[key]
                if isinstance(value, list):
                    merged[key] = sorted(set(existing + value))
                else:
                    merged[key] = sorted(set(existing + [value]))
            else:
                merged[key] = value
    return merged


def _cross_doc_signature(chunk: ChunkRecord) -> str | None:
    if chunk.chunk_type != "table_kv":
        return None

    md = chunk.metadata or {}
    field = str(md.get("field_name", "")).strip().lower()
    dosage = str(md.get("dosage", "")).strip().lower()
    section = str(md.get("section", chunk.section)).strip().lower()
    if field and dosage:
        return f"{section}|{field}|{dosage}"

    text = _normalize_token(chunk.text)
    if not text:
        return None
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:24]


def _find_repeated_lines(blocks: list[RawPageBlock]) -> set[str]:
    by_page: dict[int, set[str]] = {}
    for block in blocks:
        if block.block_type != "paragraph":
            continue
        lines = {_normalize_for_embedding(ln.strip()).lower() for ln in re.split(r"[\n\r]+", block.text) if ln.strip()}
        by_page.setdefault(block.page, set()).update(lines)

    counts: Counter[str] = Counter()
    for lines in by_page.values():
        for line in lines:
            if 5 <= len(line) <= 120:
                counts[line] += 1

    return {line for line, c in counts.items() if c >= 2}


def _infer_table_section(rows: list[list[str]], fallback: str | None) -> str:
    flat = " ".join(" ".join((cell or "").strip() for cell in row) for row in rows).lower()
    if "type de production" in flat or "dosage (ppm)" in flat:
        return "Dosages Recommandes"
    if "poids farine" in flat and "50 ppm" in flat:
        return "Table de Conversion"
    if any(token in flat for token in ("microbiology", "heavy metals", "allergens", "gmo status", "food safty data")):
        return "Food Safety Data"
    return fallback or "Table Data"


def _merge_wrapped_table_rows(rows: list[list[str]]) -> list[list[str]]:
    merged: list[list[str]] = []
    for row in rows:
        cells = [(cell or "").strip() for cell in row]
        if not any(cells):
            continue
        nonempty_idx = [idx for idx, val in enumerate(cells) if val]
        if merged and len(nonempty_idx) == 1:
            idx = nonempty_idx[0]
            # Merge wrapped single-cell continuations into previous logical row.
            merged[-1][idx] = f"{merged[-1][idx]} {cells[idx]}".strip()
            continue
        merged.append(cells)
    return merged


def _normalize_table_block(block: RawPageBlock, section: str) -> list[NormalizedBlock]:
    assert block.rows is not None
    rows = _merge_wrapped_table_rows(block.rows)
    result: list[NormalizedBlock] = []

    header: list[str] | None = None
    if len(rows) > 1 and sum(1 for c in rows[0] if c) >= 2:
        header = rows[0]
        rows = rows[1:]

    for row in rows:
        cells = [c for c in row if c]
        if not cells:
            continue

        field = cells[0]
        value = " ; ".join(cells[1:]) if len(cells) > 1 else ""
        if header and len(header) >= 2 and len(cells) >= 2:
            pairs = [f"{header[i]}={cells[i]}" for i in range(min(len(header), len(cells))) if header[i] and cells[i]]
            value = " ; ".join(pairs)

        if not value and ":" in field:
            left, right = field.split(":", 1)
            field = left.strip()
            value = right.strip()

        # Filter malformed table rows that add noise but no semantic value.
        if len(field.strip()) < 3:
            continue
        if not value:
            continue

        sentence, row_meta = _render_table_sentence(section=section, field=field, value=value)
        result.append(
            NormalizedBlock(
                doc_id=block.doc_id,
                file_name=block.file_name,
                page=block.page,
                section=section,
                chunk_type="table_kv",
                text=_normalize_for_embedding(sentence),
                text_raw=sentence,
                lang=_detect_lang(sentence),
                metadata={
                    "fields": [field],
                    "units": _extract_units(sentence),
                    "field_name": field,
                    "section": section,
                    **row_meta,
                },
            )
        )

    if result:
        summary_text = " ".join(item.text for item in result)
        result.append(
            NormalizedBlock(
                doc_id=block.doc_id,
                file_name=block.file_name,
                page=block.page,
                section=section,
                chunk_type="paragraph",
                text=_normalize_for_embedding(summary_text),
                text_raw=summary_text,
                lang=_merge_lang([item.lang for item in result]),
                metadata={"table_summary": True, "section": section},
            )
        )

    return result


def _render_table_sentence(section: str, field: str, value: str) -> tuple[str, dict[str, str]]:
    pairs: dict[str, str] = {}
    for part in value.split(";"):
        part = part.strip()
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        pairs[key.strip()] = val.strip()

    production = pairs.get("Type de Production")
    dosage = pairs.get("Dosage (ppm)")
    obs = pairs.get("Observations")
    if not production:
        production = field
    metadata: dict[str, str] = {}

    if production and dosage:
        dosage_value = dosage if "ppm" in dosage.lower() else f"{dosage} ppm"
        canonical_product = _canonical_product_type(production)
        sentence = (
            f"Dans la section « {section} », la recommandation pour « {production} » "
            f"indique un dosage de {dosage_value}."
        )
        metadata["product_type"] = production
        metadata["product_type_canonical"] = canonical_product
        metadata["dosage"] = dosage
        metadata["dosage_unit"] = "ppm"
        if obs:
            sentence += f" Cette recommandation est accompagnee de la remarque suivante : {obs}."
            metadata["observation"] = obs
        else:
            sentence += " Aucune remarque complementaire n'est precisee."
        return sentence, metadata

    sentence = (
        f"Dans la section « {section} », le champ « {field} » contient l'information suivante : {value}."
    ).strip()
    if production:
        metadata["product_type"] = production
        metadata["product_type_canonical"] = _canonical_product_type(production)
    return sentence, metadata


def _ocr_pdf_page(pdf_path: str, page_number: int, config: PrepConfig) -> str:
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore
    except ImportError:
        return ""

    with tempfile.TemporaryDirectory() as tmp_dir:
        prefix = Path(tmp_dir) / "page"
        cmd = [
            "pdftoppm",
            "-f",
            str(page_number),
            "-singlefile",
            "-png",
            pdf_path,
            str(prefix),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except Exception:  # noqa: BLE001
            return ""

        image_path = prefix.with_suffix(".png")
        if not image_path.exists():
            return ""

        try:
            return pytesseract.image_to_string(Image.open(image_path), lang=config.ocr_lang).strip()
        except Exception:  # noqa: BLE001
            return ""


def _normalize_token(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[^a-z0-9%/\-\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _canonical_product_type(text: str) -> str:
    value = _normalize_token(text)
    if any(token in value for token in ("pain", "panification", "bread")):
        return "bread"
    if any(token in value for token in ("biscuit", "cracker")):
        return "biscuit"
    if "viennoiserie" in value:
        return "viennoiserie"
    if "surgel" in value:
        return "frozen"
    return "other"


def _extract_structured_metadata(text: str, doc_id: str, file_name: str) -> dict:
    raw = f"{doc_id} {file_name} {text}"
    norm = _normalize_token(raw)
    metadata: dict[str, object] = {}

    family = _infer_product_family(norm)
    metadata["product_family"] = family

    codes = sorted(_extract_model_codes(norm))
    if codes:
        metadata["model_codes"] = codes

    dosage_fields = _extract_dosage_fields(norm)
    metadata.update(dosage_fields)

    if "food safty data" in norm or "food safety data" in norm:
        metadata["section_canonical"] = "food_safety"
    elif "dosages recommandes" in norm or "dosage" in norm:
        metadata["section_canonical"] = "dosage"
    elif "reglementation" in norm or "directive" in norm or "codex alimentarius" in norm:
        metadata["section_canonical"] = "regulatory"
    elif "storage" in norm or "minimum durability" in norm or "conservation" in norm:
        metadata["section_canonical"] = "storage"

    return metadata


def _infer_product_family(text_norm: str) -> str:
    if "acide ascorbique" in text_norm or " e300 " in f" {text_norm} ":
        return "ascorbic"
    if " hcf " in f" {text_norm} " or " xylanase " in f" {text_norm} " and "hcf" in text_norm:
        return "hcf"
    if " hcb " in f" {text_norm} ":
        return "hcb"
    if " tg " in f" {text_norm} " or "transglutaminase" in text_norm:
        return "tg"
    if " gox " in f" {text_norm} " or " glucose oxidase " in f" {text_norm} ":
        return "gox"
    if " go max " in text_norm:
        return "go"
    if " af " in f" {text_norm} " or "amylase" in text_norm and "af" in text_norm:
        return "af"
    if " amg " in f" {text_norm} " or "amyloglucosidase" in text_norm:
        return "amg"
    if " lipase " in f" {text_norm} " or " l max " in text_norm or re.search(r"\bl(?:55|65)\b", text_norm):
        return "lipase"
    if " fresh " in f" {text_norm} " or " soft " in f" {text_norm} ":
        return "fresh_soft"
    return "other"


def _extract_model_codes(text_norm: str) -> set[str]:
    codes: set[str] = set()

    for token in text_norm.split():
        if re.fullmatch(r"(?:hcf|hcb|af|amg|tg|gox|go|fresh|soft)\d{2,4}", token):
            codes.add(token)
        if re.fullmatch(r"max(?:63|64|65|x)", token):
            codes.add(token)
        if re.fullmatch(r"l(?:55|65)", token):
            codes.add(token)

    for prefix, suffix in re.findall(r"\b(hcf|hcb|tg|go|l)\s*max\s*(x|63|64|65)\b", text_norm):
        codes.add(f"{prefix}max{suffix}")

    return codes


def _extract_dosage_fields(text_norm: str) -> dict[str, str]:
    fields: dict[str, str] = {}

    patterns = {
        "dosage_standardization": r"standardization of wheat flour\s*([0-9.]+\s*(?:-\s*[0-9.]+)?\s*ppm)",
        "dosage_bread_improvement": r"bread improvement\s*([0-9.]+\s*(?:-\s*[0-9.]+)?\s*ppm)",
        "dosage_suggested_optimum": r"suggested optimum dosage\s*([0-9.]+\s*(?:-\s*[0-9.]+)?\s*ppm)",
        "dosage_suggested": r"suggest(?:ed)? dosage\s*([0-9.]+\s*(?:-\s*[0-9.]+)?\s*ppm)",
        "dosage_range": r"dosage range\s*([0-9.]+\s*(?:-\s*[0-9.]+)?\s*ppm)",
        "dosage_max_authorized": r"dosage maximum autoris[ée]?\s*(?:france belgique\s*)?:?\s*([0-9.]+\s*ppm)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text_norm)
        if match:
            fields[key] = match.group(1).strip()

    generic = re.search(r"\bdosage\s*([0-9.]+\s*(?:-\s*[0-9.]+)?\s*ppm)", text_norm)
    if generic:
        fields.setdefault("dosage_generic", generic.group(1).strip())

    return fields

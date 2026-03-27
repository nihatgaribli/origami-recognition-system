"""Hybrid Origami Assistant UI.

Features:
- Image input: upload a local image and run AI prediction (top-3).
- Text input: type origami name and search PostgreSQL for difficulty + link.
"""

import os
import sys
import json
import re
import threading
import time
import multiprocessing as mp
import traceback
from tkinter import filedialog, messagebox

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from visualization._db_config import get_connection
from ai.image_preprocessing import preprocess_rgb_image_like_training
from ai.groq_integration import generate_search_response

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

IMG_SIZE = (224, 224)
MODEL_PATH = os.path.join(BASE_DIR, "origami_model.h5")
KERAS_FALLBACK_PATH = os.path.join(BASE_DIR, "checkpoints", "best_model.keras")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "label_map.json")
PREDICTION_TIMEOUT_SECONDS = 60


def _predict_worker(model_path: str, fallback_path: str, image_path: str, top_k: int, output_queue: mp.Queue):
    """Run prediction in an isolated process so hangs can be safely terminated."""
    try:
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

        import tensorflow as tf

        if os.path.exists(fallback_path):
            model = tf.keras.models.load_model(fallback_path, compile=False)
        elif os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path, compile=False)
        else:
            raise FileNotFoundError(f"Model not found: {fallback_path} or {model_path}")

        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError("Could not read the selected image.")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        normalized = preprocess_rgb_image_like_training(image_rgb, img_size=IMG_SIZE)
        batch = np.expand_dims(normalized, axis=0)

        scores = model(batch, training=False).numpy()[0]
        top_indices = np.argsort(scores)[-top_k:][::-1]
        top_results = [(int(i), float(scores[i]) * 100.0) for i in top_indices]
        output_queue.put(("ok", top_results))
    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}".strip()
        output_queue.put(("error", error_text, traceback.format_exc()))


class HybridOrigamiAssistant(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Hybrid Origami Assistant")
        self.geometry("980x700")
        self.minsize(900, 620)

        self.model = None
        self.current_image_path = None
        self.index_to_label = None

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_header()
        self._build_chat_area()
        self._build_composer()
        self._set_processing(False)
        self._set_status("Ready")

    def _build_header(self):
        header = ctk.CTkFrame(self, corner_radius=14)
        header.grid(row=0, column=0, columnspan=2, padx=14, pady=(14, 8), sticky="ew")
        header.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            header,
            text="Hybrid Origami Assistant",
            font=ctk.CTkFont(size=28, weight="bold"),
        )
        title.grid(row=0, column=0, padx=16, pady=(12, 0), sticky="w")

        subtitle = ctk.CTkLabel(
            header,
            text="Attach an image and/or type a prompt, then press Send.",
            text_color="#9EA7B8",
            font=ctk.CTkFont(size=13),
        )
        subtitle.grid(row=1, column=0, padx=16, pady=(4, 12), sticky="w")

    def _build_chat_area(self):
        chat_frame = ctk.CTkFrame(self, corner_radius=14)
        chat_frame.grid(row=1, column=0, padx=14, pady=(6, 8), sticky="nsew")
        chat_frame.grid_rowconfigure(2, weight=1)
        chat_frame.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(chat_frame, text="Status: Ready", text_color="#9EA7B8")
        self.status_label.grid(row=0, column=0, padx=14, pady=(12, 6), sticky="w")

        self.how_to_box = ctk.CTkTextbox(chat_frame, height=96, font=ctk.CTkFont(size=12), wrap="word")
        self.how_to_box.grid(row=1, column=0, padx=14, pady=(0, 8), sticky="ew")
        self.how_to_box.insert(
            "1.0",
            "How to use:\n"
            "1. Click Attach to select an image (optional).\n"
            "2. Type an origami name or question (optional).\n"
            "3. Click Send.\n"
            "4. Wait until Status becomes Done and the assistant message appears below."
        )
        self.how_to_box.configure(state="disabled")

        self.output_box = ctk.CTkTextbox(chat_frame, font=ctk.CTkFont(size=13), wrap="word")
        self.output_box.grid(row=2, column=0, padx=14, pady=(0, 12), sticky="nsew")
        self.output_box.configure(state="disabled")
        self._configure_chat_tags()
        self._write_output(
            "Assistant is ready.\n\n"
            "Results will appear here."
        )

    def _build_composer(self):
        composer = ctk.CTkFrame(self, corner_radius=14)
        composer.grid(row=2, column=0, padx=14, pady=(0, 14), sticky="ew")
        composer.grid_columnconfigure(1, weight=1)

        self.attach_button = ctk.CTkButton(
            composer,
            text="Attach",
            width=110,
            height=40,
            command=self.on_attach_image,
        )
        self.attach_button.grid(row=0, column=0, padx=(12, 8), pady=(12, 6), sticky="w")

        self.prompt_entry = ctk.CTkEntry(
            composer,
            placeholder_text="Write a prompt (example: Crane) and press Send...",
            height=40,
        )
        self.prompt_entry.grid(row=0, column=1, padx=(0, 8), pady=(12, 6), sticky="ew")
        self.prompt_entry.bind("<Return>", lambda _e: self.on_send())

        self.send_button = ctk.CTkButton(
            composer,
            text="Send",
            width=110,
            height=40,
            command=self.on_send,
        )
        self.send_button.grid(row=0, column=2, padx=(0, 12), pady=(12, 6), sticky="e")

        self.processing_indicator = ctk.CTkProgressBar(composer, mode="indeterminate")
        self.processing_indicator.grid(row=1, column=2, padx=(0, 12), pady=(0, 12), sticky="e")
        self.processing_indicator.grid_remove()

        self.attachment_label = ctk.CTkLabel(
            composer,
            text="No image attached.",
            text_color="#9EA7B8",
        )
        self.attachment_label.grid(row=1, column=0, columnspan=2, padx=12, pady=(0, 12), sticky="w")

    def _write_output(self, text: str):
        self.output_box.configure(state="normal")
        self.output_box.delete("1.0", "end")
        self.output_box.insert("1.0", text)
        self.output_box.configure(state="disabled")

    def _configure_chat_tags(self):
        text_widget = self.output_box._textbox
        text_widget.tag_configure(
            "user_role",
            foreground="#7CC5FF",
            font=("Segoe UI", 12, "bold"),
            spacing1=8,
        )
        text_widget.tag_configure(
            "assistant_role",
            foreground="#88E0A6",
            font=("Segoe UI", 12, "bold"),
            spacing1=8,
        )
        text_widget.tag_configure(
            "message_text",
            foreground="#E8EDF5",
            font=("Segoe UI", 12),
            lmargin1=16,
            lmargin2=16,
            spacing3=8,
        )
        text_widget.tag_configure(
            "system_text",
            foreground="#9EA7B8",
            font=("Segoe UI", 11, "italic"),
            spacing3=8,
        )

    def _append_output(self, text: str):
        self.output_box.configure(state="normal")
        self.output_box.insert("end", text + "\n", "system_text")
        self.output_box.see("end")
        self.output_box.configure(state="disabled")

    def _append_chat(self, role: str, text: str):
        self.output_box.configure(state="normal")

        if role.lower() == "you":
            role_label = "You"
            role_tag = "user_role"
        else:
            role_label = "Assistant"
            role_tag = "assistant_role"

        self.output_box.insert("end", f"{role_label}\n", role_tag)
        self.output_box.insert("end", f"{text}\n", "message_text")
        self.output_box.insert("end", "\n")

        self.output_box.see("end")
        self.output_box.configure(state="disabled")

    def _set_status(self, text: str):
        self.status_label.configure(text=f"Status: {text}")

    def _set_status_async(self, text: str):
        self.after(0, lambda: self._set_status(text))

    def _append_chat_async(self, role: str, text: str):
        self.after(0, lambda: self._append_chat(role, text))

    def _set_processing(self, is_processing: bool):
        state = "disabled" if is_processing else "normal"
        self.attach_button.configure(state=state)
        self.send_button.configure(state=state)
        self.prompt_entry.configure(state=state)

        if is_processing:
            self.processing_indicator.grid()
            self.processing_indicator.start()
        else:
            self.processing_indicator.stop()
            self.processing_indicator.grid_remove()
 
    def _load_label_map_once(self):
        """Load class index->label map from training artifact if available."""
        if self.index_to_label is not None:
            return

        if os.path.exists(LABEL_MAP_PATH):
            try:
                with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
                    raw_map = json.load(f)
                self.index_to_label = {int(idx): str(label) for idx, label in raw_map.items()}
                return
            except Exception:
                # Fall back to DB/static map when label map artifact is not readable.
                pass
 
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT DISTINCT m.model_name_original
                FROM images i
                INNER JOIN models m ON i.model_id = m.model_id
                WHERE i.cloudinary_url IS NOT NULL
                  AND i.cloudinary_url <> ''
                  AND m.model_name_original IS NOT NULL
                """
            )
            labels = sorted(row[0] for row in cur.fetchall())
        finally:
            conn.close()
 
        if labels:
            self.index_to_label = {idx: label for idx, label in enumerate(labels)}
        else:
            # Keep empty mapping; caller will display class_<index> fallback.
            self.index_to_label = {}

    def _predict_top3(self, image_path: str):
        ctx = mp.get_context("spawn")
        output_queue = ctx.Queue()
        process = ctx.Process(
            target=_predict_worker,
            args=(MODEL_PATH, KERAS_FALLBACK_PATH, image_path, 3, output_queue),
            daemon=True,
        )
        process.start()
        process.join(timeout=PREDICTION_TIMEOUT_SECONDS)

        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
            raise TimeoutError(
                f"Image analysis timed out after {PREDICTION_TIMEOUT_SECONDS} seconds. "
                "Please try a smaller image or run again."
            )

        if output_queue.empty():
            raise RuntimeError(
                "Prediction process finished without returning a result. "
                f"Exit code: {process.exitcode}"
            )

        result = output_queue.get()
        status = result[0]

        if status == "error":
            error_text = result[1] if len(result) > 1 else "Unknown prediction error"
            error_trace = result[2] if len(result) > 2 else ""
            raise RuntimeError(f"{error_text}\n{error_trace}")
        if status != "ok":
            raise RuntimeError(f"Unexpected worker status: {status}")

        payload = result[1]

        top_results = []
        for idx, confidence in payload:
            label = (self.index_to_label or {}).get(int(idx), f"class_{int(idx)}")
            top_results.append((int(idx), label, float(confidence)))
        return top_results

    def _extract_search_keywords(self, text: str) -> str:
        """Extract search keywords from natural language questions using AI."""
        from ai.groq_integration import get_groq_client
        
        text = (text or "").strip()
        if not text or len(text) < 2:
            return text
        
        # Try using AI to understand what user wants
        client = get_groq_client()
        if client and client.is_available():
            try:
                extraction_prompt = f"""Extract the search keywords from this origami-related query. 
Return ONLY the search terms (model names, creator names, categories, etc), nothing else.

Query: {text}

Search terms:"""
                ai_extraction = client.generate_response(extraction_prompt, temperature=0.3, max_tokens=50)
                if ai_extraction and len(ai_extraction.strip()) > 0:
                    return ai_extraction.strip()
            except Exception:
                pass  # Fall back to simple extraction
        
        # Simple fallback if AI fails
        text_lower = text.lower()
        
        # Remove only obvious filler words, NOT names or key terms
        filler_words = [
            "show me", "find me", "send me", "please",
            "give me", "tell me about", "look for",
            "what is", "where is", "how to",
            "origami models of", "origami diagrams of", "origami by",
            r"^origami\s+", r"^models\s+of\s+", r"^diagrams\s+of\s+"
        ]
        
        result = text_lower
        for word in filler_words:
            result = result.replace(word, " ").strip()
        
        # Clean extra spaces
        result = " ".join(result.split())
        
        # If nothing left, return original
        if not result or len(result) < 2:
            return text_lower
        
        return result.strip()

    def _query_database(self, text: str):
        """Search database for models matching the query (handles natural language with AI)."""
        from ai.groq_integration import get_groq_client
        
        if not text or len(text.strip()) < 2:
            return []
        
        conn = get_connection()
        try:
            cur = conn.cursor()
            
            # **ALWAYS use AI to understand the query** - not just for complex queries
            client = get_groq_client()
            if client and client.is_available():
                try:
                    parse_prompt = f"""Parse this origami query into structured filters. Return ONLY valid JSON.

Query: {text}

Return this exact format (use null for unspecified):
{{
  "difficulty": null or 1-5,
  "paper_shape": null or "Square"/"Rectangle"/"Diamond"/etc,
  "requires_cutting": null or true or false,
  "requires_glue": null or true or false,
  "category_keyword": null or animal/category name,
  "creator": null or creator name,
  "model_name": null or model name,
  "limit": 10
}}"""
                    ai_parse = client.generate_response(parse_prompt, temperature=0.1, max_tokens=200)
                    
                    if ai_parse:
                        try:
                            import json
                            # Clean JSON from markdown code blocks
                            ai_parse_clean = ai_parse.strip()
                            if ai_parse_clean.startswith("```"):
                                ai_parse_clean = ai_parse_clean.split("```")[1]
                                if ai_parse_clean.startswith("json"):
                                    ai_parse_clean = ai_parse_clean[4:]
                                ai_parse_clean = ai_parse_clean.strip()
                            
                            filters = json.loads(ai_parse_clean)
                            
                            # Build WHERE clause dynamically
                            where_clauses = []
                            params = []
                            
                            if filters.get('model_name'):
                                where_clauses.append("m.model_name_original ILIKE %s")
                                params.append(f"%{filters['model_name']}%")
                            
                            if filters.get('creator'):
                                where_clauses.append("c.name_original ILIKE %s")
                                params.append(f"%{filters['creator']}%")
                            
                            if filters.get('category_keyword'):
                                # For animal/category terms, search for specific animals
                                category = filters['category_keyword'].lower()
                                animal_keywords = ['dog', 'cat', 'bird', 'fish', 'butterfly', 'crane', 'dragon', 
                                                 'elephant', 'lion', 'tiger', 'monkey', 'horse', 'rabbit', 'fox', 
                                                 'penguin', 'eagle', 'owl', 'snake', 'frog', 'bee']
                                
                                # If category is generic "animals", search for specific animals
                                if category in ['animals', 'animal', 'category']:
                                    animal_search = " OR ".join([f"m.model_name_original ILIKE %s" for _ in animal_keywords])
                                    where_clauses.append(f"({animal_search})")
                                    params.extend([f"%{a}%" for a in animal_keywords])
                                else:
                                    where_clauses.append("m.model_name_original ILIKE %s")
                                    params.append(f"%{filters['category_keyword']}%")
                            
                            if filters.get('paper_shape'):
                                where_clauses.append("m.paper_shape ILIKE %s")
                                params.append(f"%{filters['paper_shape']}%")
                            
                            if filters.get('difficulty') is not None:
                                where_clauses.append("m.difficulty = %s")
                                params.append(filters['difficulty'])
                            
                            if filters.get('requires_cutting') is not None:
                                where_clauses.append("m.uses_cutting = %s")
                                params.append(filters['requires_cutting'])
                            
                            if filters.get('requires_glue') is not None:
                                where_clauses.append("m.uses_glue = %s")
                                params.append(filters['requires_glue'])
                            
                            limit = filters.get('limit', 10)
                            
                            if where_clauses:
                                where_sql = " AND ".join(where_clauses)
                                query = f"""
                                    SELECT
                                        m.model_name_original,
                                        c.name_original AS creator_name,
                                        m.difficulty,
                                        m.source_url
                                    FROM models m
                                    LEFT JOIN creators c ON c.creator_id = m.creator_id
                                    WHERE {where_sql}
                                    ORDER BY m.model_name_original
                                    LIMIT %s
                                """
                                params.append(limit)
                                cur.execute(query, tuple(params))
                                results = cur.fetchall()
                                if results:
                                    return results
                        except (json.JSONDecodeError, KeyError, ValueError):
                            pass  # Fall back to simple search
                except Exception:
                    pass  # Fall back to simple search
            
            # Fallback: Simple text search if AI fails
            search_term = text.strip().lower()
            cur.execute(
                """
                SELECT
                    m.model_name_original,
                    c.name_original AS creator_name,
                    m.difficulty,
                    m.source_url
                FROM models m
                LEFT JOIN creators c ON c.creator_id = m.creator_id
                WHERE m.model_name_original ILIKE %s
                   OR m.model_name_normalized ILIKE %s
                   OR c.name_original ILIKE %s
                   OR c.name_normalized ILIKE %s
                ORDER BY m.model_name_original
                LIMIT 10
                """,
                (
                    f"%{search_term}%",
                    f"%{search_term}%",
                    f"%{search_term}%",
                    f"%{search_term.lower().replace(' ', '_')}%",
                ),
            )
            return cur.fetchall()
        finally:
            conn.close()

    def _normalize_model_name(self, text: str) -> str:
        cleaned = re.sub(r"[^a-z0-9\s_\-]", "", (text or "").lower())
        cleaned = re.sub(r"[\s\-]+", "_", cleaned).strip("_")
        return cleaned

    def _query_model_profile(self, model_label: str):
        normalized = self._normalize_model_name(model_label)

        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT
                    m.model_name_original,
                    c.name_original AS creator_name,
                    m.difficulty,
                    m.paper_shape,
                    m.pieces,
                    m.uses_cutting,
                    m.uses_glue,
                    m.source_url,
                    m.year_created
                FROM models m
                LEFT JOIN creators c ON c.creator_id = m.creator_id
                WHERE m.model_name_original ILIKE %s
                   OR m.model_name_normalized ILIKE %s
                ORDER BY
                    CASE
                        WHEN LOWER(m.model_name_original) = LOWER(%s) THEN 0
                        WHEN LOWER(m.model_name_original) LIKE LOWER(%s) THEN 1
                        WHEN m.model_name_normalized = %s THEN 2
                        ELSE 3
                    END,
                    m.difficulty NULLS LAST,
                    m.pieces NULLS LAST
                LIMIT 1
                """,
                (
                    f"%{model_label}%",
                    f"%{normalized}%",
                    model_label,
                    f"{model_label}%",
                    normalized,
                ),
            )
            return cur.fetchone()
        finally:
            conn.close()

    def _estimate_characteristics(self, label: str, confidence: float, db_profile: tuple | None) -> dict[str, str]:
        if db_profile is not None:
            (
                _name,
                creator_name,
                difficulty,
                paper_shape,
                pieces,
                uses_cutting,
                uses_glue,
                source_url,
                year_created,
            ) = db_profile
            return {
                "Creator": creator_name or "Unknown",
                "Difficulty": "Unknown" if difficulty is None else str(difficulty),
                "Paper shape": paper_shape or "Unknown",
                "Piece count": "Unknown" if pieces is None else str(pieces),
                "Cutting": "Yes" if uses_cutting else "No",
                "Glue": "Yes" if uses_glue else "No",
                "Year": "Unknown" if year_created is None else str(year_created),
                "Link": source_url or "No link available",
                "Profile source": "Database match",
            }

        label_l = (label or "").lower()
        estimated_difficulty = "Intermediate"
        if any(k in label_l for k in ("dragon", "phoenix", "tessellation", "modular", "ryujin")):
            estimated_difficulty = "Advanced"
        elif any(k in label_l for k in ("crane", "boat", "heart", "fish", "star", "butterfly")):
            estimated_difficulty = "Beginner/Intermediate"
        elif confidence < 45:
            estimated_difficulty = "Uncertain"

        confidence_band = "High" if confidence >= 80 else "Medium" if confidence >= 55 else "Low"
        return {
            "Creator": "Unknown",
            "Difficulty": estimated_difficulty,
            "Paper shape": "Likely square",
            "Piece count": "Likely 1",
            "Cutting": "Likely no",
            "Glue": "Likely no",
            "Year": "Unknown",
            "Link": "No link available",
            "Profile source": "Heuristic estimate",
            "Prediction confidence band": confidence_band,
        }

    def _get_geometric_reasoning(self, label: str) -> str:
        """Provide geometric and structural reasoning for the classification."""
        label_l = (label or "").lower()

        if "modular" in label_l:
            return "This figure matches the Modular category due to its symmetric folds and multiple components."
        elif "crane" in label_l or "phoenix" in label_l:
            return "Shows a classic bird form with sharp fold lines and a prominent head design."
        elif "tessellation" in label_l or "geometric" in label_l:
            return "Displays a tile-like pattern composed of regular polygons (triangles, squares)."
        elif "dragon" in label_l:
            return "Features a long spiral tail, spread wings, and a flying figure form at the top."
        elif "star" in label_l or "snowflake" in label_l:
            return "Shows radial symmetry with cut rays extending outward in a pattern."
        elif "animal" in label_l or "bird" in label_l or "fish" in label_l:
            return "Displays organic curves, natural geometric composition, and natural form."
        else:
            return "The folding pattern and structural composition have been analyzed."

    def _format_metadata_table(self, profiles_data: list[dict]) -> str:
        """Format prediction data and characteristics into a markdown table."""
        if not profiles_data:
            return ""

        lines = [
            "| Model | Confidence | Difficulty | Pieces | Year |",
            "|-------|------------|------------|--------|------|",
        ]
        for data in profiles_data:
            model = data.get("model", "Unknown")
            conf = data.get("confidence", "N/A")
            diff = data.get("difficulty", "Unknown")
            pieces = data.get("pieces", "Unknown")
            year = data.get("year", "Unknown")
            lines.append(f"| {model} | {conf} | {diff} | {pieces} | {year} |")

        return "\n".join(lines)

    def _show_preview(self, image_path: str):
        pil_image = Image.open(image_path).convert("RGB")
        pil_image.thumbnail((260, 180))
        return pil_image

    def on_attach_image(self):
        file_path = filedialog.askopenfilename(
            title="Choose an origami image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")],
        )
        if not file_path:
            return

        self.current_image_path = file_path
        self.attachment_label.configure(text=f"Attached: {os.path.basename(file_path)}")

    def on_send(self):
        prompt_text = self.prompt_entry.get().strip()
        image_path = self.current_image_path

        if not prompt_text and not image_path:
            messagebox.showinfo("Input needed", "Attach an image and/or type a prompt first.")
            return

        user_parts = []
        if prompt_text:
            user_parts.append(prompt_text)
        if image_path:
            user_parts.append(f"[image: {os.path.basename(image_path)}]")

        self._append_chat("You", " | ".join(user_parts))
        self._set_status("Processing...")
        self._append_chat("Assistant", "Processing your request. Please wait...")
        self._set_processing(True)

        worker = threading.Thread(
            target=self._run_hybrid_flow,
            args=(prompt_text, image_path),
            daemon=True,
        )
        worker.start()

    def _run_hybrid_flow(self, prompt_text: str, image_path: str | None):
        try:
            started_at = time.time()
            response_lines = []

            if image_path:
                self._set_status_async("Loading AI model...")
                self._append_chat_async("Assistant", "Preparing image analysis...")
                self._load_label_map_once()

                self._set_status_async("Running image prediction...")
                self._append_chat_async("Assistant", "Analyzing image...")
                top_results = self._predict_top3(image_path)

                response_lines.append("## 🎨 Image Analysis")
                response_lines.append("")

                # Best prediction
                best_idx, best_label, best_score = top_results[0]
                
                # Get database profile for this model
                profile = self._query_model_profile(best_label)
                estimated = self._estimate_characteristics(best_label, best_score, profile)
                
                # Generate professional analysis using Groq LLM
                from ai.groq_integration import generate_professional_image_analysis
                
                self._set_status_async("Generating AI analysis...")
                
                # Prepare parameters for professional analysis
                tutorial_link = estimated.get('Link', '')
                # Only pass link if it's a valid URL (not "No link available")
                if 'No link available' in tutorial_link or tutorial_link == '':
                    tutorial_link = ''
                
                professional_analysis = generate_professional_image_analysis(
                    model_name=best_label,
                    confidence=best_score,  # Already in 0-100 range
                    top_models=[(t[1], t[2]) for t in top_results[:3]],
                    creator=estimated.get('Creator', 'Unknown'),
                    difficulty=estimated.get('Difficulty', 'Intermediate'),
                    paper_shape=estimated.get('Paper shape', 'Square'),
                    uses_cutting=estimated.get('Cutting') == 'Yes',
                    uses_glue=estimated.get('Glue') == 'Yes',
                    tutorial_link=tutorial_link
                )
                
                if professional_analysis:
                    response_lines.append(professional_analysis)
                else:
                    # Fallback to simple format if Groq fails
                    response_lines.append("### Top-3 Predictions:")
                    for rank, (idx, label, score) in enumerate(top_results, 1):
                        response_lines.append(f"{rank}. **{label}** [{score:.1f}%]")
                    response_lines.append("")
                    
                    if best_score < 25:
                        response_lines.append("⚠️ **Low Confidence Detection**")
                        response_lines.append("Possible causes: Complex lighting, non-standard paper color, intricate folding pattern")
                        response_lines.append("")
                    
                    response_lines.append("### Geometric Reasoning:")
                    geometric = self._get_geometric_reasoning(best_label)
                    response_lines.append(geometric)
                    response_lines.append("")
                    
                    response_lines.append("### Technical Sheet:")
                    response_lines.append(f"- **Model:** {best_label}")
                    response_lines.append(f"- **Creator:** {estimated.get('Creator', 'Unknown')}")
                    response_lines.append(f"- **Difficulty:** {estimated.get('Difficulty', 'Unknown')}")
                    response_lines.append(f"- **Paper Shape:** {estimated.get('Paper shape', 'Unknown')}")
                    response_lines.append(f"- **Cutting:** {'Yes' if estimated.get('Cutting') == 'Yes' else 'No'}")
                    response_lines.append(f"- **Glue:** {'Yes' if estimated.get('Glue') == 'Yes' else 'No'}")
                    if estimated.get('Link'):
                        response_lines.append(f"- **[Tutorial Link]({estimated.get('Link')})**")
                    response_lines.append("")

            if prompt_text:
                self._set_status_async("Searching database...")
                self._append_chat_async("Assistant", "Searching database...")
                rows = self._query_database(prompt_text)
                response_lines.append("")
                response_lines.append("## 📚 Database Search Results")
                response_lines.append("")
                response_lines.append(f"**Query:** {prompt_text}")
                response_lines.append("")
                
                if not rows:
                    response_lines.append("❌ No matching models found. Try searching for:")
                    response_lines.append("- Model name (e.g., 'Crane', 'Dragon')")
                    response_lines.append("- Creator name (e.g., 'Robert Lang')")
                    response_lines.append("- Difficulty level")
                else:
                    response_lines.append(f"✅ Found **{len(rows)}** matching result(s):")
                    response_lines.append("")
                    
                    for idx, row_data in enumerate(rows, 1):
                        name = row_data[0]
                        creator_name = row_data[1]
                        difficulty = row_data[2]
                        source_url = row_data[3]
                        
                        diff_text = "Unknown" if difficulty is None else f"Level {difficulty}"
                        creator_text = creator_name or "Unknown Creator"
                        
                        response_lines.append(f"**{idx}. {name}**")
                        response_lines.append(f"   - Creator: {creator_text}")
                        response_lines.append(f"   - Difficulty: {diff_text}")
                        if source_url:
                            response_lines.append(f"   - [View Instructions]({source_url})")
                        response_lines.append("")
                    
                    # Generate AI-powered response using Groq LLM
                    self._set_status_async("Generating AI response...")
                    groq_response = generate_search_response(prompt_text, rows)
                    if groq_response:
                        response_lines.append("### 🤖 AI Assistant's Insight:")
                        response_lines.append(groq_response)
                        response_lines.append("")

            final_text = "\n".join(line for line in response_lines if line is not None).strip()
            elapsed = time.time() - started_at

            def update_ui():
                self._append_chat("Assistant", final_text or "No output generated.")
                self._set_status(f"Done ({elapsed:.1f}s)")
                self._set_processing(False)
                self.prompt_entry.delete(0, "end")
                self.current_image_path = None
                self.attachment_label.configure(text="No image attached.")

            self.after(0, update_ui)
        except Exception as exc:
            error_msg = f"Request failed: {str(exc)}"
            self.after(0, lambda msg=error_msg: self._handle_error(msg))

    def _handle_error(self, message: str):
        self._set_status("Error")
        self._set_processing(False)
        self._append_chat("Assistant", message)
        messagebox.showerror("Hybrid Origami Assistant", message)


def main():
    app = HybridOrigamiAssistant()
    app.mainloop()


if __name__ == "__main__":
    main()

"""MiniRAG desktop application (Tkinter)."""

import os
import shutil
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import chunker
import config
import loader
import rag
import retriever

STATUS_COLORS = {
    "READY": "#2e7d32",
    "INDEXED": "#1565c0",
    "PROCESSING": "#ef6c00",
    "ERROR": "#c62828",
}


class MiniRAGApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("MiniRAG - Offline Document Question Answering")
        self.root.geometry("920x780")
        self.root.minsize(820, 680)
        self.index = None
        self.chunks: list[dict] = []
        self.doc_count = 0
        self.busy = False
        self._build_ui()
        self._refresh_document_list()
        self._try_load_existing_index()

    def _build_ui(self):
        header = ttk.Frame(self.root, padding=(14, 12))
        header.pack(fill="x")
        ttk.Label(header, text="MiniRAG", font=("Segoe UI", 20, "bold")).pack(side="left")
        ttk.Label(header, text="   Offline Document Question Answering",
                  font=("Segoe UI", 11)).pack(side="left")
        self.status_var = tk.StringVar(value="READY")
        self.status_label = ttk.Label(header, textvariable=self.status_var,
                                      font=("Segoe UI", 10, "bold"))
        self.status_label.pack(side="right")
        ttk.Separator(self.root).pack(fill="x")

        main = ttk.Frame(self.root, padding=14)
        main.pack(fill="both", expand=True)

        doc_frame = ttk.LabelFrame(main, text="Documents", padding=8)
        doc_frame.pack(fill="x", pady=5)
        list_wrap = ttk.Frame(doc_frame)
        list_wrap.pack(side="left", fill="both", expand=True)
        self.doc_listbox = tk.Listbox(list_wrap, height=5)
        self.doc_listbox.pack(fill="both", expand=True)
        doc_btns = ttk.Frame(doc_frame)
        doc_btns.pack(side="left", padx=10)
        self.add_btn = ttk.Button(doc_btns, text="Add Documents", command=self._add_documents)
        self.add_btn.pack(fill="x", pady=2)
        self.build_btn = ttk.Button(doc_btns, text="Build / Rebuild Index",
                                    command=self._build_index_thread)
        self.build_btn.pack(fill="x", pady=2)

        self.stats_var = tk.StringVar(value="Documents: 0    Chunks: 0    Index: Not Built")
        ttk.Label(main, textvariable=self.stats_var, font=("Segoe UI", 9)).pack(
            fill="x", pady=(2, 8)
        )

        q_frame = ttk.LabelFrame(main, text="Ask a question", padding=8)
        q_frame.pack(fill="x", pady=5)
        self.question_text = tk.Text(q_frame, height=3, wrap="word")
        self.question_text.pack(side="left", fill="x", expand=True)
        self.ask_btn = ttk.Button(q_frame, text="Ask Question", command=self._ask_question_thread)
        self.ask_btn.pack(side="left", padx=10)

        ans_frame = ttk.LabelFrame(main, text="Answer", padding=8)
        ans_frame.pack(fill="both", expand=True, pady=5)
        self.answer_text = scrolledtext.ScrolledText(ans_frame, height=6, wrap="word")
        self.answer_text.pack(fill="both", expand=True)
        self.answer_text.config(state="disabled")

        src_frame = ttk.LabelFrame(main, text="Sources", padding=8)
        src_frame.pack(fill="x", pady=5)
        self.sources_text = tk.Text(src_frame, height=3, wrap="word")
        self.sources_text.pack(fill="x")
        self.sources_text.config(state="disabled")

        ret_frame = ttk.LabelFrame(main, text="Retrieval Details (Top-K Chunks)", padding=8)
        ret_frame.pack(fill="both", expand=True, pady=5)
        self.retrieval_text = scrolledtext.ScrolledText(ret_frame, height=9, wrap="word")
        self.retrieval_text.pack(fill="both", expand=True)
        self.retrieval_text.config(state="disabled")

    def _set_status(self, text: str):
        self.status_var.set(text)
        self.status_label.configure(foreground=STATUS_COLORS.get(text, "#000000"))

    def _set_text(self, widget: tk.Text, content: str):
        widget.config(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", content)
        widget.config(state="disabled")

    def _set_busy(self, busy: bool):
        self.busy = busy
        state = "disabled" if busy else "normal"
        self.add_btn.config(state=state)
        self.build_btn.config(state=state)
        self.ask_btn.config(state=state)

    def _refresh_document_list(self):
        self.doc_listbox.delete(0, "end")
        filenames = loader.list_documents(config.DOCUMENTS_DIR)
        for name in filenames:
            self.doc_listbox.insert("end", name)
        self.doc_count = len(filenames)
        self._update_stats()

    def _update_stats(self, index_ready: bool | None = None):
        if index_ready is None:
            index_ready = self.index is not None and bool(self.chunks)
        status = "Ready" if index_ready else "Not Built"
        self.stats_var.set(
            f"Documents: {self.doc_count}    Chunks: {len(self.chunks)}    Index: {status}"
        )

    def _try_load_existing_index(self):
        if retriever.index_exists():
            try:
                self.index, self.chunks = retriever.load_index()
                self._set_status("INDEXED")
            except Exception:
                self.index, self.chunks = None, []
        self._update_stats()

    def _add_documents(self):
        if self.busy:
            return
        filepaths = filedialog.askopenfilenames(
            title="Select PDF or TXT documents",
            filetypes=[("Supported documents", "*.pdf *.txt")],
        )
        if not filepaths:
            return

        copied = 0
        errors = []
        for path in filepaths:
            try:
                dest = os.path.join(config.DOCUMENTS_DIR, os.path.basename(path))
                if os.path.abspath(path) != os.path.abspath(dest):
                    shutil.copy2(path, dest)
                copied += 1
            except Exception as exc:
                errors.append(f"{os.path.basename(path)}: {exc}")

        if copied:
            # The old index may no longer match the document folder.
            self.index, self.chunks = None, []
            retriever.delete_index()
            self._refresh_document_list()
            self._set_status("READY")

        if errors:
            messagebox.showerror("Some files could not be added", "\n".join(errors))

    def _build_index_thread(self):
        if self.busy:
            return
        filenames = loader.list_documents(config.DOCUMENTS_DIR)
        if not filenames:
            messagebox.showwarning("No documents", "Please add at least one PDF or TXT document first.")
            return
        self._set_busy(True)
        self._set_status("PROCESSING")
        threading.Thread(target=self._build_index_worker, args=(filenames,), daemon=True).start()

    def _build_index_worker(self, filenames: list[str]):
        try:
            all_chunks = []
            next_id = 0
            skipped = []
            for filename in filenames:
                filepath = os.path.join(config.DOCUMENTS_DIR, filename)
                try:
                    pages = loader.load_document(filepath)
                except ValueError as exc:
                    skipped.append(f"{filename}: {exc}")
                    continue
                file_chunks = chunker.chunk_pages(pages, start_id=next_id)
                next_id += len(file_chunks)
                all_chunks.extend(file_chunks)

            if not all_chunks:
                raise ValueError("No text could be extracted from the documents.")

            index, chunks = retriever.build_index(all_chunks)
            retriever.save_index(index, chunks)
            self.root.after(0, self._finish_build, index, chunks, skipped, filenames)
        except Exception as exc:
            self.root.after(0, self._build_failed, str(exc))

    def _finish_build(self, index, chunks, skipped, filenames):
        self.index, self.chunks = index, chunks
        self.doc_count = len(filenames)
        self._update_stats(index_ready=True)
        self._set_status("INDEXED")
        self._set_busy(False)
        if skipped:
            messagebox.showwarning("Some documents were skipped", "\n".join(skipped))
        messagebox.showinfo("Index built", f"Index built successfully with {len(chunks)} chunks.")

    def _build_failed(self, error: str):
        self._set_status("ERROR")
        self._set_busy(False)
        messagebox.showerror("Indexing failed", error)

    def _ask_question_thread(self):
        if self.busy:
            return
        question = self.question_text.get("1.0", "end").strip()
        if not question:
            messagebox.showwarning("Empty question", "Please type a question.")
            return
        if self.index is None or not self.chunks:
            messagebox.showwarning("Index not built", "Please build the index before asking questions.")
            return

        self._set_busy(True)
        self._set_status("PROCESSING")
        self._set_text(self.answer_text, "Searching documents...")
        self._set_text(self.sources_text, "")
        self._set_text(self.retrieval_text, "")
        index, chunks = self.index, self.chunks
        threading.Thread(target=self._ask_question_worker,
                         args=(question, index, chunks), daemon=True).start()

    def _ask_question_worker(self, question, index, chunks):
        try:
            results = retriever.search(index, chunks, question)
            if not rag.has_sufficient_evidence(results):
                answer = "No relevant information was found in the provided documents."
                relevant = []
            else:
                relevant = [r for r in results if r["similarity"] >= config.SIMILARITY_THRESHOLD]
                answer = rag.generate_answer(question, relevant)
            self.root.after(0, self._finish_question, answer, relevant, results)
        except rag.OllamaError as exc:
            self.root.after(0, self._question_failed, "Ollama error", str(exc))
        except Exception as exc:
            self.root.after(0, self._question_failed, "Error", str(exc))

    def _finish_question(self, answer, relevant, results):
        self._set_text(self.answer_text, answer)
        self._display_retrieval(results)
        self._display_sources(relevant)
        self._set_status("INDEXED")
        self._set_busy(False)

    def _question_failed(self, title: str, error: str):
        self._set_status("ERROR")
        self._set_busy(False)
        self._set_text(self.answer_text, "")
        messagebox.showerror(title, error)

    def _display_sources(self, results: list[dict]):
        lines = [
            f"{r['source']} — Page {r['page']} — Similarity: {r['similarity']:.2f}"
            for r in results
        ]
        self._set_text(self.sources_text, "\n".join(lines))

    def _display_retrieval(self, results: list[dict]):
        if not results:
            self._set_text(self.retrieval_text, "No chunks retrieved.")
            return
        blocks = []
        for rank, r in enumerate(results, start=1):
            preview = r["text"][:220].replace("\n", " ").strip()
            passed = "yes" if r["similarity"] >= config.SIMILARITY_THRESHOLD else "no"
            blocks.append(
                f"Rank {rank}\n"
                f"  Source: {r['source']}\n"
                f"  Page: {r['page']}\n"
                f"  Similarity: {r['similarity']:.4f}   "
                f"(passes threshold {config.SIMILARITY_THRESHOLD}: {passed})\n"
                f"  Preview: {preview}...\n"
            )
        self._set_text(self.retrieval_text, "\n".join(blocks))


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    MiniRAGApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

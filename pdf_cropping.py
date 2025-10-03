# pdf_cropping_gui.py
"""
PDF Trimming GUI (Tkinter + PyMuPDF)
- Canvas上の見た目どおりにPDFをクロップ
- 回転(/Rotate)・zoom・表示縮小(display_scale)は逆行列で一括補正
- 重要: get_pixmap のピクセルは「上原点(top-left)」→ Y反転しない！

依存:
  pip install pymupdf pillow
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    import fitz  # PyMuPDF
except Exception as e:  # pragma: no cover
    raise RuntimeError("PyMuPDF(fitz) が必要です。pip install pymupdf を実行してください") from e

try:
    from PIL import Image, ImageTk
except Exception as e:  # pragma: no cover
    raise RuntimeError("Pillow が必要です。pip install pillow を実行してください") from e


# ----------------------------- 型定義 -----------------------------
RectPts = Tuple[float, float, float, float]  # (x0,y0,x1,y1) in PDF points (左下原点)


@dataclass
class PageView:
    page_index: int
    zoom: float = 2.0                  # レンダリング倍率 (2.0 ≒ 144dpi)
    display_scale: float = 1.0         # さらに縮小して Canvas に表示
    pix_width: int = 1                 # get_pixmap 後のピクセル幅（prerotate・zoom 反映後）
    pix_height: int = 1                # 同ピクセル高さ
    img_tk: Optional[ImageTk.PhotoImage] = None
    mat: Optional[fitz.Matrix] = None      # ページ → デバイス行列（zoom + rotation）
    inv_mat: Optional[fitz.Matrix] = None  # 上記の逆行列（デバイス → ページ）


@dataclass
class Selection:
    # Canvas 上座標 (上原点, ピクセル)
    x0: Optional[int] = None
    y0: Optional[int] = None
    x1: Optional[int] = None
    y1: Optional[int] = None

    def normalized(self) -> Optional[Tuple[int, int, int, int]]:
        if None in (self.x0, self.y0, self.x1, self.y1):
            return None
        x0, y0, x1, y1 = self.x0, self.y0, self.x1, self.y1
        return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    def clear(self) -> None:
        self.x0 = self.y0 = self.x1 = self.y1 = None


@dataclass
class AppState:
    doc: Optional[fitz.Document] = None
    path: Optional[str] = None
    page_view: PageView = field(default_factory=lambda: PageView(page_index=0))
    selection: Selection = field(default_factory=Selection)
    selection_rect_id: Optional[int] = None  # Canvas item id
    apply_all: Optional[tk.BooleanVar] = None


# ----------------------------- ユーティリティ -----------------------------
IMG_OFFSET_X: int = 20  # Canvas 内の画像の左上オフセット
IMG_OFFSET_Y: int = 20


def _apply_matrix_to_point(mat: "fitz.Matrix", x: float, y: float) -> "fitz.Point":
    """Matrix を点に適用（バージョン差吸収のため手計算）。"""
    return fitz.Point(
        mat.a * x + mat.c * y + mat.e,
        mat.b * x + mat.d * y + mat.f,
    )


def selection_canvas_to_page_rect(sel: Selection, pv: PageView, page: "fitz.Page") -> Optional[RectPts]:
    """
    Canvas 矩形(上原点, px) → ページ座標(points, 下原点, 未回転)。
    注意: get_pixmap のピクセル座標は「上原点」。→ Y反転は不要！
    変換手順:
      Canvas(px, top-left)
        └─(オフセット除去)→ 表示画像 px (top-left)
        └─(/display_scale)→ デバイス px (top-left)   # get_pixmap のピクセル
        └─(inv_mat)──────→ ページ座標(points)
    """
    n = sel.normalized()
    if n is None or pv.inv_mat is None:
        return None
    x0, y0, x1, y1 = n

    # 画像のキャンバス内オフセットを除去
    x0 -= IMG_OFFSET_X; y0 -= IMG_OFFSET_Y
    x1 -= IMG_OFFSET_X; y1 -= IMG_OFFSET_Y

    # 表示画像サイズ（px, top-left）
    disp_w = pv.pix_width * pv.display_scale
    disp_h = pv.pix_height * pv.display_scale

    # 表示外をクリップ
    def clamp(v, lo, hi): return max(lo, min(hi, v))
    x0, x1 = sorted((clamp(x0, 0.0, disp_w), clamp(x1, 0.0, disp_w)))
    y0, y1 = sorted((clamp(y0, 0.0, disp_h), clamp(y1, 0.0, disp_h)))
    if abs(x1 - x0) < 1e-6 or abs(y1 - y0) < 1e-6:
        return None

    # 表示 px → デバイス px（プレビュー縮小を外す）※上原点のまま
    x0_dev = x0 / pv.display_scale
    y0_dev = y0 / pv.display_scale
    x1_dev = x1 / pv.display_scale
    y1_dev = y1 / pv.display_scale

    # 逆行列で「デバイス → ページ」へ（Y反転なし！）
    p0 = _apply_matrix_to_point(pv.inv_mat, x0_dev, y0_dev)
    p1 = _apply_matrix_to_point(pv.inv_mat, x1_dev, y1_dev)

    # 正規化 & page.rect でクリップ（基準は常に page.rect に統一）
    x0p, x1p = sorted((p0.x, p1.x))
    y0p, y1p = sorted((p0.y, p1.y))
    rect = fitz.Rect(x0p, y0p, x1p, y1p) & page.rect
    if rect.is_empty or rect.width <= 0 or rect.height <= 0:
        return None
    return (rect.x0, rect.y0, rect.x1, rect.y1)


# ----------------------------- GUI 本体 -----------------------------
class PdfCropperApp(ttk.Frame):
    def __init__(self, master: tk.Tk) -> None:
        super().__init__(master)
        self.master.title("PDF Cropper (Tkinter + PyMuPDF)")
        self.master.geometry("1024x768")
        self.pack(fill=tk.BOTH, expand=True)

        self.state = AppState()
        self.state.apply_all = tk.BooleanVar(value=True)

        self._build_widgets()
        self._bind_canvas_events()

    # ------------------------- UI 構築 -------------------------
    def _build_widgets(self) -> None:
        toolbar = ttk.Frame(self)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.btn_open = ttk.Button(toolbar, text="PDF を開く", command=self.open_pdf)
        self.btn_prev = ttk.Button(toolbar, text="◀ 前のページ", command=self.prev_page, state=tk.DISABLED)
        self.btn_next = ttk.Button(toolbar, text="次のページ ▶", command=self.next_page, state=tk.DISABLED)
        self.btn_reset = ttk.Button(toolbar, text="選択をリセット", command=self.reset_selection, state=tk.DISABLED)
        self.chk_apply_all = ttk.Checkbutton(toolbar, text="全ページに適用 (比率)", variable=self.state.apply_all)
        self.btn_save = ttk.Button(toolbar, text="別名で保存", command=self.save_cropped, state=tk.DISABLED)

        for w in (self.btn_open, self.btn_prev, self.btn_next, self.btn_reset, self.chk_apply_all, self.btn_save):
            w.pack(side=tk.LEFT, padx=6, pady=6)

        self.status = ttk.Label(self, text="PDF を開いてください", anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas = tk.Canvas(self, bg="#222222", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self._canvas_img_id: Optional[int] = None

    def _bind_canvas_events(self) -> None:
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

    # ------------------------- イベント処理 -------------------------
    def on_mouse_down(self, e: tk.Event) -> None:
        if not self.state.doc:
            return
        self.state.selection.x0 = int(e.x)
        self.state.selection.y0 = int(e.y)
        self.state.selection.x1 = int(e.x)
        self.state.selection.y1 = int(e.y)
        self._draw_selection()

    def on_mouse_drag(self, e: tk.Event) -> None:
        if not self.state.doc:
            return
        if self.state.selection.x0 is None:
            return
        self.state.selection.x1 = int(e.x)
        self.state.selection.y1 = int(e.y)
        self._draw_selection()

    def on_mouse_up(self, e: tk.Event) -> None:
        if not self.state.doc:
            return
        if self.state.selection.x0 is None:
            return
        self.state.selection.x1 = int(e.x)
        self.state.selection.y1 = int(e.y)
        self._draw_selection()

    def on_canvas_resize(self, e: tk.Event) -> None:
        if self.state.doc:
            self.render_current_page()

    # ------------------------- PDF 操作 -------------------------
    def open_pdf(self) -> None:
        path = filedialog.askopenfilename(title="PDF を選択", filetypes=[("PDF files", "*.pdf")])
        if not path:
            return
        try:
            doc = fitz.open(path)
        except Exception as e:
            messagebox.showerror("エラー", f"PDF を開けませんでした:\n{e}")
            return

        self.state.doc = doc
        self.state.path = path
        self.state.page_view = PageView(page_index=0)
        self.state.selection.clear()
        self._update_nav_buttons()
        self.render_current_page()
        self.status.config(text=f"{path} を開きました (全 {doc.page_count} ページ)")

    def prev_page(self) -> None:
        if not self.state.doc:
            return
        if self.state.page_view.page_index > 0:
            self.state.page_view.page_index -= 1
            self.state.selection.clear()
            self.render_current_page()
            self._update_nav_buttons()

    def next_page(self) -> None:
        if not self.state.doc:
            return
        if self.state.page_view.page_index < (self.state.doc.page_count - 1):
            self.state.page_view.page_index += 1
            self.state.selection.clear()
            self.render_current_page()
            self._update_nav_buttons()

    def reset_selection(self) -> None:
        self.state.selection.clear()
        if self.state.selection_rect_id is not None:
            self.canvas.delete(self.state.selection_rect_id)
            self.state.selection_rect_id = None

    def save_cropped(self) -> None:
        if not self.state.doc:
            return
        pv = self.state.page_view
        page = self.state.doc.load_page(pv.page_index)

        rect_pts = selection_canvas_to_page_rect(self.state.selection, pv, page)
        if rect_pts is None:
            messagebox.showwarning("選択なし", "トリミングする領域をドラッグで選択してください。")
            return

        # page.rect 基準でクロップ
        base_rect = page.rect
        crop = (fitz.Rect(*rect_pts) & base_rect)
        if crop.is_empty or crop.width <= 0 or crop.height <= 0:
            messagebox.showwarning("選択なし", "選択領域がページ内にありません。再度選択してください。")
            return

        # 全ページ適用のため比率（page.rect 基準）
        fx0 = (crop.x0 - base_rect.x0) / float(base_rect.width)
        fy0 = (crop.y0 - base_rect.y0) / float(base_rect.height)
        fx1 = (crop.x1 - base_rect.x0) / float(base_rect.width)
        fy1 = (crop.y1 - base_rect.y0) / float(base_rect.height)

        save_path = filedialog.asksaveasfilename(
            title="保存先を選択", defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")], initialfile="cropped.pdf",
        )
        if not save_path:
            return

        try:
            out = fitz.open(self.state.path)
            if self.state.apply_all and self.state.apply_all.get():
                for i in range(out.page_count):
                    p = out.load_page(i)
                    b = p.rect  # 全ページ page.rect 基準
                    rect = fitz.Rect(
                        b.x0 + fx0 * b.width,  b.y0 + fy0 * b.height,
                        b.x0 + fx1 * b.width,  b.y0 + fy1 * b.height,
                    ) & b
                    if not rect.is_empty:
                        p.set_cropbox(rect)
                        # 必要なら MediaBox も合わせるとビューア差異がさらに減る:
                        # p.set_mediabox(rect)
            else:
                p = out.load_page(pv.page_index)
                b = p.rect
                rect = (crop & b)
                p.set_cropbox(rect)
                # p.set_mediabox(rect)  # 必要なら有効化

            out.save(save_path)
            out.close()
            messagebox.showinfo("保存完了", f"保存しました:\n{save_path}")
        except Exception as e:
            messagebox.showerror("保存エラー", f"保存に失敗しました:\n{e}")

    # ------------------------- レンダリング -------------------------
    def render_current_page(self) -> None:
        assert self.state.doc is not None
        pv = self.state.page_view
        page = self.state.doc.load_page(pv.page_index)

        # ページ→デバイス行列（回転をプレビューに反映）
        mat = fitz.Matrix(pv.zoom, pv.zoom).prerotate(page.rotation)
        pv.mat = mat
        inv = fitz.Matrix(mat); inv.invert()
        pv.inv_mat = inv

        # ビットマップ化（page.rect 基準）
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pv.pix_width, pv.pix_height = pix.width, pix.height

        # Canvas サイズに合わせて表示縮小（拡大はしない）
        c_w = max(self.canvas.winfo_width() - 40, 200)
        c_h = max(self.canvas.winfo_height() - 40, 200)
        scale_w = c_w / pv.pix_width
        scale_h = c_h / pv.pix_height
        pv.display_scale = float(min(scale_w, scale_h, 1.0))

        # PIL 画像へ変換／縮小
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        if pv.display_scale < 1.0:
            new_size = (max(1, int(pix.width * pv.display_scale)),
                        max(1, int(pix.height * pv.display_scale)))
            img = img.resize(new_size, Image.LANCZOS)
        pv.img_tk = ImageTk.PhotoImage(img)

        # Canvas に描画
        self.canvas.delete("all")
        self._canvas_img_id = self.canvas.create_image(
            IMG_OFFSET_X, IMG_OFFSET_Y, image=pv.img_tk, anchor=tk.NW
        )
        self.canvas.config(scrollregion=(0, 0, img.width + IMG_OFFSET_X * 2, img.height + IMG_OFFSET_Y * 2))

        # 既存の選択枠を再描画
        if self.state.selection_rect_id is not None:
            self.canvas.delete(self.state.selection_rect_id)
            self.state.selection_rect_id = None
        if self.state.selection.normalized() is not None:
            self._draw_selection()

        # ボタン状態・ステータス
        self.btn_prev.config(state=(tk.NORMAL if pv.page_index > 0 else tk.DISABLED))
        self.btn_next.config(state=(tk.NORMAL if pv.page_index < self.state.doc.page_count - 1 else tk.DISABLED))
        self.btn_reset.config(state=tk.NORMAL)
        self.btn_save.config(state=tk.NORMAL)

        self.status.config(
            text=f"ページ {pv.page_index + 1} / {self.state.doc.page_count} | "
                 f"zoom: {pv.zoom:.1f}x | disp: {pv.display_scale:.2f}"
        )

    def _draw_selection(self) -> None:
        n = self.state.selection.normalized()
        if n is None:
            return
        x0, y0, x1, y1 = n
        if self.state.selection_rect_id is not None:
            self.canvas.delete(self.state.selection_rect_id)
        self.state.selection_rect_id = self.canvas.create_rectangle(
            x0, y0, x1, y1, outline="#00FF88", dash=(6, 3), width=2
        )

    def _update_nav_buttons(self) -> None:
        if not self.state.doc:
            self.btn_prev.config(state=tk.DISABLED)
            self.btn_next.config(state=tk.DISABLED)
            self.btn_reset.config(state=tk.DISABLED)
            self.btn_save.config(state=tk.DISABLED)
        else:
            self.btn_prev.config(state=(tk.NORMAL if self.state.page_view.page_index > 0 else tk.DISABLED))
            self.btn_next.config(state=(tk.NORMAL if self.state.page_view.page_index < self.state.doc.page_count - 1 else tk.DISABLED))
            self.btn_reset.config(state=tk.NORMAL)
            self.btn_save.config(state=tk.NORMAL)


# ----------------------------- エントリポイント -----------------------------
def main() -> None:
    root = tk.Tk()
    try:
        root.call("tk", "scaling", 1.2)
        style = ttk.Style()
        if sys.platform == "darwin":
            style.theme_use("aqua")
        else:
            style.theme_use("clam")
    except Exception:
        pass

    app = PdfCropperApp(root)
    app.mainloop()


if __name__ == "__main__":
    main()

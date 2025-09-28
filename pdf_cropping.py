# Author: kamekingdom (2025-09-29)
"""
PDF Trimming GUI (Tkinter + PyMuPDF)

機能概要:
- PDF をファイルダイアログから選択
- ページ画像を Canvas に表示
- マウスドラッグでトリミング領域(矩形)を指定 (ラバーバンド表示)
- 前/次ページ移動、選択のリセット
- 現在ページのみ / 全ページに同一比率で適用を選択
- 別名保存でクロップ済み PDF を出力

依存関係:
  pip install pymupdf pillow
  (PyMuPDF は "fitz" 名で import)

注意:
- PyMuPDF の座標は PDF 下原点(左下が (0,0))。Canvas 上は上原点。変換ロジックで吸収しています。
- ページごとの寸法が異なる場合でも、"全ページに適用" は比率ベースで適用します。
"""
from __future__ import annotations

import io
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
RectFrac = Tuple[float, float, float, float]  # (x0/W, y0/H, x1/W, y1/H) 比率


@dataclass
class PageView:
    page_index: int
    zoom: float = 2.0  # レンダリング倍率 (2.0 ≒ 144dpi)
    display_scale: float = 1.0  # 画像のさらに縮小倍率
    pix_width: int = 1
    pix_height: int = 1
    img_tk: Optional[ImageTk.PhotoImage] = None


@dataclass
class Selection:
    # Canvas 上座標 (上原点, ピクセル単位)
    x0: Optional[int] = None
    y0: Optional[int] = None
    x1: Optional[int] = None
    y1: Optional[int] = None

    def normalized(self) -> Optional[Tuple[int, int, int, int]]:
        if None in (self.x0, self.y0, self.x1, self.y1):
            return None
        x0, y0, x1, y1 = self.x0, self.y0, self.x1, self.y1
        if x0 is None or y0 is None or x1 is None or y1 is None:
            return None
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
    apply_all: tk.BooleanVar | None = None


# ----------------------------- ユーティリティ -----------------------------
# 画像の描画原点（Canvas 内のオフセット）
IMG_OFFSET_X: int = 20
IMG_OFFSET_Y: int = 20


def display_to_points(
    x_disp: float,
    y_disp: float,
    page_rect: fitz.Rect,
    zoom: float,
    display_scale: float,
) -> Tuple[float, float]:
    """Canvas(上原点, px) → PDF points(下原点) に変換。

    変換は以下の考え方:
      pix_width  = page_rect.width  * zoom
      disp_width = pix_width * display_scale
      よって 1 disp_px = 1 / (zoom * display_scale) [points]

    また Y については PDF 下原点に合わせるため、
      y_top_pts = y_disp / (zoom * display_scale)
      y_pts = page_rect.height - y_top_pts
    """
    scale = 1.0 / (zoom * display_scale)
    x_pts_top = x_disp * scale
    y_top_pts = y_disp * scale
    x_pts = x_pts_top
    y_pts = float(page_rect.height) - y_top_pts
    return x_pts, y_pts


def selection_disp_to_points(
    sel: Selection, page_rect: fitz.Rect, zoom: float, display_scale: float
) -> Optional[RectPts]:
    """Canvas 選択(上原点, px) → PDF points(下原点) への変換。

    - Canvas では画像は (IMG_OFFSET_X, IMG_OFFSET_Y) に配置されているため、
      選択座標からこのオフセットを減算する。
    - さらに、表示画像サイズ (disp_w, disp_h) を超える分をクリップして
      MediaBox 外に出ないようにする。
    """
    n = sel.normalized()
    if n is None:
        return None
    x0, y0, x1, y1 = n

    # オフセット補正
    x0 -= IMG_OFFSET_X
    y0 -= IMG_OFFSET_Y
    x1 -= IMG_OFFSET_X
    y1 -= IMG_OFFSET_Y

    # 表示画像サイズ（px）
    disp_w = float(page_rect.width) * zoom * display_scale
    disp_h = float(page_rect.height) * zoom * display_scale

    # クリッピング（[0, disp_w/h] に制限）
    def clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    x0 = clamp(x0, 0.0, disp_w)
    x1 = clamp(x1, 0.0, disp_w)
    y0 = clamp(y0, 0.0, disp_h)
    y1 = clamp(y1, 0.0, disp_h)

    # 無効領域は破棄
    if abs(x1 - x0) < 1e-6 or abs(y1 - y0) < 1e-6:
        return None

    # px → points 変換
    scale = 1.0 / (zoom * display_scale)
    x0_top = x0 * scale
    x1_top = x1 * scale
    y0_top = y0 * scale
    y1_top = y1 * scale

    # 上原点 → 下原点
    H = float(page_rect.height)
    y0_pts = H - y0_top
    y1_pts = H - y1_top

    x0p, x1p = x0_top, x1_top
    y0p, y1p = y0_pts, y1_pts

    # 正規化
    return (min(x0p, x1p), min(y0p, y1p), max(x0p, x1p), max(y0p, y1p))


def rect_points_to_frac(rect_pts: RectPts, page_rect: fitz.Rect) -> RectFrac:
    W = float(page_rect.width)
    H = float(page_rect.height)
    x0, y0, x1, y1 = rect_pts
    return (x0 / W, y0 / H, x1 / W, y1 / H)


def rect_frac_to_points(frac: RectFrac, page_rect: fitz.Rect) -> RectPts:
    W = float(page_rect.width)
    H = float(page_rect.height)
    fx0, fy0, fx1, fy1 = frac
    return (fx0 * W, fy0 * H, fx1 * W, fy1 * H)


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
        # Top toolbar
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

        # Status bar
        self.status = ttk.Label(self, text="PDF を開いてください", anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

        # Canvas (画像表示)
        self.canvas = tk.Canvas(self, bg="#222222", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 画像オブジェクト保持用
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
        # ウィンドウサイズ変更時に現在ページを再描画 (表示倍率のみ再計算)
        if self.state.doc:
            self.render_current_page()

    # ------------------------- PDF 操作 -------------------------
    def open_pdf(self) -> None:
        path = filedialog.askopenfilename(
            title="PDF を選択",
            filetypes=[("PDF files", "*.pdf")]
        )
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
        rect_pts = selection_disp_to_points(self.state.selection, page.rect, pv.zoom, pv.display_scale)
        if rect_pts is None:
            messagebox.showwarning("選択なし", "トリミングする領域をドラッグで選択してください。")
            return

        # 比率で保持 (全ページ適用用)
        frac = rect_points_to_frac(rect_pts, page.rect)

        save_path = filedialog.asksaveasfilename(
            title="保存先を選択",
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            initialfile="cropped.pdf",
        )
        if not save_path:
            return

        try:
            # 元ドキュメントをコピーして安全に更新
            out = fitz.open(self.state.path)
            if self.state.apply_all and self.state.apply_all.get():
                for i in range(out.page_count):
                    p = out.load_page(i)
                    pts = rect_frac_to_points(frac, p.rect)
                    crop = fitz.Rect(*pts) & p.rect
                    if crop.is_empty or crop.width <= 0 or crop.height <= 0:
                        continue
                    p.set_cropbox(crop)
            else:
                p = out.load_page(pv.page_index)
                crop = (fitz.Rect(*rect_pts) & p.rect)
                if crop.is_empty or crop.width <= 0 or crop.height <= 0:
                    messagebox.showwarning("選択なし", "選択領域がページ内にありません。再度選択してください。")
                    out.close()
                    return
                p.set_cropbox(crop)

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

        # レンダリング解像度
        zoom = pv.zoom
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        pv.pix_width, pv.pix_height = pix.width, pix.height

        # Canvas 幅に合わせて表示縮小(横基準)。余裕を持って 40px 余白
        c_w = max(self.canvas.winfo_width() - 40, 200)
        c_h = max(self.canvas.winfo_height() - 40, 200)
        # 横幅基準で縮小。ただし縦がはみ出る場合は縦基準に切替
        scale_w = c_w / pv.pix_width
        scale_h = c_h / pv.pix_height
        display_scale = min(scale_w, scale_h, 1.0)  # 1.0 を超えて拡大しない
        pv.display_scale = float(display_scale)

        # PIL 画像へ
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        if display_scale < 1.0:
            new_size = (max(1, int(pix.width * display_scale)), max(1, int(pix.height * display_scale)))
            img = img.resize(new_size, Image.LANCZOS)
        pv.img_tk = ImageTk.PhotoImage(img)

        # Canvas に描画
        self.canvas.delete("all")
        self._canvas_img_id = self.canvas.create_image(
            IMG_OFFSET_X, IMG_OFFSET_Y, image=pv.img_tk, anchor=tk.NW
        )
        self.canvas.config(scrollregion=(0, 0, img.width + IMG_OFFSET_X*2, img.height + IMG_OFFSET_Y*2))

        # 選択枠の再描画
        if self.state.selection_rect_id is not None:
            self.state.selection_rect_id = None
        if self.state.selection.normalized() is not None:
            self._draw_selection()

        # ボタン状態
        self.btn_prev.config(state=(tk.NORMAL if pv.page_index > 0 else tk.DISABLED))
        self.btn_next.config(state=(tk.NORMAL if pv.page_index < self.state.doc.page_count - 1 else tk.DISABLED))
        self.btn_reset.config(state=tk.NORMAL)
        self.btn_save.config(state=tk.NORMAL)

        self.status.config(text=f"ページ {pv.page_index + 1} / {self.state.doc.page_count} | 表示倍率: {zoom:.1f}x × {display_scale:.2f}")

    def _draw_selection(self) -> None:
        n = self.state.selection.normalized()
        if n is None:
            return
        x0, y0, x1, y1 = n
        x0o, y0o, x1o, y1o = x0, y0, x1, y1
        # 既存の枠を削除して描画
        if self.state.selection_rect_id is not None:
            self.canvas.delete(self.state.selection_rect_id)
        self.state.selection_rect_id = self.canvas.create_rectangle(
            x0o, y0o, x1o, y1o,
            outline="#00FF88",
            dash=(6, 3),
            width=2
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
    # ttk の見た目
    try:
        root.call("tk", "scaling", 1.2)  # HiDPI 環境向け (必要なら)
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

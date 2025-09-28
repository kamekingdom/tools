# Author: kamekingdom (2025-09-29)
"""
複数 PDF のページを 1 ページの PDF に合成する GUI ツール
- Tkinter + PyMuPDF + Pillow
- 位置・サイズは GUI 上の見た目通りに PDF 化（前後関係を含む）

依存:
    pip install pymupdf pillow
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    import fitz  # PyMuPDF
except Exception as e:
    raise RuntimeError("PyMuPDF(fitz) が必要です。pip install pymupdf") from e

try:
    from PIL import Image, ImageTk
except Exception as e:
    raise RuntimeError("Pillow が必要です。pip install pillow") from e


# ------------------------- 型 -------------------------
Point = Tuple[float, float]          # (x, y) pt, 上原点
Rect  = Tuple[float, float, float, float]  # (x, y, w, h) pt, 上原点


@dataclass
class PdfItem:
    path: str
    page_index: int
    x: float
    y: float
    scale: float
    w: float     # 元ページ幅 pt
    h: float     # 元ページ高 pt
    thumb: Optional[ImageTk.PhotoImage] = None
    canvas_id: Optional[int] = None

    def bbox(self) -> Rect:
        return (self.x, self.y, self.w * self.scale, self.h * self.scale)

    def center(self) -> Point:
        x, y, w, h = self.bbox()
        return (x + w / 2.0, y + h / 2.0)


@dataclass
class AppState:
    items: List[PdfItem] = field(default_factory=list)
    selected: Set[int] = field(default_factory=set)

    dragging_index: Optional[int] = None
    drag_offset: Tuple[float, float] = (0.0, 0.0)

    # リサイズ用
    resizing_index: Optional[int] = None
    resizing_corner: Optional[str] = None  # 'nw','ne','sw','se'
    resize_anchor: Tuple[float, float] = (0.0, 0.0)
    resize_orig_wh: Tuple[float, float] = (1.0, 1.0)
    resize_orig_scale: float = 1.0

    view_scale: float = 0.5  # 表示倍率 px/pt
    grid: int = 10
    snap: bool = True
    snap_tol: float = 6.0


# ------------------------- アプリ -------------------------
class PdfCollageApp(ttk.Frame):
    IMG_BG = "#222222"
    GUIDE = "#00E0FF"
    HANDLE_PX = 10

    def __init__(self, master: tk.Tk) -> None:
        super().__init__(master)
        master.title("PDF Collage Composer (Tkinter + PyMuPDF)")
        master.geometry("1200x800")
        self.pack(fill=tk.BOTH, expand=True)

        self.state = AppState()
        self._guide_ids: List[int] = []

        self._build_widgets()
        self._bind()
        self._update_status()

    # ---------- UI ----------
    def _build_widgets(self) -> None:
        tb = ttk.Frame(self)
        tb.pack(side=tk.TOP, fill=tk.X)

        # ファイル操作
        ttk.Button(tb, text="PDF追加", command=self.add_pdfs).pack(side=tk.LEFT, padx=4, pady=4)
        ttk.Button(tb, text="削除", command=self.remove_selected).pack(side=tk.LEFT, padx=4)

        ttk.Separator(tb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        # 表示操作
        ttk.Button(tb, text="拡大 +", command=lambda: self.zoom_canvas(1.2)).pack(side=tk.LEFT, padx=2)
        ttk.Button(tb, text="縮小 -", command=lambda: self.zoom_canvas(1/1.2)).pack(side=tk.LEFT, padx=2)
        ttk.Button(tb, text="表示をフィット", command=self.fit_view).pack(side=tk.LEFT, padx=2)

        ttk.Separator(tb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        # 選択要素スケール
        self.scale_var: tk.DoubleVar = tk.DoubleVar(value=1.0)
        ttk.Label(tb, text="選択スケール").pack(side=tk.LEFT)
        self.scale_spin = ttk.Spinbox(
            tb, from_=0.05, to=5.0, increment=0.05,
            textvariable=self.scale_var, width=6,
            command=self.apply_scale_spin
        )
        self.scale_spin.pack(side=tk.LEFT)
        ttk.Button(tb, text="×0.9", command=lambda: self.scale_selected(1/1.111111)).pack(side=tk.LEFT, padx=2)
        ttk.Button(tb, text="×1.1", command=lambda: self.scale_selected(1.111111)).pack(side=tk.LEFT, padx=2)

        ttk.Separator(tb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        # スナップ
        self.snap_var: tk.BooleanVar = tk.BooleanVar(value=True)
        ttk.Checkbutton(tb, text="スナップ", variable=self.snap_var, command=self.toggle_snap).pack(side=tk.LEFT)
        ttk.Label(tb, text="グリッド(pt)").pack(side=tk.LEFT, padx=(8, 2))
        self.grid_var: tk.IntVar = tk.IntVar(value=10)
        ttk.Spinbox(tb, from_=2, to=100, increment=1, textvariable=self.grid_var, width=5,
                    command=self.redraw).pack(side=tk.LEFT)

        ttk.Separator(tb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        # 縦方向の整列のみ残す
        ttk.Separator(tb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        # 書き出し
        ttk.Button(tb, text="PDFとして書き出し", command=self.export_pdf).pack(side=tk.LEFT, padx=6)

        # Status
        self.status = ttk.Label(self, text="PDFを追加してください", anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

        # Canvas
        self.canvas = tk.Canvas(self, bg=self.IMG_BG, highlightthickness=0, cursor="arrow")
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def _bind(self) -> None:
        self.canvas.bind('<Configure>', lambda e: self.redraw())
        self.canvas.bind('<Motion>', self.on_motion)
        self.canvas.bind('<Button-1>', self.on_left_down)
        self.canvas.bind('<B1-Motion>', self.on_left_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_left_up)
        self.canvas.bind('<MouseWheel>', self.on_wheel)   # Windows
        self.canvas.bind('<Button-4>', self.on_wheel)     # Linux up
        self.canvas.bind('<Button-5>', self.on_wheel)     # Linux down
        self.master.bind('+', lambda e: self.zoom_canvas(1.2))
        self.master.bind('-', lambda e: self.zoom_canvas(1/1.2))

    # ---------- 座標/描画 ----------
    def _pt2px(self, v: float) -> float:
        return v * self.state.view_scale

    def _px2pt(self, v: float) -> float:
        return v / self.state.view_scale

    def _draw_guides(self, lines: List[Tuple[float, float, float, float]]) -> None:
        for gid in self._guide_ids:
            self.canvas.delete(gid)
        self._guide_ids.clear()
        for x0, y0, x1, y1 in lines:
            gid = self.canvas.create_line(x0, y0, x1, y1, fill=self.GUIDE, dash=(6, 3), width=1)
            self._guide_ids.append(gid)

    def _clear_guides(self) -> None:
        self._draw_guides([])

    def redraw(self) -> None:
        """キャンバス全体を再描画"""
        self.state.grid = int(self.grid_var.get())
        self.state.snap = bool(self.snap_var.get())
        self.canvas.delete('all')

        # グリッド
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        gpx = max(1, int(self._pt2px(self.state.grid)))
        for x in range(0, max(1, w), gpx):
            self.canvas.create_line(x, 0, x, h, fill="#333333")
        for y in range(0, max(1, h), gpx):
            self.canvas.create_line(0, y, w, y, fill="#333333")

        # アイテム（リスト順：後方ほど前面）
        for idx, it in enumerate(self.state.items):
            self._ensure_thumb(idx)
            x, y, iw, ih = it.bbox()
            xpx, ypx = self._pt2px(x), self._pt2px(y)
            wpx, hpx = self._pt2px(iw), self._pt2px(ih)
            if it.canvas_id:
                try:
                    self.canvas.delete(it.canvas_id)
                except Exception:
                    pass
            it.canvas_id = self.canvas.create_image(xpx, ypx, image=it.thumb, anchor=tk.NW)
            outline = "#00ff88" if idx in self.state.selected else "#aaaaaa"
            self.canvas.create_rectangle(xpx, ypx, xpx + wpx, ypx + hpx, outline=outline, width=2)
            # ハンドル
            if idx in self.state.selected:
                for hx, hy in [(xpx, ypx), (xpx + wpx, ypx), (xpx, ypx + hpx), (xpx + wpx, ypx + hpx)]:
                    self.canvas.create_rectangle(
                        hx - self.HANDLE_PX / 2, hy - self.HANDLE_PX / 2,
                        hx + self.HANDLE_PX / 2, hy + self.HANDLE_PX / 2,
                        fill="#00ff88", outline=""
                    )
        self._update_status()

    def _ensure_thumb(self, idx: int) -> None:
        it = self.state.items[idx]
        if it.thumb is not None:
            return
        zoom = max(0.05, min(5.0, it.scale * self.state.view_scale))
        mat = fitz.Matrix(zoom, zoom)
        with fitz.open(it.path) as d:
            p = d.load_page(it.page_index)
            pix = p.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        it.thumb = ImageTk.PhotoImage(img)

    # ---------- データ操作 ----------
    def add_pdfs(self) -> None:
        paths = filedialog.askopenfilenames(
            title="PDF を選択",
            filetypes=[("PDF files", "*.pdf")]
        )
        if not paths:
            return
        for path in paths:
            try:
                with fitz.open(path) as d:
                    p = d.load_page(0)
                    w, h = float(p.rect.width), float(p.rect.height)
            except Exception as e:
                messagebox.showerror("読み込みエラー", f"{os.path.basename(path)}: {e}")
                continue
            margin = 20.0
            x = margin + (len(self.state.items) % 5) * (w * 0.25 + margin)
            y = margin + (len(self.state.items) // 5) * (h * 0.25 + margin)
            self.state.items.append(PdfItem(path, 0, x, y, 0.25, w, h))
        self.redraw()

    def remove_selected(self) -> None:
        if not self.state.selected:
            return
        keep: List[PdfItem] = []
        for i, it in enumerate(self.state.items):
            if i not in self.state.selected:
                keep.append(it)
        self.state.items = keep
        self.state.selected.clear()
        self.redraw()

    # ---------- ヒットテスト ----------
    def _hit_test(self, px: float, py: float) -> Optional[int]:
        for i in reversed(range(len(self.state.items))):  # 前面から判定
            x, y, w, h = self.state.items[i].bbox()
            if x <= px <= x + w and y <= py <= y + h:
                return i
        return None

    def _hit_handle(self, px: float, py: float) -> Tuple[Optional[int], Optional[str]]:
        r = self.HANDLE_PX / 2.0 / self.state.view_scale  # pt
        for i in reversed(range(len(self.state.items))):
            x, y, w, h = self.state.items[i].bbox()
            corners = {
                "nw": (x, y),
                "ne": (x + w, y),
                "sw": (x, y + h),
                "se": (x + w, y + h),
            }
            for name, (cx, cy) in corners.items():
                if abs(px - cx) <= r and abs(py - cy) <= r:
                    return i, name
        return None, None

    # ---------- イベント ----------
    def on_motion(self, e: tk.Event) -> None:
        px, py = self._px2pt(e.x), self._px2pt(e.y)
        _, corner = self._hit_handle(px, py)
        self.canvas.config(cursor="sizing" if corner else "arrow")

    def on_left_down(self, e: tk.Event) -> None:
        px, py = self._px2pt(e.x), self._px2pt(e.y)

        # 先にハンドル
        hid, corner = self._hit_handle(px, py)
        if hid is not None and corner is not None:
            if hid not in self.state.selected:
                self.state.selected = {hid}
            it = self.state.items[hid]
            self.state.resizing_index = hid
            self.state.resizing_corner = corner
            x, y, w, h = it.bbox()
            anchors = {"nw": (x + w, y + h), "ne": (x, y + h), "sw": (x + w, y), "se": (x, y)}
            self.state.resize_anchor = anchors[corner]
            self.state.resize_orig_wh = (w, h)
            self.state.resize_orig_scale = it.scale
            return

        idx = self._hit_test(px, py)
        if idx is None:
            if not (e.state & 0x0001):  # Shift 未押下
                self.state.selected.clear()
            self.redraw()
            return

        if e.state & 0x0001:  # Shift でトグル
            if idx in self.state.selected:
                self.state.selected.remove(idx)
            else:
                self.state.selected.add(idx)
        else:
            if idx not in self.state.selected:
                self.state.selected = {idx}
            # クリックした要素を前面へ（描画順＝書き出し順を一致）
            self._bring_to_front(idx)
            idx = len(self.state.items) - 1
            self.state.selected = {idx}

        self.state.dragging_index = idx
        it = self.state.items[idx]
        self.state.drag_offset = (px - it.x, py - it.y)
        self.redraw()

    def on_left_drag(self, e: tk.Event) -> None:
        px, py = self._px2pt(e.x), self._px2pt(e.y)

        # リサイズ中
        if self.state.resizing_index is not None and self.state.resizing_corner:
            i = self.state.resizing_index
            it = self.state.items[i]
            ax, ay = self.state.resize_anchor
            dx = abs(px - ax)
            dy = abs(py - ay)
            w0, h0 = self.state.resize_orig_wh
            s = max(0.05, min(10.0, max(dx / max(1e-6, w0), dy / max(1e-6, h0))))
            it.scale = s * self.state.resize_orig_scale
            w, h = it.w * it.scale, it.h * it.scale
            c = self.state.resizing_corner
            if c == "nw":
                it.x = ax - w; it.y = ay - h
            elif c == "ne":
                it.x = ax;     it.y = ay - h
            elif c == "sw":
                it.x = ax - w; it.y = ay
            else:  # "se"
                it.x = ax;     it.y = ay
            it.thumb = None
            self.redraw()
            return

        # 移動中
        if self.state.dragging_index is None:
            return
        idx = self.state.dragging_index
        it = self.state.items[idx]
        nx = px - self.state.drag_offset[0]
        ny = py - self.state.drag_offset[1]

        guides: List[Tuple[float, float, float, float]] = []
        if self.state.snap:
            g = float(self.state.grid)
            nx = round(nx / g) * g
            ny = round(ny / g) * g
            tol = self.state.snap_tol
            x, y, w, h = it.bbox()
            cand_x = [nx, nx + w / 2, nx + w]
            cand_y = [ny, ny + h / 2, ny + h]
            for j, other in enumerate(self.state.items):
                if j == idx:
                    continue
                ox, oy, ow, oh = other.bbox()
                tgt_x = [ox, ox + ow / 2, ox + ow]
                tgt_y = [oy, oy + oh / 2, oy + oh]
                for cx in cand_x:
                    for tx in tgt_x:
                        if abs(cx - tx) <= tol:
                            dx2 = tx - cx
                            nx += dx2
                            x0 = self._pt2px(tx)
                            guides.append((x0, 0, x0, self.canvas.winfo_height()))
                            break
                for cy in cand_y:
                    for ty in tgt_y:
                        if abs(cy - ty) <= tol:
                            dy2 = ty - cy
                            ny += dy2
                            y0 = self._pt2px(ty)
                            guides.append((0, y0, self.canvas.winfo_width(), y0))
                            break

        it.x, it.y = nx, ny
        self._draw_guides(guides)
        self.redraw()

    def on_left_up(self, e: tk.Event) -> None:
        self.state.dragging_index = None
        self.state.resizing_index = None
        self.state.resizing_corner = None
        self._clear_guides()

    def on_wheel(self, e: tk.Event) -> None:
        if not self.state.selected:
            return
        delta = 0
        if hasattr(e, "delta") and e.delta:
            delta = e.delta
        elif getattr(e, "num", None) == 4:
            delta = 120
        elif getattr(e, "num", None) == 5:
            delta = -120
        factor = 1.0 + (0.1 if delta > 0 else -0.1)
        self.scale_selected(factor)

    # ---------- 操作 ----------
    def scale_selected(self, factor: float) -> None:
        for i in self.state.selected:
            it = self.state.items[i]
            it.scale = float(max(0.05, min(10.0, it.scale * factor)))
            it.thumb = None
        if len(self.state.selected) == 1:
            i = next(iter(self.state.selected))
            self.scale_var.set(self.state.items[i].scale)
        self.redraw()

    def apply_scale_spin(self) -> None:
        if not self.state.selected:
            return
        val = float(self.scale_var.get())
        for i in self.state.selected:
            it = self.state.items[i]
            it.scale = float(max(0.05, min(10.0, val)))
            it.thumb = None
        self.redraw()

    def toggle_snap(self) -> None:
        self.state.snap = bool(self.snap_var.get())

    def align(self, mode: str) -> None:
        idxs = sorted(self.state.selected)
        if len(idxs) < 2:
            return
        items = [self.state.items[i] for i in idxs]
        xs = [it.x for it in items]
        ys = [it.y for it in items]
        ws = [it.w * it.scale for it in items]
        hs = [it.h * it.scale for it in items]
        if mode == "top":
            top = min(ys)
            for it in items:
                it.y = top
        elif mode == "bottom":
            bottom = max(y + h for y, h in zip(ys, hs))
            for it, y, h in zip(items, ys, hs):
                it.y = bottom - h
        elif mode == "center_y":
            cy = sum(y + h / 2 for y, h in zip(ys, hs)) / len(items)
            for it, h in zip(items, hs):
                it.y = cy - h / 2
        self.redraw()

    def distribute(self, axis: str) -> None:
        # （必要なら呼ぶ）今回は UI から外しているが関数は残す
        idxs = sorted(self.state.selected)
        if len(idxs) < 3:
            return
        items = [self.state.items[i] for i in idxs]
        if axis == "x":
            items.sort(key=lambda it: it.x)
            left = items[0].x
            right = max(it.x + it.w * it.scale for it in items)
            total_w = sum(it.w * it.scale for it in items)
            gaps = len(items) - 1
            if gaps <= 0:
                return
            gap = (right - left - total_w) / gaps
            x = left
            for it in items:
                it.x = x
                x += it.w * it.scale + gap
        else:
            items.sort(key=lambda it: it.y)
            top = items[0].y
            bottom = max(it.y + it.h * it.scale for it in items)
            total_h = sum(it.h * it.scale for it in items)
            gaps = len(items) - 1
            if gaps <= 0:
                return
            gap = (bottom - top - total_h) / gaps
            y = top
            for it in items:
                it.y = y
                y += it.h * it.scale + gap
        self.redraw()

    def zoom_canvas(self, fac: float) -> None:
        self.state.view_scale = float(max(0.1, min(4.0, self.state.view_scale * fac)))
        for it in self.state.items:
            it.thumb = None
        self.redraw()

    def fit_view(self) -> None:
        if not self.state.items:
            return
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        x0, y0, x1, y1 = self._world_bounds()
        if x1 - x0 <= 1 or y1 - y0 <= 1:
            return
        scale_x = (w - 40) / (x1 - x0)
        scale_y = (h - 40) / (y1 - y0)
        self.state.view_scale = float(max(0.1, min(4.0, min(scale_x, scale_y))))
        for it in self.state.items:
            it.thumb = None
        self.redraw()

    def _world_bounds(self) -> Tuple[float, float, float, float]:
        if not self.state.items:
            return (0.0, 0.0, 100.0, 100.0)
        xs = [it.x for it in self.state.items]
        ys = [it.y for it in self.state.items]
        xe = [it.x + it.w * it.scale for it in self.state.items]
        ye = [it.y + it.h * it.scale for it in self.state.items]
        return (min(xs), min(ys), max(xe), max(ye))

    def _update_status(self) -> None:
        n = len(self.state.items)
        sel = len(self.state.selected)
        self.status.config(text=f"要素: {n} | 選択: {sel} | 表示倍率: {self.state.view_scale:.2f} px/pt")

    def _bring_to_front(self, index: int) -> None:
        if 0 <= index < len(self.state.items):
            it = self.state.items.pop(index)
            self.state.items.append(it)

    # ---------- 書き出し ----------
    def export_pdf(self) -> None:
        if not self.state.items:
            messagebox.showwarning("なし", "要素がありません。PDF を追加してください。")
            return

        margin = 20.0
        x0, y0, x1, y1 = self._world_bounds()
        width  = (x1 - x0) + margin * 2
        height = (y1 - y0) + margin * 2

        save = filedialog.asksaveasfilename(
            title="保存先", defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")], initialfile="collage.pdf"
        )
        if not save:
            return

        try:
            out = fitz.open()
            page = out.new_page(width=width, height=height)

            for it in self.state.items:
                # GUI（上原点）の左上位置
                x_gui = (it.x - x0) + margin
                y_gui_top = (it.y - y0) + margin

                # 配置サイズ（pt）
                dw = it.w * it.scale
                dh = it.h * it.scale

                # GUI（上原点）からPDF（下原点）への座標変換
                # GUIでの位置をそのままPDFの位置に対応させる
                y0_pdf = y_gui_top         # 矩形の上
                y1_pdf = y_gui_top + dh    # 矩形の下
                x0_pdf = x_gui             # 矩形の左
                x1_pdf = x_gui + dw        # 矩形の右

                dest = fitz.Rect(x0_pdf, y0_pdf, x1_pdf, y1_pdf)

                with fitz.open(it.path) as src:
                    page.show_pdf_page(dest, src, it.page_index)

            out.save(save)
            out.close()
            messagebox.showinfo("完了", f"保存しました:\n{save}")

        except Exception as e:
            messagebox.showerror("保存エラー", f"保存に失敗しました:\n{e}")


# ------------------------- main -------------------------
def main() -> None:
    root = tk.Tk()
    try:
        style = ttk.Style()
        if sys.platform == "darwin":
            style.theme_use("aqua")
        else:
            style.theme_use("clam")
    except Exception:
        pass
    app = PdfCollageApp(root)
    app.mainloop()


if __name__ == "__main__":
    main()

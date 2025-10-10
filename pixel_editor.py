# Author: kamekingdom (2025-10-09)
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import tkinter as tk
from tkinter import ttk, filedialog, colorchooser, messagebox

try:
    from PIL import Image, ImageTk
except Exception as e:
    raise RuntimeError("Pillow が必要です。pip install pillow") from e

try:
    import numpy as np
except Exception as e:
    raise RuntimeError("NumPy が必要です。pip install numpy") from e


# =============== 型定義 ===============
Color = Tuple[int, int, int]  # (R,G,B)


@dataclass
class DocState:
    img: Optional[Image.Image] = None
    orig: Optional[Image.Image] = None
    undo_stack: List[Image.Image] = field(default_factory=list)
    redo_stack: List[Image.Image] = field(default_factory=list)

    def snapshot(self) -> Image.Image:
        assert self.img is not None
        return self.img.copy()


# =============== アプリ本体 ===============
class PixelGridEditor(ttk.Frame):
    BG: str = "#202124"
    GRID: str = "#2f2f2f"

    def __init__(self, master: tk.Tk) -> None:
        super().__init__(master)
        master.title("Pixel Grid/Brush Color Editor")
        master.geometry("1250x800")
        self.pack(fill=tk.BOTH, expand=True)

        # ドキュメント
        self.doc: DocState = DocState()

        # ビュー（キャンバス→画像座標の一次変換）
        self.view_scale: float = 1.0      # px per image-pixel
        self.offset_x: float = 20.0       # 画像左上のキャンバス座標
        self.offset_y: float = 20.0

        # 選択・編集状態
        self.mode: tk.StringVar = tk.StringVar(value="cell")  # "cell" or "brush"
        self.cell_size: tk.IntVar = tk.IntVar(value=8)        # セル選択のセル幅(px, 画像ピクセル単位)
        self.brush_radius: tk.IntVar = tk.IntVar(value=1)     # ブラシ半径(px, 画像ピクセル単位)
        self.target_color: Color = (255, 0, 0)

        self.dragging: bool = False
        self.anchor_px: Optional[Tuple[int, int]] = None      # セルモードのアンカー(画像ピクセル)
        self.preview_rect: Optional[Tuple[int, int, int, int]] = None   # セルモードのプレビュー矩形 (x0,y0,x1_ex,y1_ex)
        self.last_applied_rect: Optional[Tuple[int, int, int, int]] = None
        self.preview_brush_center: Optional[Tuple[int, int]] = None     # ブラシ点線円
        self.last_brush_circle: Optional[Tuple[int, int, int]] = None   # (cx,cy,r) 実線円

        self._build_ui()
        self._bind()
        self._redraw()

    # ---------- UI 構築 ----------
    def _build_ui(self) -> None:
        pane = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True)

        # 左: キャンバス
        left = ttk.Frame(pane)
        self.canvas = tk.Canvas(left, bg=self.BG, highlightthickness=0, cursor="arrow")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        pane.add(left, weight=4)

        # 右: サイドバー
        right = ttk.Frame(pane, width=330)
        pane.add(right, weight=0)

        # ファイル
        lf_file = ttk.Labelframe(right, text="ファイル")
        lf_file.pack(fill=tk.X, padx=8, pady=(8, 4))
        ttk.Button(lf_file, text="画像を開く", command=self.open_image).pack(side=tk.LEFT, padx=6, pady=8)
        ttk.Button(lf_file, text="保存", command=self.save_image).pack(side=tk.LEFT, padx=6, pady=8)

        # 表示
        lf_view = ttk.Labelframe(right, text="表示")
        lf_view.pack(fill=tk.X, padx=8, pady=4)
        ttk.Button(lf_view, text="ズーム +", command=lambda: self._zoom_at(None, 1.25)).pack(side=tk.LEFT, padx=4, pady=6)
        ttk.Button(lf_view, text="ズーム -", command=lambda: self._zoom_at(None, 0.8)).pack(side=tk.LEFT, padx=4, pady=6)
        ttk.Button(lf_view, text="全体表示", command=self.fit_view).pack(side=tk.LEFT, padx=4, pady=6)

        # モード
        lf_mode = ttk.Labelframe(right, text="モード")
        lf_mode.pack(fill=tk.X, padx=8, pady=4)
        ttk.Radiobutton(lf_mode, text="セル選択（クリック=1セル / ドラッグ=矩形）", value="cell", variable=self.mode).pack(anchor=tk.W, padx=8, pady=2)
        ttk.Radiobutton(lf_mode, text="ブラシ（ピクセル）", value="brush", variable=self.mode).pack(anchor=tk.W, padx=8, pady=(2, 6))

        # セルサイズ
        lf_cell = ttk.Labelframe(right, text="セルサイズ（px）")
        lf_cell.pack(fill=tk.X, padx=8, pady=4)
        sp = ttk.Spinbox(lf_cell, from_=1, to=256, increment=1, textvariable=self.cell_size, width=6, command=self._redraw)
        sp.pack(side=tk.LEFT, padx=8, pady=8)
        sc = ttk.Scale(lf_cell, from_=1, to=256, variable=self.cell_size, command=lambda _ : self._redraw())
        sc.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 8))

        # ブラシ半径
        lf_brush = ttk.Labelframe(right, text="ブラシ半径（px）")
        lf_brush.pack(fill=tk.X, padx=8, pady=4)
        sb = ttk.Spinbox(lf_brush, from_=1, to=128, increment=1, textvariable=self.brush_radius, width=6, command=self._redraw)
        sb.pack(side=tk.LEFT, padx=8, pady=8)
        ss = ttk.Scale(lf_brush, from_=1, to=128, variable=self.brush_radius, command=lambda _ : self._redraw())
        ss.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 8))

        # パレット（常時表示）
        lf_color = ttk.Labelframe(right, text="パレット")
        lf_color.pack(fill=tk.X, padx=8, pady=4)
        self.swatch = tk.Canvas(lf_color, width=48, height=32, bg=self._rgb_hex(self.target_color),
                                highlightthickness=1, highlightbackground="#555")
        self.swatch.grid(row=0, column=0, padx=(8, 6), pady=8)
        ttk.Button(lf_color, text="色を選ぶ", command=self.pick_target_color).grid(row=0, column=1, sticky=tk.W, padx=4, pady=8)
        lf_color.columnconfigure(1, weight=1)

        # 履歴
        lf_hist = ttk.Labelframe(right, text="履歴")
        lf_hist.pack(fill=tk.X, padx=8, pady=(4, 8))
        ttk.Button(lf_hist, text="元に戻す (Ctrl+Z)", command=self.undo).pack(side=tk.LEFT, padx=8, pady=8)
        ttk.Button(lf_hist, text="やり直す (Ctrl+Y)", command=self.redo).pack(side=tk.LEFT, padx=8, pady=8)

        # ステータス
        self.status = ttk.Label(right, text="画像を開いてください", anchor=tk.W)
        self.status.pack(fill=tk.X, padx=8, pady=(0, 8))

    # ---------- イベント束縛 ----------
    def _bind(self) -> None:
        self.canvas.bind("<Configure>", lambda e: self._redraw())
        self.canvas.bind("<Motion>", self.on_motion)

        # 左クリック（選択・確定）
        self.canvas.bind("<Button-1>", self.on_left_down)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_up)

        # パン（中ボタン or Spaceドラッグ）
        self.canvas.bind("<Button-2>", self.on_mid_down)
        self.canvas.bind("<B2-Motion>", self.on_mid_drag)
        self.canvas.bind("<ButtonRelease-2>", self.on_mid_up)
        self.master.bind("<space>", lambda e: self.canvas.config(cursor="fleur"))
        self.master.bind("<KeyRelease-space>", lambda e: self.canvas.config(cursor="arrow"))

        # ズーム
        self.canvas.bind("<MouseWheel>", self.on_wheel)   # Win
        self.canvas.bind("<Button-4>", self.on_wheel)     # Linux up
        self.canvas.bind("<Button-5>", self.on_wheel)     # Linux down

        # Undo / Redo
        self.master.bind("<Control-z>", lambda e: self.undo())
        self.master.bind("<Control-Z>", lambda e: self.undo())
        self.master.bind("<Control-y>", lambda e: self.redo())
        self.master.bind("<Control-Y>", lambda e: self.redo())
        self.master.bind("<Control-Shift-Z>", lambda e: self.redo())
        # mac
        self.master.bind("<Command-z>", lambda e: self.undo())
        self.master.bind("<Command-Shift-Z>", lambda e: self.redo())

        # ブラシ半径ショートカット
        self.master.bind("[", lambda e: self._nudge_brush(-1))
        self.master.bind("]", lambda e: self._nudge_brush(+1))
        self.master.bind("<Shift-bracketleft>", lambda e: self._nudge_brush(-5))
        self.master.bind("<Shift-bracketright>", lambda e: self._nudge_brush(+5))

    # ---------- ファイル ----------
    def open_image(self) -> None:
        path = filedialog.askopenfilename(title="画像を選択", filetypes=[
            ("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"), ("All files", "*.*")
        ])
        if not path:
            return
        img = Image.open(path).convert("RGB")
        self.doc = DocState(img=img.copy(), orig=img.copy(), undo_stack=[], redo_stack=[])
        # 初期ズームと配置
        W = self.canvas.winfo_width() or 1000
        H = self.canvas.winfo_height() or 700
        sx = (W - 60) / img.width
        sy = (H - 60) / img.height
        self.view_scale = float(max(1.0, min(32.0, min(sx, sy))))
        self.offset_x = (W - img.width * self.view_scale) / 2
        self.offset_y = (H - img.height * self.view_scale) / 2
        # クリア
        self.preview_rect = None
        self.last_applied_rect = None
        self.preview_brush_center = None
        self.last_brush_circle = None
        self._redraw()
        self._set_status(f"{os.path.basename(path)} | {img.width}×{img.height}px")

    def save_image(self) -> None:
        if self.doc.img is None:
            return
        path = filedialog.asksaveasfilename(
            title="保存先", defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg"), ("BMP", "*.bmp"), ("TIFF", "*.tif;*.tiff")]
        )
        if not path:
            return
        try:
            self.doc.img.save(path)
            self._set_status(f"保存しました: {path}")
        except Exception as e:
            messagebox.showerror("保存エラー", str(e))

    # ---------- Undo/Redo ----------
    def _push_undo(self) -> None:
        if self.doc.img is None:
            return
        self.doc.undo_stack.append(self.doc.snapshot())
        if len(self.doc.undo_stack) > 100:
            self.doc.undo_stack.pop(0)
        self.doc.redo_stack.clear()

    def undo(self) -> None:
        if not self.doc.undo_stack:
            return
        cur = self.doc.snapshot()
        img = self.doc.undo_stack.pop()
        self.doc.redo_stack.append(cur)
        self.doc.img = img
        self._set_status("Undo")
        self._redraw()

    def redo(self) -> None:
        if not self.doc.redo_stack:
            return
        cur = self.doc.snapshot()
        img = self.doc.redo_stack.pop()
        self.doc.undo_stack.append(cur)
        self.doc.img = img
        self._set_status("Redo")
        self._redraw()

    # ---------- 色 ----------
    def pick_target_color(self) -> None:
        rgb, _ = colorchooser.askcolor(color=self._rgb_hex(self.target_color), title="置換する色を選ぶ")
        if rgb is None:
            return
        self.target_color = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        self.swatch.configure(bg=self._rgb_hex(self.target_color))
        self._set_status(f"ターゲット色: {self.target_color}")

    # ---------- 入力（セル／ブラシ） ----------
    def on_motion(self, e: tk.Event) -> None:
        x, y = self._canvas_to_img(e.x, e.y)
        if self.doc.img is not None and 0 <= x < self.doc.img.width and 0 <= y < self.doc.img.height:
            rgb = self.doc.img.getpixel((x, y))
            self._set_status(f"({x},{y}) RGB{rgb} | mode={self.mode.get()} | cell={self.cell_size.get()} | r={self.brush_radius.get()} | zoom={self.view_scale:.2f}")

        if self.mode.get() == "brush":
            self.preview_brush_center = (max(0, min((self.doc.img.width - 1 if self.doc.img else 0), x)),
                                         max(0, min((self.doc.img.height - 1 if self.doc.img else 0), y)))
            self._redraw()  # 点線円のみ更新（軽量）

    def on_left_down(self, e: tk.Event) -> None:
        if self.doc.img is None:
            return
        self.dragging = True
        self._push_undo()

        ix, iy = self._canvas_to_img(e.x, e.y, clamp=True)
        if self.mode.get() == "cell":
            # クリック＝1セル、ドラッグ開始
            self.anchor_px = (ix, iy)
            self.preview_rect = self._rect_from_cells(self.anchor_px, (ix, iy))
            self._redraw()
        else:
            # ブラシ一発適用（実線円表示用に記録）
            self._apply_brush(ix, iy)
            self.last_brush_circle = (ix, iy, int(self.brush_radius.get()))
            self._redraw()

    def on_left_drag(self, e: tk.Event) -> None:
        if not self.dragging or self.doc.img is None:
            return
        ix, iy = self._canvas_to_img(e.x, e.y, clamp=True)

        if self.mode.get() == "cell":
            # 点線矩形プレビュー（未適用）
            self.preview_rect = self._rect_from_cells(self.anchor_px, (ix, iy))
            self._redraw()
        else:
            # ブラシは移動しながら逐次適用（即時置換）
            self._apply_brush(ix, iy)
            self.last_brush_circle = (ix, iy, int(self.brush_radius.get()))
            self._redraw()

    def on_left_up(self, e: tk.Event) -> None:
        if not self.dragging or self.doc.img is None:
            return
        self.dragging = False

        if self.mode.get() == "cell" and self.preview_rect is not None:
            # プレビューしていた矩形を確定し、実線表示＋画素を置換
            x0, y0, x1, y1 = self.preview_rect
            self._fill_rect(x0, y0, x1, y1, self.target_color)
            self.last_applied_rect = (x0, y0, x1, y1)
            self.preview_rect = None
            self._redraw()
        else:
            # ブラシは既に適用済み。last_brush_circle は _apply_brush で更新済み
            self._redraw()

    # ---------- パン／ズーム ----------
    def on_mid_down(self, e: tk.Event) -> None:
        self._pan_anchor: Optional[Tuple[int, int]] = (e.x, e.y)

    def on_mid_drag(self, e: tk.Event) -> None:
        if not hasattr(self, "_pan_anchor") or self._pan_anchor is None:
            return
        dx = e.x - self._pan_anchor[0]
        dy = e.y - self._pan_anchor[1]
        self.offset_x += dx
        self.offset_y += dy
        self._pan_anchor = (e.x, e.y)
        self._redraw()

    def on_mid_up(self, e: tk.Event) -> None:
        self._pan_anchor = None

    def on_wheel(self, e: tk.Event) -> None:
        if self.doc.img is None:
            return
        fac = 1.25 if (getattr(e, "delta", 0) > 0 or getattr(e, "num", 0) == 4) else 0.8
        self._zoom_at((e.x, e.y), fac)

    # ---------- 実処理（セル／ブラシ） ----------
    def _rect_from_cells(self, p0: Tuple[int, int], p1: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """セルグリッドへスナップした矩形（画像座標、右下は排他端）。"""
        cs = int(max(1, self.cell_size.get()))
        x0 = (min(p0[0], p1[0]) // cs) * cs
        y0 = (min(p0[1], p1[1]) // cs) * cs
        x1 = ((max(p0[0], p1[0]) // cs) + 1) * cs
        y1 = ((max(p0[1], p1[1]) // cs) + 1) * cs
        # 画像境界にクランプ
        assert self.doc.img is not None
        x0 = max(0, min(self.doc.img.width, x0))
        y0 = max(0, min(self.doc.img.height, y0))
        x1 = max(0, min(self.doc.img.width, x1))
        y1 = max(0, min(self.doc.img.height, y1))
        return (x0, y0, x1, y1)

    def _fill_rect(self, x0: int, y0: int, x1: int, y1: int, color: Color) -> None:
        """矩形領域を即時置換。y1/x1 は排他端。"""
        assert self.doc.img is not None
        arr = np.asarray(self.doc.img).copy()
        arr[y0:y1, x0:x1] = np.array(color, dtype=np.uint8)
        self.doc.img = Image.fromarray(arr, mode="RGB")

    def _apply_brush(self, cx: int, cy: int) -> None:
        """中心 (cx,cy)、半径 r の円形ブラシで即時置換。"""
        assert self.doc.img is not None
        r = int(max(1, self.brush_radius.get()))
        h, w = self.doc.img.height, self.doc.img.width
        x0 = max(0, cx - r); x1 = min(w - 1, cx + r)
        y0 = max(0, cy - r); y1 = min(h - 1, cy + r)
        if x0 > x1 or y0 > y1:
            return
        arr = np.asarray(self.doc.img).copy()
        yy, xx = np.ogrid[y0:y1 + 1, x0:x1 + 1]
        mask = (xx - cx) * (xx - cx) + (yy - cy) * (yy - cy) <= r * r
        arr[y0:y1 + 1, x0:x1 + 1][mask] = np.array(self.target_color, dtype=np.uint8)
        self.doc.img = Image.fromarray(arr, mode="RGB")

    # ---------- ビュー座標 ----------
    def _canvas_to_img(self, cx: int, cy: int, clamp: bool = False) -> Tuple[int, int]:
        x = int((cx - self.offset_x) / self.view_scale)
        y = int((cy - self.offset_y) / self.view_scale)
        if clamp and self.doc.img is not None:
            x = max(0, min(self.doc.img.width - 1, x))
            y = max(0, min(self.doc.img.height - 1, y))
        return x, y

    def _zoom_at(self, canvas_xy: Optional[Tuple[int, int]], fac: float) -> None:
        if self.doc.img is None:
            return
        if canvas_xy is None:
            cx = self.canvas.winfo_width() // 2
            cy = self.canvas.winfo_height() // 2
        else:
            cx, cy = canvas_xy
        ix, iy = self._canvas_to_img(cx, cy, clamp=True)
        new_scale = float(max(0.5, min(64.0, self.view_scale * fac)))
        if abs(new_scale - self.view_scale) < 1e-6:
            return
        self.view_scale = new_scale
        self.offset_x = cx - ix * self.view_scale
        self.offset_y = cy - iy * self.view_scale
        self._redraw()

    def fit_view(self) -> None:
        if self.doc.img is None:
            return
        W = self.canvas.winfo_width(); H = self.canvas.winfo_height()
        sx = (W - 60) / self.doc.img.width
        sy = (H - 60) / self.doc.img.height
        self.view_scale = float(max(0.5, min(64.0, min(sx, sy))))
        self.offset_x = (W - self.doc.img.width * self.view_scale) / 2
        self.offset_y = (H - self.doc.img.height * self.view_scale) / 2
        self._redraw()

    # ---------- 描画 ----------
    def _redraw(self) -> None:
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()

        # 背景グリッド（キャンバス座標）
        step = 20
        for x in range(0, w, step):
            self.canvas.create_line(x, 0, x, h, fill=self.GRID)
        for y in range(0, h, step):
            self.canvas.create_line(0, y, w, y, fill=self.GRID)

        if self.doc.img is None:
            return

        # 画像描画（最近傍）
        disp = self.doc.img.resize(
            (int(self.doc.img.width * self.view_scale), int(self.doc.img.height * self.view_scale)),
            resample=Image.NEAREST
        )
        self._tk_img = ImageTk.PhotoImage(disp)
        self.canvas.create_image(self.offset_x, self.offset_y, image=self._tk_img, anchor=tk.NW)

        # セルモード：プレビュー（点線）
        if self.mode.get() == "cell" and self.preview_rect is not None:
            x0, y0, x1, y1 = self.preview_rect
            cx0, cy0 = self.offset_x + x0 * self.view_scale, self.offset_y + y0 * self.view_scale
            cx1, cy1 = self.offset_x + x1 * self.view_scale, self.offset_y + y1 * self.view_scale
            self.canvas.create_rectangle(cx0, cy0, cx1, cy1, outline="#00C2FF", width=2, dash=(6, 3))

        # セルモード：確定（実線）
        if self.last_applied_rect is not None:
            x0, y0, x1, y1 = self.last_applied_rect
            cx0, cy0 = self.offset_x + x0 * self.view_scale, self.offset_y + y0 * self.view_scale
            cx1, cy1 = self.offset_x + x1 * self.view_scale, self.offset_y + y1 * self.view_scale
            self.canvas.create_rectangle(cx0, cy0, cx1, cy1, outline="#00FF7F", width=2)

        # ブラシ：点線プレビュー（カーソル位置）
        if self.mode.get() == "brush" and self.preview_brush_center is not None:
            bx, by = self.preview_brush_center
            r = int(self.brush_radius.get())
            cx0 = self.offset_x + (bx - r) * self.view_scale
            cy0 = self.offset_y + (by - r) * self.view_scale
            cx1 = self.offset_x + (bx + r + 1) * self.view_scale
            cy1 = self.offset_y + (by + r + 1) * self.view_scale
            self.canvas.create_oval(cx0, cy0, cx1, cy1, outline="#00C2FF", width=2, dash=(6, 3))

        # ブラシ：最後に適用した円（実線）
        if self.last_brush_circle is not None:
            bx, by, r = self.last_brush_circle
            cx0 = self.offset_x + (bx - r) * self.view_scale
            cy0 = self.offset_y + (by - r) * self.view_scale
            cx1 = self.offset_x + (bx + r + 1) * self.view_scale
            cy1 = self.offset_y + (by + r + 1) * self.view_scale
            self.canvas.create_oval(cx0, cy0, cx1, cy1, outline="#00FF7F", width=2)

    # ---------- utils ----------
    def _rgb_hex(self, c: Color) -> str:
        return "#%02x%02x%02x" % c

    def _set_status(self, s: str) -> None:
        self.status.config(text=s)

    def _nudge_brush(self, d: int) -> None:
        v = max(1, min(128, int(self.brush_radius.get()) + d))
        self.brush_radius.set(v)
        self._set_status(f"ブラシ半径: {v}px")
        self._redraw()


# =============== エントリポイント ===============
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
    app = PixelGridEditor(root)
    root.mainloop()


if __name__ == "__main__":
    main()

"""Tkinter overlay for live metrics cards and a heading minimap."""

from __future__ import annotations

import math
import tkinter as tk

from quadruped_rl_genesis.interface.telemetry import (
    CardData,
    MinimapState,
    SectionData,
)

OVERLAY_PAD = 20
OVERLAY_GAP = 20
OVERLAY_TITLE_BAR_OFFSET = 27
OVERLAY_MIN_WIDTH = 420
OVERLAY_MIN_HEIGHT = 180
OVERLAY_DEFAULT_WIDTH = 680
OVERLAY_DEFAULT_HEIGHT = 420
OVERLAY_COLUMN_GAP = 16

MAP_CANVAS_SIZE = 280
MAP_PAD = 12

COLOR_ROOT = "#1a1a1a"
COLOR_PANEL = "#252525"
COLOR_DIVIDER = "#333333"
COLOR_LABEL = "#909090"
COLOR_VALUE = "#e0e0e0"
COLOR_TITLE = "#ffffff"

FONT_TITLE = ("Consolas", 9, "bold")
FONT_ROW = ("Consolas", 9)
FONT_MAP_LEGEND = ("Consolas", 8)
PAD_SECTION = 8
PAD_ROW = 2
DIVIDER_HEIGHT = 1
PANEL_BORDER = 1
PANEL_RELIEF = "flat"

COLOR_MAP_BG = "#1e1e1e"
COLOR_MAP_BORDER = "#555555"
COLOR_MAP_MARGIN = "#3a3a3a"
COLOR_MAP_GRID = "#2a2a2a"
COLOR_HEADING = "#ffeb3b"
COLOR_GOAL = "#e53935"


class MetricsOverlayWindow:
    """Floating Tk window that shows metrics columns and a heading minimap.

    The window stays top-most, supports user goal picking on the minimap canvas,
    and can be positioned next to the Genesis viewer via :meth:`reposition_next_to`.
    """

    def __init__(self) -> None:
        """Create the root window, layout, metric columns, and minimap canvas.

        The window is created top-most with a default screen position offset from
        the right edge.
        """
        self._root = tk.Tk()
        self._root.title("Live metrics")
        self._root.configure(bg=COLOR_ROOT)
        self._root.attributes("-topmost", True)
        self._root.resizable(True, True)
        self._root.minsize(OVERLAY_MIN_WIDTH, OVERLAY_MIN_HEIGHT + MAP_CANVAS_SIZE + 72)

        self._pending_goal_xy: tuple[float, float] | None = None
        self._last_minimap_state: MinimapState | None = None

        self._container = tk.Frame(
            self._root,
            bg=COLOR_ROOT,
            padx=PAD_SECTION,
            pady=PAD_SECTION,
        )

        self._map_outer = tk.Frame(
            self._root,
            bg=COLOR_DIVIDER,
            padx=PANEL_BORDER,
            pady=PANEL_BORDER,
        )
        self._map_outer.pack(side=tk.BOTTOM, fill=tk.X)
        self._container.pack(fill=tk.BOTH, expand=True)

        self._map_inner = tk.Frame(self._map_outer, bg=COLOR_PANEL)
        self._map_inner.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            self._map_inner,
            text="Top view",
            font=FONT_TITLE,
            fg=COLOR_TITLE,
            bg=COLOR_PANEL,
            anchor="w",
        ).pack(fill=tk.X, padx=PAD_SECTION, pady=(PAD_ROW, 0))

        self._map_canvas = tk.Canvas(
            self._map_inner,
            width=MAP_CANVAS_SIZE,
            height=MAP_CANVAS_SIZE,
            bg=COLOR_MAP_BG,
            highlightthickness=0,
        )
        self._map_canvas.pack(
            padx=PAD_SECTION,
            pady=(PAD_ROW, PAD_ROW),
        )

        legend = tk.Frame(self._map_inner, bg=COLOR_PANEL)
        legend.pack(fill=tk.X, padx=PAD_SECTION, pady=(0, PAD_SECTION))
        tk.Label(
            legend,
            text=(
                "Solid: terrain · Dashed: goal margin · Yellow arrow: robot forward · Red: goal"
            ),
            font=FONT_MAP_LEGEND,
            fg=COLOR_LABEL,
            bg=COLOR_PANEL,
            anchor="w",
            justify=tk.LEFT,
        ).pack(fill=tk.X)

        self._map_canvas.bind("<Button-1>", self._on_map_click)
        self._map_canvas.bind("<B1-Motion>", self._on_map_drag)
        self._map_canvas.bind("<Configure>", self._on_map_configure)

        self._last_shape: tuple[
            tuple[tuple[str, int], ...], tuple[tuple[str, int], ...]
        ] = (
            (),
            (),
        )
        self._robot_widgets: list[tuple[tk.Label, list[tk.Label]]] = []
        self._rl_widgets: list[tuple[tk.Label, list[tk.Label]]] = []
        self._root.update_idletasks()
        sw = self._root.winfo_screenwidth()
        fallback_x = max(0, sw - OVERLAY_DEFAULT_WIDTH - OVERLAY_PAD)
        fallback_y = OVERLAY_PAD
        self._root.geometry(f"+{fallback_x}+{fallback_y}")
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self) -> None:
        self._root.withdraw()

    def _on_map_configure(self, _event: tk.Event) -> None:
        if self._last_minimap_state is not None:
            self._draw_minimap(self._last_minimap_state)

    def _on_map_click(self, event: tk.Event) -> None:
        self._pick_goal_from_canvas(event.x, event.y)

    def _on_map_drag(self, event: tk.Event) -> None:
        self._pick_goal_from_canvas(event.x, event.y)

    def _pick_goal_from_canvas(self, cx: int, cy: int) -> None:
        state = self._last_minimap_state
        if state is None:
            return
        w = self._map_canvas.winfo_width()
        h = self._map_canvas.winfo_height()
        if w <= 1 or h <= 1:
            return
        xy = self._canvas_to_world(float(cx), float(cy), state, float(w), float(h))
        if xy is not None:
            self._pending_goal_xy = xy

    def consume_pending_goal_xy(self) -> tuple[float, float] | None:
        """Pop and return a user-picked goal from the minimap.

        Returns:
            tuple[float, float] | None: World-frame ``(x, y)`` meters if the user
                clicked or dragged since the last call, otherwise ``None``.
        """
        if self._pending_goal_xy is None:
            return None
        out = self._pending_goal_xy
        self._pending_goal_xy = None
        return out

    def _world_bounds(self, state: MinimapState) -> tuple[float, float, float, float]:
        width_m = state["width_m"]
        length_m = state["length_m"]
        return (
            -0.5 * width_m,
            0.5 * width_m,
            -0.5 * length_m,
            0.5 * length_m,
        )

    def _margin_bounds(self, state: MinimapState) -> tuple[float, float, float, float]:
        min_x, max_x, min_y, max_y = self._world_bounds(state)
        m = float(state["margin_m"])
        return (min_x + m, max_x - m, min_y + m, max_y - m)

    def _square_viewport(
        self, canvas_w: float, canvas_h: float
    ) -> tuple[float, float, float]:
        aw = max(canvas_w - 2 * MAP_PAD, 0.0)
        ah = max(canvas_h - 2 * MAP_PAD, 0.0)
        side = min(aw, ah)
        left = MAP_PAD + (aw - side) / 2
        top = MAP_PAD + (ah - side) / 2
        return (left, top, side)

    def _uniform_world_scale(
        self, state: MinimapState, side: float
    ) -> tuple[float, float, float]:
        min_x, max_x, min_y, max_y = self._world_bounds(state)
        span_x = max_x - min_x
        span_y = max_y - min_y
        if span_x <= 0 or span_y <= 0:
            return (1.0, 0.0, 0.0)
        scale = min(side / span_x, side / span_y)
        return (scale, span_x * scale, span_y * scale)

    def _canvas_to_world(
        self,
        cx: float,
        cy: float,
        state: MinimapState,
        canvas_w: float,
        canvas_h: float,
    ) -> tuple[float, float] | None:
        min_x, _max_x, _min_y, max_y = self._world_bounds(state)
        left, top, side = self._square_viewport(canvas_w, canvas_h)
        if side <= 0:
            return None
        scale, draw_w, draw_h = self._uniform_world_scale(state, side)
        ox = left + (side - draw_w) / 2
        oy = top + (side - draw_h) / 2
        if cx < ox or cx > ox + draw_w or cy < oy or cy > oy + draw_h:
            return None
        wx = min_x + (cx - ox) / scale
        wy = max_y - (cy - oy) / scale
        mmin_x, mmax_x, mmin_y, mmax_y = self._margin_bounds(state)
        wx = min(max(wx, mmin_x), mmax_x)
        wy = min(max(wy, mmin_y), mmax_y)
        return (wx, wy)

    def _world_to_canvas(
        self,
        wx: float,
        wy: float,
        state: MinimapState,
        canvas_w: float,
        canvas_h: float,
    ) -> tuple[float, float]:
        min_x, _max_x, _min_y, max_y = self._world_bounds(state)
        left, top, side = self._square_viewport(canvas_w, canvas_h)
        scale, draw_w, draw_h = self._uniform_world_scale(state, side)
        ox = left + (side - draw_w) / 2
        oy = top + (side - draw_h) / 2
        cx = ox + (wx - min_x) * scale
        cy = oy + (max_y - wy) * scale
        return (cx, cy)

    def _draw_minimap(self, state: MinimapState) -> None:
        self._last_minimap_state = state
        c = self._map_canvas
        c.delete("all")
        w = float(c.winfo_width())
        h = float(c.winfo_height())
        if w <= 1 or h <= 1:
            w, h = float(MAP_CANVAS_SIZE), float(MAP_CANVAS_SIZE)

        min_x, max_x, min_y, max_y = self._world_bounds(state)
        mmin_x, mmax_x, mmin_y, mmax_y = self._margin_bounds(state)

        x0, y0c = self._world_to_canvas(min_x, max_y, state, w, h)
        x1, y1c = self._world_to_canvas(max_x, min_y, state, w, h)
        c.create_rectangle(x0, y0c, x1, y1c, outline=COLOR_MAP_BORDER, width=2)

        mx0, my0c = self._world_to_canvas(mmin_x, mmax_y, state, w, h)
        mx1, my1c = self._world_to_canvas(mmax_x, mmin_y, state, w, h)
        c.create_rectangle(
            mx0,
            my0c,
            mx1,
            my1c,
            outline=COLOR_MAP_MARGIN,
            dash=(4, 4),
            width=1,
        )

        if abs(min_x) < 1e6 and abs(min_y) < 1e6:
            zx, zy0 = self._world_to_canvas(0.0, min_y, state, w, h)
            zx, zy1 = self._world_to_canvas(0.0, max_y, state, w, h)
            c.create_line(zx, zy0, zx, zy1, fill=COLOR_MAP_GRID, width=1)
            zx0, zy = self._world_to_canvas(min_x, 0.0, state, w, h)
            zx1, zy = self._world_to_canvas(max_x, 0.0, state, w, h)
            c.create_line(zx0, zy, zx1, zy, fill=COLOR_MAP_GRID, width=1)

        rx, ry = state["robot_x"], state["robot_y"]
        rcx, rcy = self._world_to_canvas(rx, ry, state, w, h)

        yaw = float(state["robot_yaw_rad"])
        heading_m = 0.9
        hx = rx + heading_m * math.cos(yaw)
        hy = ry + heading_m * math.sin(yaw)
        hcx, hcy = self._world_to_canvas(hx, hy, state, w, h)
        c.create_line(rcx, rcy, hcx, hcy, fill=COLOR_HEADING, width=2, arrow=tk.LAST)

        gx, gy = state["goal_x"], state["goal_y"]
        gcx, gcy = self._world_to_canvas(gx, gy, state, w, h)
        r_goal = 4.0
        c.create_oval(
            gcx - r_goal,
            gcy - r_goal,
            gcx + r_goal,
            gcy + r_goal,
            fill=COLOR_GOAL,
            outline=COLOR_GOAL,
            width=1,
        )

        ltx, lty = self._world_to_canvas(min_x, max_y, state, w, h)
        rtx, rty = self._world_to_canvas(max_x, max_y, state, w, h)
        _, rby = self._world_to_canvas(max_x, min_y, state, w, h)
        top_mid_x = (ltx + rtx) * 0.5
        right_mid_y = (rty + rby) * 0.5
        c.create_text(
            top_mid_x,
            lty + 10,
            text="+Y",
            font=FONT_MAP_LEGEND,
            fill=COLOR_LABEL,
            anchor="n",
        )
        c.create_text(
            rtx - 8,
            right_mid_y,
            text="+X",
            font=FONT_MAP_LEGEND,
            fill=COLOR_LABEL,
            anchor="e",
        )

    def _section_frame(
        self, parent: tk.Widget, title: str
    ) -> tuple[tk.Frame, tk.Frame, tk.Label]:
        outer = tk.Frame(
            parent,
            bg=COLOR_DIVIDER,
            padx=PANEL_BORDER,
            pady=PANEL_BORDER,
        )
        inner = tk.Frame(outer, bg=COLOR_PANEL, padx=PAD_SECTION, pady=PAD_ROW)
        inner.pack(fill=tk.BOTH, expand=True)
        title_lbl = tk.Label(
            inner,
            text=title,
            font=FONT_TITLE,
            fg=COLOR_TITLE,
            bg=COLOR_PANEL,
            anchor="w",
        )
        title_lbl.pack(fill=tk.X, pady=(PAD_ROW, PAD_ROW))
        sep = tk.Frame(inner, height=DIVIDER_HEIGHT, bg=COLOR_DIVIDER)
        sep.pack(fill=tk.X, pady=(0, PAD_ROW))
        return outer, inner, title_lbl

    def _row(self, parent: tk.Frame, label: str, value: str) -> tk.Label:
        f = tk.Frame(parent, bg=COLOR_PANEL)
        f.pack(fill=tk.X, pady=PAD_ROW)
        tk.Label(
            f,
            text=label,
            font=FONT_ROW,
            fg=COLOR_LABEL,
            bg=COLOR_PANEL,
            anchor="w",
        ).pack(side=tk.LEFT)
        val_lbl = tk.Label(
            f,
            text=value,
            font=FONT_ROW,
            fg=COLOR_VALUE,
            bg=COLOR_PANEL,
            anchor="e",
        )
        val_lbl.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        return val_lbl

    def reposition_next_to(
        self, genesis_x: int, genesis_y: int, genesis_w: int, genesis_h: int
    ) -> None:
        """Place the overlay beside the Genesis viewer bounds, clamped to the screen.

        Args:
            genesis_x (int): Viewer window left edge in screen pixels.
            genesis_y (int): Viewer window top edge in screen pixels.
            genesis_w (int): Viewer width in pixels.
            genesis_h (int): Viewer height in pixels.
        """
        self._root.update_idletasks()
        our_w = self._root.winfo_width()
        our_h = self._root.winfo_height()
        if our_w <= 1:
            our_w = OVERLAY_DEFAULT_WIDTH
        if our_h <= 1:
            our_h = OVERLAY_DEFAULT_HEIGHT
        x = genesis_x + genesis_w + OVERLAY_GAP
        y = genesis_y - OVERLAY_TITLE_BAR_OFFSET
        sw = self._root.winfo_screenwidth()
        sh = self._root.winfo_screenheight()
        if x + our_w > sw:
            x = genesis_x - our_w - OVERLAY_GAP
        if x < 0:
            x = 0
        if y + our_h > sh:
            y = max(0, sh - our_h)
        if y < 0:
            y = 0
        self._root.geometry(f"+{x}+{y}")

    def _build_column(
        self, parent: tk.Frame, sections: SectionData
    ) -> list[tuple[tk.Label, list[tk.Label]]]:
        widgets: list[tuple[tk.Label, list[tk.Label]]] = []
        for section_title, rows in sections:
            outer, inner, sec_title_lbl = self._section_frame(parent, section_title)
            outer.pack(fill=tk.X, pady=(0, PAD_SECTION))
            value_lbls = [self._row(inner, lbl, val) for lbl, val in rows]
            widgets.append((sec_title_lbl, value_lbls))
        return widgets

    def _rebuild(self, content: CardData) -> None:
        for w in self._container.winfo_children():
            w.destroy()
        self._robot_widgets = []
        self._rl_widgets = []
        robot_sections, rl_sections = content

        title_lbl = tk.Label(
            self._container,
            text="Live metrics",
            font=FONT_TITLE,
            fg=COLOR_TITLE,
            bg=COLOR_ROOT,
        )
        title_lbl.pack(pady=(0, PAD_SECTION))
        sep = tk.Frame(self._container, height=DIVIDER_HEIGHT, bg=COLOR_DIVIDER)
        sep.pack(fill=tk.X, pady=(0, PAD_SECTION))

        columns_frame = tk.Frame(self._container, bg=COLOR_ROOT)
        columns_frame.pack(fill=tk.BOTH, expand=True)

        left_col = tk.Frame(columns_frame, bg=COLOR_ROOT)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        robot_header = tk.Label(
            left_col,
            text="Robot",
            font=FONT_TITLE,
            fg=COLOR_TITLE,
            bg=COLOR_ROOT,
        )
        robot_header.pack(pady=(0, PAD_ROW))
        self._robot_widgets = self._build_column(left_col, robot_sections)

        right_col = tk.Frame(columns_frame, bg=COLOR_ROOT)
        right_col.pack(
            side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(OVERLAY_COLUMN_GAP, 0)
        )
        rl_header = tk.Label(
            right_col,
            text="Reinforcement Learning",
            font=FONT_TITLE,
            fg=COLOR_TITLE,
            bg=COLOR_ROOT,
        )
        rl_header.pack(pady=(0, PAD_ROW))
        self._rl_widgets = self._build_column(right_col, rl_sections)

        self._last_shape = (
            tuple((title, len(rows)) for title, rows in robot_sections),
            tuple((title, len(rows)) for title, rows in rl_sections),
        )

    def update(
        self,
        content: str | CardData,
        minimap_state: MinimapState | None = None,
    ) -> None:
        """Refresh labels from plain text or structured card data and redraw the minimap.

        Args:
            content (str | CardData): Either a fallback message string or structured
                ``(robot_sections, rl_sections)`` card data from telemetry.
            minimap_state (MinimapState | None): When set, redraws the minimap; when
                ``None`` with string ``content``, clears the map canvas.
        """
        if isinstance(content, str):
            for w in self._container.winfo_children():
                w.destroy()
            self._last_shape = ((), ())
            self._robot_widgets = []
            self._rl_widgets = []
            tk.Label(
                self._container,
                text=content,
                font=FONT_ROW,
                fg=COLOR_VALUE,
                bg=COLOR_ROOT,
                justify=tk.LEFT,
                anchor="nw",
            ).pack(fill=tk.BOTH, expand=True)
            self._root.update_idletasks()
            if minimap_state is not None:
                self._draw_minimap(minimap_state)
            else:
                self._last_minimap_state = None
                self._map_canvas.delete("all")
            return
        robot_sections, rl_sections = content
        shape = (
            tuple((title, len(rows)) for title, rows in robot_sections),
            tuple((title, len(rows)) for title, rows in rl_sections),
        )
        if shape == self._last_shape and self._robot_widgets and self._rl_widgets:
            for (section_title, rows), (title_lbl, value_lbls) in zip(
                robot_sections, self._robot_widgets
            ):
                title_lbl.config(text=section_title)
                for (_, value_text), val_lbl in zip(rows, value_lbls):
                    val_lbl.config(text=value_text)
            for (section_title, rows), (title_lbl, value_lbls) in zip(
                rl_sections, self._rl_widgets
            ):
                title_lbl.config(text=section_title)
                for (_, value_text), val_lbl in zip(rows, value_lbls):
                    val_lbl.config(text=value_text)
        else:
            self._rebuild(content)
        self._root.update_idletasks()
        if minimap_state is not None:
            self._draw_minimap(minimap_state)

    def pump_events(self) -> None:
        """Process pending Tk events so the window stays responsive.

        Call periodically from the main training or visualization loop.
        """
        self._root.update()

    def destroy(self) -> None:
        """Destroy the Tk root window and release resources.

        Ignores ``TclError`` if the window was already destroyed.
        """
        try:
            self._root.destroy()
        except tk.TclError:
            pass

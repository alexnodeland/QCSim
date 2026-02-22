import marimo

__generated_with = "0.19.8"
app = marimo.App(width="full", app_title="QCSim: Explorable Quantum Computing")


# ══════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════
@app.cell
def _():
    import marimo as mo
    import altair as alt
    import polars as pl
    import numpy as np
    import sys
    sys.path.insert(0, ".")
    import qcsim
    from qcsim import Gate, QuantumRegister, QuantumCircuit, Result

    try:
        import anywidget
        import traitlets
        has_3d = True
    except ImportError:
        anywidget = None
        traitlets = None
        has_3d = False

    return Gate, QuantumCircuit, QuantumRegister, Result, alt, anywidget, has_3d, mo, np, pl, qcsim, traitlets


# ══════════════════════════════════════════════════════════════
# BLOCH SPHERE WIDGET (anywidget + three.js)
# ══════════════════════════════════════════════════════════════
@app.cell
def _(anywidget, has_3d, traitlets):
    if has_3d:
        class BlochSphere(anywidget.AnyWidget):
            _esm = r"""
            import * as THREE from "https://esm.sh/three@0.162.0";
            import {OrbitControls} from "https://esm.sh/three@0.162.0/examples/jsm/controls/OrbitControls.js";

            function render({model, el}) {
                const W = 400, H = 400;
                el.style.width = W + "px";
                el.style.height = H + "px";

                const scene = new THREE.Scene();
                scene.background = new THREE.Color(0xfafafa);

                const camera = new THREE.PerspectiveCamera(35, W / H, 0.1, 100);
                camera.position.set(2.8, 2.0, 2.8);
                camera.lookAt(0, 0, 0);

                const renderer = new THREE.WebGLRenderer({antialias: true});
                renderer.setSize(W, H);
                renderer.setPixelRatio(window.devicePixelRatio);
                el.appendChild(renderer.domElement);

                const controls = new OrbitControls(camera, renderer.domElement);
                controls.enablePan = false;
                controls.enableZoom = true;

                // Wireframe sphere
                const sGeo = new THREE.SphereGeometry(1, 32, 24);
                const sMat = new THREE.MeshBasicMaterial({
                    color: 0xcccccc, wireframe: true, transparent: true, opacity: 0.15
                });
                scene.add(new THREE.Mesh(sGeo, sMat));

                // Great circles
                function addCircle(fn, color, op) {
                    const pts = [];
                    for (let i = 0; i <= 128; i++) {
                        const a = (i / 128) * Math.PI * 2;
                        pts.push(fn(a));
                    }
                    const g = new THREE.BufferGeometry().setFromPoints(pts);
                    scene.add(new THREE.Line(g, new THREE.LineBasicMaterial({
                        color, transparent: true, opacity: op
                    })));
                }
                addCircle(a => new THREE.Vector3(Math.cos(a), 0, Math.sin(a)), 0xdddddd, 0.4);
                addCircle(a => new THREE.Vector3(Math.cos(a), Math.sin(a), 0), 0xdddddd, 0.3);
                addCircle(a => new THREE.Vector3(0, Math.sin(a), Math.cos(a)), 0xdddddd, 0.3);

                // Axes
                const AL = 1.3;
                function addAxis(f, t, c) {
                    const g = new THREE.BufferGeometry().setFromPoints([
                        new THREE.Vector3(...f), new THREE.Vector3(...t)
                    ]);
                    scene.add(new THREE.Line(g, new THREE.LineBasicMaterial({color: c})));
                }
                addAxis([-AL,0,0],[AL,0,0], 0xaaaaaa);
                addAxis([0,-AL,0],[0,AL,0], 0xaaaaaa);
                addAxis([0,0,-AL],[0,0,AL], 0xaaaaaa);

                // Labels
                function makeLabel(text, pos, color) {
                    const c = document.createElement("canvas");
                    c.width = 128; c.height = 128;
                    const ctx = c.getContext("2d");
                    ctx.font = "bold 72px 'Helvetica Neue', Arial, sans-serif";
                    ctx.fillStyle = color;
                    ctx.textAlign = "center";
                    ctx.textBaseline = "middle";
                    ctx.fillText(text, 64, 64);
                    const t = new THREE.CanvasTexture(c);
                    const m = new THREE.SpriteMaterial({map: t, depthTest: false});
                    const s = new THREE.Sprite(m);
                    s.position.set(...pos);
                    s.scale.set(0.3, 0.3, 0.3);
                    scene.add(s);
                }
                makeLabel("X", [1.5, 0, 0], "#888");
                makeLabel("Y", [0, 0, 1.5], "#888");
                makeLabel("|0\u27E9", [0, 1.55, 0], "#4c78a8");
                makeLabel("|1\u27E9", [0, -1.55, 0], "#e45756");

                // State arrow
                let arrow = new THREE.ArrowHelper(
                    new THREE.Vector3(0, 1, 0),
                    new THREE.Vector3(0, 0, 0),
                    1.0, 0xe45756, 0.1, 0.06
                );
                scene.add(arrow);

                const dotG = new THREE.SphereGeometry(0.045, 16, 16);
                const dotM = new THREE.MeshBasicMaterial({color: 0xe45756});
                const dot = new THREE.Mesh(dotG, dotM);
                scene.add(dot);

                function update() {
                    const theta = model.get("theta");
                    const phi = model.get("phi");
                    const bx = Math.sin(theta) * Math.cos(phi);
                    const by = Math.cos(theta);
                    const bz = Math.sin(theta) * Math.sin(phi);
                    scene.remove(arrow);
                    const dir = new THREE.Vector3(bx, by, bz);
                    const len = dir.length();
                    if (len > 0.001) dir.normalize();
                    arrow = new THREE.ArrowHelper(dir, new THREE.Vector3(0,0,0), len > 0.001 ? 1.0 : 0.001, 0xe45756, 0.1, 0.06);
                    scene.add(arrow);
                    dot.position.set(bx, by, bz);
                }
                update();
                model.on("change:theta", update);
                model.on("change:phi", update);

                let animId;
                function animate() {
                    animId = requestAnimationFrame(animate);
                    controls.update();
                    renderer.render(scene, camera);
                }
                animate();
                return () => { cancelAnimationFrame(animId); renderer.dispose(); controls.dispose(); };
            }
            export default {render};
            """
            theta = traitlets.Float(0.6).tag(sync=True)
            phi = traitlets.Float(0.0).tag(sync=True)
    else:
        BlochSphere = None
    return (BlochSphere,)


# ══════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════
@app.cell
def _(mo):
    mo.sidebar([
        mo.md("## **QCSim**\n*Explorable Quantum Computing*"),
        mo.nav_menu({
            "#/": "Home",
            "#/qubit": "1 \u2014 Qubit Explorer",
            "#/gates": "2 \u2014 Gate Transforms",
            "#/rotations": "3 \u2014 Rotation Gates",
            "#/entanglement": "4 \u2014 Entanglement",
            "#/bell": "5 \u2014 Bell States",
            "#/grover": "6 \u2014 Grover Search",
            "#/sandbox": "7 \u2014 Circuit Sandbox",
            "#/references": "References",
        }, orientation="vertical"),
    ])
    return


# ══════════════════════════════════════════════════════════════
# HOME PAGE
# ══════════════════════════════════════════════════════════════
@app.cell
def _(mo):
    home_page = mo.vstack([
        mo.md(r"""
# Explorable Quantum Computing

<div style="font-size:1.15em; line-height:1.7; max-width:52em; color:#444;">

A quantum computer is not a faster classical computer.
It is a **different kind of machine** — one that computes with *amplitudes* instead of bits,
where information is carried by the **complex coefficients** of a state vector,
and where the act of *looking* at a result **changes** the result.

This notebook is an **explorable explanation** in the spirit of
[Bret Victor](http://worrydream.com/ExplorableExplanations/):
every concept here is a **live, manipulable thing**. Drag sliders,
change gates, build circuits — and watch the math respond in real time.

</div>
"""),
        mo.callout(
            mo.md(
                '*"The best way to understand something is to **change** it '
                'and see what happens."*\n\n'
                '— Bret Victor, *Inventing on Principle*'
            ),
            kind="info",
        ),
        mo.md(r"""
**How to use this notebook**: Use the sidebar to navigate between sections.
Every section contains interactive controls — change a parameter and all
downstream visualizations update instantly. Read the prose, play with the
controls, build intuition.
"""),
    ])
    return (home_page,)


# ══════════════════════════════════════════════════════════════
# SECTION 1: WHAT IS A QUBIT?
# ══════════════════════════════════════════════════════════════
@app.cell
def _(mo):
    theta_slider = mo.ui.slider(
        start=0, stop=314, step=1, value=60,
        label="\u03b8 (polar angle)", show_value=True, full_width=True,
    )
    phi_slider = mo.ui.slider(
        start=0, stop=628, step=1, value=0,
        label="\u03c6 (azimuthal angle)", show_value=True, full_width=True,
    )
    return phi_slider, theta_slider


@app.cell
def _(BlochSphere, alt, has_3d, mo, np, pl, phi_slider, theta_slider):
    _theta = theta_slider.value / 100.0
    _phi = phi_slider.value / 100.0

    _alpha = np.cos(_theta / 2)
    _beta = np.exp(1j * _phi) * np.sin(_theta / 2)
    _p0 = float(np.abs(_alpha) ** 2)
    _p1 = float(np.abs(_beta) ** 2)

    _bx = float(np.sin(_theta) * np.cos(_phi))
    _by = float(np.sin(_theta) * np.sin(_phi))
    _bz = float(np.cos(_theta))

    # ── 3D Bloch sphere ──
    if has_3d and BlochSphere is not None:
        _bloch_widget = mo.ui.anywidget(BlochSphere(theta=_theta, phi=_phi))
    else:
        _bloch_widget = None

    # ── 2D Bloch projections (fallback / alternate view) ──
    _t_circle = np.linspace(0, 2 * np.pi, 200)
    _circle_df = pl.DataFrame({"cx": np.cos(_t_circle).tolist(), "cy": np.sin(_t_circle).tolist()})

    _point_xz = pl.DataFrame({"x": [_bx], "z": [_bz], "label": ["|ψ⟩"]})
    _point_xy = pl.DataFrame({"x": [_bx], "y": [_by], "label": ["|ψ⟩"]})
    _arrow_xz = pl.DataFrame({"x": [0, _bx], "z": [0, _bz]})
    _arrow_xy = pl.DataFrame({"x": [0, _bx], "y": [0, _by]})

    def _bloch_view(circle_df, arrow_df, point_df, xcol, ycol, title):
        _base = (
            alt.Chart(circle_df).mark_line(color="#ddd", strokeWidth=1.5)
            .encode(x=alt.X("cx:Q", axis=None, scale=alt.Scale(domain=[-1.4, 1.4])),
                    y=alt.Y("cy:Q", axis=None, scale=alt.Scale(domain=[-1.4, 1.4])))
        )
        _ah = alt.Chart(pl.DataFrame({"x": [-1, 1], "y": [0, 0]})).mark_line(
            color="#ccc", strokeDash=[4, 4], strokeWidth=1).encode(x="x:Q", y="y:Q")
        _av = alt.Chart(pl.DataFrame({"x": [0, 0], "y": [-1, 1]})).mark_line(
            color="#ccc", strokeDash=[4, 4], strokeWidth=1).encode(x="x:Q", y="y:Q")
        _ar = (alt.Chart(arrow_df).mark_line(color="#e45756", strokeWidth=2.5)
               .encode(x=alt.X(f"{xcol}:Q"), y=alt.Y(f"{ycol}:Q")))
        _pt = (alt.Chart(point_df).mark_circle(size=100, color="#e45756")
               .encode(x=alt.X(f"{xcol}:Q"), y=alt.Y(f"{ycol}:Q")))
        _lb = (alt.Chart(point_df).mark_text(dx=14, dy=-10, fontSize=13, fontWeight="bold", color="#e45756")
               .encode(x=alt.X(f"{xcol}:Q"), y=alt.Y(f"{ycol}:Q"), text="label:N"))
        return ((_base + _ah + _av + _ar + _pt + _lb)
                .properties(width=220, height=220, title=title)
                .configure_view(strokeWidth=0))

    _chart_xz = _bloch_view(_circle_df, _arrow_xz, _point_xz, "x", "z", "Side view (X\u2013Z)")
    _chart_xy = _bloch_view(_circle_df, _arrow_xy, _point_xy, "x", "y", "Top view (X\u2013Y)")
    _flat_views = mo.hstack([_chart_xz, _chart_xy], justify="center", gap=0.5)

    # ── Bloch sphere display: tabs if 3D available, otherwise flat views ──
    if _bloch_widget is not None:
        _sphere_display = mo.ui.tabs({
            "3D Bloch Sphere": mo.center(_bloch_widget),
            "2D Projections": _flat_views,
        })
    else:
        _sphere_display = _flat_views

    # ── Probability bars with hover ──
    _hover_q = alt.selection_point(on="pointerover", nearest=True, empty=False)
    _prob_df = pl.DataFrame({"State": ["|0⟩", "|1⟩"], "Probability": [_p0, _p1]})
    _prob_chart = (
        alt.Chart(_prob_df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("State:N", axis=alt.Axis(labelFontSize=14, title=None)),
            y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1]),
                     axis=alt.Axis(format=".0%", title="Measurement Probability")),
            color=alt.Color("State:N", scale=alt.Scale(
                domain=["|0⟩", "|1⟩"], range=["#4c78a8", "#e45756"]), legend=None),
            opacity=alt.condition(_hover_q, alt.value(1), alt.value(0.6)),
            tooltip=[alt.Tooltip("State:N"), alt.Tooltip("Probability:Q", format=".4f")],
        )
        .add_params(_hover_q)
        .properties(width=160, height=260, title="Measurement Outcome")
    )

    # ── Stat cards ──
    _alpha_str = f"{_alpha:.4f}"
    _beta_r = f"{_beta.real:.4f}"
    _beta_i = f"{_beta.imag:+.4f}"
    _beta_str = f"{_beta_r}{_beta_i}i" if abs(_beta.imag) > 1e-6 else _beta_r

    _stats = mo.hstack([
        mo.stat(value=f"{_p0:.1%}", label="P(|0⟩)", bordered=True),
        mo.stat(value=f"{_p1:.1%}", label="P(|1⟩)", bordered=True),
        mo.stat(value=f"({_bx:.2f}, {_by:.2f}, {_bz:.2f})", label="Bloch (x,y,z)", bordered=True),
    ], justify="center", gap=1)

    qubit_page = mo.vstack([
        mo.md(r"""
## 1. What Is a Qubit?

A classical bit is either **0** or **1**. A qubit is a unit vector in a
two-dimensional complex vector space:

$$|\psi\rangle = \alpha\,|0\rangle + \beta\,|1\rangle
\qquad\text{where}\quad |\alpha|^2 + |\beta|^2 = 1$$

- $\alpha$ and $\beta$ are **complex amplitudes** (not probabilities!)
- $|\alpha|^2$ is the probability of measuring **0**
- $|\beta|^2$ is the probability of measuring **1**

We can parameterize any single-qubit state with two angles, $\theta$ and $\phi$:

$$|\psi\rangle = \cos\!\left(\tfrac{\theta}{2}\right)|0\rangle
+ e^{i\phi}\sin\!\left(\tfrac{\theta}{2}\right)|1\rangle$$

This maps every qubit state to a point on the **Bloch sphere**.
"""),
        mo.callout(mo.md(
            "**Key insight**: Before measurement, the qubit exists in a superposition "
            "of both states simultaneously. The amplitudes carry *phase* information "
            "that enables interference — the engine of quantum speedup."
        ), kind="info"),
        mo.md(f"""
### Qubit State Explorer

{theta_slider}
*\u03b8 in hundredths of radians — range [0, \u03c0]*

{phi_slider}
*\u03c6 in hundredths of radians — range [0, 2\u03c0]*
"""),
        mo.hstack([_sphere_display, mo.vstack([_prob_chart, _stats])], justify="center", gap=1.5),
        mo.md(f"**State vector**: |\u03c8⟩ = {_alpha_str} |0⟩ + ({_beta_str}) |1⟩"),
    ])
    return (qubit_page,)


# ══════════════════════════════════════════════════════════════
# SECTION 2: QUANTUM GATES AS TRANSFORMATIONS
# ══════════════════════════════════════════════════════════════
@app.cell
def _(mo):
    gate_selector = mo.ui.dropdown(
        options=["X", "Y", "Z", "H", "S", "T"], value="H", label="Choose a gate",
    )
    input_selector = mo.ui.dropdown(
        options={"|0⟩": "0", "|1⟩": "1", "|+⟩ = (|0⟩+|1⟩)/√2": "+", "|\u2212⟩ = (|0⟩\u2212|1⟩)/√2": "-"},
        value="|0⟩", label="Input state",
    )
    show_all_gates = mo.ui.switch(label="Compare all gates on this input")
    return gate_selector, input_selector, show_all_gates


@app.cell
def _(Gate, alt, gate_selector, input_selector, mo, np, pl, show_all_gates):
    _gate_map = {"X": Gate.X, "Y": Gate.Y, "Z": Gate.Z, "H": Gate.H, "S": Gate.S, "T": Gate.T}
    _gate = _gate_map[gate_selector.value]

    _input_map = {
        "0": np.array([[1], [0]], dtype=complex),
        "1": np.array([[0], [1]], dtype=complex),
        "+": np.array([[1], [1]], dtype=complex) / np.sqrt(2),
        "-": np.array([[1], [-1]], dtype=complex) / np.sqrt(2),
    }
    _in_state = _input_map[input_selector.value]
    _out_state = _gate @ _in_state

    # ── Gate matrix heatmap with hover ──
    _mat_rows = []
    for _i in range(2):
        for _j in range(2):
            _val = complex(_gate[_i, _j])
            _mat_rows.append({
                "row": f"|{_i}⟩", "col": f"⟨{_j}|",
                "magnitude": float(np.abs(_val)),
                "phase_deg": float(np.angle(_val, deg=True)),
                "label": (f"{_val.real:.2f}" if abs(_val.imag) < 1e-10
                          else f"{_val.imag:+.2f}i" if abs(_val.real) < 1e-10
                          else f"{_val.real:.2f}{_val.imag:+.2f}i"),
            })
    _mat_df = pl.DataFrame(_mat_rows)
    _mat_hover = alt.selection_point(on="pointerover", empty=False)
    _heatmap = (
        alt.Chart(_mat_df).mark_rect(cornerRadius=6)
        .encode(
            x=alt.X("col:N", title=None, axis=alt.Axis(orient="top", labelFontSize=14)),
            y=alt.Y("row:N", title=None, axis=alt.Axis(labelFontSize=14)),
            color=alt.Color("magnitude:Q", scale=alt.Scale(scheme="blues", domain=[0, 1]),
                            legend=alt.Legend(title="Magnitude")),
            opacity=alt.condition(_mat_hover, alt.value(1), alt.value(0.7)),
            tooltip=["row:N", "col:N", "label:N",
                      alt.Tooltip("phase_deg:Q", title="Phase (\u00b0)", format=".1f")],
        )
        .add_params(_mat_hover)
    )
    _mat_text = (
        alt.Chart(_mat_df).mark_text(fontSize=15, fontWeight="bold")
        .encode(x="col:N", y="row:N", text="label:N",
                color=alt.condition(alt.datum.magnitude > 0.5, alt.value("white"), alt.value("#333")))
    )
    _matrix_chart = ((_heatmap + _mat_text)
                     .properties(width=170, height=170, title=f"{gate_selector.value} Gate Matrix")
                     .configure_view(strokeWidth=0))

    # ── Before/after bar helper ──
    def _state_bars(state_vec, title, colors):
        _labels_sb = ["|0⟩", "|1⟩"]
        _rows_sb = []
        for _ki, _li in enumerate(_labels_sb):
            _vi = complex(state_vec[_ki, 0])
            _rows_sb.append({
                "Basis": _li,
                "Probability": float(np.abs(_vi) ** 2),
                "Amp": (f"{_vi.real:.3f}" if abs(_vi.imag) < 1e-10
                        else f"{_vi.imag:+.3f}i" if abs(_vi.real) < 1e-10
                        else f"{_vi.real:.3f}{_vi.imag:+.3f}i"),
            })
        _dfb = pl.DataFrame(_rows_sb)
        _hov = alt.selection_point(on="pointerover", nearest=True, empty=False)
        return (
            alt.Chart(_dfb).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            .encode(
                x=alt.X("Basis:N", title=None, axis=alt.Axis(labelFontSize=14)),
                y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1]),
                         axis=alt.Axis(format=".0%", title="P")),
                color=alt.Color("Basis:N", scale=alt.Scale(
                    domain=["|0⟩", "|1⟩"], range=colors), legend=None),
                opacity=alt.condition(_hov, alt.value(1), alt.value(0.6)),
                tooltip=["Basis:N", alt.Tooltip("Probability:Q", format=".4f"), "Amp:N"],
            )
            .add_params(_hov)
            .properties(width=130, height=170, title=title)
        )

    _in_chart = _state_bars(_in_state, "Input", ["#93c4e2", "#f5a6a0"])
    _out_chart = _state_bars(_out_state, "Output", ["#4c78a8", "#e45756"])

    _arrow_md = mo.md(
        f'<div style="font-size:2.5em; padding-top:50px; color:#888;">\u2192</div>'
        f'<div style="font-size:0.85em; text-align:center; color:#666; margin-top:-8px;">'
        f'{gate_selector.value}</div>'
    )

    # ── Single gate view ──
    _single_view = mo.hstack(
        [_in_chart, _arrow_md, _matrix_chart, _arrow_md, _out_chart],
        justify="center", gap=0.5,
    )

    # ── All-gates comparison (faceted) ──
    _all_rows = []
    for _gn in ["X", "Y", "Z", "H", "S", "T"]:
        _go = _gate_map[_gn] @ _in_state
        for _ki in range(2):
            _vi = complex(_go[_ki, 0])
            _all_rows.append({
                "Gate": _gn, "Basis": ["|0⟩", "|1⟩"][_ki],
                "Probability": float(np.abs(_vi) ** 2),
            })
    _all_df = pl.DataFrame(_all_rows)
    _compare_chart = (
        alt.Chart(_all_df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("Basis:N", title=None, axis=alt.Axis(labelFontSize=12)),
            y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1]),
                     axis=alt.Axis(format=".0%")),
            color=alt.Color("Basis:N", scale=alt.Scale(
                domain=["|0⟩", "|1⟩"], range=["#4c78a8", "#e45756"]), legend=None),
            tooltip=["Gate:N", "Basis:N", alt.Tooltip("Probability:Q", format=".4f")],
        )
        .properties(width=80, height=160)
        .facet(column=alt.Column("Gate:N", title=None, sort=["X", "Y", "Z", "H", "S", "T"]))
        .configure_view(strokeWidth=0)
    )

    _viz = _single_view if not show_all_gates.value else _compare_chart

    gates_page = mo.vstack([
        mo.md(r"""
## 2. Quantum Gates: Transformations on Amplitudes

A quantum gate is a **unitary matrix** — a transformation that preserves the
total probability (the length of the state vector). Every gate is *reversible*.

| Gate | Matrix | What it does |
|------|--------|-------------|
| **X** (NOT) | $\begin{pmatrix}0&1\\1&0\end{pmatrix}$ | Flips \|0⟩ ↔ \|1⟩ |
| **Z** (Phase) | $\begin{pmatrix}1&0\\0&-1\end{pmatrix}$ | Flips the sign of \|1⟩ |
| **H** (Hadamard) | $\frac{1}{\sqrt{2}}\begin{pmatrix}1&1\\1&-1\end{pmatrix}$ | Creates equal superposition |
| **S** | $\begin{pmatrix}1&0\\0&i\end{pmatrix}$ | Quarter-turn around Z |
| **T** | $\begin{pmatrix}1&0\\0&e^{i\pi/4}\end{pmatrix}$ | Eighth-turn around Z |
"""),
        mo.hstack([gate_selector, input_selector, show_all_gates], justify="start", gap=2),
        _viz,
    ])
    return (gates_page,)


# ══════════════════════════════════════════════════════════════
# SECTION 3: ROTATION GATES — CONTINUOUS EXPLORATION
# ══════════════════════════════════════════════════════════════
@app.cell
def _(mo):
    rot_gate_select = mo.ui.dropdown(
        options=["RX", "RY", "RZ"], value="RX", label="Rotation axis",
    )
    rot_angle_slider = mo.ui.slider(
        start=0, stop=628, step=1, value=157,
        label="Angle (hundredths of radians, range [0, 2\u03c0])",
        show_value=True, full_width=True,
    )
    return rot_angle_slider, rot_gate_select


@app.cell
def _(Gate, alt, mo, np, pl, rot_angle_slider, rot_gate_select):
    _angle = rot_angle_slider.value / 100.0
    _rot_map = {"RX": Gate.RX, "RY": Gate.RY, "RZ": Gate.RZ}
    _rot_gate = _rot_map[rot_gate_select.value](_angle)
    _in_vec = np.array([[1], [0]], dtype=complex)
    _out_vec = _rot_gate @ _in_vec

    # ── Sweep data ──
    _angles = np.linspace(0, 2 * np.pi, 300)
    _sweep_rows = []
    for _a in _angles:
        _g = _rot_map[rot_gate_select.value](_a)
        _out = _g @ _in_vec
        _sweep_rows.append({
            "angle": float(_a),
            "P(|0⟩)": float(np.abs(_out[0, 0]) ** 2),
            "P(|1⟩)": float(np.abs(_out[1, 0]) ** 2),
        })
    _sweep_df = pl.DataFrame(_sweep_rows)
    _sweep_long = _sweep_df.unpivot(index="angle", variable_name="State", value_name="Probability")

    # ── Nearest-point hover layers ──
    _nearest = alt.selection_point(nearest=True, on="pointerover", fields=["angle"], empty=False)

    _base_line = (
        alt.Chart(_sweep_long).mark_line(strokeWidth=2.5, opacity=0.7)
        .encode(
            x=alt.X("angle:Q", title="Rotation Angle (radians)",
                     axis=alt.Axis(
                         values=[0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
                         labelExpr="datum.value == 0 ? '0' : datum.value == 1.5707963267948966 ? '\u03c0/2' : datum.value == 3.141592653589793 ? '\u03c0' : datum.value == 4.71238898038469 ? '3\u03c0/2' : '2\u03c0'")),
            y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1]),
                     axis=alt.Axis(format=".0%")),
            color=alt.Color("State:N", scale=alt.Scale(
                domain=["P(|0⟩)", "P(|1⟩)"], range=["#4c78a8", "#e45756"])),
        )
    )

    # Invisible selectors
    _selectors = (
        alt.Chart(_sweep_long).mark_point(size=1)
        .encode(x="angle:Q", opacity=alt.value(0))
        .add_params(_nearest)
    )

    # Points at hover
    _hover_pts = (
        alt.Chart(_sweep_long).mark_circle(size=80, strokeWidth=1.5, stroke="white")
        .encode(
            x="angle:Q", y="Probability:Q",
            color=alt.Color("State:N", scale=alt.Scale(
                domain=["P(|0⟩)", "P(|1⟩)"], range=["#4c78a8", "#e45756"]), legend=None),
        )
        .transform_filter(_nearest)
    )

    # Text labels at hover
    _hover_text = (
        alt.Chart(_sweep_long).mark_text(align="left", dx=8, dy=-10, fontSize=12, fontWeight="bold")
        .encode(
            x="angle:Q", y="Probability:Q",
            text=alt.Text("Probability:Q", format=".3f"),
            color=alt.Color("State:N", scale=alt.Scale(
                domain=["P(|0⟩)", "P(|1⟩)"], range=["#4c78a8", "#e45756"]), legend=None),
        )
        .transform_filter(_nearest)
    )

    # Vertical rule at hover
    _hover_rule = (
        alt.Chart(_sweep_long).mark_rule(color="#ccc", strokeDash=[4, 4])
        .encode(x="angle:Q")
        .transform_filter(_nearest)
    )

    # Current angle marker (from slider)
    _current_marker = pl.DataFrame({
        "angle": [_angle, _angle],
        "State": ["P(|0⟩)", "P(|1⟩)"],
        "Probability": [float(np.abs(_out_vec[0, 0]) ** 2), float(np.abs(_out_vec[1, 0]) ** 2)],
    })
    _marker = (
        alt.Chart(_current_marker).mark_circle(size=120, strokeWidth=2, stroke="black")
        .encode(x="angle:Q", y="Probability:Q",
                color=alt.Color("State:N", scale=alt.Scale(
                    domain=["P(|0⟩)", "P(|1⟩)"], range=["#4c78a8", "#e45756"]), legend=None))
    )
    _slider_rule = (
        alt.Chart(pl.DataFrame({"angle": [_angle]}))
        .mark_rule(color="#999", strokeDash=[6, 3])
        .encode(x="angle:Q")
    )

    _sweep_chart = (
        alt.layer(_base_line, _selectors, _hover_rule, _hover_pts, _hover_text, _slider_rule, _marker)
        .properties(width=600, height=260,
                    title=f"{rot_gate_select.value}(\u03b8) applied to |0⟩ — hover to explore, dot marks slider angle")
        .configure_view(strokeWidth=0)
    )

    _p0r = float(np.abs(_out_vec[0, 0]) ** 2)
    _p1r = float(np.abs(_out_vec[1, 0]) ** 2)

    rotations_page = mo.vstack([
        mo.md(r"""
## 3. Rotation Gates: Continuous Parameterized Transformations

The gates above are *discrete* — fixed rotations. But quantum computing also
needs **parameterized** gates that rotate by an *arbitrary* angle.

$$R_X(\theta) = \begin{pmatrix}\cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\ -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2}\end{pmatrix}
\qquad
R_Y(\theta) = \begin{pmatrix}\cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\ \sin\frac{\theta}{2} & \cos\frac{\theta}{2}\end{pmatrix}
\qquad
R_Z(\phi) = \begin{pmatrix}e^{-i\phi/2} & 0 \\ 0 & e^{i\phi/2}\end{pmatrix}$$
"""),
        mo.callout(mo.md(
            "**Drag the angle slider** and watch the output state change continuously. "
            "**Hover over the chart** to read exact probabilities at any angle."
        ), kind="info"),
        mo.vstack([rot_gate_select, rot_angle_slider]),
        _sweep_chart,
        mo.hstack([
            mo.stat(value=f"{_angle:.2f} rad", label="Current \u03b8", bordered=True),
            mo.stat(value=f"{np.degrees(_angle):.1f}\u00b0", label="Degrees", bordered=True),
            mo.stat(value=f"{_p0r:.4f}", label="P(|0⟩)", bordered=True),
            mo.stat(value=f"{_p1r:.4f}", label="P(|1⟩)", bordered=True),
        ], justify="center", gap=1),
    ])
    return (rotations_page,)


# ══════════════════════════════════════════════════════════════
# SECTION 4: MULTI-QUBIT STATES & ENTANGLEMENT
# ══════════════════════════════════════════════════════════════
@app.cell
def _(mo):
    q0_gate = mo.ui.dropdown(
        options=["I (identity)", "X", "Y", "Z", "H", "S", "T"], value="H",
        label="Gate on qubit 0",
    )
    q1_gate = mo.ui.dropdown(
        options=["I (identity)", "X", "Y", "Z", "H", "S", "T"], value="I (identity)",
        label="Gate on qubit 1",
    )
    entangle_gate = mo.ui.dropdown(
        options=["None", "CX (CNOT: 0\u21921)", "CX (CNOT: 1\u21920)", "CZ", "SWAP"],
        value="CX (CNOT: 0\u21921)", label="Two-qubit gate",
    )
    init_state = mo.ui.dropdown(
        options=["00", "01", "10", "11"], value="00", label="Initial state",
    )
    n_shots = mo.ui.slider(
        start=100, stop=10000, step=100, value=4000,
        label="Measurement shots", show_value=True,
    )
    return entangle_gate, init_state, n_shots, q0_gate, q1_gate


@app.cell
def _(
    QuantumCircuit, QuantumRegister, Result, alt, entangle_gate,
    init_state, mo, n_shots, np, pl, q0_gate, q1_gate,
):
    _q = QuantumRegister(init_state.value)
    _qc = QuantumCircuit(_q)

    _g_map = {"X": "X", "Y": "Y", "Z": "Z", "H": "H", "S": "S", "T": "T"}
    if q0_gate.value != "I (identity)":
        getattr(_qc, _g_map[q0_gate.value])(0)
    if q1_gate.value != "I (identity)":
        getattr(_qc, _g_map[q1_gate.value])(1)

    if entangle_gate.value == "CX (CNOT: 0\u21921)":
        _qc.CX(0, 1)
    elif entangle_gate.value == "CX (CNOT: 1\u21920)":
        _qc.CX(1, 0)
    elif entangle_gate.value == "CZ":
        _qc.CZ(0, 1)
    elif entangle_gate.value == "SWAP":
        _qc.SWAP(0, 1)

    _sv = Result.get_statevector(_qc)
    _labels = ["|00⟩", "|01⟩", "|10⟩", "|11⟩"]
    _colors4 = ["#4c78a8", "#72b7b2", "#f58518", "#e45756"]

    # ── Probability bars with hover ──
    _amp_rows = []
    for _k, _lbl in enumerate(_labels):
        _val = complex(_sv[_k, 0])
        _amp_rows.append({
            "State": _lbl,
            "Probability": float(np.abs(_val) ** 2),
            "Amplitude": (f"{_val.real:.3f}" if abs(_val.imag) < 1e-10
                          else f"{_val.imag:+.3f}i" if abs(_val.real) < 1e-10
                          else f"{_val.real:.3f}{_val.imag:+.3f}i"),
        })
    _amp_df = pl.DataFrame(_amp_rows)
    _amp_hov = alt.selection_point(on="pointerover", nearest=True, empty=False)
    _prob_bars = (
        alt.Chart(_amp_df).mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
        .encode(
            x=alt.X("State:N", title=None, sort=_labels,
                     axis=alt.Axis(labelFontSize=14, labelAngle=0)),
            y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1]),
                     axis=alt.Axis(format=".0%", title="Probability")),
            color=alt.Color("State:N", scale=alt.Scale(domain=_labels, range=_colors4), legend=None),
            opacity=alt.condition(_amp_hov, alt.value(1), alt.value(0.6)),
            tooltip=["State:N", "Amplitude:N", alt.Tooltip("Probability:Q", format=".4f")],
        )
        .add_params(_amp_hov)
        .properties(width=280, height=250, title="State Amplitudes (Probability)")
    )

    # ── Phase display ──
    _phase_rows = []
    for _k, _lbl in enumerate(_labels):
        _val = complex(_sv[_k, 0])
        _mag = float(np.abs(_val))
        _phase = float(np.angle(_val))
        _phase_rows.append({
            "State": _lbl, "Magnitude": _mag, "Phase (rad)": _phase,
            "px": _mag * np.cos(_phase), "py": _mag * np.sin(_phase),
        })
    _phase_df = pl.DataFrame(_phase_rows)
    _pct = np.linspace(0, 2 * np.pi, 100)
    _pc_df = pl.DataFrame({"cx": np.cos(_pct).tolist(), "cy": np.sin(_pct).tolist()})
    _pbg = (alt.Chart(_pc_df).mark_line(color="#eee", strokeWidth=1)
            .encode(x=alt.X("cx:Q", axis=None, scale=alt.Scale(domain=[-1.3, 1.3])),
                    y=alt.Y("cy:Q", axis=None, scale=alt.Scale(domain=[-1.3, 1.3]))))
    _ppts = (alt.Chart(_phase_df).mark_circle(size=150, strokeWidth=1.5, stroke="white")
             .encode(x="px:Q", y="py:Q",
                     color=alt.Color("State:N", scale=alt.Scale(domain=_labels, range=_colors4)),
                     tooltip=["State:N", alt.Tooltip("Magnitude:Q", format=".3f"),
                              alt.Tooltip("Phase (rad):Q", format=".3f")]))
    _plbs = (alt.Chart(_phase_df).mark_text(dx=12, dy=-10, fontSize=12, fontWeight="bold")
             .encode(x="px:Q", y="py:Q", text="State:N",
                     color=alt.Color("State:N", scale=alt.Scale(domain=_labels, range=_colors4), legend=None)))
    _phase_chart = ((_pbg + _ppts + _plbs)
                    .properties(width=250, height=250, title="Amplitude (magnitude & phase)")
                    .configure_view(strokeWidth=0))

    # ── Measurement ──
    _counts = _qc.measure(n_shots.value)
    _meas_rows = [{"Outcome": _k2, "Count": _v2, "Frequency": _v2 / n_shots.value}
                  for _k2, _v2 in sorted(_counts.items())]
    _meas_df = pl.DataFrame(_meas_rows)
    _meas_chart = (
        alt.Chart(_meas_df).mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
        .encode(
            x=alt.X("Outcome:N", title=None, sort=["00", "01", "10", "11"],
                     axis=alt.Axis(labelFontSize=14, labelAngle=0)),
            y=alt.Y("Frequency:Q", scale=alt.Scale(domain=[0, 1]),
                     axis=alt.Axis(format=".0%", title="Frequency")),
            color=alt.Color("Outcome:N", scale=alt.Scale(
                domain=["00", "01", "10", "11"], range=_colors4), legend=None),
            tooltip=["Outcome:N", "Count:Q", alt.Tooltip("Frequency:Q", format=".4f")],
        )
        .properties(width=280, height=250, title=f"Measurement ({n_shots.value} shots)")
    )

    # ── Circuit diagram (styled HTML) ──
    _q0l = "q\u2080: "
    _q1l = "q\u2081: "
    _q0l += f"\u2500[{q0_gate.value.split()[0]}]\u2500" if q0_gate.value != "I (identity)" else "\u2500\u2500\u2500\u2500\u2500"
    _q1l += f"\u2500[{q1_gate.value.split()[0]}]\u2500" if q1_gate.value != "I (identity)" else "\u2500\u2500\u2500\u2500\u2500"
    if entangle_gate.value != "None":
        if "0\u21921" in entangle_gate.value:
            _q0l += "\u2500\u25cf\u2500\u2500\u2500"; _q1l += "\u2500\u2295\u2500\u2500\u2500"
        elif "1\u21920" in entangle_gate.value:
            _q0l += "\u2500\u2295\u2500\u2500\u2500"; _q1l += "\u2500\u25cf\u2500\u2500\u2500"
        elif entangle_gate.value == "CZ":
            _q0l += "\u2500\u25cf\u2500\u2500\u2500"; _q1l += "\u2500\u25cf\u2500\u2500\u2500"
        elif entangle_gate.value == "SWAP":
            _q0l += "\u2500\u2715\u2500\u2500\u2500"; _q1l += "\u2500\u2715\u2500\u2500\u2500"
    _q0l += "\u2500 \u25b8 measure"
    _q1l += "\u2500 \u25b8 measure"

    _vert_line = "     \u2502" if entangle_gate.value != "None" else ""
    _circuit_html = mo.md(f"""
<div style="font-family:monospace; font-size:1.1em; line-height:2.2em; padding:1em; background:#f8f8f8; border-radius:8px; display:inline-block;">
<div>|{init_state.value[0]}⟩  {_q0l}</div>
<div style="color:#aaa;">{_vert_line}</div>
<div>|{init_state.value[1]}⟩  {_q1l}</div>
</div>
""")

    entanglement_page = mo.vstack([
        mo.md(r"""
## 4. Multi-Qubit States and Entanglement

Two qubits live in a **4-dimensional** vector space: |00⟩, |01⟩, |10⟩, |11⟩.
When two qubits are **independent**, the state factors as a tensor product.
But quantum mechanics allows **entangled** states that *cannot* be factored —
measuring one qubit **instantly determines** the other.

$$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$
"""),
        mo.hstack([init_state, q0_gate, q1_gate, entangle_gate, n_shots], justify="start", gap=1),
        _circuit_html,
        mo.ui.tabs({
            "Probabilities": _prob_bars,
            "Phase Space": _phase_chart,
            "Measurement": _meas_chart,
        }),
    ])
    return (entanglement_page,)


# ══════════════════════════════════════════════════════════════
# SECTION 5: BELL STATES
# ══════════════════════════════════════════════════════════════
@app.cell
def _(mo):
    bell_select = mo.ui.radio(
        options={"\u03a6\u207a (|00⟩+|11⟩)/\u221a2": "phi+",
                 "\u03a6\u207b (|00⟩\u2212|11⟩)/\u221a2": "phi-",
                 "\u03a8\u207a (|01⟩+|10⟩)/\u221a2": "psi+",
                 "\u03a8\u207b (|01⟩\u2212|10⟩)/\u221a2": "psi-"},
        value="\u03a6\u207a (|00⟩+|11⟩)/\u221a2", label="Bell state", inline=True,
    )
    return (bell_select,)


@app.cell
def _(alt, bell_select, mo, np, pl):
    _bell_states = {
        "phi+": np.array([[1], [0], [0], [1]], dtype=complex) / np.sqrt(2),
        "phi-": np.array([[1], [0], [0], [-1]], dtype=complex) / np.sqrt(2),
        "psi+": np.array([[0], [1], [1], [0]], dtype=complex) / np.sqrt(2),
        "psi-": np.array([[0], [1], [-1], [0]], dtype=complex) / np.sqrt(2),
    }
    _sv = _bell_states[bell_select.value]
    _labels = ["|00⟩", "|01⟩", "|10⟩", "|11⟩"]

    # ── Amplitude bars with hover ──
    _rows = []
    for _k, _lbl in enumerate(_labels):
        _val = complex(_sv[_k, 0])
        _rows.append({"State": _lbl, "Amplitude": float(_val.real),
                       "Probability": float(np.abs(_val) ** 2)})
    _df = pl.DataFrame(_rows)
    _bhov = alt.selection_point(on="pointerover", nearest=True, empty=False)
    _amp_chart = (
        alt.Chart(_df)
        .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8,
                  cornerRadiusBottomLeft=8, cornerRadiusBottomRight=8)
        .encode(
            x=alt.X("State:N", title=None, sort=_labels,
                     axis=alt.Axis(labelFontSize=15, labelAngle=0)),
            y=alt.Y("Amplitude:Q", scale=alt.Scale(domain=[-0.8, 0.8]),
                     axis=alt.Axis(title="Amplitude (real part)")),
            color=alt.condition(alt.datum.Amplitude > 0, alt.value("#4c78a8"), alt.value("#e45756")),
            opacity=alt.condition(_bhov, alt.value(1), alt.value(0.6)),
            tooltip=["State:N", alt.Tooltip("Amplitude:Q", format=".4f"),
                      alt.Tooltip("Probability:Q", format=".4f")],
        )
        .add_params(_bhov)
        .properties(width=300, height=250, title="Bell State Amplitudes")
    )
    _zero_rule = (alt.Chart(pl.DataFrame({"y": [0]}))
                  .mark_rule(color="#999", strokeDash=[2, 2]).encode(y="y:Q"))

    # ── Density matrix with hover ──
    _dm = _sv @ _sv.conj().T
    _dm_rows = []
    for _i in range(4):
        for _j in range(4):
            _val = complex(_dm[_i, _j])
            _dm_rows.append({
                "row": _labels[_i], "col": _labels[_j],
                "value": float(_val.real),
                "label": f"{_val.real:.2f}" if abs(_val.imag) < 1e-10 else f"{_val.real:.2f}{_val.imag:+.2f}i",
            })
    _dm_df = pl.DataFrame(_dm_rows)
    _dmhov = alt.selection_point(on="pointerover", empty=False)
    _dm_heat = (
        alt.Chart(_dm_df).mark_rect(cornerRadius=4)
        .encode(
            x=alt.X("col:N", title=None, sort=_labels, axis=alt.Axis(orient="top", labelFontSize=12)),
            y=alt.Y("row:N", title=None, sort=_labels, axis=alt.Axis(labelFontSize=12)),
            color=alt.Color("value:Q", scale=alt.Scale(scheme="redblue", domain=[-0.5, 0.5]),
                            legend=alt.Legend(title="Value")),
            opacity=alt.condition(_dmhov, alt.value(1), alt.value(0.6)),
            tooltip=["row:N", "col:N", "label:N"],
        )
        .add_params(_dmhov)
    )
    _dm_text = (
        alt.Chart(_dm_df).mark_text(fontSize=11)
        .encode(x=alt.X("col:N", sort=_labels), y=alt.Y("row:N", sort=_labels), text="label:N",
                color=alt.condition(alt.datum.value > 0.25, alt.value("white"),
                                    alt.condition(alt.datum.value < -0.25, alt.value("white"), alt.value("#333"))))
    )
    _dm_chart = ((_dm_heat + _dm_text)
                 .properties(width=250, height=250, title="Density Matrix \u03c1 = |\u03c8⟩⟨\u03c8|")
                 .configure_view(strokeWidth=0))

    _name_map = {"phi+": "\u03a6\u207a", "phi-": "\u03a6\u207b", "psi+": "\u03a8\u207a", "psi-": "\u03a8\u207b"}

    bell_page = mo.vstack([
        mo.md(r"""
## 5. The Bell States: Maximal Entanglement

There are four maximally entangled 2-qubit states, called the **Bell states**.
They form an orthonormal basis for the 2-qubit space:

| Bell State | Formula | Correlation |
|-----------|---------|-------------|
| $\|\Phi^+\rangle$ | $\frac{1}{\sqrt{2}}(\|00\rangle + \|11\rangle)$ | Same outcome, + phase |
| $\|\Phi^-\rangle$ | $\frac{1}{\sqrt{2}}(\|00\rangle - \|11\rangle)$ | Same outcome, − phase |
| $\|\Psi^+\rangle$ | $\frac{1}{\sqrt{2}}(\|01\rangle + \|10\rangle)$ | Opposite outcome, + phase |
| $\|\Psi^-\rangle$ | $\frac{1}{\sqrt{2}}(\|01\rangle - \|10\rangle)$ | Opposite outcome, − phase |

These states are the backbone of **quantum teleportation**, **superdense coding**, and **quantum key distribution**.
"""),
        bell_select,
        mo.hstack([_amp_chart + _zero_rule, _dm_chart], justify="center", gap=2),
        mo.callout(mo.md(
            f"### |{_name_map[bell_select.value]}⟩\n\n"
            "**Key insight**: Each qubit is individually maximally mixed (50/50), "
            "but the *pair* is perfectly correlated. This is the essence of entanglement — "
            "the whole contains more information than the sum of its parts.\n\n"
            "The density matrix makes this visible: the off-diagonal elements (coherences) "
            "carry the entanglement information that gets destroyed upon measurement."
        ), kind="success"),
    ])
    return (bell_page,)


# ══════════════════════════════════════════════════════════════
# SECTION 6: GROVER'S ALGORITHM
# ══════════════════════════════════════════════════════════════
@app.cell
def _(mo):
    grover_target = mo.ui.dropdown(
        options={"00": 0, "01": 1, "10": 2, "11": 3}, value="00",
        label="Target state to find",
    )
    grover_step = mo.ui.slider(
        start=0, stop=3, step=1, value=0,
        label="Algorithm step", show_value=True,
    )
    return grover_step, grover_target


@app.cell
def _(QuantumCircuit, QuantumRegister, Result, alt, grover_step, grover_target, mo, np, pl):
    _target_idx = grover_target.value
    _step = grover_step.value
    _labels = ["|00⟩", "|01⟩", "|10⟩", "|11⟩"]
    _step_names = [
        "Initial state |00⟩",
        "After Hadamard (equal superposition)",
        "After Oracle (phase flip on target)",
        "After Diffusion (amplitude amplification)",
    ]

    _q = QuantumRegister("00")
    _qc = QuantumCircuit(_q)
    _snapshots = []

    _snapshots.append(Result.get_statevector(_qc).copy())
    _qc.H(0); _qc.H(1)
    _snapshots.append(Result.get_statevector(_qc).copy())

    _oracle = np.eye(4, dtype=complex)
    _oracle[_target_idx, _target_idx] = -1
    _qc.state_vec = _oracle @ _qc.state_vec
    _snapshots.append(Result.get_statevector(_qc).copy())

    _qc.H(0); _qc.H(1); _qc.X(0); _qc.X(1)
    _qc.CZ(0, 1)
    _qc.X(0); _qc.X(1); _qc.H(0); _qc.H(1)
    _snapshots.append(Result.get_statevector(_qc).copy())

    # ── All steps data ──
    _all_rows = []
    for _si in range(_step + 1):
        _ss = _snapshots[_si]
        for _k, _lbl in enumerate(_labels):
            _val = complex(_ss[_k, 0])
            _all_rows.append({
                "Step": f"{_si}: {_step_names[_si]}", "step_num": _si,
                "State": _lbl, "Amplitude": float(_val.real),
                "Probability": float(np.abs(_val) ** 2),
            })
    _all_df = pl.DataFrame(_all_rows)

    # ── Current step chart with hover ──
    _csv = _snapshots[_step]
    _cur_rows = []
    for _k, _lbl in enumerate(_labels):
        _val = complex(_csv[_k, 0])
        _cur_rows.append({"State": _lbl, "Amplitude": float(_val.real),
                          "Probability": float(np.abs(_val) ** 2)})
    _cur_df = pl.DataFrame(_cur_rows)

    _ghov = alt.selection_point(on="pointerover", nearest=True, empty=False)
    _main_chart = (
        alt.Chart(_cur_df)
        .mark_bar(cornerRadiusTopLeft=10, cornerRadiusTopRight=10,
                  cornerRadiusBottomLeft=10, cornerRadiusBottomRight=10)
        .encode(
            x=alt.X("State:N", title=None, sort=_labels,
                     axis=alt.Axis(labelFontSize=16, labelAngle=0)),
            y=alt.Y("Amplitude:Q", scale=alt.Scale(domain=[-0.6, 1.1]),
                     axis=alt.Axis(title="Amplitude")),
            color=alt.condition(alt.datum.State == _labels[_target_idx],
                                alt.value("#e45756"), alt.value("#4c78a8")),
            opacity=alt.condition(_ghov, alt.value(1), alt.value(0.7)),
            tooltip=["State:N", alt.Tooltip("Amplitude:Q", format=".4f"),
                      alt.Tooltip("Probability:Q", format=".4f")],
        )
        .add_params(_ghov)
        .properties(width=350, height=300, title=f"Step {_step}: {_step_names[_step]}")
    )
    _zero_l = alt.Chart(pl.DataFrame({"y": [0]})).mark_rule(color="#999").encode(y="y:Q")
    _mean_val = float(np.mean([complex(_csv[_k2, 0]).real for _k2 in range(4)]))
    _mean_l = (alt.Chart(pl.DataFrame({"y": [_mean_val]}))
               .mark_rule(color="#f58518", strokeDash=[6, 3], strokeWidth=2).encode(y="y:Q"))

    # ── Small multiples ──
    _small_chart = (
        alt.Chart(_all_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4,
                  cornerRadiusBottomLeft=4, cornerRadiusBottomRight=4)
        .encode(
            x=alt.X("State:N", title=None, sort=_labels,
                     axis=alt.Axis(labelFontSize=10, labelAngle=0)),
            y=alt.Y("Amplitude:Q", scale=alt.Scale(domain=[-0.6, 1.1]),
                     axis=alt.Axis(title="Amp")),
            color=alt.condition(alt.datum.State == _labels[_target_idx],
                                alt.value("#e45756"), alt.value("#4c78a8")),
        )
        .properties(width=120, height=120)
        .facet(column=alt.Column("Step:N", title=None,
                                 sort=[f"{_si2}: {_step_names[_si2]}" for _si2 in range(4)]))
    )

    _target_prob = float(np.abs(complex(_csv[_target_idx, 0])) ** 2)

    _callout_kinds = ["neutral", "info", "warn", "success"]
    _callout_msgs = [
        "All amplitudes start at zero except |00⟩.",
        "Hadamard creates equal superposition: every state has amplitude 1/2.",
        "The oracle flips the sign of the target state. Notice the negative bar.",
        "Diffusion reflects amplitudes about the mean, boosting the target to ~100%!",
    ]

    grover_page = mo.vstack([
        mo.md(r"""
## 6. Grover's Algorithm: Quantum Search

Grover's algorithm finds a marked item in an unsorted list of $N$ items in only
$O(\sqrt{N})$ queries — a **quadratic speedup** over classical search.

1. **Superposition**: Put all qubits in equal superposition with Hadamard gates
2. **Oracle**: Flip the phase of the target state (mark it with −1)
3. **Diffusion**: Amplify the marked amplitude (reflect about the mean)

For 2 qubits ($N=4$ states), **one iteration** is optimal.
"""),
        mo.hstack([grover_target, grover_step], justify="start", gap=2),
        mo.hstack([
            _main_chart + _zero_l + _mean_l,
            mo.vstack([
                mo.stat(value=_labels[_target_idx], label="Target", bordered=True),
                mo.stat(value=f"{_target_prob:.1%}", label="P(target)", bordered=True),
                mo.md('<span style="color:#f58518;">\u2501\u2501\u2501</span> Orange = mean amplitude'),
            ]),
        ], justify="center", gap=1.5),
        mo.callout(mo.md(f"**Step {_step}**: {_callout_msgs[_step]}"), kind=_callout_kinds[_step]),
        _small_chart,
    ])
    return (grover_page,)


# ══════════════════════════════════════════════════════════════
# SECTION 7: CIRCUIT SANDBOX
# ══════════════════════════════════════════════════════════════
@app.cell
def _(mo):
    num_qubits_select = mo.ui.slider(
        start=2, stop=4, step=1, value=2,
        label="Number of qubits", show_value=True,
    )
    sandbox_shots = mo.ui.slider(
        start=500, stop=10000, step=500, value=4000,
        label="Measurement shots", show_value=True,
    )
    _gate_options = ["I", "X", "Y", "Z", "H", "S", "T"]
    _tq_options = ["\u2014", "CX\u2193", "CX\u2191", "CZ"]

    layer_gates = mo.ui.dictionary({
        f"q{_q}_step{_s}": mo.ui.dropdown(options=_gate_options, value="I")
        for _q in range(4) for _s in range(6)
    })
    layer_tq = mo.ui.dictionary({
        f"tq{_q}_{_q+1}_step{_s}": mo.ui.dropdown(options=_tq_options, value="\u2014")
        for _q in range(3) for _s in range(6)
    })
    return layer_gates, layer_tq, num_qubits_select, sandbox_shots


@app.cell
def _(layer_gates, layer_tq, mo, num_qubits_select):
    _nq = num_qubits_select.value
    _rows_html = []
    for _s in range(6):
        _row = f"<td style='padding:4px 8px; font-weight:bold; color:#888;'>Step {_s+1}</td>"
        for _qi in range(_nq):
            _row += f"<td style='padding:4px;'>{layer_gates[f'q{_qi}_step{_s}']}</td>"
        for _qi in range(_nq - 1):
            _row += f"<td style='padding:4px;'>{layer_tq[f'tq{_qi}_{_qi+1}_step{_s}']}</td>"
        _rows_html.append(f"<tr>{_row}</tr>")

    _header = "<th></th>"
    for _qi in range(_nq):
        _header += f"<th style='padding:4px 10px;'>q{_qi}</th>"
    for _qi in range(_nq - 1):
        _header += f"<th style='padding:4px 10px; color:#888; font-size:0.85em;'>q{_qi}\u2194q{_qi+1}</th>"

    sandbox_grid = mo.md(f"""
### Gate Selection Grid

<table style="border-collapse:collapse; margin:0 auto;">
<tr>{_header}</tr>
{''.join(_rows_html)}
</table>
""")
    return (sandbox_grid,)


@app.cell
def _(
    QuantumCircuit, QuantumRegister, alt, layer_gates, layer_tq,
    mo, np, num_qubits_select, pl, sandbox_shots,
):
    _nq = num_qubits_select.value
    _qr = QuantumRegister(_nq)
    _qc = QuantumCircuit(_qr)
    _gate_methods = {"X": "X", "Y": "Y", "Z": "Z", "H": "H", "S": "S", "T": "T"}

    for _s in range(6):
        for _qi in range(_nq):
            _gname = layer_gates.value[f"q{_qi}_step{_s}"]
            if _gname != "I" and _gname in _gate_methods:
                getattr(_qc, _gate_methods[_gname])(_qi)
        for _qi in range(_nq - 1):
            _tqname = layer_tq.value[f"tq{_qi}_{_qi+1}_step{_s}"]
            if _tqname == "CX\u2193":
                _qc.CX(_qi, _qi + 1)
            elif _tqname == "CX\u2191":
                _qc.CX(_qi + 1, _qi)
            elif _tqname == "CZ":
                _qc.CZ(_qi, _qi + 1)

    _sv = _qc.state_vec
    _n_states = 2 ** _nq
    _labels = [format(_i, f"0{_nq}b") for _i in range(_n_states)]

    _state_rows = []
    for _k in range(_n_states):
        _val = complex(_sv[_k, 0])
        _state_rows.append({
            "State": f"|{_labels[_k]}⟩",
            "Probability": float(np.abs(_val) ** 2),
            "Amplitude": (f"{_val.real:.3f}" if abs(_val.imag) < 1e-10
                          else f"{_val.imag:+.3f}i" if abs(_val.real) < 1e-10
                          else f"{_val.real:.3f}{_val.imag:+.3f}i"),
        })
    _state_df = pl.DataFrame(_state_rows)

    _thov = alt.selection_point(on="pointerover", nearest=True, empty=False)
    _theory_chart = (
        alt.Chart(_state_df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("State:N", title=None, sort=[f"|{_l}⟩" for _l in _labels],
                     axis=alt.Axis(labelFontSize=11, labelAngle=-45)),
            y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1]),
                     axis=alt.Axis(format=".0%", title="Probability")),
            color=alt.Color("Probability:Q", scale=alt.Scale(scheme="viridis"), legend=None),
            opacity=alt.condition(_thov, alt.value(1), alt.value(0.7)),
            tooltip=["State:N", "Amplitude:N", alt.Tooltip("Probability:Q", format=".4f")],
        )
        .add_params(_thov)
        .properties(width=max(300, _n_states * 35), height=260,
                    title="Theoretical Probability Distribution")
    )

    _counts = _qc.measure(sandbox_shots.value)
    _meas_rows = [{"Outcome": f"|{_lbl}⟩", "Count": _counts.get(_lbl, 0),
                   "Frequency": _counts.get(_lbl, 0) / sandbox_shots.value}
                  for _lbl in _labels]
    _meas_df = pl.DataFrame(_meas_rows)
    _meas_chart = (
        alt.Chart(_meas_df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, opacity=0.8)
        .encode(
            x=alt.X("Outcome:N", title=None, sort=[f"|{_l}⟩" for _l in _labels],
                     axis=alt.Axis(labelFontSize=11, labelAngle=-45)),
            y=alt.Y("Frequency:Q", scale=alt.Scale(domain=[0, 1]),
                     axis=alt.Axis(format=".0%", title="Frequency")),
            color=alt.Color("Frequency:Q", scale=alt.Scale(scheme="viridis"), legend=None),
            tooltip=["Outcome:N", "Count:Q", alt.Tooltip("Frequency:Q", format=".4f")],
        )
        .properties(width=max(300, _n_states * 35), height=260,
                    title=f"Measurement Simulation ({sandbox_shots.value} shots)")
    )

    sandbox_results = mo.ui.tabs({
        "Probability Distribution": mo.hstack([_theory_chart, _meas_chart], justify="center", gap=1.5),
        "State Vector": mo.ui.table(_state_df),
    })
    return (sandbox_results,)


@app.cell
def _(mo, num_qubits_select, sandbox_grid, sandbox_results, sandbox_shots):
    sandbox_page = mo.vstack([
        mo.md(r"""
## 7. Circuit Sandbox: Build and Measure

Build a quantum circuit with up to **4 qubits** and **6 gate layers**.
Select gates for each qubit at each step, then observe the resulting
probability distribution.
"""),
        mo.callout(mo.md(
            "**Try these experiments:**\n"
            "- **Bell state**: H on q0, then CX\u2193 on q0\u2194q1 \u2192 observe correlated 00/11\n"
            "- **GHZ state**: H on q0, CX\u2193(0,1), CX\u2193(1,2) \u2192 three-way entanglement\n"
            "- **Phase kickback**: X on q1, H on q0, CX\u2193(0,1) \u2192 phase appears on q0"
        ), kind="info"),
        mo.hstack([num_qubits_select, sandbox_shots], justify="start", gap=2),
        sandbox_grid,
        sandbox_results,
    ])
    return (sandbox_page,)


# ══════════════════════════════════════════════════════════════
# SECTION 8: REFERENCES
# ══════════════════════════════════════════════════════════════
@app.cell
def _(mo):
    references_page = mo.vstack([
        mo.md("## References & Further Reading"),
        mo.md(
            "This notebook was built with "
            "[QCSim](https://github.com/alexnodeland/QCSim), "
            "a minimal quantum circuit simulator in Python."
        ),
        mo.accordion({
            "Textbooks": mo.md("""
- **Nielsen & Chuang**, *Quantum Computation and Quantum Information* (Cambridge, 2010) — The standard reference.
- **Yanofsky & Mannucci**, *Quantum Computing for Computer Scientists* (Cambridge, 2008) — Gentler introduction.
- **Mermin**, *Quantum Computer Science* (Cambridge, 2007) — Lean and elegant.
"""),
            "Interactive Resources": mo.md("""
- [IBM Quantum Composer](https://quantum.ibm.com/composer) — Build circuits on real hardware.
- [Quirk](https://algassert.com/quirk) — Beautiful drag-and-drop circuit simulator.
- [Quantum Country](https://quantum.country/) — Spaced-repetition essays by Andy Matuschak & Michael Nielsen.
"""),
            "On Explorable Explanations": mo.md("""
- [Bret Victor, *Explorable Explanations*](http://worrydream.com/ExplorableExplanations/)
- [Bret Victor, *Inventing on Principle*](https://vimeo.com/36579366)
- [Nicky Case, *Explorable Explanations*](https://explorabl.es/)
"""),
        }, multiple=True),
        mo.md(
            "*Built with [marimo](https://marimo.io), "
            "[Altair](https://altair-viz.github.io/), and "
            "[QCSim](https://github.com/alexnodeland/QCSim).*"
        ),
    ])
    return (references_page,)


# ══════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════
@app.cell
def _(
    mo, home_page, qubit_page, gates_page, rotations_page,
    entanglement_page, bell_page, grover_page, sandbox_page, references_page,
):
    mo.routes({
        "#/": home_page,
        "#/qubit": qubit_page,
        "#/gates": gates_page,
        "#/rotations": rotations_page,
        "#/entanglement": entanglement_page,
        "#/bell": bell_page,
        "#/grover": grover_page,
        "#/sandbox": sandbox_page,
        "#/references": references_page,
    })
    return


if __name__ == "__main__":
    app.run()

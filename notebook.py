import marimo

__generated_with = "0.19.8"
app = marimo.App(width="medium", app_title="QCSim: Explorable Quantum Computing")


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

    return Gate, QuantumCircuit, QuantumRegister, Result, alt, mo, np, pl, qcsim


# ─────────────────────────────────────────────────────────────
# SECTION 0: TITLE & MANIFESTO
# ─────────────────────────────────────────────────────────────
@app.cell
def _(mo):
    mo.md(
        r"""
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

    ---

    **How to use this notebook**: Every section below contains interactive controls.
    Change a parameter and all downstream visualizations update instantly.
    Read the prose, play with the controls, build intuition.

    > *"The best way to understand something is to **change** it and see what happens."*
    > — Bret Victor, *Inventing on Principle*
    """
    )
    return


# ─────────────────────────────────────────────────────────────
# SECTION 1: WHAT IS A QUBIT?
# ─────────────────────────────────────────────────────────────
@app.cell
def _(mo):
    mo.md(
        r"""
    ---

    ## 1. What Is a Qubit?

    A classical bit is either **0** or **1**. A qubit is a unit vector in a
    two-dimensional complex vector space:

    $$|\psi\rangle = \alpha\,|0\rangle + \beta\,|1\rangle
    \qquad\text{where}\quad |\alpha|^2 + |\beta|^2 = 1$$

    - $\alpha$ and $\beta$ are **complex amplitudes** (not probabilities!)
    - $|\alpha|^2$ is the probability of measuring **0**
    - $|\beta|^2$ is the probability of measuring **1**

    The key insight: **before measurement, the qubit exists in a superposition of
    both states simultaneously.** The amplitudes carry *phase* information that
    enables interference — the engine of quantum speedup.

    We can parameterize any single-qubit state with two angles, $\theta$ and $\phi$:

    $$|\psi\rangle = \cos\!\left(\tfrac{\theta}{2}\right)|0\rangle
    + e^{i\phi}\sin\!\left(\tfrac{\theta}{2}\right)|1\rangle$$

    This maps every qubit state to a point on the **Bloch sphere** — a beautiful
    geometric picture where gates become rotations.

    **Try it below**: drag $\theta$ and $\phi$ and watch how the state, probabilities,
    and Bloch sphere all change together.
    """
    )
    return


@app.cell
def _(mo):
    theta_slider = mo.ui.slider(
        start=0, stop=314, step=1, value=60,
        label="θ (polar angle)", show_value=True, full_width=True,
    )
    phi_slider = mo.ui.slider(
        start=0, stop=628, step=1, value=0,
        label="φ (azimuthal angle)", show_value=True, full_width=True,
    )
    mo.md(
        f"""
    ### Qubit State Explorer

    {theta_slider}
    *θ shown as hundredths of radians — range [0, π]*

    {phi_slider}
    *φ shown as hundredths of radians — range [0, 2π]*
    """
    )
    return phi_slider, theta_slider


@app.cell
def _(alt, mo, np, pl, phi_slider, theta_slider):
    _theta = theta_slider.value / 100.0
    _phi = phi_slider.value / 100.0

    _alpha = np.cos(_theta / 2)
    _beta = np.exp(1j * _phi) * np.sin(_theta / 2)

    _p0 = float(np.abs(_alpha) ** 2)
    _p1 = float(np.abs(_beta) ** 2)

    # ── Bloch sphere projection (top-down XY and side XZ) ──
    _bx = float(np.sin(_theta) * np.cos(_phi))
    _by = float(np.sin(_theta) * np.sin(_phi))
    _bz = float(np.cos(_theta))

    # Circle outline for unit sphere
    _t_circle = np.linspace(0, 2 * np.pi, 200)
    _circle_df = pl.DataFrame({"cx": np.cos(_t_circle), "cy": np.sin(_t_circle)})

    _point_xz = pl.DataFrame({"x": [_bx], "z": [_bz], "label": ["| ψ ⟩"]})
    _point_xy = pl.DataFrame({"x": [_bx], "y": [_by], "label": ["| ψ ⟩"]})
    _arrow_xz = pl.DataFrame({"x": [0, _bx], "z": [0, _bz]})
    _arrow_xy = pl.DataFrame({"x": [0, _bx], "y": [0, _by]})

    def _bloch_view(circle_df, arrow_df, point_df, xcol, ycol, title):
        base_circle = (
            alt.Chart(circle_df)
            .mark_line(color="#ddd", strokeWidth=1.5)
            .encode(x=alt.X("cx:Q", axis=None, scale=alt.Scale(domain=[-1.4, 1.4])),
                    y=alt.Y("cy:Q", axis=None, scale=alt.Scale(domain=[-1.4, 1.4])))
        )
        axis_h = alt.Chart(pl.DataFrame({"x": [-1, 1], "y": [0, 0]})).mark_line(
            color="#ccc", strokeDash=[4, 4], strokeWidth=1
        ).encode(x="x:Q", y="y:Q")
        axis_v = alt.Chart(pl.DataFrame({"x": [0, 0], "y": [-1, 1]})).mark_line(
            color="#ccc", strokeDash=[4, 4], strokeWidth=1
        ).encode(x="x:Q", y="y:Q")
        arrow = (
            alt.Chart(arrow_df)
            .mark_line(color="#e45756", strokeWidth=2.5)
            .encode(x=alt.X(f"{xcol}:Q"), y=alt.Y(f"{ycol}:Q"))
        )
        point = (
            alt.Chart(point_df)
            .mark_circle(size=100, color="#e45756")
            .encode(x=alt.X(f"{xcol}:Q"), y=alt.Y(f"{ycol}:Q"))
        )
        label = (
            alt.Chart(point_df)
            .mark_text(dx=14, dy=-10, fontSize=13, fontWeight="bold", color="#e45756")
            .encode(x=alt.X(f"{xcol}:Q"), y=alt.Y(f"{ycol}:Q"), text="label:N")
        )
        return (
            (base_circle + axis_h + axis_v + arrow + point + label)
            .properties(width=220, height=220, title=title)
            .configure_view(strokeWidth=0)
        )

    _chart_xz = _bloch_view(_circle_df, _arrow_xz, _point_xz, "x", "z", "Side view (X–Z)")
    _chart_xy = _bloch_view(_circle_df, _arrow_xy, _point_xy, "x", "y", "Top view (X–Y)")

    # ── Probability bar chart ──
    _prob_df = pl.DataFrame({
        "State": ["|0⟩", "|1⟩"],
        "Probability": [_p0, _p1],
    })
    _prob_chart = (
        alt.Chart(_prob_df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("State:N", axis=alt.Axis(labelFontSize=14, title=None)),
            y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1]),
                     axis=alt.Axis(format=".0%", title="Measurement Probability")),
            color=alt.Color("State:N", scale=alt.Scale(
                domain=["|0⟩", "|1⟩"], range=["#4c78a8", "#e45756"]
            ), legend=None),
            tooltip=[alt.Tooltip("State:N"), alt.Tooltip("Probability:Q", format=".4f")],
        )
        .properties(width=160, height=220, title="Measurement Outcome")
    )

    # State vector text
    _alpha_str = f"{_alpha:.4f}"
    _beta_r = f"{_beta.real:.4f}"
    _beta_i = f"{_beta.imag:+.4f}"
    _beta_str = f"{_beta_r}{_beta_i}i" if abs(_beta.imag) > 1e-6 else _beta_r

    _state_md = mo.md(
        f"""
    **State vector**:  |ψ⟩ = {_alpha_str} |0⟩ + ({_beta_str}) |1⟩

    | | Amplitude | Probability |
    |---|---|---|
    | **|0⟩** | {_alpha_str} | {_p0:.4f} |
    | **|1⟩** | {_beta_str} | {_p1:.4f} |

    **Bloch coordinates**: x={_bx:.3f}, y={_by:.3f}, z={_bz:.3f}
    """
    )

    mo.hstack(
        [_chart_xz, _chart_xy, mo.vstack([_prob_chart, _state_md])],
        justify="center", gap=1.5,
    )
    return


# ─────────────────────────────────────────────────────────────
# SECTION 2: QUANTUM GATES AS TRANSFORMATIONS
# ─────────────────────────────────────────────────────────────
@app.cell
def _(mo):
    mo.md(
        r"""
    ---

    ## 2. Quantum Gates: Transformations on Amplitudes

    A quantum gate is a **unitary matrix** — a transformation that preserves the
    total probability (the length of the state vector). Every gate is *reversible*:
    you can always undo it.

    Think of it this way: a gate **rotates** the qubit's state vector in its
    complex vector space. Different gates rotate along different axes.

    | Gate | Matrix | What it does |
    |------|--------|-------------|
    | **X** (NOT) | $\begin{pmatrix}0&1\\1&0\end{pmatrix}$ | Flips \|0⟩ ↔ \|1⟩. Rotation by π around X-axis. |
    | **Z** (Phase) | $\begin{pmatrix}1&0\\0&-1\end{pmatrix}$ | Flips the sign of \|1⟩. Rotation by π around Z-axis. |
    | **H** (Hadamard) | $\frac{1}{\sqrt{2}}\begin{pmatrix}1&1\\1&-1\end{pmatrix}$ | Creates equal superposition. Rotation by π around the X+Z axis. |
    | **S** | $\begin{pmatrix}1&0\\0&i\end{pmatrix}$ | Quarter-turn around Z-axis. S² = Z. |
    | **T** | $\begin{pmatrix}1&0\\0&e^{i\pi/4}\end{pmatrix}$ | Eighth-turn around Z-axis. T² = S. |

    **Select a gate below** and watch how it transforms the state vector.
    The heatmap shows the gate's matrix — the *magnitude* and *phase* of each entry.
    """
    )
    return


@app.cell
def _(mo):
    gate_selector = mo.ui.dropdown(
        options=["X", "Y", "Z", "H", "S", "T"],
        value="H",
        label="Choose a gate",
    )
    input_selector = mo.ui.dropdown(
        options={"|0⟩": "0", "|1⟩": "1", "|+⟩ = (|0⟩+|1⟩)/√2": "+", "|−⟩ = (|0⟩−|1⟩)/√2": "-"},
        value="|0⟩",
        label="Input state",
    )
    mo.hstack([gate_selector, input_selector], justify="start", gap=2)
    return gate_selector, input_selector


@app.cell
def _(Gate, alt, gate_selector, input_selector, mo, np, pl):
    # Get gate matrix
    _gate_map = {"X": Gate.X, "Y": Gate.Y, "Z": Gate.Z, "H": Gate.H, "S": Gate.S, "T": Gate.T}
    _gate = _gate_map[gate_selector.value]

    # Get input state
    _input_map = {
        "0": np.array([[1], [0]], dtype=complex),
        "1": np.array([[0], [1]], dtype=complex),
        "+": np.array([[1], [1]], dtype=complex) / np.sqrt(2),
        "-": np.array([[1], [-1]], dtype=complex) / np.sqrt(2),
    }
    _in_state = _input_map[input_selector.value]
    _out_state = _gate @ _in_state

    # ── Gate matrix heatmap ──
    _rows = []
    for _i in range(2):
        for _j in range(2):
            _val = complex(_gate[_i, _j])
            _rows.append({
                "row": f"|{_i}⟩",
                "col": f"⟨{_j}|",
                "magnitude": float(np.abs(_val)),
                "phase_deg": float(np.angle(_val, deg=True)),
                "label": (
                    f"{_val.real:.2f}" if abs(_val.imag) < 1e-10
                    else f"{_val.imag:+.2f}i" if abs(_val.real) < 1e-10
                    else f"{_val.real:.2f}{_val.imag:+.2f}i"
                ),
            })
    _mat_df = pl.DataFrame(_rows)

    _heatmap = (
        alt.Chart(_mat_df)
        .mark_rect(cornerRadius=6)
        .encode(
            x=alt.X("col:N", title=None, axis=alt.Axis(orient="top", labelFontSize=14)),
            y=alt.Y("row:N", title=None, axis=alt.Axis(labelFontSize=14)),
            color=alt.Color("magnitude:Q", scale=alt.Scale(scheme="blues", domain=[0, 1]),
                            legend=alt.Legend(title="Magnitude")),
            tooltip=["row:N", "col:N", "label:N", alt.Tooltip("phase_deg:Q", title="Phase (°)", format=".1f")],
        )
        .properties(width=170, height=170, title=f"{gate_selector.value} Gate Matrix")
    )
    _text_layer = (
        alt.Chart(_mat_df)
        .mark_text(fontSize=15, fontWeight="bold")
        .encode(
            x="col:N",
            y="row:N",
            text="label:N",
            color=alt.condition(
                alt.datum.magnitude > 0.5,
                alt.value("white"),
                alt.value("#333"),
            ),
        )
    )
    _matrix_chart = (_heatmap + _text_layer).configure_view(strokeWidth=0)

    # ── Before/after bar chart ──
    def _state_bars(state_vec, title, color_scheme):
        _labels_sb = ["|0⟩", "|1⟩"]
        _rows_sb = []
        for _ki, _li in enumerate(_labels_sb):
            _vi = complex(state_vec[_ki, 0])
            _rows_sb.append({
                "Basis": _li,
                "Probability": float(np.abs(_vi) ** 2),
                "Real": float(_vi.real),
                "Imag": float(_vi.imag),
                "Amp": (
                    f"{_vi.real:.3f}" if abs(_vi.imag) < 1e-10
                    else f"{_vi.imag:+.3f}i" if abs(_vi.real) < 1e-10
                    else f"{_vi.real:.3f}{_vi.imag:+.3f}i"
                ),
            })
        df_state = pl.DataFrame(_rows_sb)
        bars = (
            alt.Chart(df_state)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            .encode(
                x=alt.X("Basis:N", title=None, axis=alt.Axis(labelFontSize=14)),
                y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1]),
                         axis=alt.Axis(format=".0%", title="P")),
                color=alt.Color("Basis:N", scale=alt.Scale(
                    domain=["|0⟩", "|1⟩"], range=color_scheme
                ), legend=None),
                tooltip=["Basis:N", alt.Tooltip("Probability:Q", format=".4f"), "Amp:N"],
            )
            .properties(width=130, height=170, title=title)
        )
        return bars

    _in_chart = _state_bars(_in_state, "Input", ["#93c4e2", "#f5a6a0"])
    _out_chart = _state_bars(_out_state, "Output", ["#4c78a8", "#e45756"])

    # Arrow
    _arrow_md = mo.md(
        f"""
    <div style="font-size:2.5em; padding-top:50px; color:#888;">→</div>
    <div style="font-size:0.85em; text-align:center; color:#666; margin-top:-8px;">{gate_selector.value}</div>
    """
    )

    mo.hstack(
        [_in_chart, _arrow_md, _matrix_chart, _arrow_md, _out_chart],
        justify="center", gap=0.5,
    )
    return


# ─────────────────────────────────────────────────────────────
# SECTION 3: THE ROTATION GATES — CONTINUOUS EXPLORATION
# ─────────────────────────────────────────────────────────────
@app.cell
def _(mo):
    mo.md(
        r"""
    ---

    ## 3. Rotation Gates: Continuous Parameterized Transformations

    The gates above are *discrete* — fixed rotations. But quantum computing also
    needs **parameterized** gates that rotate by an *arbitrary* angle.

    $$R_X(\theta) = \begin{pmatrix}\cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\ -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2}\end{pmatrix}
    \qquad
    R_Y(\theta) = \begin{pmatrix}\cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\ \sin\frac{\theta}{2} & \cos\frac{\theta}{2}\end{pmatrix}
    \qquad
    R_Z(\phi) = \begin{pmatrix}e^{-i\phi/2} & 0 \\ 0 & e^{i\phi/2}\end{pmatrix}$$

    **Drag the angle below** and watch the output state change continuously.
    At $\theta = \pi$, $R_X$ becomes the X gate (up to global phase).
    At $\theta = \pi$, $R_Y$ swaps |0⟩ and |1⟩.
    """
    )
    return


@app.cell
def _(mo):
    rot_gate_select = mo.ui.dropdown(
        options=["RX", "RY", "RZ"],
        value="RX",
        label="Rotation axis",
    )
    rot_angle_slider = mo.ui.slider(
        start=0, stop=628, step=1, value=157,
        label="Angle (hundredths of radians, range [0, 2π])",
        show_value=True, full_width=True,
    )
    mo.vstack([rot_gate_select, rot_angle_slider])
    return rot_angle_slider, rot_gate_select


@app.cell
def _(Gate, alt, mo, np, pl, rot_angle_slider, rot_gate_select):
    _angle = rot_angle_slider.value / 100.0
    _rot_map = {"RX": Gate.RX, "RY": Gate.RY, "RZ": Gate.RZ}
    _rot_gate = _rot_map[rot_gate_select.value](_angle)

    _in_vec = np.array([[1], [0]], dtype=complex)  # |0⟩
    _out_vec = _rot_gate @ _in_vec

    # Sweep the angle from 0 to 2π and show how probabilities evolve
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

    _current_marker = pl.DataFrame({
        "angle": [_angle, _angle],
        "State": ["P(|0⟩)", "P(|1⟩)"],
        "Probability": [
            float(np.abs(_out_vec[0, 0]) ** 2),
            float(np.abs(_out_vec[1, 0]) ** 2),
        ],
    })

    _sweep_chart = (
        alt.Chart(_sweep_long)
        .mark_line(strokeWidth=2.5, opacity=0.7)
        .encode(
            x=alt.X("angle:Q", title="Rotation Angle (radians)",
                     axis=alt.Axis(values=[0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
                                   labelExpr="datum.value == 0 ? '0' : datum.value == 1.5707963267948966 ? 'π/2' : datum.value == 3.141592653589793 ? 'π' : datum.value == 4.71238898038469 ? '3π/2' : '2π'")),
            y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1]),
                     axis=alt.Axis(format=".0%")),
            color=alt.Color("State:N", scale=alt.Scale(
                domain=["P(|0⟩)", "P(|1⟩)"], range=["#4c78a8", "#e45756"]
            )),
        )
        .properties(width=550, height=250)
    )

    _marker = (
        alt.Chart(_current_marker)
        .mark_circle(size=120, strokeWidth=2, stroke="black")
        .encode(
            x="angle:Q",
            y="Probability:Q",
            color=alt.Color("State:N", scale=alt.Scale(
                domain=["P(|0⟩)", "P(|1⟩)"], range=["#4c78a8", "#e45756"]
            ), legend=None),
        )
    )

    _rule = (
        alt.Chart(pl.DataFrame({"angle": [_angle]}))
        .mark_rule(color="#999", strokeDash=[4, 4])
        .encode(x="angle:Q")
    )

    _full_chart = (
        (_sweep_chart + _rule + _marker)
        .properties(title=f"{rot_gate_select.value}(θ) applied to |0⟩ — probability vs. angle")
        .configure_view(strokeWidth=0)
    )

    _p0 = float(np.abs(_out_vec[0, 0]) ** 2)
    _p1 = float(np.abs(_out_vec[1, 0]) ** 2)

    _info = mo.md(
        f"""
    **Current angle**: θ = {_angle:.2f} rad ({np.degrees(_angle):.1f}°)

    | | Probability |
    |---|---|
    | **|0⟩** | {_p0:.4f} |
    | **|1⟩** | {_p1:.4f} |

    *The curve shows how the measurement probabilities change as θ sweeps from 0 to 2π.
    The dot marks the current angle.*
    """
    )

    mo.vstack([_full_chart, _info])
    return


# ─────────────────────────────────────────────────────────────
# SECTION 4: MULTI-QUBIT STATES & ENTANGLEMENT
# ─────────────────────────────────────────────────────────────
@app.cell
def _(mo):
    mo.md(
        r"""
    ---

    ## 4. Multi-Qubit States and Entanglement

    Two qubits live in a **4-dimensional** vector space with basis states
    |00⟩, |01⟩, |10⟩, |11⟩. A general 2-qubit state is:

    $$|\psi\rangle = \alpha_{00}|00\rangle + \alpha_{01}|01\rangle + \alpha_{10}|10\rangle + \alpha_{11}|11\rangle$$

    When two qubits are **independent** (separable), the state factors:
    $|\psi\rangle = |\psi_A\rangle \otimes |\psi_B\rangle$.

    But quantum mechanics allows **entangled** states that *cannot* be factored —
    where measuring one qubit **instantly determines** the other. The most famous is
    the **Bell state**:

    $$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$

    Here the qubits are perfectly correlated: if you measure one as 0, the other
    is *always* 0. If you measure 1, the other is *always* 1. And yet each
    individual qubit appears completely random!

    **Build a 2-qubit circuit below** and visualize the resulting state.
    """
    )
    return


@app.cell
def _(mo):
    q0_gate = mo.ui.dropdown(
        options=["I (identity)", "X", "Y", "Z", "H", "S", "T"],
        value="H",
        label="Gate on qubit 0",
    )
    q1_gate = mo.ui.dropdown(
        options=["I (identity)", "X", "Y", "Z", "H", "S", "T"],
        value="I (identity)",
        label="Gate on qubit 1",
    )
    entangle_gate = mo.ui.dropdown(
        options=["None", "CX (CNOT: 0→1)", "CX (CNOT: 1→0)", "CZ", "SWAP"],
        value="CX (CNOT: 0→1)",
        label="Two-qubit gate",
    )
    init_state = mo.ui.dropdown(
        options=["00", "01", "10", "11"],
        value="00",
        label="Initial state",
    )
    n_shots = mo.ui.slider(
        start=100, stop=10000, step=100, value=4000,
        label="Measurement shots", show_value=True,
    )
    mo.hstack([init_state, q0_gate, q1_gate, entangle_gate, n_shots], justify="start", gap=1)
    return entangle_gate, init_state, n_shots, q0_gate, q1_gate


@app.cell
def _(
    QuantumCircuit, QuantumRegister, Result, alt, entangle_gate,
    init_state, mo, n_shots, np, pl, q0_gate, q1_gate,
):
    _q = QuantumRegister(init_state.value)
    _qc = QuantumCircuit(_q)

    # Apply single-qubit gates
    _g_map = {"X": "X", "Y": "Y", "Z": "Z", "H": "H", "S": "S", "T": "T"}
    if q0_gate.value != "I (identity)":
        getattr(_qc, _g_map[q0_gate.value])(0)
    if q1_gate.value != "I (identity)":
        getattr(_qc, _g_map[q1_gate.value])(1)

    # Apply two-qubit gate
    if entangle_gate.value == "CX (CNOT: 0→1)":
        _qc.CX(0, 1)
    elif entangle_gate.value == "CX (CNOT: 1→0)":
        _qc.CX(1, 0)
    elif entangle_gate.value == "CZ":
        _qc.CZ(0, 1)
    elif entangle_gate.value == "SWAP":
        _qc.SWAP(0, 1)

    _sv = Result.get_statevector(_qc)
    _labels = ["|00⟩", "|01⟩", "|10⟩", "|11⟩"]

    # ── State vector amplitude chart ──
    _amp_rows = []
    for _k, _lbl in enumerate(_labels):
        _val = complex(_sv[_k, 0])
        _amp_rows.append({
            "State": _lbl,
            "Real": float(_val.real),
            "Imaginary": float(_val.imag),
            "Probability": float(np.abs(_val) ** 2),
            "Amplitude": (
                f"{_val.real:.3f}" if abs(_val.imag) < 1e-10
                else f"{_val.imag:+.3f}i" if abs(_val.real) < 1e-10
                else f"{_val.real:.3f}{_val.imag:+.3f}i"
            ),
        })
    _amp_df = pl.DataFrame(_amp_rows)

    _prob_bars = (
        alt.Chart(_amp_df)
        .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
        .encode(
            x=alt.X("State:N", title=None, sort=_labels,
                     axis=alt.Axis(labelFontSize=14, labelAngle=0)),
            y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1]),
                     axis=alt.Axis(format=".0%", title="Probability")),
            color=alt.Color("State:N", scale=alt.Scale(
                domain=_labels, range=["#4c78a8", "#72b7b2", "#f58518", "#e45756"]
            ), legend=None),
            tooltip=[
                "State:N", "Amplitude:N",
                alt.Tooltip("Probability:Q", format=".4f"),
            ],
        )
        .properties(width=280, height=250, title="State Amplitudes (Probability)")
    )

    # ── Amplitude phase/magnitude polar-ish display ──
    _phase_rows = []
    for _k, _lbl in enumerate(_labels):
        _val = complex(_sv[_k, 0])
        _mag = float(np.abs(_val))
        _phase = float(np.angle(_val))
        _phase_rows.append({
            "State": _lbl,
            "Magnitude": _mag,
            "Phase (rad)": _phase,
            "px": _mag * np.cos(_phase),
            "py": _mag * np.sin(_phase),
        })
    _phase_df = pl.DataFrame(_phase_rows)

    _phase_chart_circle_t = np.linspace(0, 2 * np.pi, 100)
    _phase_circle = pl.DataFrame({"cx": np.cos(_phase_chart_circle_t), "cy": np.sin(_phase_chart_circle_t)})
    _phase_bg = (
        alt.Chart(_phase_circle)
        .mark_line(color="#eee", strokeWidth=1)
        .encode(
            x=alt.X("cx:Q", axis=None, scale=alt.Scale(domain=[-1.3, 1.3])),
            y=alt.Y("cy:Q", axis=None, scale=alt.Scale(domain=[-1.3, 1.3])),
        )
    )
    _phase_points = (
        alt.Chart(_phase_df)
        .mark_circle(size=150, strokeWidth=1.5, stroke="white")
        .encode(
            x="px:Q", y="py:Q",
            color=alt.Color("State:N", scale=alt.Scale(
                domain=_labels, range=["#4c78a8", "#72b7b2", "#f58518", "#e45756"]
            )),
            tooltip=["State:N", alt.Tooltip("Magnitude:Q", format=".3f"),
                      alt.Tooltip("Phase (rad):Q", format=".3f")],
        )
    )
    _phase_labels = (
        alt.Chart(_phase_df)
        .mark_text(dx=12, dy=-10, fontSize=12, fontWeight="bold")
        .encode(x="px:Q", y="py:Q", text="State:N",
                color=alt.Color("State:N", scale=alt.Scale(
                    domain=_labels, range=["#4c78a8", "#72b7b2", "#f58518", "#e45756"]
                ), legend=None))
    )
    _amp_phase_chart = (
        (_phase_bg + _phase_points + _phase_labels)
        .properties(width=250, height=250, title="Amplitude (magnitude & phase)")
        .configure_view(strokeWidth=0)
    )

    # ── Measurement simulation ──
    _counts = _qc.measure(n_shots.value)
    _meas_rows = [{"Outcome": _k, "Count": _v, "Frequency": _v / n_shots.value}
                  for _k, _v in sorted(_counts.items())]
    _meas_df = pl.DataFrame(_meas_rows)

    _meas_chart = (
        alt.Chart(_meas_df)
        .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
        .encode(
            x=alt.X("Outcome:N", title=None, sort=["00", "01", "10", "11"],
                     axis=alt.Axis(labelFontSize=14, labelAngle=0)),
            y=alt.Y("Frequency:Q", scale=alt.Scale(domain=[0, 1]),
                     axis=alt.Axis(format=".0%", title="Frequency")),
            color=alt.Color("Outcome:N", scale=alt.Scale(
                domain=["00", "01", "10", "11"], range=["#4c78a8", "#72b7b2", "#f58518", "#e45756"]
            ), legend=None),
            tooltip=["Outcome:N", "Count:Q", alt.Tooltip("Frequency:Q", format=".4f")],
        )
        .properties(width=280, height=250, title=f"Measurement ({n_shots.value} shots)")
    )

    # ── Circuit text diagram ──
    _q0_line = "q₀: "
    _q1_line = "q₁: "
    _q0_line += f"─[{q0_gate.value.split()[0]}]─" if q0_gate.value != "I (identity)" else "─────"
    _q1_line += f"─[{q1_gate.value.split()[0]}]─" if q1_gate.value != "I (identity)" else "─────"
    if entangle_gate.value != "None":
        _egate = entangle_gate.value.split("(")[0].strip()
        if "0→1" in entangle_gate.value:
            _q0_line += "─●───"
            _q1_line += "─⊕───"
        elif "1→0" in entangle_gate.value:
            _q0_line += "─⊕───"
            _q1_line += "─●───"
        elif entangle_gate.value == "CZ":
            _q0_line += "─●───"
            _q1_line += "─●───"
        elif entangle_gate.value == "SWAP":
            _q0_line += "─✕───"
            _q1_line += "─✕───"
    _q0_line += "─ ▸ measure"
    _q1_line += "─ ▸ measure"

    _circuit_md = mo.md(
        f"""
    **Circuit**:
    ```
    |{init_state.value[0]}⟩  {_q0_line}
         {'│' if entangle_gate.value != 'None' else ' '}
    |{init_state.value[1]}⟩  {_q1_line}
    ```
    """
    )

    mo.vstack([
        _circuit_md,
        mo.hstack([_prob_bars, _amp_phase_chart, _meas_chart], justify="center", gap=1),
    ])
    return


# ─────────────────────────────────────────────────────────────
# SECTION 5: ENTANGLEMENT DEEP DIVE — BELL STATES
# ─────────────────────────────────────────────────────────────
@app.cell
def _(mo):
    mo.md(
        r"""
    ---

    ## 5. The Bell States: Maximal Entanglement

    There are four maximally entangled 2-qubit states, called the **Bell states**.
    They form an orthonormal basis for the 2-qubit space:

    | Bell State | Formula | Correlation |
    |-----------|---------|-------------|
    | $\|\Phi^+\rangle$ | $\frac{1}{\sqrt{2}}(\|00\rangle + \|11\rangle)$ | Same outcome, + phase |
    | $\|\Phi^-\rangle$ | $\frac{1}{\sqrt{2}}(\|00\rangle - \|11\rangle)$ | Same outcome, − phase |
    | $\|\Psi^+\rangle$ | $\frac{1}{\sqrt{2}}(\|01\rangle + \|10\rangle)$ | Opposite outcome, + phase |
    | $\|\Psi^-\rangle$ | $\frac{1}{\sqrt{2}}(\|01\rangle - \|10\rangle)$ | Opposite outcome, − phase |

    These states are the **backbone of quantum teleportation**, **superdense coding**,
    and **quantum key distribution**. Each qubit individually looks completely random
    (50/50), yet they are perfectly correlated.

    **Select a Bell state below** to see its structure.
    """
    )
    return


@app.cell
def _(mo):
    bell_select = mo.ui.radio(
        options={"Φ⁺ (|00⟩+|11⟩)/√2": "phi+",
                 "Φ⁻ (|00⟩−|11⟩)/√2": "phi-",
                 "Ψ⁺ (|01⟩+|10⟩)/√2": "psi+",
                 "Ψ⁻ (|01⟩−|10⟩)/√2": "psi-"},
        value="Φ⁺ (|00⟩+|11⟩)/√2",
        label="Bell state",
        inline=True,
    )
    bell_select
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

    # Amplitude bar chart (signed real values for bell states)
    _rows = []
    for _k, _lbl in enumerate(_labels):
        _val = complex(_sv[_k, 0])
        _rows.append({
            "State": _lbl,
            "Amplitude": float(_val.real),
            "Probability": float(np.abs(_val) ** 2),
        })
    _df = pl.DataFrame(_rows)

    _amp_chart = (
        alt.Chart(_df)
        .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8,
                  cornerRadiusBottomLeft=8, cornerRadiusBottomRight=8)
        .encode(
            x=alt.X("State:N", title=None, sort=_labels,
                     axis=alt.Axis(labelFontSize=15, labelAngle=0)),
            y=alt.Y("Amplitude:Q", scale=alt.Scale(domain=[-0.8, 0.8]),
                     axis=alt.Axis(title="Amplitude (real part)")),
            color=alt.condition(
                alt.datum.Amplitude > 0,
                alt.value("#4c78a8"),
                alt.value("#e45756"),
            ),
            tooltip=["State:N", alt.Tooltip("Amplitude:Q", format=".4f"),
                      alt.Tooltip("Probability:Q", format=".4f")],
        )
        .properties(width=300, height=250, title="Bell State Amplitudes")
    )

    _zero_rule = (
        alt.Chart(pl.DataFrame({"y": [0]}))
        .mark_rule(color="#999", strokeDash=[2, 2])
        .encode(y="y:Q")
    )

    # Density matrix heatmap
    _dm = _sv @ _sv.conj().T
    _dm_rows = []
    for _i in range(4):
        for _j in range(4):
            _val = complex(_dm[_i, _j])
            _dm_rows.append({
                "row": _labels[_i],
                "col": _labels[_j],
                "value": float(_val.real),
                "label": f"{_val.real:.2f}" if abs(_val.imag) < 1e-10 else f"{_val.real:.2f}{_val.imag:+.2f}i",
            })
    _dm_df = pl.DataFrame(_dm_rows)

    _dm_heat = (
        alt.Chart(_dm_df)
        .mark_rect(cornerRadius=4)
        .encode(
            x=alt.X("col:N", title=None, sort=_labels, axis=alt.Axis(orient="top", labelFontSize=12)),
            y=alt.Y("row:N", title=None, sort=_labels, axis=alt.Axis(labelFontSize=12)),
            color=alt.Color("value:Q", scale=alt.Scale(scheme="redblue", domain=[-0.5, 0.5]),
                            legend=alt.Legend(title="Value")),
            tooltip=["row:N", "col:N", "label:N"],
        )
        .properties(width=250, height=250, title="Density Matrix ρ = |ψ⟩⟨ψ|")
    )
    _dm_text = (
        alt.Chart(_dm_df)
        .mark_text(fontSize=11)
        .encode(
            x=alt.X("col:N", sort=_labels),
            y=alt.Y("row:N", sort=_labels),
            text="label:N",
            color=alt.condition(
                alt.datum.value > 0.25,
                alt.value("white"),
                alt.condition(alt.datum.value < -0.25, alt.value("white"), alt.value("#333"))
            ),
        )
    )

    _name_map = {"phi+": "Φ⁺", "phi-": "Φ⁻", "psi+": "Ψ⁺", "psi-": "Ψ⁻"}
    _desc = mo.md(
        f"""
    ### |{_name_map[bell_select.value]}⟩

    **Key insight**: Each qubit is individually maximally mixed (50/50 for 0 or 1),
    but the *pair* is perfectly correlated. This is the essence of entanglement —
    the whole contains more information than the sum of its parts.

    The density matrix makes this visible: the off-diagonal elements (coherences)
    carry the entanglement information that gets destroyed upon measurement.
    """
    )

    mo.vstack([
        mo.hstack([_amp_chart + _zero_rule, _dm_heat + _dm_text], justify="center", gap=2),
        _desc,
    ])
    return


# ─────────────────────────────────────────────────────────────
# SECTION 6: GROVER'S ALGORITHM — STEP BY STEP
# ─────────────────────────────────────────────────────────────
@app.cell
def _(mo):
    mo.md(
        r"""
    ---

    ## 6. Grover's Algorithm: Quantum Search

    Grover's algorithm finds a marked item in an unsorted list of $N$ items in only
    $O(\sqrt{N})$ queries — a **quadratic speedup** over classical search.

    The algorithm has three phases:
    1. **Superposition**: Put all qubits in equal superposition with Hadamard gates
    2. **Oracle**: Flip the phase of the target state (mark it with a −1)
    3. **Diffusion**: Amplify the marked amplitude (reflect about the mean)

    Steps 2–3 are repeated $\approx \frac{\pi}{4}\sqrt{N}$ times. For 2 qubits
    ($N=4$ states), **one iteration** is optimal.

    **Choose which state to search for**, then step through the algorithm and watch
    the amplitudes change.
    """
    )
    return


@app.cell
def _(mo):
    grover_target = mo.ui.dropdown(
        options={"00": 0, "01": 1, "10": 2, "11": 3},
        value="00",
        label="Target state to find",
    )
    grover_step = mo.ui.slider(
        start=0, stop=3, step=1, value=0,
        label="Algorithm step",
        show_value=True,
    )
    mo.hstack([grover_target, grover_step], justify="start", gap=2)
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

    # Build circuit step by step
    _q = QuantumRegister("00")
    _qc = QuantumCircuit(_q)

    _snapshots = []

    # Step 0: initial
    _snapshots.append(Result.get_statevector(_qc).copy())

    # Step 1: Hadamard
    _qc.H(0)
    _qc.H(1)
    _snapshots.append(Result.get_statevector(_qc).copy())

    # Step 2: Oracle — flip phase of target
    # We'll manually apply the phase oracle
    _oracle = np.eye(4, dtype=complex)
    _oracle[_target_idx, _target_idx] = -1
    _qc.state_vec = _oracle @ _qc.state_vec
    _snapshots.append(Result.get_statevector(_qc).copy())

    # Step 3: Diffusion — reflect about mean
    _qc.H(0)
    _qc.H(1)
    _qc.X(0)
    _qc.X(1)
    _qc.CZ(0, 1)
    _qc.X(0)
    _qc.X(1)
    _qc.H(0)
    _qc.H(1)
    _snapshots.append(Result.get_statevector(_qc).copy())

    # Build visualization for all steps up to current
    _all_rows = []
    for _step_i in range(_step + 1):
        _sv_snap = _snapshots[_step_i]
        for _k, _lbl in enumerate(_labels):
            _val = complex(_sv_snap[_k, 0])
            _all_rows.append({
                "Step": f"{_step_i}: {_step_names[_step_i]}",
                "step_num": _step_i,
                "State": _lbl,
                "Amplitude": float(_val.real),
                "Probability": float(np.abs(_val) ** 2),
            })
    _all_df = pl.DataFrame(_all_rows)

    # Current step chart (large)
    _current_sv = _snapshots[_step]
    _current_rows = []
    for _k, _lbl in enumerate(_labels):
        _val = complex(_current_sv[_k, 0])
        _current_rows.append({
            "State": _lbl,
            "Amplitude": float(_val.real),
            "Probability": float(np.abs(_val) ** 2),
        })
    _current_df = pl.DataFrame(_current_rows)

    _main_chart = (
        alt.Chart(_current_df)
        .mark_bar(cornerRadiusTopLeft=10, cornerRadiusTopRight=10,
                  cornerRadiusBottomLeft=10, cornerRadiusBottomRight=10)
        .encode(
            x=alt.X("State:N", title=None, sort=_labels,
                     axis=alt.Axis(labelFontSize=16, labelAngle=0)),
            y=alt.Y("Amplitude:Q", scale=alt.Scale(domain=[-0.6, 1.1]),
                     axis=alt.Axis(title="Amplitude")),
            color=alt.condition(
                alt.datum.State == _labels[_target_idx],
                alt.value("#e45756"),
                alt.value("#4c78a8"),
            ),
            tooltip=["State:N", alt.Tooltip("Amplitude:Q", format=".4f"),
                      alt.Tooltip("Probability:Q", format=".4f")],
        )
        .properties(width=350, height=300, title=f"Step {_step}: {_step_names[_step]}")
    )
    _zero_line = alt.Chart(pl.DataFrame({"y": [0]})).mark_rule(color="#999").encode(y="y:Q")
    _mean_val = float(np.mean([complex(_current_sv[_k2, 0]).real for _k2 in range(4)]))
    _mean_line = (
        alt.Chart(pl.DataFrame({"y": [_mean_val]}))
        .mark_rule(color="#f58518", strokeDash=[6, 3], strokeWidth=2)
        .encode(y="y:Q")
    )

    # Small multiples: all steps
    _small_chart = (
        alt.Chart(_all_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4,
                  cornerRadiusBottomLeft=4, cornerRadiusBottomRight=4)
        .encode(
            x=alt.X("State:N", title=None, sort=_labels,
                     axis=alt.Axis(labelFontSize=10, labelAngle=0)),
            y=alt.Y("Amplitude:Q", scale=alt.Scale(domain=[-0.6, 1.1]),
                     axis=alt.Axis(title="Amp")),
            color=alt.condition(
                alt.datum.State == _labels[_target_idx],
                alt.value("#e45756"),
                alt.value("#4c78a8"),
            ),
        )
        .properties(width=120, height=120)
        .facet(
            column=alt.Column("Step:N", title=None,
                              sort=[f"{_si}: {_step_names[_si]}" for _si in range(4)]),
        )
    )

    _target_label = _labels[_target_idx]
    _target_prob = float(np.abs(complex(_current_sv[_target_idx, 0])) ** 2)

    _explanation = mo.md(
        f"""
    **Target**: {_target_label} (highlighted in red)
    **Target probability**: **{_target_prob:.1%}**
    <span style="color:#f58518;">━━━</span> Orange dashed line = mean amplitude

    {"**→ All amplitudes start at zero except |00⟩.**" if _step == 0 else ""}
    {"**→ Hadamard creates equal superposition: every state has amplitude 1/2.**" if _step == 1 else ""}
    {"**→ The oracle flips the sign of the target state. Notice the negative bar.**" if _step == 2 else ""}
    {"**→ Diffusion reflects amplitudes about the mean, boosting the target to ~100%!**" if _step == 3 else ""}
    """
    )

    mo.vstack([
        mo.hstack([_main_chart + _zero_line + _mean_line, _explanation], justify="center", gap=1.5),
        _small_chart,
    ])
    return


# ─────────────────────────────────────────────────────────────
# SECTION 7: BUILD YOUR OWN CIRCUIT
# ─────────────────────────────────────────────────────────────
@app.cell
def _(mo):
    mo.md(
        r"""
    ---

    ## 7. Circuit Sandbox: Build and Measure

    Now it's your turn. Build a quantum circuit with up to **4 qubits** and **6 gate
    layers**. Select gates for each qubit at each step, then observe the resulting
    probability distribution.

    This is where intuition is built — try these experiments:
    - **Bell state**: H on q0, then CX(0,1) → observe correlated 00/11
    - **GHZ state**: H on q0, CX(0,1), CX(1,2) → three-way entanglement
    - **Phase kickback**: X on q1, H on q0, CX(0,1) → phase appears on q0
    """
    )
    return


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
    _tq_options = ["—", "CX↓", "CX↑", "CZ"]

    # 4 qubits × 6 layers of single-qubit gates
    layer_gates = mo.ui.dictionary({
        f"q{q}_step{s}": mo.ui.dropdown(options=_gate_options, value="I")
        for q in range(4)
        for s in range(6)
    })
    # 6 layers of two-qubit gates (between adjacent pairs)
    layer_tq = mo.ui.dictionary({
        f"tq{q}_{q+1}_step{s}": mo.ui.dropdown(options=_tq_options, value="—")
        for q in range(3)
        for s in range(6)
    })
    mo.hstack([num_qubits_select, sandbox_shots], justify="start", gap=2)
    return layer_gates, layer_tq, num_qubits_select, sandbox_shots


@app.cell
def _(layer_gates, layer_tq, mo, num_qubits_select):
    _nq = num_qubits_select.value
    _rows_html = []
    for _s in range(6):
        _row = f"<td style='padding:4px 8px; font-weight:bold; color:#888;'>Step {_s+1}</td>"
        for _q in range(_nq):
            _key = f"q{_q}_step{_s}"
            _row += f"<td style='padding:4px;'>{layer_gates[_key]}</td>"
        # Two-qubit gates between pairs
        for _q in range(_nq - 1):
            _tq_key = f"tq{_q}_{_q+1}_step{_s}"
            _row += f"<td style='padding:4px;'>{layer_tq[_tq_key]}</td>"
        _rows_html.append(f"<tr>{_row}</tr>")

    _header = "<th></th>"
    for _q in range(_nq):
        _header += f"<th style='padding:4px 10px;'>q{_q}</th>"
    for _q in range(_nq - 1):
        _header += f"<th style='padding:4px 10px; color:#888; font-size:0.85em;'>q{_q}↔q{_q+1}</th>"

    _table_html = f"""
    <table style="border-collapse:collapse; margin:0 auto;">
    <tr>{_header}</tr>
    {''.join(_rows_html)}
    </table>
    """
    mo.md(
        f"""
    ### Gate Selection Grid

    Select single-qubit gates for each qubit at each step, and optionally
    two-qubit gates between adjacent pairs.

    {_table_html}
    """
    )
    return


@app.cell
def _(
    QuantumCircuit, QuantumRegister, alt, layer_gates, layer_tq,
    mo, np, num_qubits_select, pl, sandbox_shots,
):
    _nq = num_qubits_select.value
    _q = QuantumRegister(_nq)
    _qc = QuantumCircuit(_q)

    _gate_methods = {"X": "X", "Y": "Y", "Z": "Z", "H": "H", "S": "S", "T": "T"}

    for _s in range(6):
        # Single-qubit gates
        for _q in range(_nq):
            _key = f"q{_q}_step{_s}"
            _gname = layer_gates.value[_key]
            if _gname != "I" and _gname in _gate_methods:
                getattr(_qc, _gate_methods[_gname])(_q)
        # Two-qubit gates
        for _q in range(_nq - 1):
            _tq_key = f"tq{_q}_{_q+1}_step{_s}"
            _tqname = layer_tq.value[_tq_key]
            if _tqname == "CX↓":
                _qc.CX(_q, _q + 1)
            elif _tqname == "CX↑":
                _qc.CX(_q + 1, _q)
            elif _tqname == "CZ":
                _qc.CZ(_q, _q + 1)

    _sv = _qc.state_vec
    _n_states = 2 ** _nq
    _labels = [format(i, f"0{_nq}b") for i in range(_n_states)]

    _state_rows = []
    for _k in range(_n_states):
        _val = complex(_sv[_k, 0])
        _state_rows.append({
            "State": f"|{_labels[_k]}⟩",
            "Probability": float(np.abs(_val) ** 2),
            "Amplitude": (
                f"{_val.real:.3f}" if abs(_val.imag) < 1e-10
                else f"{_val.imag:+.3f}i" if abs(_val.real) < 1e-10
                else f"{_val.real:.3f}{_val.imag:+.3f}i"
            ),
        })
    _state_df = pl.DataFrame(_state_rows)

    _theory_chart = (
        alt.Chart(_state_df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("State:N", title=None, sort=[f"|{_l}⟩" for _l in _labels],
                     axis=alt.Axis(labelFontSize=11, labelAngle=-45)),
            y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1]),
                     axis=alt.Axis(format=".0%", title="Probability")),
            color=alt.Color("Probability:Q", scale=alt.Scale(scheme="viridis"),
                            legend=None),
            tooltip=["State:N", "Amplitude:N", alt.Tooltip("Probability:Q", format=".4f")],
        )
        .properties(width=max(300, _n_states * 35), height=260,
                    title="Theoretical Probability Distribution")
    )

    _counts = _qc.measure(sandbox_shots.value)
    _meas_rows = []
    for _lbl in _labels:
        _meas_rows.append({
            "Outcome": f"|{_lbl}⟩",
            "Count": _counts.get(_lbl, 0),
            "Frequency": _counts.get(_lbl, 0) / sandbox_shots.value,
        })
    _meas_df = pl.DataFrame(_meas_rows)

    _meas_chart = (
        alt.Chart(_meas_df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, opacity=0.8)
        .encode(
            x=alt.X("Outcome:N", title=None, sort=[f"|{_l}⟩" for _l in _labels],
                     axis=alt.Axis(labelFontSize=11, labelAngle=-45)),
            y=alt.Y("Frequency:Q", scale=alt.Scale(domain=[0, 1]),
                     axis=alt.Axis(format=".0%", title="Frequency")),
            color=alt.Color("Frequency:Q", scale=alt.Scale(scheme="viridis"),
                            legend=None),
            tooltip=["Outcome:N", "Count:Q", alt.Tooltip("Frequency:Q", format=".4f")],
        )
        .properties(width=max(300, _n_states * 35), height=260,
                    title=f"Measurement Simulation ({sandbox_shots.value} shots)")
    )

    mo.vstack([
        mo.hstack([_theory_chart, _meas_chart], justify="center", gap=1.5),
    ])
    return


# ─────────────────────────────────────────────────────────────
# SECTION 8: REFERENCES & FURTHER READING
# ─────────────────────────────────────────────────────────────
@app.cell
def _(mo):
    mo.md(
        r"""
    ---

    ## References & Further Reading

    This notebook was built with [QCSim](https://github.com/alexnodeland/QCSim),
    a minimal quantum circuit simulator in Python.

    ### Textbooks
    - **Nielsen & Chuang**, *Quantum Computation and Quantum Information* (Cambridge, 2010) — The standard reference.
    - **Yanofsky & Mannucci**, *Quantum Computing for Computer Scientists* (Cambridge, 2008) — Gentler introduction.
    - **Mermin**, *Quantum Computer Science* (Cambridge, 2007) — Lean and elegant.

    ### Interactive Resources
    - [IBM Quantum Composer](https://quantum.ibm.com/composer) — Build circuits on real hardware.
    - [Quirk](https://algassert.com/quirk) — Beautiful drag-and-drop circuit simulator.
    - [Quantum Country](https://quantum.country/) — Spaced-repetition essays by Andy Matuschak & Michael Nielsen.

    ### On Explorable Explanations
    - [Bret Victor, *Explorable Explanations*](http://worrydream.com/ExplorableExplanations/)
    - [Bret Victor, *Inventing on Principle*](https://vimeo.com/36579366)
    - [Nicky Case, *Explorable Explanations*](https://explorabl.es/)

    ---

    *Built with [marimo](https://marimo.io), [Altair](https://altair-viz.github.io/), and [QCSim](https://github.com/alexnodeland/QCSim).*
    """
    )
    return


if __name__ == "__main__":
    app.run()

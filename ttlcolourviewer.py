import streamlit as st
import requests
import rdflib
import pandas as pd
import numpy as np
import re
import colorsys
from typing import Optional, Tuple, Dict, Any, List

# Patch numpy.asscalar for compatibility with colormath
if not hasattr(np, 'asscalar'):
    np.asscalar = lambda x: x.item()

from rdflib.namespace import Namespace
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

# Import colormath for advanced color difference calculations and conversions
from colormath.color_objects import LabColor, sRGBColor, LCHabColor, XYZColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000, delta_e_cmc

# Define a dictionary to map common color names to HEX codes
COLOR_NAME_TO_HEX = {
    "orange": "#ffa500",
    "blue": "#0000ff",
    "green": "#008000",
    "red": "#ff0000",
    "yellow": "#ffff00"
}

# -----------------------------------------------------------------------------
# Utility Functions for Color Science & Conversions
# -----------------------------------------------------------------------------
def validate_lab_values(l: float, a: float, b: float) -> bool:
    """Validate LAB values are within the expected ranges."""
    valid = True
    if not (0 <= l <= 100):
        st.warning(f"Lightness value L={l} is out of range (0-100).")
        valid = False
    if not (-128 <= a <= 128):
        st.warning(f"a value {a} is out of range (-128 to 128).")
        valid = False
    if not (-128 <= b <= 128):
        st.warning(f"b value {b} is out of range (-128 to 128).")
        valid = False
    return valid

def validate_hex(hex_code: str) -> bool:
    """Validate hex code format."""
    if not re.match(r'^#([A-Fa-f0-9]{6})$', hex_code):
        st.warning(f"Hex code {hex_code} is not valid.")
        return False
    return True

def delta_e_cie76(lab1: Tuple[float, float, float], lab2: Tuple[float, float, float]) -> float:
    """Calculate color difference using CIE76 (ΔE*ab)."""
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    return np.sqrt((L1 - L2)**2 + (a1 - a2)**2 + (b1 - b2)**2)

def delta_e_ciede2000_wrapper(lab1: Tuple[float, float, float], lab2: Tuple[float, float, float]) -> float:
    """Calculate color difference using CIEDE2000."""
    color1 = LabColor(lab_l=lab1[0], lab_a=lab1[1], lab_b=lab1[2])
    color2 = LabColor(lab_l=lab2[0], lab_a=lab2[1], lab_b=lab2[2])
    return delta_e_cie2000(color1, color2)

def delta_e_cmc_wrapper(lab1: Tuple[float, float, float], lab2: Tuple[float, float, float],
                         l: float = 2, c: float = 1) -> float:
    """Calculate color difference using CMC(l:c) with adjustable parameters."""
    color1 = LabColor(lab_l=lab1[0], lab_a=lab1[1], lab_b=lab1[2])
    color2 = LabColor(lab_l=lab2[0], lab_a=lab2[1], lab_b=lab2[2])
    return delta_e_cmc(color1, color2, l, c)  # positional args

def lab_to_hex(l: float, a: float, b: float) -> str:
    """Convert LAB color to HEX code using sRGB conversion."""
    lab = LabColor(lab_l=l, lab_a=a, lab_b=b)
    rgb = convert_color(lab, sRGBColor)
    r = int(max(0, min(1, rgb.rgb_r)) * 255)
    g = int(max(0, min(1, rgb.rgb_g)) * 255)
    b_val = int(max(0, min(1, rgb.rgb_b)) * 255)
    return f'#{r:02x}{g:02x}{b_val:02x}'

def hex_to_lab(hex_color: str) -> Optional[Tuple[float, float, float]]:
    """Convert HEX color to LAB tuple."""
    if not hex_color.startswith("#"):
        hex_color = COLOR_NAME_TO_HEX.get(hex_color.lower(), "")
    if not validate_hex(hex_color):
        return None
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    rgb = sRGBColor(r, g, b, is_upscaled=True)
    lab = convert_color(rgb, LabColor)
    return (lab.lab_l, lab.lab_a, lab.lab_b)

def convert_color_spaces(input_color: str, input_space: str) -> Dict[str, Any]:
    """
    Convert an input color from HEX or LAB to multiple color spaces.
    Returns a dict with keys: sRGB, LAB, XYZ, HSL, LCh.
    """
    result = {}
    if input_space == "HEX":
        if not validate_hex(input_color):
            return result
        r = int(input_color[1:3], 16)
        g = int(input_color[3:5], 16)
        b = int(input_color[5:7], 16)
        srgb = sRGBColor(r, g, b, is_upscaled=True)
        lab = convert_color(srgb, LabColor)
    elif input_space == "LAB":
        try:
            parts = [float(x.strip()) for x in input_color.split(",")]
            if len(parts) != 3 or not validate_lab_values(*parts):
                return result
            lab = LabColor(lab_l=parts[0], lab_a=parts[1], lab_b=parts[2])
            srgb = convert_color(lab, sRGBColor)
        except Exception as e:
            st.error(f"Invalid LAB input: {e}")
            return result
    else:
        st.error("Unsupported color space.")
        return result

    r = int(max(0, min(1, srgb.rgb_r)) * 255)
    g = int(max(0, min(1, srgb.rgb_g)) * 255)
    b_val = int(max(0, min(1, srgb.rgb_b)) * 255)
    result["sRGB"] = f"RGB({r}, {g}, {b_val})"
    result["HEX"] = f'#{r:02x}{g:02x}{b_val:02x}'
    result["LAB"] = f"L:{lab.lab_l:.2f}, a:{lab.lab_a:.2f}, b:{lab.lab_b:.2f}"
    xyz = convert_color(srgb, XYZColor)
    result["XYZ"] = f"X:{xyz.xyz_x:.2f}, Y:{xyz.xyz_y:.2f}, Z:{xyz.xyz_z:.2f}"
    r_norm, g_norm, b_norm = r/255, g/255, b_val/255
    h, l_val, s = colorsys.rgb_to_hls(r_norm, g_norm, b_norm)
    result["HSL"] = f"H:{h*360:.0f}°, S:{s*100:.0f}%, L:{l_val*100:.0f}%"
    lch = convert_color(lab, LCHabColor)
    result["LCh"] = f"L:{lch.lch_l:.2f}, C:{lch.lch_c:.2f}, h:{lch.lch_h:.2f}°"
    return result

def apply_chromatic_adaptation(lab: LabColor, source_wp: Tuple[float, float, float],
                               target_wp: Tuple[float, float, float]) -> LabColor:
    """
    Apply a simple Bradford chromatic adaptation from source white point to target white point.
    Note: This is a basic implementation.
    """
    M = np.array([[ 0.8951,  0.2664, -0.1614],
                  [-0.7502,  1.7135,  0.0367],
                  [ 0.0389, -0.0685,  1.0296]])
    M_inv = np.linalg.inv(M)
    srgb = convert_color(lab, sRGBColor)
    xyz = convert_color(srgb, XYZColor)
    source = np.array(source_wp)
    target = np.array(target_wp)
    cone_source = M.dot(source)
    cone_target = M.dot(target)
    adapt_ratio = cone_target / cone_source
    xyz_vals = np.array([xyz.xyz_x, xyz.xyz_y, xyz.xyz_z])
    cone_vals = M.dot(xyz_vals)
    adapted_cone = cone_vals * adapt_ratio
    adapted_xyz = M_inv.dot(adapted_cone)
    adapted_XYZ = XYZColor(*adapted_xyz)
    adapted_lab = convert_color(adapted_XYZ, LabColor)
    return adapted_lab

def simulate_color_blindness(color: str, cb_type: str) -> str:
    """
    Simulate color blindness on a color provided as a HEX string or a common color name.
    Supported cb_type: "None", "Protanopia", "Deuteranopia", "Tritanopia".
    Uses simple matrix approximations.
    """
    if not color.startswith("#"):
        color = COLOR_NAME_TO_HEX.get(color.lower(), color)
    if cb_type == "None":
        return color
    r = int(color[1:3], 16)
    g = int(color[3:5], 16)
    b = int(color[5:7], 16)
    rgb = sRGBColor(r, g, b, is_upscaled=True)
    rgb_arr = np.array([rgb.rgb_r, rgb.rgb_g, rgb.rgb_b])
    matrices = {
        "Protanopia": np.array([[0.56667, 0.43333, 0.0],
                                  [0.55833, 0.44167, 0.0],
                                  [0.0,     0.24167, 0.75833]]),
        "Deuteranopia": np.array([[0.625, 0.375, 0.0],
                                  [0.70,  0.30,  0.0],
                                  [0.0,   0.30,  0.70]]),
        "Tritanopia": np.array([[0.95, 0.05, 0.0],
                                [0.0, 0.43333, 0.56667],
                                [0.0, 0.475, 0.525]])
    }
    new_rgb = matrices[cb_type].dot(rgb_arr)
    new_rgb = np.clip(new_rgb, 0, 1)
    r_new = int(new_rgb[0] * 255)
    g_new = int(new_rgb[1] * 255)
    b_new = int(new_rgb[2] * 255)
    return f'#{r_new:02x}{g_new:02x}{b_new:02x}'

# -----------------------------------------------------------------------------
# Page Configuration & Custom Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Library Special Collections Fading Data Repository",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
        .main-header {
            font-family: 'Helvetica Neue', sans-serif;
            color: #1E1E1E;
            background: linear-gradient(90deg, #f8f9fa, #e9ecef);
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        }
        .card {
            background-color: white;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
            margin-bottom: 1rem;
        }
        /* Hide empty cards */
        .card:empty {
            display: none;
        }
        .stat-box {
            background: white;
            border-left: 5px solid #4361ee;
            padding: 1rem;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            margin-bottom: 1rem;
        }
        .footnote {
            font-size: 0.8rem;
            color: #6c757d;
            font-style: italic;
            margin-top: 2rem;
            border-top: 1px solid #dee2e6;
            padding-top: 1rem;
        }
        .github-corner svg {
            fill: #4361ee;
            color: #fff;
            position: absolute;
            top: 0;
            border: 0;
            right: 0;
        }
        .color-chip {
            display: inline-block;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin-right: 8px;
            vertical-align: middle;
            border: 1px solid rgba(0,0,0,0.1);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# Define Namespaces
# -----------------------------------------------------------------------------
LAB = Namespace("http://example.org/lab#")
EX = Namespace("http://example.org/")
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")

# -----------------------------------------------------------------------------
# Initialize Session State
# -----------------------------------------------------------------------------
if "current_graph" not in st.session_state:
    st.session_state.current_graph = None
if "color_terms" not in st.session_state:
    st.session_state.color_terms = pd.DataFrame()
if "fading_simulations" not in st.session_state:
    st.session_state.fading_simulations = pd.DataFrame()
if "restoration_simulations" not in st.session_state:
    st.session_state.restoration_simulations = pd.DataFrame()
if "metadata" not in st.session_state:
    st.session_state.metadata = {}

# -----------------------------------------------------------------------------
# Data Loading & Extraction Functions
# -----------------------------------------------------------------------------
def load_ttl_from_url(url: str) -> Optional[rdflib.Graph]:
    """Fetch and parse a TTL file from the given URL with error handling."""
    with st.spinner("Loading data from repository..."):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            ttl_content = response.text
            graph = rdflib.Graph()
            graph.parse(data=ttl_content, format="turtle")
            return graph
        except requests.RequestException as e:
            st.error(f"Network error: {e}")
        except Exception as e:
            st.error(f"Failed to parse TTL data: {e}")
    return None

def list_ttl_files(repo_owner: str = "hsandaver", repo_name: str = "fadingrepo") -> List[dict]:
    """List all TTL files in the GitHub repository."""
    repo_api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents"
    try:
        response = requests.get(repo_api_url, timeout=10)
        response.raise_for_status()
        files = response.json()
        ttl_files = [file for file in files if file.get("name", "").endswith(".ttl")]
        return ttl_files
    except requests.RequestException as e:
        st.error(f"Error connecting to GitHub API: {e}")
    except ValueError as e:
        st.error(f"Error parsing repository data: {e}")
    return []

def extract_metadata(graph: rdflib.Graph) -> Dict[str, str]:
    """Extract provenance metadata from the graph if available."""
    query = """
        PREFIX dct: <http://purl.org/dc/terms/>
        SELECT ?institution ?date ?method WHERE {
            ?s dct:publisher ?institution ;
               dct:created ?date ;
               dct:description ?method .
        } LIMIT 1
    """
    metadata = {}
    try:
        results = graph.query(query)
        for row in results:
            metadata['institution'] = str(row.institution)
            metadata['date'] = str(row.date)
            metadata['method'] = str(row.method)
    except Exception as e:
        st.info("No provenance metadata found in the TTL file.")
    return metadata

def extract_color_terms(graph: rdflib.Graph) -> pd.DataFrame:
    """Extract ColorTerm data from the graph and return as a DataFrame."""
    query = """
        PREFIX lab: <http://example.org/lab#>
        PREFIX ex: <http://example.org/>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        SELECT ?uri ?a ?b ?l ?hex ?label WHERE {
            ?uri a ex:ColorTerm ;
                 lab:hasA ?a ;
                 lab:hasB ?b ;
                 lab:hasL ?l ;
                 lab:hasHex ?hex ;
                 skos:prefLabel ?label .
        }
    """
    results = graph.query(query)
    data = []
    for row in results:
        l_val = float(row.l)
        a_val = float(row.a)
        b_val = float(row.b)
        hex_code = str(row.hex)
        if validate_lab_values(l_val, a_val, b_val) and validate_hex(hex_code):
            data.append({
                "uri": str(row.uri),
                "a": a_val,
                "b": b_val,
                "l": l_val,
                "hex": hex_code,
                "label": str(row.label)
            })
    return pd.DataFrame(data) if data else pd.DataFrame()

def extract_fading_simulations(graph: rdflib.Graph) -> pd.DataFrame:
    """Extract FadingSimulation data from the graph, including optional simulation parameters."""
    query = """
        PREFIX lab: <http://example.org/lab#>
        PREFIX ex: <http://example.org/>
        SELECT ?uri ?a ?b ?l ?dyeType ?exposureTime ?intensity WHERE {
            ?uri a ex:FadingSimulation ;
                 lab:fadingColorA ?a ;
                 lab:fadingColorB ?b ;
                 lab:fadingColorL ?l ;
                 lab:dyeType ?dyeType .
            OPTIONAL { ?uri lab:exposureTime ?exposureTime . }
            OPTIONAL { ?uri lab:intensity ?intensity . }
        }
    """
    results = graph.query(query)
    data = []
    for row in results:
        l_val = float(row.l)
        a_val = float(row.a)
        b_val = float(row.b)
        if validate_lab_values(l_val, a_val, b_val):
            data.append({
                "uri": str(row.uri),
                "a": a_val,
                "b": b_val,
                "l": l_val,
                "dyeType": str(row.dyeType),
                "exposureTime": float(row.exposureTime) if row.exposureTime else None,
                "intensity": float(row.intensity) if row.intensity else None
            })
    return pd.DataFrame(data) if data else pd.DataFrame()

def extract_restoration_simulations(graph: rdflib.Graph) -> pd.DataFrame:
    """Extract RestorationSimulation data from the graph."""
    query = """
        PREFIX lab: <http://example.org/lab#>
        PREFIX ex: <http://example.org/>
        SELECT ?uri ?origA ?origB ?origL ?targetA ?targetB ?targetL WHERE {
            ?uri a ex:RestorationSimulation ;
                 lab:originalColorA ?origA ;
                 lab:originalColorB ?origB ;
                 lab:originalColorL ?origL ;
                 lab:targetColorA ?targetA ;
                 lab:targetColorB ?targetB ;
                 lab:targetColorL ?targetL .
        }
    """
    results = graph.query(query)
    data = []
    for row in results:
        origL = float(row.origL)
        origA = float(row.origA)
        origB = float(row.origB)
        targetL = float(row.targetL)
        targetA = float(row.targetA)
        targetB = float(row.targetB)
        if validate_lab_values(origL, origA, origB) and validate_lab_values(targetL, targetA, targetB):
            data.append({
                "uri": str(row.uri),
                "origA": origA,
                "origB": origB,
                "origL": origL,
                "targetA": targetA,
                "targetB": targetB,
                "targetL": targetL
            })
    return pd.DataFrame(data) if data else pd.DataFrame()

# -----------------------------------------------------------------------------
# Visualization Functions with Customization & Accessibility
# -----------------------------------------------------------------------------
def create_interactive_3d_plot(
    color_df: pd.DataFrame,
    fading_df: pd.DataFrame = None,
    restoration_df: pd.DataFrame = None,
    marker_size: int = 12,
    marker_opacity: float = 0.8,
    cb_simulation: str = "None"
) -> Optional[go.Figure]:
    """Create an interactive 3D plot of LAB color space using Plotly."""
    if color_df.empty:
        return None

    fig = go.Figure()

    for _, row in color_df.iterrows():
        col_hex = simulate_color_blindness(row["hex"], cb_simulation)
        fig.add_trace(go.Scatter3d(
            x=[row["a"]],
            y=[row["b"]],
            z=[row["l"]],
            mode="markers",
            marker=dict(
                size=marker_size,
                color=col_hex,
                opacity=marker_opacity,
                line=dict(color="black", width=1)
            ),
            text=row["label"],
            hoverinfo="text",
            hovertext=(
                f"<b>{row['label']}</b><br>L: {row['l']:.1f}<br>"
                f"a: {row['a']:.1f}<br>b: {row['b']:.1f}<br>hex: {row['hex']}"
            ),
            name=row["label"]
        ))

    if fading_df is not None and not fading_df.empty:
        for _, row in fading_df.iterrows():
            extra_info = ""
            if row.get("exposureTime") is not None:
                extra_info += f"<br>Exposure: {row['exposureTime']}"
            if row.get("intensity") is not None:
                extra_info += f"<br>Intensity: {row['intensity']}"
            fig.add_trace(go.Scatter3d(
                x=[row["a"]],
                y=[row["b"]],
                z=[row["l"]],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    symbol="diamond",
                    color=simulate_color_blindness("orange", cb_simulation),
                    opacity=marker_opacity - 0.1,
                    line=dict(color="black", width=1)
                ),
                hoverinfo="text",
                hovertext=(
                    f"<b>Fading Point</b><br>L: {row['l']:.1f}<br>"
                    f"a: {row['a']:.1f}<br>b: {row['b']:.1f}<br>"
                    f"Dye Type: {row['dyeType']}{extra_info}"
                ),
                name=f"Fading ({row['dyeType']})"
            ))

    if restoration_df is not None and not restoration_df.empty:
        for _, row in restoration_df.iterrows():
            d_e = delta_e_cie76(
                (row["origL"], row["origA"], row["origB"]),
                (row["targetL"], row["targetA"], row["targetB"])
            )
            fig.add_trace(go.Scatter3d(
                x=[row["origA"]],
                y=[row["origB"]],
                z=[row["origL"]],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    symbol="square",
                    color=simulate_color_blindness("blue", cb_simulation),
                    opacity=marker_opacity - 0.1,
                    line=dict(color="black", width=1)
                ),
                hoverinfo="text",
                hovertext=(
                    f"<b>Original Color</b><br>L: {row['origL']:.1f}<br>"
                    f"a: {row['origA']:.1f}<br>b: {row['origB']:.1f}"
                ),
                name="Original Color"
            ))
            fig.add_trace(go.Scatter3d(
                x=[row["targetA"]],
                y=[row["targetB"]],
                z=[row["targetL"]],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    symbol="circle",
                    color=simulate_color_blindness("green", cb_simulation),
                    opacity=marker_opacity - 0.1,
                    line=dict(color="black", width=1)
                ),
                hoverinfo="text",
                hovertext=(
                    f"<b>Target Color</b><br>L: {row['targetL']:.1f}<br>"
                    f"a: {row['targetA']:.1f}<br>b: {row['targetB']:.1f}<br>"
                    f"ΔE*ab: {d_e:.2f}"
                ),
                name="Target Color"
            ))
            fig.add_trace(go.Scatter3d(
                x=[row["origA"], row["targetA"]],
                y=[row["origB"], row["targetB"]],
                z=[row["origL"], row["targetL"]],
                mode="lines",
                line=dict(
                    color="rgba(0,100,0,0.5)",
                    width=4,
                    dash="dot"
                ),
                hoverinfo="none",
                showlegend=False
            ))

    fig.update_layout(
        title={
            "text": "LAB Color Space Visualization",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(size=24)
        },
        scene=dict(
            xaxis_title="a (green-red)",
            yaxis_title="b (blue-yellow)",
            zaxis_title="L (lightness)",
            xaxis=dict(range=[-128, 128], gridcolor="#eeeeee"),
            yaxis=dict(range=[-128, 128], gridcolor="#eeeeee"),
            zaxis=dict(range=[0, 100], gridcolor="#eeeeee"),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            aspectmode="cube"
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1
        ),
        template="plotly_white",
        height=700
    )
    return fig

def create_color_palette_chart(color_df: pd.DataFrame, cb_simulation: str = "None") -> Optional[go.Figure]:
    """Create a visual color palette display as a stacked bar chart."""
    if color_df.empty:
        return None

    fig = go.Figure()
    for _, row in color_df.iterrows():
        col_hex = simulate_color_blindness(row["hex"], cb_simulation)
        fig.add_trace(go.Bar(
            x=[row["label"]],
            y=[1],
            marker_color=col_hex,
            name=row["label"],
            hoverinfo="text",
            hovertext=(
                f"<b>{row['label']}</b><br>L: {row['l']:.1f}<br>"
                f"a: {row['a']:.1f}<br>b: {row['b']:.1f}<br>hex: {row['hex']}"
            ),
            showlegend=False
        ))
    fig.update_layout(
        title={
            "text": "Color Palette",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(size=18)
        },
        barmode="stack",
        height=200,
        margin=dict(l=10, r=10, t=50, b=40),
        xaxis=dict(showgrid=False, showticklabels=True, tickangle=45),
        yaxis=dict(showgrid=False, showticklabels=False, showline=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig

def create_2d_ab_plot(
    color_df: pd.DataFrame,
    fading_df: pd.DataFrame = None,
    restoration_df: pd.DataFrame = None,
    cb_simulation: str = "None"
) -> Optional[go.Figure]:
    """Create a 2D scatter plot of the a*b* plane in LAB color space."""
    if color_df.empty:
        return None

    fig = go.Figure()
    for _, row in color_df.iterrows():
        col_hex = simulate_color_blindness(row["hex"], cb_simulation)
        fig.add_trace(go.Scatter(
            x=[row["a"]],
            y=[row["b"]],
            mode="markers",
            marker=dict(
                size=15,
                color=col_hex,
                symbol="circle",
                line=dict(color="black", width=1)
            ),
            text=row["label"],
            hoverinfo="text",
            hovertext=(
                f"<b>{row['label']}</b><br>L: {row['l']:.1f}<br>"
                f"a: {row['a']:.1f}<br>b: {row['b']:.1f}"
            ),
            name=row["label"]
        ))

    if fading_df is not None and not fading_df.empty:
        for _, row in fading_df.iterrows():
            extra_info = ""
            if row.get("exposureTime") is not None:
                extra_info += f"<br>Exposure: {row['exposureTime']}"
            if row.get("intensity") is not None:
                extra_info += f"<br>Intensity: {row['intensity']}"
            fig.add_trace(go.Scatter(
                x=[row["a"]],
                y=[row["b"]],
                mode="markers",
                marker=dict(
                    size=12,
                    symbol="diamond",
                    color=simulate_color_blindness("orange", cb_simulation),
                    line=dict(color="black", width=1)
                ),
                hoverinfo="text",
                hovertext=(
                    f"<b>Fading Point</b><br>L: {row['l']:.1f}<br>"
                    f"a: {row['a']:.1f}<br>b: {row['b']:.1f}<br>"
                    f"Dye Type: {row['dyeType']}{extra_info}"
                ),
                name=f"Fading ({row['dyeType']})"
            ))
    if restoration_df is not None and not restoration_df.empty:
        for _, row in restoration_df.iterrows():
            d_e = delta_e_cie76(
                (row["origL"], row["origA"], row["origB"]),
                (row["targetL"], row["targetA"], row["targetB"])
            )
            fig.add_trace(go.Scatter(
                x=[row["origA"]],
                y=[row["origB"]],
                mode="markers",
                marker=dict(
                    size=12,
                    symbol="square",
                    color=simulate_color_blindness("blue", cb_simulation),
                    line=dict(color="black", width=1)
                ),
                hoverinfo="text",
                hovertext=(
                    f"<b>Original Color</b><br>L: {row['origL']:.1f}<br>"
                    f"a: {row['origA']:.1f}<br>b: {row['origB']:.1f}"
                ),
                name="Original Color"
            ))
            fig.add_trace(go.Scatter(
                x=[row["targetA"]],
                y=[row["targetB"]],
                mode="markers",
                marker=dict(
                    size=12,
                    symbol="circle",
                    color=simulate_color_blindness("green", cb_simulation),
                    line=dict(color="black", width=1)
                ),
                hoverinfo="text",
                hovertext=(
                    f"<b>Target Color</b><br>L: {row['targetL']:.1f}<br>"
                    f"a: {row['targetA']:.1f}<br>b: {row['targetB']:.1f}<br>"
                    f"ΔE*ab: {d_e:.2f}"
                ),
                name="Target Color"
            ))
            fig.add_trace(go.Scatter(
                x=[row["origA"], row["targetA"]],
                y=[row["origB"], row["targetB"]],
                mode="lines+markers",
                line=dict(color="rgba(0,100,0,0.5)", width=2, dash="dot"),
                marker=dict(size=0),
                hoverinfo="none",
                showlegend=False
            ))

    fig.add_shape(
        type="line", line=dict(dash="dash", width=1, color="gray"),
        x0=0, y0=-100, x1=0, y1=100
    )
    fig.add_shape(
        type="line", line=dict(dash="dash", width=1, color="gray"),
        x0=-100, y0=0, x1=100, y1=0
    )
    fig.add_annotation(x=80, y=10, text="Red", showarrow=False, font=dict(color="red"))
    fig.add_annotation(x=-80, y=10, text="Green", showarrow=False, font=dict(color="green"))
    fig.add_annotation(x=10, y=80, text="Yellow", showarrow=False, font=dict(color="goldenrod"))
    fig.add_annotation(x=10, y=-80, text="Blue", showarrow=False, font=dict(color="blue"))

    fig.update_layout(
        title={
            "text": "a*b* Color Plane (L* value filtered)",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(size=18)
        },
        xaxis=dict(
            title="a* (green-red)",
            range=[-100, 100],
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor="gray",
            gridcolor="#eeeeee"
        ),
        yaxis=dict(
            title="b* (blue-yellow)",
            range=[-100, 100],
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor="gray",
            gridcolor="#eeeeee",
            scaleanchor="x",
            scaleratio=1
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1
        ),
        height=500,
        margin=dict(l=40, r=40, t=60, b=40),
        plot_bgcolor="rgba(245,245,245,0.8)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="closest"
    )
    return fig

# -----------------------------------------------------------------------------
# GitHub Corner Link
# -----------------------------------------------------------------------------
st.markdown(
    """
    <a href="https://github.com/hsandaver/fadingrepo" class="github-corner" aria-label="View source on GitHub">
      <svg width="80" height="80" viewBox="0 0 250 250" aria-hidden="true">
        <path d="M0,0 L115,115 L130,115 L142,142 L250,250 L250,0 Z"></path>
        <path d="M128.3,109.0 C113.8,99.7 119.0,89.6 119.0,89.6 C122.0,82.7 120.5,78.6 120.5,78.6 C119.2,72.0 123.4,76.3 123.4,76.3 C127.3,80.9 125.5,87.3 125.5,87.3 C122.9,97.6 130.6,101.9 134.4,103.2" fill="currentColor" style="transform-origin: 130px 106px;" class="octo-arm"></path>
        <path d="M115.0,115.0 C114.9,115.1 118.7,116.5 119.8,115.4 L133.7,101.6 C136.9,99.2 139.9,98.4 142.2,98.6 C133.8,88.0 127.5,74.4 143.8,58.0 C148.5,53.4 154.0,51.2 159.7,51.0 C160.3,49.4 163.2,43.6 171.4,40.1 C171.4,40.1 176.1,42.5 178.8,56.2 C183.1,58.6 187.2,61.8 190.9,65.4 C194.5,69.0 197.7,73.2 200.1,77.6 C213.8,80.2 216.3,84.9 216.3,84.9 C212.7,93.1 206.9,96.0 205.4,96.6 C205.1,102.4 203.0,107.8 198.3,112.5 C181.9,128.9 168.3,122.5 157.7,114.1 C157.9,116.9 156.7,120.9 152.7,124.9 L141.0,136.5 C139.8,137.7 141.6,141.9 141.8,141.8 Z" fill="currentColor" class="octo-body"></path>
      </svg>
    </a>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# Main App Layout
# -----------------------------------------------------------------------------
st.markdown(
    '<div class="main-header"><h1 style="text-align: center;">Library Special Collections Fading Data Repository</h1>'
    '<p style="text-align: center;">Store and analyze fading data for your library\'s special collection</p></div>',
    unsafe_allow_html=True
)

with st.sidebar:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🗂️ TTL File Selection")
    ttl_files = list_ttl_files()
    if ttl_files:
        file_names = [file["name"] for file in ttl_files]
        selected_file_name = st.selectbox("Select a TTL dataset:", file_names)
        selected_file = next(file for file in ttl_files if file["name"] == selected_file_name)
        url = selected_file.get("download_url")
        if st.button("Load Selected File", key="load_button", use_container_width=True):
            graph = load_ttl_from_url(url)
            if graph:
                st.session_state.current_graph = graph
                st.session_state.color_terms = extract_color_terms(graph)
                st.session_state.fading_simulations = extract_fading_simulations(graph)
                st.session_state.restoration_simulations = extract_restoration_simulations(graph)
                st.session_state.metadata = extract_metadata(graph)
                st.success(f"✅ Successfully loaded {len(graph)} triples!")
            else:
                st.error("Failed to load graph data.")
    else:
        st.warning("No TTL files found in the repository.")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.metadata:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📝 Data Provenance")
        meta = st.session_state.metadata
        st.markdown(f"**Institution:** {meta.get('institution', 'Unknown')}")
        st.markdown(f"**Date:** {meta.get('date', 'Unknown')}")
        st.markdown(f"**Method:** {meta.get('method', 'Not provided')}")
        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("ℹ️ About this App"):
        st.markdown("""
        **Library Special Collections Fading Data Repository** is designed for cultural heritage research. 
        
        **Key Features:**
        - Store and analyze fading data from special collections.
        - Advanced ΔE metrics for precise color difference calculations.
        - Conversion between multiple color spaces.
        - Chromatic adaptation and color blindness simulations.
        
        *Note: Detailed documentation is available for further information.*
        """)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Dataset Statistics")
    col_stats1, col_stats2 = st.columns(2)
    with col_stats1:
        st.markdown(
            f'<div class="stat-box"><h3 style="margin:0">Colors</h3>'
            f'<h2 style="margin:0;color:#4361ee">{len(st.session_state.color_terms)}</h2></div>',
            unsafe_allow_html=True
        )
    with col_stats2:
        total_sims = (len(st.session_state.fading_simulations) +
                      len(st.session_state.restoration_simulations))
        st.markdown(
            f'<div class="stat-box"><h3 style="margin:0">Simulations</h3>'
            f'<h2 style="margin:0;color:#4361ee">{total_sims}</h2></div>',
            unsafe_allow_html=True
        )
    st.subheader("🎛️ Display Options")
    view_type = st.radio(
        "Select visualization type:",
        ["3D LAB Space", "2D a*b* Plane", "Color Palette", "Data Table"]
    )
    marker_size = st.slider("Marker Size", 5, 20, 12)
    marker_opacity = st.slider("Marker Opacity", 0.1, 1.0, 0.8)
    cb_simulation = st.selectbox("Color Blindness Simulation:", ["None", "Protanopia", "Deuteranopia", "Tritanopia"],
                                 help="Simulate how colors appear for different types of color vision deficiencies.")
    
    l_filter = None
    if view_type == "2D a*b* Plane":
        l_filter = st.slider("Filter by L value (±10):", 0, 100, 50, 5)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧮 Dynamic ΔE Calculator")
    col_dE1, col_dE2 = st.columns(2)
    with col_dE1:
        color1_hex = st.color_picker("Select Color 1", "#ff0000")
    with col_dE2:
        color2_hex = st.color_picker("Select Color 2", "#00ff00")
    de_metric = st.selectbox("Select ΔE Metric:", ["CIE76", "CIEDE2000", "CMC(l:c)"],
                             help="CIE76 is simple Euclidean; CIEDE2000 offers improved perceptual uniformity; CMC(l:c) allows parameter tuning.")
    lab1 = hex_to_lab(color1_hex)
    lab2 = hex_to_lab(color2_hex)
    if lab1 and lab2:
        if de_metric == "CIE76":
            d_e_value = delta_e_cie76(lab1, lab2)
        elif de_metric == "CIEDE2000":
            d_e_value = delta_e_ciede2000_wrapper(lab1, lab2)
        else:
            d_e_value = delta_e_cmc_wrapper(lab1, lab2)
        st.markdown(f"**ΔE ({de_metric}): {d_e_value:.2f}**")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔄 Color Conversion Utility")
    input_space = st.selectbox("Input Color Space:", ["HEX", "LAB"])
    if input_space == "HEX":
        input_color = st.text_input("Enter HEX code:", "#3366ff")
    else:
        input_color = st.text_input("Enter LAB values (comma separated, e.g. 50, 0, 0):", "50, 0, 0")
    if st.button("Convert Color"):
        conversions = convert_color_spaces(input_color, input_space)
        if conversions:
            st.markdown("**Conversions:**")
            for key, val in conversions.items():
                st.markdown(f"- **{key}:** {val}")
        else:
            st.error("Conversion failed. Please check your input.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔍 Dataset Filter & Download")
    filter_text = st.text_input("Filter color terms by label:")
    if filter_text and not st.session_state.color_terms.empty:
        filtered_df = st.session_state.color_terms[st.session_state.color_terms["label"].str.contains(filter_text, case=False)]
        st.dataframe(filtered_df)
    if st.button("Download Color Data as CSV"):
        csv = st.session_state.color_terms.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "color_data.csv", "text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.current_graph is not None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    if not st.session_state.color_terms.empty:
        palette_fig = create_color_palette_chart(st.session_state.color_terms, cb_simulation)
        if palette_fig:
            st.plotly_chart(palette_fig, use_container_width=True, config={'displayModeBar': False})

    if view_type == "3D LAB Space":
        fig = create_interactive_3d_plot(
            st.session_state.color_terms,
            st.session_state.fading_simulations,
            st.session_state.restoration_simulations,
            marker_size=marker_size,
            marker_opacity=marker_opacity,
            cb_simulation=cb_simulation
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    elif view_type == "2D a*b* Plane":
        filtered_colors = st.session_state.color_terms[
            (st.session_state.color_terms["l"] >= l_filter - 10) &
            (st.session_state.color_terms["l"] <= l_filter + 10)
        ].copy()
        filtered_fading = (
            st.session_state.fading_simulations[
                (st.session_state.fading_simulations["l"] >= l_filter - 10) &
                (st.session_state.fading_simulations["l"] <= l_filter + 10)
            ].copy() if not st.session_state.fading_simulations.empty else None
        )
        fig = create_2d_ab_plot(filtered_colors, filtered_fading, st.session_state.restoration_simulations, cb_simulation)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No colors found with L value around {l_filter}. Try adjusting the filter.")

    elif view_type == "Color Palette":
        st.info("Color palette displayed above.")
    
    elif view_type == "Data Table":
        st.markdown("### Color Terms")
        st.dataframe(st.session_state.color_terms)
        if not st.session_state.fading_simulations.empty:
            st.markdown("### Fading Simulations")
            st.dataframe(st.session_state.fading_simulations)
        if not st.session_state.restoration_simulations.empty:
            st.markdown("### Restoration Simulations")
            st.dataframe(st.session_state.restoration_simulations)
            st.markdown("### Restoration Color Comparisons")
            for index, row in st.session_state.restoration_simulations.iterrows():
                orig_hex = lab_to_hex(row['origL'], row['origA'], row['origB'])
                target_hex = lab_to_hex(row['targetL'], row['targetA'], row['targetB'])
                d_e = delta_e_cie76((row['origL'], row['origA'], row['origB']),
                                     (row['targetL'], row['targetA'], row['targetB']))
                st.markdown(f"""
                    <div style="display: flex; align-items: center; margin-bottom: 10px;">
                        <div style="width: 50px; height: 50px; background-color: {orig_hex}; border: 1px solid #000; margin-right: 10px;"></div>
                        <div style="width: 50px; height: 50px; background-color: {target_hex}; border: 1px solid #000; margin-right: 10px;"></div>
                        <span>ΔE*ab: {d_e:.2f}</span>
                    </div>
                """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

import streamlit as st

st.title("Package Test")

packages = [
    ("streamlit", "st"),
    ("pandas", "pd"),
    ("numpy", "np"),
    ("plotly.express", "px"),
    ("plotly.graph_objects", "go"),
    ("matplotlib.pyplot", "plt"),
    ("seaborn", "sns"),
    ("sklearn", "sklearn"),
]

for package, alias in packages:
    try:
        if "." in package:
            # Handle submodules like plotly.express
            parts = package.split(".")
            exec(f"import {parts[0]}")
            exec(f"{parts[0]} = __import__('{parts[0]}')")
            st.success(f"✅ {package}")
        else:
            __import__(package)
            st.success(f"✅ {package}")
    except ImportError as e:
        st.error(f"❌ {package}: {e}")

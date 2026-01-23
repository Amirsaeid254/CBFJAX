#!/usr/bin/env python3
"""
CBFJAX Map Editor - Visual GUI tool for creating map configurations.

A canvas-based editor for creating and editing map configurations
with support for all CBFJAX geometry types.

Features:
    - Visual placement and manipulation of obstacles
    - Support for: cylinder, norm_box, box, ellipse
    - Support for boundaries: norm_boundary, boundary
    - Load existing map configurations (Python or JSON)
    - Export to Python map_config.py format
    - Random map generation
    - Grid layout generation
    - Pan, zoom, and grid display

Usage:
    python run_map_editor.py

The editor will open in your default web browser.
"""

import os
import sys
import webbrowser

def main():
    """Launch the map editor."""
    # Get the path to the HTML file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(script_dir, 'map_editor', 'map_editor.html')

    if not os.path.exists(html_path):
        print(f"Error: Could not find map_editor.html at {html_path}")
        sys.exit(1)

    # Convert to file:// URL
    file_url = 'file://' + os.path.abspath(html_path)

    print("""
    ==========================================
    CBFJAX Map Editor
    ==========================================

    Opening editor in your default web browser...

    If the browser doesn't open automatically,
    open this file manually:
    {}

    ==========================================
    """.format(html_path))

    # Open in default browser
    webbrowser.open(file_url)


if __name__ == '__main__':
    main()

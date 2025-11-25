import argparse
import os

import matplotlib.pyplot as plt

# Image name: "russia_map.png"

# Path to the base map image
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAP_PATH = os.path.join(BASE_DIR, "DataSet/KeyImages", "russia_map.png")


# These bounds roughly match the visible Russia area in the map image
MIN_LAT, MAX_LAT = 40.0, 82.0      # bottom, top
MIN_LON, MAX_LON = 20.0, 190.0     # left, right


def latlon_to_pixels(lat: float, lon: float, width: int, height: int):
    """
    Convert latitude/longitude to pixel coordinates on the map image,
    assuming a simple linear mapping between the defined bounds.
    """
    # Normalize lon/lat into [0, 1]
    x_norm = (lon - MIN_LON) / (MAX_LON - MIN_LON)
    y_norm = (MAX_LAT - lat) / (MAX_LAT - MIN_LAT)  # invert latitude for image y-axis

    # Convert to pixel coordinates
    x_px = x_norm * width
    y_px = y_norm * height
    return x_px, y_px


def plot_point_on_map(lat: float, lon: float, output_path: str):
    """
    Draw a SINGLE small red circle at (lat, lon) on the Russia map.
    """
    if not os.path.exists(MAP_PATH):
        raise FileNotFoundError(f"Map image not found at: {MAP_PATH}")

    img = plt.imread(MAP_PATH)
    height, width = img.shape[0], img.shape[1]

    # Convert coordinates
    x_px, y_px = latlon_to_pixels(lat, lon, width, height)

    # Create figure sized roughly to the image
    dpi = 100
    fig_width = width / dpi
    fig_height = height / dpi
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

    # Show map
    ax.imshow(img)
    ax.axis("off")

    # ---- SMALL RED DOT (for CV) ----
    # s controls size in points^2; tweak if Tevin wants bigger/smaller
    ax.scatter([x_px], [y_px], s=50, c="red")  # single solid red dot

    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Saved map with point at {lat}, {lon} -> {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot a single small red circle on the Russia map at given lat/lon."
    )
    parser.add_argument("--lat", type=float, required=True, help="Latitude in degrees")
    parser.add_argument("--lon", type=float, required=True, help="Longitude in degrees")
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/point.png",
        help="Output image path (PNG recommended)",
    )

    args = parser.parse_args()
    plot_point_on_map(args.lat, args.lon, args.out)


if __name__ == "__main__":
    main()

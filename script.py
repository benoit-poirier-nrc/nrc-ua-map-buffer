#!/usr/bin/env python3

import os
import sys
import logging
import requests
from datetime import datetime
import time
import shutil

import geopandas as gpd
import pandas as pd
from shapely.geometry import shape
from shapely.geometry import JOIN_STYLE
from shapely.ops import unary_union

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
API_URL = "https://deepstatemap.live/api/history/last"
RUSSIA_BORDER_FILE = "input/rus_adm0_osm_clipped.geojson"
UKRAINE_BORDER_FILE = "input/ukr_adm0_no_blacksea_osm.geojson"
OUTPUT_DIR = "data"
OUTPUT_LATEST_FILENAME = "buffer_zones_data_latest.geojson"
OUTPUT_TODAY_FILENAME = f"buffer_zones_data_{datetime.now().strftime('%Y%m%d')}.geojson"
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# Buffer distances (in meters)
BUFFER_DISTANCES = {
    "Critical": 9000,   # 9 km
    "High": 40000,      # 40 km
    "Moderate": 60000   # 60 km
}

def fetch_deepstatemap_data():
    """Fetches data from Deepstatemap API with retries."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows Phone 10.0; Android 6.0.1; Microsoft; RM-1152) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 "
            "Mobile Safari/537.36 Edge/15.15254"
        )
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(API_URL, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as exc:
            logger.warning(f"API request failed (attempt {attempt}/{MAX_RETRIES}): {exc}")
            if attempt < MAX_RETRIES:
                logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                logger.error("All API request attempts failed.")
                return None

def extract_occupied_territories(raw_data):
    """Creates a GeoDataFrame of occupied territories from Deepstatemap data."""
    try:
        features = raw_data.get("map", {}).get("features", [])
        territories = []
        for feature in features:
            name = feature["properties"]["name"].split("///")[1].strip()
            if name in {"CADR and CALR", "Occupied", "Occupied Crimea"}:
                geom = shape(feature["geometry"])
                if geom.is_valid:
                    territories.append(geom)
        return gpd.GeoDataFrame(geometry=territories, crs="EPSG:4326")
    except Exception as exc:
        logger.error(f"Error extracting occupied territories: {exc}")
        return None

def load_geojson_as_gdf(file_path):
    """Loads a GeoJSON file into a GeoDataFrame with error handling."""

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None
    try:
        gdf = gpd.read_file(file_path)
        if gdf.empty:
            logger.warning(f"GeoJSON file is empty: {file_path}")
            return None
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)
        return gdf
    except Exception as exc:
        logger.error(f"Failed to load GeoJSON file {file_path}: {exc}")
        return None

def merge_and_clean_geometries(occupied_gdf, russia_gdf):
    """Merges and cleans frontlines by applying buffers and removing artifacts."""

    try:
        occupied_gdf = occupied_gdf.to_crs(epsg=32637)
        russia_gdf = russia_gdf.to_crs(epsg=32637)

        merged_geometry = unary_union(
            occupied_gdf.geometry.tolist() + russia_gdf.geometry.tolist()
        ).buffer(400).buffer(-400)

        cleaned_geometry = merged_geometry.buffer(0.000009, resolution=1, join_style=JOIN_STYLE.mitre)\
            .buffer(-0.000009, resolution=1, join_style=JOIN_STYLE.mitre)

        return gpd.GeoDataFrame(geometry=[cleaned_geometry], crs="EPSG:32637").to_crs(epsg=4326)
    except Exception as exc:
        logger.error(f"Error merging geometries: {exc}")
        return None

def create_buffer_zones(merged_gdf, ukraine_gdf):
    """Creates buffer zones along the merged Russia-Occupied border, inside Ukraine."""

    try:
        buffer_layers = []
        merged_geom_utm = merged_gdf.to_crs(epsg=32637).geometry.iloc[0]
        ukraine_geom_utm = ukraine_gdf.to_crs(epsg=32637)

        # Buffer distances
        buffer_distances_sorted = [
            ("Critical", 0, BUFFER_DISTANCES["Critical"]),
            ("High", BUFFER_DISTANCES["Critical"], BUFFER_DISTANCES["High"]),
            ("Moderate", BUFFER_DISTANCES["High"], BUFFER_DISTANCES["Moderate"])
        ]

        for zone, inner, outer in buffer_distances_sorted:
            outer_buffer = merged_geom_utm.buffer(outer)
            inner_buffer = merged_geom_utm.buffer(inner)
            ring_buffer = outer_buffer.difference(inner_buffer)

            # Ensure buffers are only inside Ukraine
            ring_buffer_clipped = gpd.clip(
                gpd.GeoDataFrame(geometry=[ring_buffer], crs="EPSG:32637"),
                ukraine_geom_utm
            )
            ring_buffer_clipped["zone"] = zone
            buffer_layers.append(ring_buffer_clipped)

        return gpd.GeoDataFrame(pd.concat(buffer_layers, ignore_index=True), crs="EPSG:32637").to_crs(epsg=4326)
    except Exception as exc:
        logger.error(f"Error creating buffer zones: {exc}")
        return None

def main():
    """Main function that orchestrates data processing."""

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Fetch DeepStateMap data
    logger.info("Fetching Deepstatemap data...")
    raw_data = fetch_deepstatemap_data()
    if not raw_data:
        logger.error("Failed to retrieve Deepstatemap data.")
        sys.exit(1)

    # Extract occupied territories from DeepStateMap data
    occupied_gdf = extract_occupied_territories(raw_data)
    if occupied_gdf is None:
        logger.error("Extracted occupied territories data is empty.")
        sys.exit(1)

    # Load border files that will be used for cropping the buffer zones
    logger.info("Loading border data...")
    russia_gdf = load_geojson_as_gdf(RUSSIA_BORDER_FILE)
    ukraine_gdf = load_geojson_as_gdf(UKRAINE_BORDER_FILE)
    if russia_gdf is None or ukraine_gdf is None:
        logger.error("At leas one border data file could not be loaded.")
        sys.exit(1)

    # Merge and clean geometries
    logger.info("Merging and cleaning geometries...")
    merged_gdf = merge_and_clean_geometries(occupied_gdf, russia_gdf)
    if merged_gdf is None:
        logger.error("Failed to merge geometries.")
        sys.exit(1)
 


    # Create the buffer zones in Ukraine only
    logger.info("Creating buffer zones...")
    buffer_gdf = create_buffer_zones(merged_gdf, ukraine_gdf)
    if buffer_gdf is None:
        logger.error("Failed to create buffer zones.")
        sys.exit(1)

    # Save the results in a file that will be used for visualisation
    output_latest_path = os.path.join(OUTPUT_DIR, OUTPUT_LATEST_FILENAME)
    buffer_gdf.to_file(output_latest_path, driver="GeoJSON")

    # Duplicate file to keep a copy for each day
    output_today_path = os.path.join(OUTPUT_DIR, OUTPUT_TODAY_FILENAME)
    shutil.copyfile(output_latest_path, output_today_path)

    logger.info(f"Buffer zones saved")

if __name__ == "__main__":
    main()

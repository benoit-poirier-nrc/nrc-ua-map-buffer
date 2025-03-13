#!/usr/bin/env python3

import os
import sys
import logging
import time
import shutil
from datetime import datetime

import requests
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape
from shapely.geometry import JOIN_STYLE
from shapely.ops import unary_union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
API_URL = "https://deepstatemap.live/api/history/last"
RUSSIA_BORDER_FILE = "input/rus_adm0_osm_clipped.geojson"
UKRAINE_BORDER_FILE = "input/ukr_adm0_no_blacksea_buffer_osm.geojson"
OUTPUT_DIR_LAST = "data/last"
OUTPUT_DIR_ARCHIVE = "data/archive"
OUTPUT_LAST_FILENAME = "buffer_zones_data_last.geojson"
OUTPUT_ARCHIVE_FILENAME = f"buffer_zones_data_{datetime.now().strftime('%Y%m%d')}.geojson"
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
OCCUPIED_TERRITORIES = frozenset({"CADR and CALR", "Occupied", "Occupied Crimea"})

# Buffer distances (in meters)
BUFFER_DISTANCES = {
    "Critical": 9000,   # 9 km
    "High": 40000,      # 40 km
    "Moderate": 60000   # 60 km
}


def fetch_deepstatemap_data() -> dict:
    """
    Fetches data from the Deepstatemap API with retries.

    Returns:
        dict: A JSON document containing the API response.
              Example:
              {
                  "id": 1741301225,
                  "map": {
                      "type": "FeatureCollection",
                      "features": [
                          {
                              "type": "Feature",
                              "geometry": { ... },
                              "properties": { ... }
                          },
                          ...
                      ]
                  }
              }

    Notes:
        In case of failure, an empty dictionary is returned.
    """
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
                return {}
    return {}

def extract_occupied_territories(raw_data: dict) -> gpd.GeoDataFrame:
    """Creates a GeoDataFrame of occupied territories from Deepstatemap data."""
    try:
        features = raw_data.get("map", {}).get("features", [])
        territories = []
        for feature in features:
            properties = feature.get("properties", {})
            # Extract the name after splitting by '///'
            name_parts = properties.get("name", "").split("///")
            if len(name_parts) > 1:
                name = name_parts[1].strip()
                if name in OCCUPIED_TERRITORIES:
                    geom = shape(feature["geometry"])
                    if geom.is_valid:
                        territories.append(geom)
        return gpd.GeoDataFrame(geometry=territories, crs="EPSG:4326")
    except Exception as exc:
        logger.error(f"Error extracting occupied territories: {exc}")
        return gpd.GeoDataFrame()


def load_geojson_as_gdf(file_path: str) -> gpd.GeoDataFrame:
    """Loads a GeoJSON file into a GeoDataFrame with error handling."""
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return gpd.GeoDataFrame()
    try:
        gdf = gpd.read_file(file_path)
        if gdf.empty:
            logger.warning(f"GeoJSON file is empty: {file_path}")
            return gpd.GeoDataFrame()
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)
        return gdf
    except Exception as exc:
        logger.error(f"Failed to load GeoJSON file {file_path}: {exc}")
        return gpd.GeoDataFrame()


def merge_and_clean_geometries(occupied_gdf: gpd.GeoDataFrame,
                               russia_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Merges and cleans frontlines by applying buffers and removing artifacts."""
    try:
        occupied_utm = occupied_gdf.to_crs(epsg=32637)
        russia_utm = russia_gdf.to_crs(epsg=32637)

        # Merging the occupied territories and Russia border with a large buffer of 400 meters
        # The large buffer is needed to remove most gaps between the two geometries
        merged_geometry = unary_union(
            list(occupied_utm.geometry) + list(russia_utm.geometry)
        ).buffer(400).buffer(-400)

        # Remove artifacts from the merged geometry
        cleaned_geometry = (merged_geometry.buffer(0.000009, resolution=1, join_style=JOIN_STYLE.mitre)
                            .buffer(-0.000009, resolution=1, join_style=JOIN_STYLE.mitre))
        merged_gdf = gpd.GeoDataFrame(geometry=[cleaned_geometry], crs="EPSG:32637")
        
        return merged_gdf.to_crs(epsg=4326)
    except Exception as exc:
        logger.error(f"Error merging geometries: {exc}")
        return gpd.GeoDataFrame()


def create_buffer_zones(merged_gdf: gpd.GeoDataFrame,
                        ukraine_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Creates buffer zones along the merged Russia-Occupied border, inside Ukraine."""
    try:
        buffer_layers = []
        merged_utm = merged_gdf.to_crs(epsg=32637)
        merged_geom_utm = merged_utm.geometry.iloc[0]
        ukraine_utm = ukraine_gdf.to_crs(epsg=32637)

        buffer_definitions = [
            ("Critical", 0, BUFFER_DISTANCES["Critical"]),
            ("High", BUFFER_DISTANCES["Critical"], BUFFER_DISTANCES["High"]),
            ("Moderate", BUFFER_DISTANCES["High"], BUFFER_DISTANCES["Moderate"])
        ]

        for zone, inner, outer in buffer_definitions:
            outer_buffer = merged_geom_utm.buffer(outer)
            inner_buffer = merged_geom_utm.buffer(inner)
            ring_buffer = outer_buffer.difference(inner_buffer)

            # Ensure buffers are only inside Ukraine
            ring_buffer_gdf = gpd.GeoDataFrame(geometry=[ring_buffer], crs="EPSG:32637")
            ring_buffer_clipped = gpd.clip(ring_buffer_gdf, ukraine_utm)
            ring_buffer_clipped["zone"] = zone
            buffer_layers.append(ring_buffer_clipped)

        combined_buffers = gpd.GeoDataFrame(
            pd.concat(buffer_layers, ignore_index=True), crs="EPSG:32637"
        )
        return combined_buffers.to_crs(epsg=4326)
    except Exception as exc:
        logger.error(f"Error creating buffer zones: {exc}")
        return gpd.GeoDataFrame()


def main() -> None:
    """Main function that orchestrates data processing."""
    os.makedirs(OUTPUT_DIR_LAST, exist_ok=True)
    os.makedirs(OUTPUT_DIR_ARCHIVE, exist_ok=True)

    logger.info("Fetching Deepstatemap data...")
    raw_data = fetch_deepstatemap_data()
    if not raw_data:
        logger.error("Failed to retrieve Deepstatemap data.")
        sys.exit(1)

    occupied_gdf = extract_occupied_territories(raw_data)
    if occupied_gdf.empty:
        logger.error("Extracted occupied territories data is empty.")
        sys.exit(1)

    logger.info("Loading border data...")
    russia_gdf = load_geojson_as_gdf(RUSSIA_BORDER_FILE)
    ukraine_gdf = load_geojson_as_gdf(UKRAINE_BORDER_FILE)
    if russia_gdf.empty or ukraine_gdf.empty:
        logger.error("At least one border data file could not be loaded.")
        sys.exit(1)

    logger.info("Merging and cleaning geometries...")
    merged_gdf = merge_and_clean_geometries(occupied_gdf, russia_gdf)
    if merged_gdf.empty:
        logger.error("Failed to merge geometries.")
        sys.exit(1)

    logger.info("Creating buffer zones...")
    buffer_gdf = create_buffer_zones(merged_gdf, ukraine_gdf)
    if buffer_gdf.empty:
        logger.error("Failed to create buffer zones.")
        sys.exit(1)

    logger.info("Generating last geojson buffer file...")
    output_last_path = os.path.join(OUTPUT_DIR_LAST, OUTPUT_LAST_FILENAME)
    buffer_gdf.to_file(output_last_path, driver="GeoJSON")

    logger.info("Saving an archive of the geojson buffer file...")
    output_archive_path = os.path.join(OUTPUT_DIR_ARCHIVE, OUTPUT_ARCHIVE_FILENAME)
    shutil.copyfile(output_last_path, output_archive_path)

    logger.info("Buffer zones saved successfully.")


if __name__ == "__main__":
    main()

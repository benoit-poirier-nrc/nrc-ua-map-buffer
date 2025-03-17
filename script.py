#!/usr/bin/env python3

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys
import logging
import time
import shutil
from datetime import datetime

import requests
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape, CAP_STYLE, JOIN_STYLE
from shapely.ops import unary_union

@dataclass(frozen=True)
class Config:
    """Configuration parameters for buffer zone processing."""
    api_url: str = "https://deepstatemap.live/api/history/last"
    russia_border_file: Path = Path("input/rus_adm0_clipped_osm.geojson")
    ukraine_border_file: Path = Path("input/ukr_adm0_mask_osm.geojson")
    output_dir_last: Path = Path("data/last")
    output_dir_archive: Path = Path("data/archive")
    output_last_filename: str = "buffer_zones_data_last.geojson"
    max_retries: int = 3
    retry_delay: int = 5
    occupied_territories: frozenset = frozenset({"CADR and CALR", "Occupied", "Occupied Crimea"})
    buffer_distances: Dict[str, int] = {
        "Critical": 9000,    # 9 km
        "High": 40000,      # 40 km
        "Moderate": 80000   # 80 km
    }

    @property
    def output_archive_filename(self) -> str:
        """Generate archive filename with current date."""
        return f"buffer_zones_data_{datetime.now().strftime('%Y%m%d')}.geojson"

# Initialize configuration and logging
config = Config()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def fetch_deepstatemap_data() -> Dict[str, Any]:
    """Fetch data from the Deepstatemap API with retries."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows Phone 10.0; Android 6.0.1; Microsoft; RM-1152) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 "
            "Mobile Safari/537.36 Edge/15.15254"
        )
    }
    
    for attempt in range(1, config.max_retries + 1):
        try:
            response = requests.get(config.api_url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as exc:
            logger.warning(f"API request failed (attempt {attempt}/{config.max_retries}): {exc}")
            if attempt < config.max_retries:
                logger.info(f"Retrying in {config.retry_delay} seconds...")
                time.sleep(config.retry_delay)
    
    logger.error("All API request attempts failed.")
    return {}

def extract_occupied_territories(raw_data: Dict[str, Any]) -> gpd.GeoDataFrame:
    """Create a GeoDataFrame of occupied territories from Deepstatemap data."""
    try:
        features = raw_data.get("map", {}).get("features", [])
        territories = [
            shape(feature["geometry"])
            for feature in features
            if (name := feature.get("properties", {}).get("name", "").split("///")[1].strip())
            in config.occupied_territories
            and shape(feature["geometry"]).is_valid
        ]
        return gpd.GeoDataFrame(geometry=territories, crs="EPSG:4326")
    except Exception as exc:
        logger.error(f"Error extracting occupied territories: {exc}")
        return gpd.GeoDataFrame()

def load_geojson_as_gdf(file_path: Path) -> gpd.GeoDataFrame:
    """Load a GeoJSON file into a GeoDataFrame with error handling."""
    if not file_path.exists():
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
    """Merge and clean frontlines by applying buffers and removing artifacts."""
    try:
        logger.info(f"Input geometries - Occupied: {len(occupied_gdf)}, Russia: {len(russia_gdf)}")
        occupied_utm = occupied_gdf.to_crs(epsg=32637)
        russia_utm = russia_gdf.to_crs(epsg=32637)

        merged_geometry = unary_union(
            list(occupied_utm.geometry) + list(russia_utm.geometry)
        ).buffer(400).buffer(-400)

        logger.info("Cleaning geometry with fine adjustment buffers")
        cleaned_geometry = (merged_geometry
                          .buffer(0.000009, join_style=JOIN_STYLE.mitre)
                          .buffer(-0.000009, join_style=JOIN_STYLE.mitre))
        
        merged_gdf = gpd.GeoDataFrame(geometry=[cleaned_geometry], crs="EPSG:32637")
        result = merged_gdf.to_crs(epsg=4326)
        
        logger.info(f"Successfully merged and cleaned geometries: {len(result)} features")
        return result
    except Exception as exc:
        logger.error(f"Error merging geometries: {exc}", exc_info=True)
        return gpd.GeoDataFrame()

def create_buffer_zones(merged_gdf: gpd.GeoDataFrame,
                       ukraine_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Create buffer zones along the merged Russia-Occupied border, inside Ukraine."""
    try:
        buffer_layers = []
        merged_utm = merged_gdf.to_crs(epsg=32637)
        merged_geom_utm = merged_utm.geometry.iloc[0]
        ukraine_utm = ukraine_gdf.to_crs(epsg=32637)

        buffer_definitions = [
            ("Critical", 0, config.buffer_distances["Critical"]),
            ("High", config.buffer_distances["Critical"], config.buffer_distances["High"]),
            ("Moderate", config.buffer_distances["High"], config.buffer_distances["Moderate"])
        ]

        for zone, inner, outer in buffer_definitions:
            outer_buffer = merged_geom_utm.buffer(
                outer,
                cap_style=CAP_STYLE.round,
                join_style=JOIN_STYLE.mitre,
                mitre_limit=2.0
            )
            inner_buffer = merged_geom_utm.buffer(
                inner,
                cap_style=CAP_STYLE.round,
                join_style=JOIN_STYLE.mitre,
                mitre_limit=2.0
            )
            ring_buffer = outer_buffer.difference(inner_buffer)

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
    """Process and generate buffer zones around occupied territories."""
    config.output_dir_last.mkdir(exist_ok=True)
    config.output_dir_archive.mkdir(exist_ok=True)

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
    russia_gdf = load_geojson_as_gdf(config.russia_border_file)
    ukraine_gdf = load_geojson_as_gdf(config.ukraine_border_file)
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
    output_last_path = config.output_dir_last / config.output_last_filename
    buffer_gdf.to_file(output_last_path, driver="GeoJSON")

    logger.info("Saving an archive of the geojson buffer file...")
    output_archive_path = config.output_dir_archive / config.output_archive_filename
    shutil.copyfile(output_last_path, output_archive_path)

    logger.info("Buffer zones saved successfully.")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("Fatal error occurred")
        sys.exit(1)
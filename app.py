import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sentinelhub import SentinelHubRequest, DataCollection, bbox_to_dimensions, MimeType, BBox, CRS, SHConfig
import cv2
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import io

# Sentinel Hub Configuration
config = SHConfig()
config.instance_id = '1b15053c-957c-431a-bbbe-c2ddeb71f099'
config.sh_client_id = 'b653dfc8-9935-4448-bfe4-b3b85a270611'
config.sh_client_secret = 'zfLAJoeNqibAnEbZ4Tib9qnTU4JV7ClQ'
config.save()

# Streamlit Interface

# Sidebar for User Inputs
st.sidebar.title('Satellite Image Analysis for Water and Plastic Segmentation')
resolution = st.sidebar.slider('Resolution (in meters)', 10, 1000, 100)

# Region of Interest Input
lat_min = st.sidebar.number_input('Min Latitude', value=15.0)
lat_max = st.sidebar.number_input('Max Latitude', value=16.0)
lon_min = st.sidebar.number_input('Min Longitude', value=72.7)
lon_max = st.sidebar.number_input('Max Longitude', value=74.7)

# Convert to Bounding Box
bbox = BBox(bbox=[lon_min, lat_min, lon_max, lat_max], crs=CRS.WGS84)


def fetch_image(bbox, resolution):
    """Fetches satellite image for the specified bounding box and resolution."""
    try:
        request = SentinelHubRequest(
            evalscript=""" 
                //VERSION=3 
                function setup() {
                    return {
                        input: ["B04", "B03", "B02", "B08"],
                        output: { bands: 4 }
                    };
                }
                function evaluatePixel(sample) {
                    return [sample.B04, sample.B03, sample.B02, sample.B08];
                }
            """,
            input_data=[SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=('2023-01-01', '2023-02-01'),
                mosaicking_order='mostRecent'
            )],
            bbox=bbox,
            size=bbox_to_dimensions(bbox, resolution),
            responses=[SentinelHubRequest.output_response(
                'default', MimeType.PNG)],
            config=config
        )
        image = request.get_data()[0]
        return image
    except Exception as e:
        st.error(f"Error fetching image: {e}")
        return None

# Functions for Image Processing


def preprocess_image(image):
    """Applies preprocessing steps like blurring and contrast enhancement."""
    if image is None:
        raise ValueError("Image is None. Cannot preprocess.")
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    enhanced = cv2.convertScaleAbs(blurred, alpha=1.5, beta=0)
    return enhanced


def calculate_ndwi(image):
    """Calculate Normalized Difference Water Index (NDWI) for water detection."""
    # Extract Green (B03) and NIR (B08) bands
    green_band = image[..., 1]  # Green band (B03)
    nir_band = image[..., 3]    # NIR band (B08)

    # Calculate NDWI
    ndwi = (green_band - nir_band) / (green_band +
                                      nir_band + 1e-5)  # Avoid division by zero
    return ndwi


def segment_water(ndwi, threshold=0.3):
    """Segment water areas using NDWI threshold."""
    # Apply threshold to NDWI to create a binary mask
    water_mask = (ndwi > threshold).astype(np.uint8) * 255

    # Morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)

    return water_mask


def segment_plastic(image, lower_bound, upper_bound):
    """Segments the image based on HSV color bounds for plastic detection."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    surface_area = sum(cv2.contourArea(c) for c in contours)
    return mask, contours, surface_area


def highlight_segments(image, mask, color):
    """Highlights segmented areas on the original image."""
    highlighted = image.copy()

    # Check if the image has an alpha channel (i.e., RGBA)
    if highlighted.shape[2] == 4:
        # Discard alpha channel for highlighting
        highlighted = highlighted[:, :, :3]

    # Apply the color to the highlighted regions (where mask > 0)
    highlighted[mask > 0] = color
    return highlighted


def convert_pixels_to_sq_km(pixel_area, resolution_meters):
    """
    Converts the pixel area to square kilometers based on resolution.

    Args:
        pixel_area (int): The number of pixels in the mask.
        resolution_meters (float): The resolution of the image in meters (per pixel).

    Returns:
        float: The area in square kilometers.
    """
    # Calculate pixel area in square meters
    pixel_area_m2 = pixel_area * (resolution_meters ** 2)

    # Convert square meters to square kilometers
    pixel_area_km2 = pixel_area_m2 / 1_000_000

    return pixel_area_km2


def generate_report(water_area, plastic_area, original_image, preprocessed_image, water_mask, plastic_mask, water_area_km2, plastic_area_km2):
    """Generates a comprehensive PDF report with all relevant details."""
    data = {
        "Metric": ["Water Surface Area", "Plastic Surface Area"],
        "Value (pixels)": [np.sum(water_mask > 0), plastic_area],
        "Value (sq km)": [water_area_km2, plastic_area_km2]
    }
    metrics_df = pd.DataFrame(data)

    # Create a BytesIO buffer to save the PDF in memory
    pdf_buffer = io.BytesIO()

    with PdfPages(pdf_buffer) as pdf:
        # Title Page
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.7, 'Satellite Image Analysis Report',
                 fontsize=18, ha='center')
        plt.text(0.5, 0.5, 'Water and Plastic Waste Detection',
                 fontsize=12, ha='center')
        plt.axis('off')
        pdf.savefig()
        plt.close()

        # Original Image
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(original_image)
        ax.set_title('Original Image')
        ax.axis('off')
        pdf.savefig()
        plt.close()

        # Preprocessed Image
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(preprocessed_image)
        ax.set_title('Preprocessed Image')
        ax.axis('off')
        pdf.savefig()
        plt.close()

        # # NDWI Calculation Image
        # fig, ax = plt.subplots(figsize=(8, 6))
        # ax.imshow(calculate_ndwi(preprocessed_image), cmap='viridis')
        # ax.set_title('NDWI Calculation')
        # ax.axis('off')
        # pdf.savefig()
        # plt.close()

        # Water Mask Image
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(water_mask, cmap="Blues")
        ax.set_title('Water Mask')
        ax.axis('off')
        pdf.savefig()
        plt.close()

        # Plastic Mask Image
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(plastic_mask, cmap="Reds")
        ax.set_title('Plastic Mask')
        ax.axis('off')
        pdf.savefig()
        plt.close()

        # Metrics Page (Table)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('tight')
        ax.axis('off')
        ax.table(cellText=metrics_df.values,
                 colLabels=metrics_df.columns, loc='center')
        pdf.savefig()
        plt.close()

        # Highlighted Water Areas
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(highlight_segments(original_image, water_mask, [0, 255, 0]))
        ax.set_title('Highlighted Water Areas')
        ax.axis('off')
        pdf.savefig()
        plt.close()

        # Highlighted Plastic Areas
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(highlight_segments(
            original_image, plastic_mask, [255, 0, 0]))
        ax.set_title('Highlighted Plastic Areas')
        ax.axis('off')
        pdf.savefig()
        plt.close()

    pdf_buffer.seek(0)  # Rewind buffer to the beginning
    return pdf_buffer


# Fetch Image
st.title('Fetch and Process Satellite Image')
original_image = fetch_image(bbox, resolution)

if original_image is None:
    st.error("Failed to fetch image. Try adjusting the region or resolution.")
else:
    st.image(original_image, caption="Original Satellite Image")

    # Preprocess Image
    st.title('Image Preprocessing')
    preprocessed_image = preprocess_image(original_image)

    st.image(preprocessed_image, caption="Preprocessed Image")

    # Water Detection using NDWI
    st.title('Water Detection')
    ndwi = calculate_ndwi(preprocessed_image)
    water_mask = segment_water(ndwi, threshold=0.3)

    st.image(water_mask, caption="Water Mask", use_column_width=True)

    # Plastic Detection
    st.title('Plastic Detection')
    plastic_mask, _, plastic_area = segment_plastic(
        preprocessed_image, lower_bound=np.array([0, 0, 200]), upper_bound=np.array([180, 30, 255])
    )
    st.image(plastic_mask, caption="Plastic Mask", use_column_width=True)

    # Convert pixel area to square kilometers
    water_pixel_count = np.sum(water_mask > 0)
    water_area_km2 = convert_pixels_to_sq_km(water_pixel_count, resolution)
    plastic_area_km2 = convert_pixels_to_sq_km(plastic_area, resolution)

    # Highlighted Water and Plastic Areas
    st.title('Highlighted Areas')
    st.image(highlight_segments(original_image, water_mask, [
             0, 255, 0]), caption="Highlighted Water Areas", use_column_width=True)
    st.image(highlight_segments(original_image, plastic_mask, [
             255, 0, 0]), caption="Highlighted Plastic Areas", use_column_width=True)

    # Display Areas
    st.subheader('Detected Areas')
    st.write(f"Water Surface Area: {water_area_km2:.6f} square kilometers")
    st.write(f"Plastic Surface Area: {plastic_area_km2:.6f} square kilometers")

    # Option to Download Report
    st.title('Download Report')
    if st.button("Generate Report"):
        pdf_buffer = generate_report(
            water_area=water_area_km2,
            plastic_area=plastic_area_km2,
            original_image=original_image,
            preprocessed_image=preprocessed_image,
            water_mask=water_mask,
            plastic_mask=plastic_mask,
            water_area_km2=water_area_km2,
            plastic_area_km2=plastic_area_km2
        )
        st.download_button(
            label="Download Report",
            data=pdf_buffer,
            file_name="detection_report.pdf",
            mime="application/pdf"
        )

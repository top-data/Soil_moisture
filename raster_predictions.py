import gdal, osr
import numpy as np
import pandas as pd

# Function to save raster
def save_raster(input, array, output_file):
    """Save raster data to a file.

    Parameters:
    input (str): Path to the input raster file.
    array (np.array): Array containing raster data.
    output_file (str): Path to the output raster file.

    Returns:
    None
    """

    # Open the input raster file
    raster = gdal.Open(input)
    # Get the geotransform information and projection
    geo = raster.GetGeoTransform()
    wkt = raster.GetProjection()
    band = raster.GetRasterBand(1)
    driver = gdal.GetDriverByName("GTiff")

    # Create a new raster file with the same size as the input
    dst_ds = driver.Create(output_file,
                           band.XSize,
                           band.YSize,
                           1,
                           gdal.GDT_Float32)

    # Write the array data to the raster file
    dst_ds.GetRasterBand(1).WriteArray(array)
    # Set NoData value
    dst_ds.GetRasterBand(1).SetNoDataValue(-999)
    # Set the geotransform and projection
    dst_ds.SetGeoTransform(geo)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    dst_ds.SetProjection(srs.ExportToWkt())
    # Close the raster files
    ds = None
    dst_ds = None

# Function to create and save raster predictions
def raster_predictions(file_path, output_path, dataframe, inputs, model):
    '''
    Creates and saves raster of predictions
    Inputs:
        file_path: string of inputs path
        file_path: string of output path
        dataframe: pandas dataframe with predictors
        inputs: list of covariate names
        model: trained model
    Outputs:
        saves raster
    '''
    # Open the input raster file
    raster = gdal.Open(file_path)
    # Read raster data as array
    array = raster.ReadAsArray()[:, :, :]
    # Normalize raster data based on min-max scaling of input data
    mins = dataframe[inputs].min().values
    maxs = dataframe[inputs].max().values
    normalised = ((array - mins.reshape((mins.shape[0], 1, 1)))/(maxs.reshape((maxs.shape[0], 1, 1)) - mins.reshape((mins.shape[0], 1, 1))))
    container = []
    # Iterate through each band in the raster
    for n in range(array.shape[2]):
        # Predict using the model for each band and flatten the results
        container.append(model.predict(np.swapaxes(normalised[:,:,n], 1, 0))[:, 0].ravel())
    # Save the raster with predicted values
    save_raster(file_path,
                np.swapaxes(np.vstack(container), 1, 0),
                output_path)

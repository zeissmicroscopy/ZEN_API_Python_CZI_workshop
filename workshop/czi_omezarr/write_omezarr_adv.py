from czitools.read_tools import read_tools
from pathlib import Path
from typing import Optional
import sys

# Add the workshop directory to the Python path BEFORE importing from utils
workshop_dir = Path(__file__).parent.parent
sys.path.insert(0, str(workshop_dir))

# Now import from utils (after sys.path is modified)
from utils.ome_zarr_utils import write_omezarr

# Point to the main data folder (two directories up from demo/omezarr_testing)
#filepath: str = str(Path(__file__).parent.parent.parent / "workshop" / "data" / "CellDivision_T10_Z15_CH2_DCV_small.czi")
filepath: str = str(Path(__file__).parent.parent.parent / "workshop" / "data" / "WP96_4Pos_B4-10_DAPI.czi")

show_napari: bool = True  # Whether to display the result in napari viewer
scene_id: int = 0

# Read the CZI file and return a 6D array with dimension order STCZYX(A)
array, mdata = read_tools.read_6darray(filepath, use_xarray=True)
array = array[scene_id, ...]
zarr_path: Path = Path(str(filepath)[:-4] + ".ome.zarr")

print(f"Array Type: {type(array)}, Shape: {array.shape}, Dtype: {array.dtype}")

# Write OME-ZARR using utility function
result_zarr_path: Optional[str] = write_omezarr(array, zarr_path=str(zarr_path), metadata=mdata, overwrite=True)
print(f"Written OME-ZARR using ome-zarr-py: {result_zarr_path}")

# Optional: Visualize the plate data using napari
if show_napari:
    import napari

    viewer: napari.Viewer = napari.Viewer()
    viewer.open(result_zarr_path, plugin="napari-ome-zarr")
    napari.run()

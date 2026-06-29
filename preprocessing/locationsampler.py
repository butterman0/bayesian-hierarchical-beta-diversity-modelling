import numpy as np
import xarray as xr
import random
from scipy.ndimage import label
import arviz as az
from matplotlib.path import Path
import warnings

class LocationSampler:
    def __init__(self, config):
        self.config = config

    def sample(self, n=None, exclude=None, random_seed=None, area=None):
        """
        Create a DataFrame of sampled locations by randomly selecting valid (xc, yc) coordinates and assigning a time index.

        Uses:
            self.config.predictor_data_path: Path to SINMOD dataset (biomod).
            self.config.train_size: Number of sites to sample.
            self.config.time_idx: Time index or list of indices to assign.
            self.config.random_seed: Random seed for reproducibility (optional).

        Returns:
            dict: Dictionary of DataFrames for each time index if multiple are specified.
                  Keys are time indices; values are DataFrames.
        """
        no_sites = self.config.train_size if n is None else n
        self.config.no_sites = n
        time_indices = self.config.time_idx
        random_seed = self.config.random_seed if random_seed is None else random_seed

        # Open the SINMOD dataset
        with xr.open_dataset(self.config.predictor_data_path).isel(time=self.config.time_idx[0]) as ds:
        
            valid_mask = ds.vorticity.notnull()
            
            structure = np.ones((3, 3), dtype=bool)  # 8-connected neighbours
            connected_labels, num_labels = label(valid_mask, structure=structure)
            
            # Count pixels in each connected region (excluding 0 = land)
            counts = np.bincount(connected_labels.ravel())
            counts[0] = 0  # ignore land label      

            # Find the label of the largest ocean region
            largest_label = np.argmax(counts)
            
            # Update valid_mask to only include the largest regions
            valid_mask.values = connected_labels == largest_label
            
            # TODO Incorporate buffer zone into config
            n_buffer = 10

            interior_mask = xr.zeros_like(valid_mask, dtype=bool)
            interior_mask.values[n_buffer:-n_buffer, n_buffer:-n_buffer] = True

            if self.config.location_buffer:
                print("Applying location buffer to valid mask.")
                print(valid_mask.sum().values, "valid locations before applying buffer.")
                valid_mask = valid_mask & interior_mask
                print(valid_mask.sum().values, "valid locations after applying buffer.")
            
            # Further restrict valid locations to those with non-null values for all predictor variables
            if self.config.predictor_variables is not None:
                for var in self.config.predictor_variables:
                    
                    if var == 'vorticity':
                        continue
                    
                    # TODO Handle this technical debt. Some variables are missing from the dataset. Shouldn't all predictor be in same dataset? They are actually in the response in the case of SINMOD....
                    if var not in ds:
                        warnings.warn(f"Variable {var} not found in predictor dataset. Skipping.")
                        continue

                    if np.any(valid_mask.values & ds[var].isnull().values):
                        print(f"Updating valid mask with variable {var}")
                        
                        sum = np.sum(valid_mask.values != ds[var].notnull().values)
                        print(sum)
                        
                        # set valid_mask to null (NaN) where the predictor is null
                        valid_mask = valid_mask & ds[var].notnull()
                        
                        print(valid_mask.sum())

            if area is not None:
                
                polygon_latlon = self.config.NAMED_AREAS.get(area, None)
                                
                # Load 2D lat/lon arrays
                ds = xr.open_dataset(self.config.latlonpath)
                lats = ds.gridLats.isel(xc=slice(None, -1), yc=slice(None, -1)).values
                lons = ds.gridLons.isel(xc=slice(None, -1), yc=slice(None, -1)).values

                # Get the xc and yc indices for the corner points
                corner_indices = np.array([
                    np.unravel_index(np.argmin((lats - lat)**2 + (lons - lon)**2), lats.shape)
                    for lat, lon in polygon_latlon
                ])
                
                # Swap columns in corner_indices to get (yc, xc) instead of (xc, yc)
                corner_indices = corner_indices[:, ::-1]
                
                # Convert polygon to Path object
                polygon_path = Path(corner_indices*800)  # expects list of (lat, lon) pairs
                
                # Filter valid_coordinates to only those that are inside the polygon
                valid_mask = polygon_path.contains_points(valid_coordinates)
                valid_coordinates = valid_coordinates[valid_mask]

                print(valid_coordinates.shape, "valid coordinates found in the specified area.")
        
        # Further restrict to all valid locations for the response variable
        with xr.open_dataset(self.config.response_data_path).isel(time=self.config.time_idx[0]) as ds:
            
            for var in self.config.response_variables:

                if np.any(valid_mask.values & ds[var].isnull().values):
                    print(f"Updating valid mask with response variable {var}")
                    valid_mask.values = valid_mask.values & ds[var].notnull().values
        
        # Stack dimensions into pairs and filter valid locations
        stacked = valid_mask.stack(z=('xc', 'yc'))
        valid_coordinates = np.stack(stacked[stacked.values].z.values).astype(np.float32)
        
        print(valid_coordinates.shape, "valid coordinates found in the dataset.")
        
        

        if exclude is not None:
            # Exclude specified coordinates
            exclude_set = set(map(tuple, exclude[:, :2]))  # Convert to set of tuples for fast lookup
            valid_set = set(map(tuple, valid_coordinates))
            
            # Filter out excluded coordinates
            filtered_set = list(valid_set - exclude_set)
            
            if len(filtered_set) < no_sites:
                raise ValueError("Not enough valid coordinates left after applying exclusions.")

            valid_coordinates = np.array(filtered_set)
            
            print(valid_coordinates.shape, "valid coordinates found after excluding specified locations.")
    
        # Check the number of valid coordinates
        if no_sites > len(valid_coordinates):
            raise ValueError("Number of sites exceeds the available valid coordinates")

        # Random seed for reproducibility, this will select the same locations each time
        random.seed(random_seed)

        # Handle single or multiple time indices
        if isinstance(time_indices, int):
            time_indices = [time_indices]

        sampled_data = []
        if self.config.randomise_sites:
            for time_index in time_indices:
                # random.seed(hash((random_seed, time_index)))
                sampled_coords = random.sample(valid_coordinates.tolist(), no_sites // len(time_indices))
                sampled_array = np.array([[x, y, time_index] for x, y in sampled_coords])
                sampled_data.append(sampled_array)

            # random.seed(random_seed)
        else:
            sampled_coords = random.sample(valid_coordinates.tolist(), no_sites // len(time_indices))
            for time_index in time_indices:
                sampled_array = np.array([[x, y, time_index] for x, y in sampled_coords])
                sampled_data.append(sampled_array)
            
        return np.vstack(sampled_data)
"""
Data loading utilities for battery datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import warnings
import zipfile
from scipy.io import loadmat
import os
warnings.filterwarnings('ignore')


class BatteryDataLoader:
    """Load and prepare battery degradation data"""
    
    def __init__(self, data_path: str):
        """
        Initialize data loader
        
        Args:
            data_path: Path to the data directory
        """
        self.data_path = Path(data_path)
        self.data = None
        
    def extract_zip_files(self) -> Path:
        """
        Extract all ZIP files in the data directory
        
        Returns:
            Path to extracted folder
        """
        extracted_path = self.data_path / "extracted"
        extracted_path.mkdir(exist_ok=True)
        
        zip_files = list(self.data_path.glob("*.zip"))
        if not zip_files:
            return extracted_path
        
        print(f"Found {len(zip_files)} ZIP file(s). Extracting...")
        for zip_file in zip_files:
            try:
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(extracted_path)
                print(f"  Extracted: {zip_file.name}")
            except Exception as e:
                print(f"  Warning: Could not extract {zip_file.name}: {e}")
        
        return extracted_path
    
    def load_nasa_mat_file(self, mat_path: Path, battery_id: str) -> pd.DataFrame:
        """
        Load a single NASA .mat file and convert to DataFrame
        
        Args:
            mat_path: Path to .mat file
            battery_id: Battery identifier
            
        Returns:
            DataFrame with battery cycle data
        """
        try:
            mat_data = loadmat(str(mat_path), simplify_cells=False)
            
            # The structure is: mat_data[battery_id] -> structured array with 'cycle' field
            battery_key = battery_id
            if battery_key not in mat_data:
                # Try to find the key (case insensitive)
                for key in mat_data.keys():
                    if not key.startswith('__') and key.upper() == battery_key.upper():
                        battery_key = key
                        break
            
            if battery_key not in mat_data:
                print(f"  Warning: Could not find battery key in {mat_path}")
                return pd.DataFrame()
            
            battery_struct = mat_data[battery_key]
            
            # Access cycles from structured array
            if hasattr(battery_struct, 'dtype') and battery_struct.dtype.names and 'cycle' in battery_struct.dtype.names:
                cycles_array = battery_struct['cycle']
                # Extract the actual cycles - handle nested structure
                if cycles_array.size == 1:
                    cycles = cycles_array.item()
                    # Might be nested - check if it's still an array
                    if isinstance(cycles, np.ndarray):
                        if cycles.size == 1:
                            cycles = cycles.item()
                        else:
                            cycles = cycles.flatten()
                else:
                    # Flatten to get all cycles
                    cycles = cycles_array.flatten()
            else:
                print(f"  Warning: Unexpected structure in {mat_path}")
                return pd.DataFrame()
            
            # Ensure cycles is iterable numpy array
            if not isinstance(cycles, np.ndarray):
                if isinstance(cycles, (list, tuple)):
                    cycles = np.array(cycles)
                else:
                    cycles = np.array([cycles])
            
            data = []
            initial_capacity = None
            capacities = []
            discharge_count = 0
            
            # Debug: check cycle count
            if len(cycles) == 0:
                print(f"    Warning: No cycles found in {battery_id}")
                return pd.DataFrame()
            
            for cycle_idx, cycle in enumerate(cycles):
                # Access structured array fields
                cycle_type = ''
                cycle_data = None
                
                if hasattr(cycle, 'dtype') and cycle.dtype.names:
                    # Get type field
                    if 'type' in cycle.dtype.names:
                        type_val = cycle['type']
                        if isinstance(type_val, np.ndarray):
                            if type_val.size > 0:
                                cycle_type = str(type_val.item() if type_val.size == 1 else type_val[0])
                            else:
                                cycle_type = ''
                        else:
                            cycle_type = str(type_val)
                    
                    # Get data field
                    if 'data' in cycle.dtype.names:
                        cycle_data = cycle['data']
                        if isinstance(cycle_data, np.ndarray) and cycle_data.size == 1:
                            cycle_data = cycle_data.item()
                else:
                    # Try dictionary-like access
                    cycle_type = cycle.get('type', '') if hasattr(cycle, 'get') else ''
                    cycle_data = cycle.get('data', {}) if hasattr(cycle, 'get') else {}
                
                # Focus on discharge cycles (they have capacity data)
                if cycle_type.lower() == 'discharge' and cycle_data is not None:
                    discharge_count += 1
                    # Access data fields - cycle_data is a tuple: (Voltage, Current, Temp, Current_charge, Voltage_charge, Time, Capacity)
                    if isinstance(cycle_data, tuple) and len(cycle_data) >= 7:
                        # Tuple structure: [0]=Voltage_measured, [1]=Current_measured, [2]=Temperature_measured, 
                        # [3]=Current_charge, [4]=Voltage_charge, [5]=Time, [6]=Capacity
                        voltage_data = cycle_data[0]  # Voltage_measured
                        current_data = cycle_data[1]  # Current_measured
                        temp_data = cycle_data[2]     # Temperature_measured
                        capacity = cycle_data[6]      # Capacity (last item)
                    elif hasattr(cycle_data, 'dtype') and cycle_data.dtype.names:
                        # Structured array access
                        capacity = cycle_data['Capacity'].item() if 'Capacity' in cycle_data.dtype.names else None
                        voltage_data = cycle_data['Voltage_measured'].item() if 'Voltage_measured' in cycle_data.dtype.names else np.array([3.7])
                        current_data = cycle_data['Current_measured'].item() if 'Current_measured' in cycle_data.dtype.names else np.array([0])
                        temp_data = cycle_data['Temperature_measured'].item() if 'Temperature_measured' in cycle_data.dtype.names else np.array([25])
                    else:
                        # Dictionary-like access
                        capacity = cycle_data.get('Capacity', None) if hasattr(cycle_data, 'get') else None
                        voltage_data = cycle_data.get('Voltage_measured', np.array([3.7])) if hasattr(cycle_data, 'get') else np.array([3.7])
                        current_data = cycle_data.get('Current_measured', np.array([0])) if hasattr(cycle_data, 'get') else np.array([0])
                        temp_data = cycle_data.get('Temperature_measured', np.array([25])) if hasattr(cycle_data, 'get') else np.array([25])
                    
                    if capacity is not None:
                        # Handle capacity - could be array or scalar
                        if isinstance(capacity, np.ndarray):
                            if len(capacity) > 0:
                                capacity = float(capacity[-1])
                            else:
                                continue
                        else:
                            capacity = float(capacity)
                        
                        capacities.append(capacity)
                        
                        if initial_capacity is None:
                            initial_capacity = capacity
                        
                        # Get mean values for voltage, current, temperature
                        voltage = float(np.mean(voltage_data)) if isinstance(voltage_data, np.ndarray) else float(voltage_data)
                        current = float(np.mean(current_data)) if isinstance(current_data, np.ndarray) else float(current_data)
                        temperature = float(np.mean(temp_data)) if isinstance(temp_data, np.ndarray) else float(temp_data)
                        
                        # Calculate SOH (State of Health)
                        soh = capacity / initial_capacity if initial_capacity > 0 else 1.0
                        
                        # Calculate degradation rate (change in capacity)
                        if len(capacities) > 1:
                            degradation_rate = (capacities[-2] - capacity) / initial_capacity
                        else:
                            degradation_rate = 0.0
                        
                        # Estimate range (km) - simplified calculation
                        efficiency = 0.15  # kWh/km (typical EV efficiency)
                        energy_kwh = (capacity * 3.7) / 1000  # Convert Ah*V to kWh
                        range_km = energy_kwh / efficiency
                        
                        data.append({
                            'battery_id': battery_id,
                            'cycle': cycle_idx,
                            'voltage': voltage,
                            'current': current,
                            'temperature': temperature,
                            'capacity': capacity,
                            'soh': soh,
                            'range_km': range_km,
                            'degradation_rate': abs(degradation_rate)
                        })
            
            if len(data) > 0:
                print(f"    Loaded {len(data)} discharge cycles from {battery_id}")
            elif discharge_count > 0:
                print(f"    Warning: Found {discharge_count} discharge cycles but extracted 0 records from {battery_id}")
            return pd.DataFrame(data)
            
        except Exception as e:
            print(f"Error loading {mat_path}: {e}")
            return pd.DataFrame()
    
    def load_nasa_dataset(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load NASA battery dataset from .mat files
        
        Args:
            file_path: Optional path override (not used for .mat files)
            
        Returns:
            DataFrame with battery data
        """
        # First, try to extract ZIP files
        extracted_path = self.extract_zip_files()
        
        # Find all .mat files
        mat_files = list(extracted_path.rglob("*.mat"))
        
        if not mat_files:
            # Try direct .mat files in raw folder
            mat_files = list(self.data_path.glob("*.mat"))
        
        if not mat_files:
            print("No NASA .mat files found. Generating simulated data...")
            return self.generate_simulated_data()
        
        print(f"Found {len(mat_files)} .mat file(s). Loading...")
        
        all_data = []
        for mat_file in mat_files:
            battery_id = mat_file.stem  # e.g., "B0005"
            print(f"  Loading: {battery_id}")
            df_battery = self.load_nasa_mat_file(mat_file, battery_id)
            if not df_battery.empty:
                all_data.append(df_battery)
        
        if not all_data:
            print("No valid data loaded. Generating simulated data...")
            return self.generate_simulated_data()
        
        df = pd.concat(all_data, ignore_index=True)
        print(f"Successfully loaded {len(df)} records from NASA dataset")
        return df
    
    def generate_simulated_data(self, n_cycles: int = 1000, n_batteries: int = 5) -> pd.DataFrame:
        """
        Generate simulated battery degradation data
        
        Args:
            n_cycles: Number of charge cycles
            n_batteries: Number of battery cells
            
        Returns:
            Simulated battery dataset
        """
        np.random.seed(42)
        data = []
        
        for battery_id in range(n_batteries):
            # Initial capacity (Ah)
            initial_capacity = np.random.uniform(1.8, 2.2)
            
            # Degradation parameters
            degradation_rate = np.random.uniform(0.0001, 0.0003)
            noise_level = 0.02
            
            for cycle in range(n_cycles):
                # Calculate SOH (State of Health)
                soh = 1.0 - (cycle * degradation_rate) + np.random.normal(0, noise_level)
                soh = max(0.3, min(1.0, soh))  # Clamp between 30% and 100%
                
                # Current capacity
                capacity = initial_capacity * soh
                
                # Voltage (decreases with degradation)
                voltage = np.random.uniform(3.2, 4.2) * soh
                
                # Current (varies with cycle)
                current = np.random.uniform(-2.0, 2.0)
                
                # Temperature (increases with degradation)
                temperature = 25 + (1 - soh) * 15 + np.random.normal(0, 2)
                
                # Range estimation (km) - based on capacity and efficiency
                efficiency = 0.85  # kWh/km
                range_km = (capacity * voltage / 1000) / efficiency * 100
                
                # Degradation rate (change in SOH per cycle)
                degradation_rate_cycle = degradation_rate + np.random.normal(0, 0.00001)
                
                data.append({
                    'battery_id': battery_id,
                    'cycle': cycle,
                    'voltage': voltage,
                    'current': current,
                    'temperature': temperature,
                    'capacity': capacity,
                    'soh': soh,
                    'range_km': range_km,
                    'degradation_rate': degradation_rate_cycle
                })
        
        df = pd.DataFrame(data)
        return df
    
    def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Main method to load data (tries NASA, falls back to simulated)
        
        Args:
            file_path: Optional path to data file
            
        Returns:
            Battery dataset DataFrame
        """
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
            except:
                self.data = self.generate_simulated_data()
        else:
            self.data = self.load_nasa_dataset()
        
        return self.data
    
    def get_features(self) -> np.ndarray:
        """
        Extract feature matrix
        
        Returns:
            Feature array: [voltage, current, temperature, cycle_count, capacity]
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        features = self.data[['voltage', 'current', 'temperature', 'cycle', 'capacity']].values
        return features
    
    def get_targets(self) -> np.ndarray:
        """
        Extract target variables
        
        Returns:
            Target array: [SOH, Range, Degradation Rate]
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        targets = self.data[['soh', 'range_km', 'degradation_rate']].values
        return targets
    
    def calculate_rul(self, soh_threshold: float = 0.7) -> pd.DataFrame:
        """
        Calculate Remaining Useful Life (RUL) for each battery
        
        Args:
            soh_threshold: SOH threshold below which battery is considered end-of-life
            
        Returns:
            DataFrame with RUL added
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        df = self.data.copy()
        df['rul'] = np.nan
        
        for battery_id in df['battery_id'].unique():
            battery_data = df[df['battery_id'] == battery_id].copy()
            battery_data = battery_data.sort_values('cycle')
            
            # Find where SOH drops below threshold
            eol_cycle = battery_data[battery_data['soh'] <= soh_threshold]['cycle']
            
            if len(eol_cycle) > 0:
                eol_cycle = eol_cycle.iloc[0]  # First cycle below threshold
                # RUL = cycles until EOL
                battery_data['rul'] = eol_cycle - battery_data['cycle']
                battery_data['rul'] = battery_data['rul'].clip(lower=0)  # No negative RUL
            else:
                # If never reaches threshold, RUL is cycles remaining to max cycle
                max_cycle = battery_data['cycle'].max()
                battery_data['rul'] = max_cycle - battery_data['cycle']
            
            df.loc[df['battery_id'] == battery_id, 'rul'] = battery_data['rul'].values
        
        self.data = df
        return df
    
    def save_processed_data(self, output_path: str):
        """
        Save processed data to file
        
        Args:
            output_path: Path to save processed data
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self.data.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    loader = BatteryDataLoader("data/raw")
    df = loader.load_data()
    print(f"Loaded {len(df)} records")
    print(df.head())
    print(f"\nFeatures shape: {loader.get_features().shape}")
    print(f"Targets shape: {loader.get_targets().shape}")


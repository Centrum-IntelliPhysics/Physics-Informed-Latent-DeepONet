import torch
import numpy as np
from sklearn.model_selection import train_test_split
import math

def convert_metadata_to_torch(npz_data, device):
    torch_metadata = {}

    # Convert each field in the metadata to a PyTorch tensor and move to the specified device
    for key, value in npz_data.items():
        if key == 'shapes':
            # Process the 'shapes' field (dictionary-like object)
            shapes_dict = value.item()  # Extract the dictionary
            # Convert the dictionary values to PyTorch tensors where possible
            torch_metadata[key] = {
                shape_key: torch.tensor(shape_value).float().to(device) if shape_value is not None else None
                for shape_key, shape_value in shapes_dict.items()
            }
        else:
            # For other fields, convert them directly to a tensor
            torch_metadata[key] = torch.tensor(value, dtype=torch.float32).to(device)
    
    return torch_metadata
#======================================================================================================================#

# Define the shape-to-numeric mapping
shape_map = {
    "circle": 0,
    "half-circle": 1,
    "isosceles-triangle": 2,
    "right-angled-triangle": 3,
    "rectangle": 4,
    "rhombus": 5,
    "polygon-3": 6,
    "polygon-4": 7,
    "polygon-5": 8,
    "polygon-6": 9
}

def convert_groupdict_to_torch(npz_data, device):
    key = list(npz_data.keys())[0]  # Get the first key
    data_dict = npz_data[key].item()

    # Initialize dictionary to hold PyTorch tensors
    torch_data_dict = {}

    # Iterate through the groups and process each group's data
    for group_idx, group_data in data_dict.items():
        # print(f"Converting Group {group_idx} to PyTorch tensors...")

        # Map Input Parameters into (15, 4) format
        input_parameters = []
        for params in group_data['input_parameters']:
            shape, num_sides, r, a_value = params
            mapped_row = [shape_map[shape], num_sides, r, a_value]
            input_parameters.append(mapped_row)

        input_parameters_tensor = torch.tensor(input_parameters, dtype=torch.float32).to(device)
        input_samples_tensor = torch.from_numpy(group_data['input_samples']).float().to(device)

        # Store the converted tensors into the new dictionary
        group_tensors = {
            'input_parameters': input_parameters_tensor,  # Shape (15, 4)
            'input_samples': input_samples_tensor
        }

        # Check if 'output_samples' exists and convert it if present
        if 'output_samples' in group_data:
            output_samples_tensor = torch.from_numpy(group_data['output_samples']).float().to(device)
            group_tensors['output_samples'] = output_samples_tensor

        # Store the processed group data in the dictionary
        torch_data_dict[group_idx] = group_tensors

        # Details
        # print(f"  Input Parameters shape: {input_parameters_tensor.shape}, device: {input_parameters_tensor.device}")
        # print(f"  Input Samples shape: {input_samples_tensor.shape}, device: {input_samples_tensor.device}")
        # if 'output_samples' in group_data:
        #    print(f"  Output Samples shape: {output_samples_tensor.shape}, device: {output_samples_tensor.device}")
        # print("-" * 50)
    
    return torch_data_dict

def check_shapes(torch_data_dict):

    for group_idx, group_data in torch_data_dict.items():
        print(f"Group {group_idx}:")
        print(f"  Input Parameters shape: {group_data['input_parameters'].shape}")
        print(f"  Input Samples shape: {group_data['input_samples'].shape}")
        if 'output_samples' in group_data:
            print(f"  Output Samples shape: {group_data['output_samples'].shape}")
        if group_idx == 1:
            break
#======================================================================================================================#

def split_groups(stove_full_solution_fields_groups, seed, train_size=50, test_size=10):
    """
    Splits the groups into training and testing sets.

    Parameters:
        stove_full_solution_fields_groups (dict): Dictionary containing the full solution fields.
        train_size (int): Number of groups to use for training.
        test_size (int): Number of groups to use for testing.

    Returns:
        dict: A tuple containing training and testing sets.
    """
    # Extract the group indices
    group_indices = list(stove_full_solution_fields_groups.keys())

    # Shuffle the group indices randomly
    train_indices, test_indices = train_test_split(group_indices, train_size=train_size, test_size=test_size, random_state=seed)

    # Split the data based on the shuffled indices
    train_data = {k: stove_full_solution_fields_groups[k] for k in train_indices}
    test_data = {k: stove_full_solution_fields_groups[k] for k in test_indices}

    return train_data, test_data

def print_group_shapes(group, num_groups=2):
    print("Group Data Shapes:")
    count = 0
    for group_idx, group_data in group.items():
        if count >= num_groups:
            break
        print(f"Group {group_idx}:")
        print(f"  Input Parameters shape: {group_data['input_parameters'].shape}")
        print(f"  Input Samples shape: {group_data['input_samples'].shape}")
        print(f"  Output Samples shape: {group_data['output_samples'].shape}")
        print("-" * 50)
        count += 1
#======================================================================================================================#

def load_and_combine_groups(groups, title, combine=True, device='cpu'):
    """
    Load and combine data from multiple groups, and transfer tensors to the specified device.

    Args:
        groups (dict): Dictionary containing groups of data (input_parameters, input_samples, output_samples).
        title(string): Title.
        combine (bool): Whether to combine the data across groups.
        device (str or torch.device): The device to transfer the tensors to (e.g., 'cuda' or 'cpu').

    Returns:
        dict or None: Combined data dictionary with tensors on the specified device, or None if combine is False.
    """
    
    # Initialize lists to store combined data
    all_input_parameters = []
    all_input_samples = []
    all_output_samples = []

    # Iterate through each group
    for group_idx, group_data in groups.items():
        # print(group_idx)
        # Collect data for combining later
        all_input_parameters.append(group_data['input_parameters'])
        all_input_samples.append(group_data['input_samples'])
        if 'output_samples' in group_data:
            all_output_samples.append(group_data['output_samples'])

    # Combine data if needed
    if combine:
        # Convert lists to tensors and concatenate across the first axis
        combined_data = {
            'input_parameters': torch.cat([torch.tensor(data, dtype=torch.float32).to(device) for data in all_input_parameters], dim=0),
            'input_samples': torch.cat([torch.tensor(data, dtype=torch.float32).to(device) for data in all_input_samples], dim=0)
        }
        
        if all_output_samples:  # Only add output_samples if they exist
            combined_data['output_samples'] = torch.cat([torch.tensor(data, dtype=torch.float32).to(device) for data in all_output_samples], dim=0)
        
        # Print the shapes of the combined data
        print(title+" Data Shapes:")
        print(f"  Input Parameters Shape: {combined_data['input_parameters'].shape}")
        print(f"  Input Samples Shape: {combined_data['input_samples'].shape}")
        if 'output_samples' in combined_data:
            print(f"  Output Samples Shape: {combined_data['output_samples'].shape}")
        
        return combined_data
    else:
        return None
#======================================================================================================================#

class Source:
    def __init__(self, a=5.0, r=1.0, x=None, y=None, xc=None, yc=None, device ='cpu'):
        self.a = a  # Intensity factor for shape contour
        self.r = r  # Size of the shape (radius, side length, etc.)
        self.xc = xc  # Center x-coordinate for source  
        self.yc = yc  # Center y-coordinate for source  
        self.x = x - self.xc 
        self.y = y - self.yc 
        self.device = device

    def circle(self):
        """Circle as the source term."""
        return torch.exp(-torch.abs(self.a * (torch.sqrt(self.x**2 + self.y**2) - self.r)))
    
    def half_circle(self):
        """Half-circle as the source term centered at (0, 0)."""
        # Shift coordinates to center the half-circle at (0, 0)
        x_shifted = self.x
        y_shifted = self.y + self.r / 2  # Move the top of the circle down to center its centroid

        # Define the half-circle boundary
        circle_boundary = torch.sqrt(x_shifted**2 + y_shifted**2) - self.r  # Circle boundary
        half_circle_boundary = torch.maximum(circle_boundary, -y_shifted)  # Limit to top half (y >= 0)

        return torch.exp(-torch.abs(self.a * half_circle_boundary))

    def isosceles_triangle(self):
        """Isosceles triangle as the source term."""
        triangle_boundary = torch.maximum(
            torch.maximum(
                self.y - (torch.sqrt(torch.tensor(3.0)) * self.x + self.r) / 2,  # Line: y = sqrt(3)*x + r/2
                self.y - (-torch.sqrt(torch.tensor(3.0)) * self.x + self.r) / 2  # Line: y = -sqrt(3)*x + r/2
            ),
            -self.y - self.r / 2  # Line: y = -r/2 (horizontal bottom line)
        )
        return torch.exp(-torch.abs(self.a * triangle_boundary))
    
    def right_angle_triangle(self):
        """Right-angled isosceles triangle as the source term centered at (0, 0)."""
        # Shift the coordinate system so the centroid is at (0, 0)
        x_shifted = self.x + self.r / 3
        y_shifted = self.y + self.r / 3

        # Define the triangle boundary in the shifted coordinate system
        triangle_boundary = torch.maximum(
            torch.maximum(-y_shifted, -x_shifted),  # Bottom edge (y = 0) and left edge (x = 0)
            x_shifted + y_shifted - self.r  # Hypotenuse (x + y = r)
        )

        return torch.exp(-torch.abs(self.a * triangle_boundary))
    
    def rectangle(self):
        """Rectangle as the source term."""
        r_x, r_y = self.r, self.r / 2  # Half-width and half-height of the rectangle
        return torch.exp(-torch.abs(self.a * (torch.maximum(torch.abs(self.x) / r_x, torch.abs(self.y) / r_y) - 1)))
    
    def square(self):
        """Square as the source term."""
        return torch.exp(-torch.abs(self.a * (torch.maximum(torch.abs(self.x), torch.abs(self.y)) - self.r / 2)))
    
    def rhombus(self):
        """Rhombus as the source term."""
        return torch.exp(-torch.abs(self.a * (torch.abs(self.x) + torch.abs(self.y) - self.r)))
    
    @staticmethod
    def signed_distance(x, y, x1, y1, x2, y2):
        """Compute signed distance from a point (x, y) to a line segment (x1, y1) to (x2, y2)."""
        px, py = x2 - x1, y2 - y1
        norm = px ** 2 + py ** 2
        u = torch.clamp(((x - x1) * px + (y - y1) * py) / norm, 0, 1)
        dx, dy = x1 + u * px - x, y1 + u * py - y
        return torch.sqrt(dx ** 2 + dy ** 2)

    def polygon(self, num_sides):
        """Regular polygon with a given number of sides centered at (xc, yc)."""
        base_angles = torch.linspace(0, 2 * math.pi, num_sides+1)[:-1]
        rotation_angle = math.pi / num_sides  # Rotate to align one side horizontally
        angles = (base_angles + rotation_angle).to(self.device)

        # Compute vertices, accounting for center (xc, yc)
        vertices = torch.stack([self.r * torch.cos(angles), self.r * torch.sin(angles)], dim=1)
        vertices = torch.cat([vertices, vertices[None, 0]], dim=0)  # Closing the polygon

        # Initialize distance array
        distance = torch.full_like(self.x, float("inf"))
        for i in range(num_sides):  
            x1, y1 = vertices[i]
            x2, y2 = vertices[i + 1]
            distance = torch.minimum(distance, self.signed_distance(self.x, self.y, x1, y1, x2, y2))

        return torch.exp(-self.a * distance), vertices
    
    def type_source(self, shape, num_sides=None):
        """General method to calculate source term based on shape."""
        shape_methods = {
            "circle": self.circle(),
            "half-circle": self.half_circle(),
            "isosceles-triangle": self.isosceles_triangle(),
            "right-angled-triangle": self.right_angle_triangle(),
            "rectangle": self.rectangle(),
            "square": self.square(),
            "rhombus": self.rhombus(),
        }
        
        if shape in shape_methods:
            return shape_methods[shape]
        
        # Handle polygons separately
        if shape.startswith("polygon"):
            if num_sides is None:
                raise ValueError(f"Polygon shapes require the number of sides (e.g., num_sides=3, 4, 5, 6, 7, 8, 9, 10, etc.)")
            return self.polygon(num_sides)[0]
        
        # If shape is not in the recognized methods, raise an error
        raise ValueError(f"Unsupported shape: {shape}")

# Function to find the key based on value
def get_key_from_value(d, value):
    return next((key for key, val in d.items() if val == value), None)
#======================================================================================================================#



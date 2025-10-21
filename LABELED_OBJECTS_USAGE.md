# Labeled Object Management - Usage Guide

The labeled object management functionality has been integrated into the existing codebase. Each labeled object (e.g., each ball) is represented as a separate `LabeledObject` class instance that contains both control points and Gaussians with the same label. The `LabeledObjectManager` manages multiple `LabeledObject` instances and allows you to apply 6DOF transformations to move and rotate entire objects.

## Integration Points

### 1. `scene/gaussian_model.py`
- **`LabeledObject` class**: Represents a single labeled object (e.g., one ball) containing control points and Gaussians with the same label
- **`LabeledObjectManager` class**: Manages multiple `LabeledObject` instances
- **`create_labeled_object_manager()` method**: Added to `GaussianModel` class for easy creation

### 2. `train_gui.py`
- **Interactive manipulation methods**: Added to `GUI` class
- **Automatic integration**: Creates object manager after segmentation
- **Real-time updates**: Transforms are applied to actual models

## Usage

### Basic Setup

After segmentation is complete (when you have labeled control points and Gaussians), the system automatically creates a labeled object manager:

```python
# This happens automatically in train_node_rendering_step() after segmentation
gui.create_labeled_object_manager()
```

The `LabeledObjectManager` creates a separate `LabeledObject` instance for each unique label. Each `LabeledObject` stores:
- References to control points and Gaussians with the same label
- Transformation parameters (translation and rotation) shared by all points in that object
- Methods to get/set positions and apply transformations

This architecture allows independent manipulation of each object (ball) while sharing parameters within each object.

### Interactive Manipulation

Once the object manager is created, you can manipulate objects:

```python
# Move Ball 1 by (2, 0, 0)
gui.translate_object(1, 2.0, 0.0, 0.0)

# Rotate Ball 2 by 90 degrees around Z-axis
gui.rotate_object(2, 'z', 90.0)

# Set absolute position of Ball 1
gui.set_object_position(1, 3.0, 1.0, 0.0)

# Reset Ball 1 to original position
gui.reset_object(1)

# Reset all objects
gui.reset_all_objects()

# Check current status
gui.print_object_status()
```

### Object Replacement

You can replace entire `LabeledObject` instances by copying all properties from one object to another:

```python
# Replace entire red ball (label 1) with blue ball (label 3)
# This copies all properties (positions, features, scaling, rotation, opacity, transformations)
# from the blue ball LabeledObject to the red ball LabeledObject
gui.replace_red_ball_with_blue_ball()

# Replace entire blue ball (label 3) with red ball (label 1)
gui.replace_blue_ball_with_red_ball()

# Replace any object with another
gui.replace_object_with_another(from_label=1, to_label=2)  # Replace red with green

# Get current object information
gui.get_object_info()
```

**How it works:**
- Gets the `LabeledObject` instances for both labels
- **Deletes the old `LabeledObject`** (from_label)
- **Creates a new `LabeledObject`** with the same label and indices
- Copies all properties from the 'to' object to the new object:
  - Control point properties (xyz, features, scaling, rotation, opacity)
  - Gaussian properties (xyz, features, scaling, rotation, opacity)
  - Object center and transformation parameters
- Replaces the old object in the manager with the new object
- The 'to' object remains unchanged

**Why erase and recreate?**
- Cleaner memory management - no lingering references
- Fresh object instance with correct initialization
- More explicit about the replacement operation

### Programmatic Usage

You can also use the object manager and individual objects directly:

```python
# Get the object manager
manager = gui.object_manager

# Get a specific LabeledObject instance
obj = manager.get_object(1)  # Get red ball (label 1)

# Access object properties
control_xyz = obj.get_control_xyz()
gaussian_xyz = obj.get_gaussian_xyz()
print(f"Object center: {obj.center}")
print(f"Translation: {obj.translation}")
print(f"Rotation: {obj.rotation}")

# Apply transformations (automatically updates the Gaussian models)
manager.set_transformation(1, translation_vector, rotation_quaternion)

# Apply transformations to all objects
manager.apply_all_transformations()

# Get all object labels
labels = manager.get_all_labels()

# Get object information
info = manager.get_object_info()
```

Each `LabeledObject` is a separate `nn.Module` with its own parameters, allowing for independent manipulation and parameter sharing within each object.

### Learning 6DOF Transformations

You can learn a 6DOF transformation to move one object to another's position:

```python
# Learn transformation at the LabeledObject level
# The learned parameters (translation, rotation) are shared by ALL Gaussians and control points
gui.learn_object_transformation(from_label=1, to_label=3, num_iterations=100, lr=0.01)

# Copy learned transformation from one object to another
gui.copy_object_transformation(from_label=2, to_label=1)

# Or use the object manager directly
obj1 = gui.object_manager.get_object(1)
obj3 = gui.object_manager.get_object(3)
loss_history = obj1.learn_transformation_to_target(obj3, num_iterations=100, lr=0.01)
```

**Key Benefits:**
- Operates at the **LabeledObject level** - not individual point level
- **Shared parameters**: One set of transformation parameters for all points in the object
- **Efficient optimization**: Updates all Gaussians and control points together
- **Differentiable**: Can be integrated into training loops

## Key Features

1. **Separate Class Instances**: Each labeled object is a separate `LabeledObject` class instance
2. **Parameter Sharing**: Control points and Gaussians within the same object share transformation parameters
3. **6DOF Transformations**: 3D translation + 3D rotation (quaternion-based)
4. **Direct Model Integration**: Works directly with Gaussian model classes, not just positions
5. **Real-time Updates**: Changes are immediately applied to the actual model data
6. **Interactive Control**: Easy manipulation through simple method calls
7. **Object Replacement**: Replace entire objects (positions, properties, transformations)

## Example Workflow

1. **Run training** with segmentation enabled
2. **After segmentation completes**, the object manager is automatically created
3. **Manipulate objects** using the provided methods
4. **Replace entire objects** (e.g., replace red ball with blue ball)
5. **View results** in the GUI or save the transformed models

## Integration with Training

The system is integrated into the training process:

```python
# In train_node_rendering_step(), after segmentation:
if self.iteration_node_rendering == self.opt.iterations_node_rendering - 1:
    # ... segmentation code ...
    
    # Create labeled object manager
    self.create_labeled_object_manager()
    
    # Example: Move ball 1 by (2, 0, 0)
    if hasattr(self, 'object_manager') and self.object_manager is not None:
        self.translate_object(1, 2.0, 0.0, 0.0)
        self.print_object_status()
    
    # Replace red ball with blue ball
    self.replace_red_ball_with_blue_ball()
    self.get_object_info()
```

## Benefits

- **Unified Manipulation**: Move entire objects (control points + Gaussians) together
- **6DOF Control**: Full 3D translation and rotation
- **Label-based Organization**: Automatically groups related points
- **Real-time Updates**: Changes are immediately visible
- **Easy Integration**: Works with existing training pipeline

## Notes

- The system automatically excludes background points (label 0)
- Transformations are applied around object centers
- Quaternion-based rotations avoid gimbal lock
- All operations are differentiable for gradient-based optimization
- The system is integrated into the existing codebase without creating new files
- **Fixed**: The `torch.cuda.FloatTensor is not a Module subclass` error has been resolved by properly registering parameters with the module

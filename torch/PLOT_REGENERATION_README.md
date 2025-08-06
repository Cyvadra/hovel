# Plot Regeneration Functionality

This document describes the new plot regeneration functionality added to `test_model_sizes.py`.

## Overview

The script now supports regenerating `plot_predictions` for all model sizes when started with specific command-line flags. This is useful when you want to:

- Update plot styles or formats without retraining models
- Regenerate plots for models that were trained previously
- Fix plot generation issues without the computational cost of retraining

## New Command-Line Arguments

### `--regenerate-plots`
Regenerate plots for all existing models without retraining. This flag will:
- Check for existing model files in the current directory
- Load each trained model
- Regenerate both loss plots and prediction plots
- Skip models that don't have saved loss data

### `--plot-only`
Only regenerate plots, do not train new models. This is useful when you only want to update existing plots.

### `--regenerate-pattern`
Specify a glob pattern to match model files for plot regeneration. Default is `"model_*_layers_*_best_model.pth"`.

### `--hidden-sizes` and `--num-layers`
Specify which model sizes to check for regeneration. These work with `--regenerate-plots`.

## Usage Examples

### 1. Regenerate plots for all existing models
```bash
python test_model_sizes.py --plot-only
```

### 2. Regenerate plots for specific model sizes
```bash
python test_model_sizes.py --regenerate-plots --hidden-sizes 1024 256 --num-layers 16 4
```

### 3. Regenerate plots using a custom pattern
```bash
python test_model_sizes.py --plot-only --regenerate-pattern "my_model_*_best.pth"
```

### 4. Regenerate plots and then continue with normal training
```bash
python test_model_sizes.py --regenerate-plots --hidden-sizes 1024 256 --num-layers 16 4
```

### 5. Normal training mode (unchanged)
```bash
python test_model_sizes.py --hidden-sizes 1024 256 --num-layers 16 4
```

## New Functions Added

### `regenerate_all_plots(hidden_sizes, num_layers)`
Regenerates plots for all existing models with the specified hidden sizes and layer counts.

### `regenerate_plots_by_pattern(pattern)`
Regenerates plots for all models matching a specific glob pattern.

### `save_loss_data(train_losses, val_losses, model_name)`
Saves training and validation loss data to a JSON file for later use.

### `load_loss_data(model_name)`
Loads training and validation loss data from a JSON file.

## File Structure

When training models, the script now creates these additional files:
- `{model_name}_loss_data.json` - Contains training and validation loss history
- `{model_name}_predictions.png` - Prediction plots (regenerated)
- `{model_name}_loss_analysis.png` - Loss analysis plots (regenerated)

## Error Handling

The regeneration functions include robust error handling:
- Models without saved loss data will skip loss plot generation
- Missing model files are reported but don't stop the process
- Individual model failures don't affect other models
- Detailed summaries show which models were processed successfully

## Testing

A test script `test_plot_regeneration.py` is provided to demonstrate all the new functionality:

```bash
python test_plot_regeneration.py
```

This will run through various scenarios and show the expected output.

## Benefits

1. **Time Efficiency**: No need to retrain models just to update plots
2. **Flexibility**: Can regenerate plots for any subset of existing models
3. **Robustness**: Handles missing files and data gracefully
4. **Consistency**: Ensures all plots use the same style and format
5. **Debugging**: Easy to regenerate plots if there were issues during initial generation

## Notes

- Loss data is automatically saved during training for future plot regeneration
- The script will create loss data files for new models but can work with existing models that don't have loss data
- GPU memory is managed efficiently during regeneration
- All existing functionality remains unchanged 